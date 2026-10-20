//! zai-sub — Gray provider sidecar for Z.AI (GLM).
//!
//! Host speaks OpenAI-Responses to a per-turn loopback relay
//! (`acp_bridge::relay`); `openai_compat` translates to the upstream
//! OpenAI-compatible chat-completions endpoint.
//!
//! Config: ~/.gray/zai/config.json  {"base_url","api_key",
//!         "accounts":[{label,key,…}],"strategy","preferred"}
//! Env:    ZAI_BASE_URL (full …/v1 base) · ZAI_API_KEY

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use acp_bridge::chat;
use acp_bridge::manifest as pmanifest;
use acp_bridge::relay::{self, Intents, TurnFn};
use openai_compat::{
    compat_turn, credential_key, probe_models, to_chat_request, to_turn_result, Compat,
};
use serde_json::{json, Value};

const NAME: &str = "zai";
const PROVIDER_ID: &str = "zai";
const DISPLAY: &str = "Z.AI (GLM)";
const AUTH_ID: &str = "zai-api-key";
const AUTH_NAME: &str = "Z.AI API key (ZAI_API_KEY)";
const TOKEN_PREFIX: &str = "zai-sub-";
const DEFAULT_BASE: &str = "https://api.z.ai/api/paas/v4";
const ENV_BASE: &str = "ZAI_BASE_URL";
const ENV_KEY: &str = "ZAI_API_KEY";
/// Z.AI's implicit prefix cache is node-local and undocumented in
/// lifetime — measured evicting within minutes under load; 5 minutes is
/// the industry-norm window the host's warming replay refreshes at.
const CACHE_TTL_SECS: u64 = 300;

/// Upstream POST timeout — same as `Compat`'s default.
const POOL_TIMEOUT: Duration = Duration::from_secs(300);
/// Cooldown after an upstream 401 (seconds).
const TTL_401: u64 = 300;
/// Cooldown after an upstream 429 (seconds).
const TTL_429: u64 = 3600;
/// Sole-account pools cap any cooldown here so the real error stays
/// visible instead of hiding behind "all accounts locked".
const SOLE_CAP: u64 = 60;

/// Static catalog used only when the /models probe can't reach a server.
const FALLBACK_MODELS: &[&str] = &[
    "glm-4.6", "glm-4.5", "glm-4.5-air", "glm-4.5-flash",
];

fn manifest() -> Value {
    json!({
        "name": NAME,
        "display_name": DISPLAY,
        "version": env!("CARGO_PKG_VERSION"),
        "protocol": "1.2",
        "capabilities": ["provider.credentials"],
        "providers": [pmanifest::provider_decl_http(
            PROVIDER_ID, DISPLAY, AUTH_ID, AUTH_NAME, "api_key",
            &["models", "chat"], CACHE_TTL_SECS,
        )],
        "commands": ["/zai"],
        "subcommands": ["zai"],
    })
}

// ---------- config ----------

fn state_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("GRAY_HOME") {
        return std::path::PathBuf::from(dir).join(NAME);
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(std::path::PathBuf::from)
        // No user home: a cwd-relative config would scribble key
        // material into whatever directory the host ran from.
        .unwrap_or_else(std::env::temp_dir);
    home.join(".gray").join(NAME)
}

fn config_path() -> std::path::PathBuf {
    state_dir().join("config.json")
}

fn read_config() -> Value {
    std::fs::read_to_string(config_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}))
}

fn write_config(v: &Value) -> std::io::Result<()> {
    std::fs::create_dir_all(state_dir())?;
    std::fs::write(config_path(), format!("{}\n", serde_json::to_string_pretty(v)?))?;
    // Key material may live here — lock it down.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(config_path(), std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// flock on a sibling lockfile; the returned File's drop releases it.
#[cfg(unix)]
fn config_lock() -> Option<std::fs::File> {
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn flock(fd: i32, op: i32) -> i32;
    }
    const LOCK_EX: i32 = 2;
    std::fs::create_dir_all(state_dir()).ok()?;
    let f = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(state_dir().join("config.lock"))
        .ok()?;
    unsafe {
        flock(f.as_raw_fd(), LOCK_EX);
    }
    Some(f)
}

#[cfg(not(unix))]
fn config_lock() -> Option<std::fs::File> {
    None
}

/// Locked read-modify-write of config.json — relay threads marking
/// cooldowns and CLI `account` verbs both mutate pool state.
fn edit_config<R>(f: impl FnOnce(&mut Value) -> R) -> R {
    let _lock = config_lock();
    let mut cfg = read_config();
    let r = f(&mut cfg);
    let _ = write_config(&cfg);
    r
}

/// Persist a host-delivered credential so `provider/chat` (which carries
/// none) can still authenticate. Never logs the value.
fn cache_credential(params: &Value) {
    if let Some(key) = credential_key(params) {
        let mut cfg = read_config();
        cfg["api_key"] = json!(key);
        let _ = write_config(&cfg);
    }
}

// ---------- account pool ----------

/// One resolved pool entry: label + usable key material.
struct Account {
    label: String,
    key: String,
}

fn accounts_of(cfg: &Value) -> Vec<Value> {
    cfg["accounts"].as_array().cloned().unwrap_or_default()
}

/// Pool is authoritative once `accounts` is a non-empty array; the env
/// key still overrides everything (ad-hoc key, same as today).
fn pool_mode(cfg: &Value) -> bool {
    std::env::var(ENV_KEY).ok().filter(|s| !s.is_empty()).is_none()
        && cfg["accounts"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false)
}

/// Resolve an entry's `key`: raw key material, or `env:NAME` /
/// `file:PATH` indirections.
fn entry_key(e: &Value) -> Option<String> {
    let raw = e["key"].as_str()?;
    if let Some(name) = raw.strip_prefix("env:") {
        return std::env::var(name).ok().filter(|s| !s.is_empty());
    }
    if let Some(path) = raw.strip_prefix("file:") {
        return std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
    }
    (!raw.is_empty()).then(|| raw.to_string())
}

/// Ordered candidates for one relay mint: cooled accounts (`reset_at`
/// in the future) and key-less entries drop out; `preferred` pins
/// first; then `strategy` — `fill_first` (array order, default) or
/// `round_robin` (least recently used first).
fn pick_accounts(cfg: &Value) -> Vec<Account> {
    let now = chat::now_secs();
    let pref = cfg["preferred"].as_str();
    let mut live: Vec<Value> = accounts_of(cfg)
        .into_iter()
        .filter(|e| e["reset_at"].as_u64().unwrap_or(0) <= now)
        .filter(|e| entry_key(e).is_some())
        .collect();
    if cfg["strategy"].as_str() == Some("round_robin") {
        live.sort_by_key(|e| e["last_used_at"].as_u64().unwrap_or(0));
    }
    live.sort_by_key(|e| usize::from(e["label"].as_str() != pref));
    live.into_iter()
        .filter_map(|e| {
            Some(Account {
                label: e["label"].as_str()?.to_string(),
                key: entry_key(&e)?,
            })
        })
        .collect()
}

/// Bench one account after an upstream 401/429. Retry-After (already
/// parsed to seconds) wins over the computed TTL; a sole-account pool
/// is capped short so the error resurfaces quickly.
fn mark_cooled(label: &str, http: u16, retry_after: Option<u64>, msg: &str) {
    edit_config(|cfg| {
        let sole = accounts_of(cfg).len() <= 1;
        let ttl = retry_after.unwrap_or(if http == 429 { TTL_429 } else { TTL_401 });
        let ttl = if sole { ttl.min(SOLE_CAP) } else { ttl };
        if let Some(e) = cfg["accounts"]
            .as_array_mut()
            .and_then(|a| a.iter_mut().find(|e| e["label"].as_str() == Some(label)))
        {
            e["status"] = json!("cooled");
            e["reset_at"] = json!(chat::now_secs() + ttl);
            let mut m = msg.to_string();
            m.truncate(160);
            e["last_error"] = json!(format!("HTTP {http}: {m}"));
        }
    });
}

/// Success: clear the account's bench and bump its usage stats.
fn mark_used(label: &str) {
    edit_config(|cfg| {
        if let Some(e) = cfg["accounts"]
            .as_array_mut()
            .and_then(|a| a.iter_mut().find(|e| e["label"].as_str() == Some(label)))
        {
            e["status"] = json!("ok");
            e["reset_at"] = json!(0);
            e["last_used_at"] = json!(chat::now_secs());
            let n = e["request_count"].as_u64().unwrap_or(0);
            e["request_count"] = json!(n + 1);
            if let Some(m) = e.as_object_mut() {
                m.remove("last_error");
            }
        }
    });
}

struct PoolErr {
    msg: String,
    /// Upstream says this account is out (401/429) — cool it and fail
    /// over. Any other failure is request-scoped and never cools.
    cool: bool,
    http: u16,
    retry_after: Option<u64>,
}

fn pool_err(msg: String) -> PoolErr {
    PoolErr { msg, cool: false, http: 0, retry_after: None }
}

/// Same POST as `openai_compat::post_chat` but keeps the status code
/// and Retry-After so the pool can bench the right account.
fn post_pooled(base: &str, key: &str, body: &Value) -> Result<Value, PoolErr> {
    let url = format!("{base}/chat/completions");
    let req = ureq::post(&url)
        .timeout(POOL_TIMEOUT)
        .set("Content-Type", "application/json")
        .set("Authorization", &format!("Bearer {key}"));
    match req.send_json(body) {
        Ok(r) => r
            .into_json()
            .map_err(|e| pool_err(format!("upstream JSON decode: {e}"))),
        Err(ureq::Error::Status(code, resp)) => {
            let retry_after = resp
                .header("retry-after")
                .and_then(|v| v.trim().parse::<u64>().ok());
            let mut text = resp.into_string().unwrap_or_default();
            if text.len() > 600 {
                let mut end = 600;
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                text.truncate(end);
                text.push('…');
            }
            Err(PoolErr {
                msg: format!("upstream HTTP {code}: {text}"),
                cool: matches!(code, 401 | 429),
                http: code,
                retry_after,
            })
        }
        Err(e) => Err(pool_err(format!("upstream: {e}"))),
    }
}

/// Pooled `TurnFn`: the same Responses→chat-completions flow as
/// `openai_compat::compat_turn`, but the upstream POST walks the mint's
/// candidate list — a 401/429 benches that account and the next
/// candidate serves the same request.
fn pool_turn(candidates: Vec<Account>, base: String) -> TurnFn {
    Arc::new(move |raw: &[u8], cancel: &AtomicBool| {
        if cancel.load(Ordering::SeqCst) {
            return Err("turn cancelled".into());
        }
        let body: Value =
            serde_json::from_slice(raw).map_err(|e| format!("request JSON: {e}"))?;
        let wire_model = body["model"].as_str().unwrap_or_default().to_string();
        let prepared = chat::prepare_turn(&body, &wire_model)?;
        let req = to_chat_request(&prepared, &prepared.model, &body);
        if cancel.load(Ordering::SeqCst) {
            return Err("turn cancelled".into());
        }
        let mut last = "no pool accounts available".to_string();
        for cand in &candidates {
            match post_pooled(&base, &cand.key, &req) {
                Ok(resp) => {
                    mark_used(&cand.label);
                    let result = to_turn_result(&resp);
                    let id = format!("resp_{}", chat::rand_hex(12));
                    return Ok(chat::fold_result(&result, &prepared.model, &id));
                }
                Err(e) if e.cool => {
                    mark_cooled(&cand.label, e.http, e.retry_after, &e.msg);
                    last = e.msg;
                }
                Err(e) => return Err(e.msg),
            }
        }
        Err(last)
    })
}

fn resolve_base(file: &Value) -> String {
    std::env::var(ENV_BASE)
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| file["base_url"].as_str().map(str::to_string))
        .unwrap_or_else(|| DEFAULT_BASE.to_string())
}

/// Key ladder for probes: env override → first live pool account →
/// cached `api_key` (host envelope / `setup --key`).
fn resolve() -> Compat {
    let file = read_config();
    let base = resolve_base(&file);
    let key = std::env::var(ENV_KEY)
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| pick_accounts(&file).into_iter().next().map(|a| a.key))
        .or_else(|| file["api_key"].as_str().map(str::to_string));
    Compat::new(base, key)
}

// ---------- provider ops ----------

fn err(code: &str, msg: impl Into<String>, retryable: bool) -> Value {
    json!({"code": code, "message": msg.into(), "retryable": retryable,
           "terminal": !retryable})
}

fn provider_models(params: &Value) -> Result<Value, Value> {
    cache_credential(params);
    let mut cfg = resolve();
    if cfg.api_key.is_none() {
        cfg.api_key = credential_key(params);
    }
    match probe_models(&cfg) {
        Ok(rows) if !rows.is_empty() => Ok(json!({"models": rows})),
        Ok(_) | Err(_) => {
            let rows: Vec<Value> = FALLBACK_MODELS
                .iter()
                .map(|id| json!({"id": id, "name": id}))
                .collect();
            if rows.is_empty() {
                Err(err("unavailable",
                    format!("couldn't list models from {}", cfg.upstream_base), true))
            } else {
                Ok(json!({"models": rows}))
            }
        }
    }
}

fn provider_chat(intents: &Intents, params: &Value) -> Result<Value, Value> {
    let model = params["model"].as_str().unwrap_or_default().to_string();
    let file = read_config();
    let base = resolve_base(&file);
    let turn: TurnFn = if pool_mode(&file) {
        let candidates = pick_accounts(&file);
        if candidates.is_empty() {
            return Err(err(
                "unavailable",
                format!(
                    "all {DISPLAY} accounts are cooling — see `{NAME}-sub account status`"
                ),
                true,
            ));
        }
        pool_turn(candidates, base)
    } else {
        let key = std::env::var(ENV_KEY)
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| file["api_key"].as_str().map(str::to_string));
        match key {
            None => {
                return Err(err(
                    "unavailable",
                    format!(
                        "no {DISPLAY} API key — set ${ENV_KEY} or run `{NAME}-sub setup --key K`"
                    ),
                    true,
                ))
            }
            Some(k) => compat_turn(Arc::new(Compat::new(base, Some(k))), None),
        }
    };
    let (relay_url, relay_token) = relay::mint_turn(intents, &model, turn, TOKEN_PREFIX)
        .map_err(|e| err("unavailable", e, true))?;
    Ok(json!({"relay_url": relay_url, "relay_token": relay_token}))
}

fn provider_revoke(_params: &Value) -> Result<Value, Value> {
    let _ = edit_config(|cfg| {
        if let Some(m) = cfg.as_object_mut() {
            m.remove("api_key");
            m.remove("accounts");
            m.remove("preferred");
        }
    });
    Ok(json!({"status": "revoked"}))
}

// ---------- command surface ----------

fn cli_help() -> String {
    format!(
        "{NAME}-sub — {DISPLAY} provider for gray\n\
         usage: {NAME}-sub manifest | sidecar | setup [--base URL] [--key K] | status | models\n\
         \x20      {NAME}-sub account add <label> [--key K|env:NAME|file:PATH] | list\n\
         \x20      {NAME}-sub account use <label>|--clear | remove <label> | status\n\
         slash:  /{NAME} status | models | setup | account …\n\
         config: ~/.gray/{NAME}/config.json · env {ENV_BASE} {ENV_KEY}"
    )
}

fn cli_status() -> String {
    let file = read_config();
    let cfg = resolve();
    let key_src = if cfg.api_key.is_some() {
        key_source_label(&file)
    } else {
        String::new()
    };
    let no_key = format!("none — set ${ENV_KEY} or `{NAME}-sub setup --key K`");
    let probe = if cfg.api_key.is_none() {
        "skipped (no key)".to_string()
    } else {
        match probe_models(&cfg) {
            Ok(rows) => format!("{} models", rows.len()),
            Err(e) => format!("unreachable ({e})"),
        }
    };
    let mut out = format!(
        "{DISPLAY}\n  base: {}\n  key:  {}\n  probe: {probe}",
        cfg.upstream_base,
        if cfg.api_key.is_some() {
            key_src.as_str()
        } else {
            no_key.as_str()
        },
    );
    if pool_mode(&file) {
        let live = pick_accounts(&file);
        let serving = live
            .first()
            .map(|a| a.label.clone())
            .unwrap_or_else(|| "none (all cooling)".to_string());
        out.push_str(&format!(
            "\n  pool: {} account(s) · strategy {} · serving {serving}",
            accounts_of(&file).len(),
            file["strategy"].as_str().unwrap_or("fill_first"),
        ));
    }
    out
}

/// Which rung of the credential ladder answered — shown in `status` so
/// "set" never leaves the user guessing where the key lives.
fn key_source_label(file: &Value) -> String {
    if std::env::var(ENV_KEY).ok().filter(|s| !s.is_empty()).is_some() {
        format!("env {ENV_KEY}")
    } else if let Some(a) = pick_accounts(file).into_iter().next() {
        format!("account {}", a.label)
    } else {
        "config api_key".to_string()
    }
}

/// Apply `f` to config under the lock and flatten the result.
fn account_edit(f: impl FnOnce(&mut Value) -> Result<String, String>) -> Result<String, String> {
    edit_config(f)
}

fn account_state_line(e: &Value) -> String {
    let now = chat::now_secs();
    let reset = e["reset_at"].as_u64().unwrap_or(0);
    let state = if reset > now {
        format!("cooling {}s", reset - now)
    } else {
        match e["status"].as_str().unwrap_or("ok") {
            "cooled" => "ok".to_string(), // bench expired; ready again
            s => s.to_string(),
        }
    };
    let mut line = format!(
        "  {:<16} {:<14} reqs {}",
        e["label"].as_str().unwrap_or("?"),
        state,
        e["request_count"].as_u64().unwrap_or(0),
    );
    if let Some(le) = e["last_error"].as_str() {
        line.push_str(&format!("  ({le})"));
    }
    line
}

fn cli_account(args: &[String]) -> Result<String, String> {
    match args.first().map(String::as_str) {
        Some("add") => {
            let label = args
                .get(1)
                .filter(|s| !s.starts_with('-'))
                .ok_or("account add needs a label")?
                .clone();
            let mut key: Option<String> = None;
            let mut it = args[2..].iter();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--key" => key = Some(it.next().ok_or("--key needs a value")?.clone()),
                    other if !other.starts_with('-') && key.is_none() => {
                        key = Some(other.to_string())
                    }
                    other => return Err(format!("unknown flag {other}")),
                }
            }
            let key = key.ok_or("account add needs a key — `--key K` (or env:NAME / file:PATH)")?;
            account_edit(|cfg| {
                if !cfg["accounts"].is_array() {
                    cfg["accounts"] = json!([]);
                }
                let arr = cfg["accounts"].as_array_mut().unwrap();
                if arr.iter().any(|e| e["label"].as_str() == Some(label.as_str())) {
                    return Err(format!("account {label:?} exists — remove it first"));
                }
                if let Some(dup) = arr
                    .iter()
                    .find(|e| e["key"].as_str() == Some(key.as_str()))
                    .and_then(|e| e["label"].as_str())
                {
                    return Err(format!("that key is already pooled as {dup:?}"));
                }
                arr.push(json!({
                    "label": label, "key": key, "status": "ok",
                    "reset_at": 0, "request_count": 0, "last_used_at": 0,
                }));
                Ok(format!("added account {label:?} ({} total)", arr.len()))
            })
        }
        Some("list") => {
            let cfg = read_config();
            let accounts = accounts_of(&cfg);
            if accounts.is_empty() {
                return Ok("no accounts — single-key mode (env/config api_key)".to_string());
            }
            let mut out = format!(
                "{} account(s) · strategy {}",
                accounts.len(),
                cfg["strategy"].as_str().unwrap_or("fill_first")
            );
            for e in &accounts {
                out.push('\n');
                out.push_str(&account_state_line(e));
            }
            Ok(out)
        }
        Some("use") => {
            let label = args.get(1).map(String::as_str);
            match label {
                None | Some("--clear") => {
                    account_edit(|cfg| {
                        if let Some(m) = cfg.as_object_mut() {
                            m.remove("preferred");
                        }
                        Ok("preferred account cleared — back to strategy order".to_string())
                    })
                }
                Some(l) => account_edit(|cfg| {
                    if !accounts_of(cfg)
                        .iter()
                        .any(|e| e["label"].as_str() == Some(l))
                    {
                        return Err(format!("no account {l:?} — `account list`"));
                    }
                    cfg["preferred"] = json!(l);
                    Ok(format!("serving account pinned to {l:?}"))
                }),
            }
        }
        Some("remove") => {
            let label = args.get(1).ok_or("account remove needs a label")?.clone();
            account_edit(|cfg| {
                let before = accounts_of(cfg).len();
                if let Some(arr) = cfg["accounts"].as_array_mut() {
                    arr.retain(|e| e["label"].as_str() != Some(label.as_str()));
                }
                if accounts_of(cfg).len() == before {
                    return Err(format!("no account {label:?}"));
                }
                if cfg["preferred"].as_str() == Some(label.as_str()) {
                    if let Some(m) = cfg.as_object_mut() {
                        m.remove("preferred");
                    }
                }
                Ok(format!("removed account {label:?}"))
            })
        }
        Some("status") => {
            let cfg = read_config();
            if !pool_mode(&cfg) {
                return Ok(
                    "single-key mode (no accounts) — `account add <label> --key K`".to_string(),
                );
            }
            let live = pick_accounts(&cfg);
            let serving = live
                .first()
                .map(|a| a.label.clone())
                .unwrap_or_else(|| "none (all cooling)".to_string());
            let mut out = format!("serving: {serving}");
            for e in &accounts_of(&cfg) {
                out.push('\n');
                out.push_str(&account_state_line(e));
            }
            Ok(out)
        }
        Some(other) => Err(format!(
            "unknown account verb {other:?} — add|list|use|remove|status"
        )),
        None => Err("account verb needed — add|list|use|remove|status".to_string()),
    }
}

fn run_cli(argv: &[String]) -> Result<String, String> {
    match argv.first().map(String::as_str) {
        Some("status") | None => Ok(cli_status()),
        Some("account") => cli_account(&argv[1..]),
        Some("models") => {
            let cfg = resolve();
            if cfg.api_key.is_none() {
                return Err(format!(
                    "no {DISPLAY} API key — set ${ENV_KEY} or run `{NAME}-sub setup --key K`"
                ));
            }
            match probe_models(&cfg) {
                Ok(rows) => Ok(rows
                    .iter()
                    .filter_map(|m| m["id"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n")),
                Err(e) => Err(e),
            }
        }
        Some("setup") => {
            let mut cfg = read_config();
            let mut it = argv[1..].iter();
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--base" => {
                        let v = it.next().ok_or("--base needs a URL")?;
                        if !v.starts_with("http://") && !v.starts_with("https://") {
                            return Err(format!("invalid --base {v:?} — expected http(s) URL"));
                        }
                        cfg["base_url"] = json!(v);
                    }
                    "--key" => cfg["api_key"] = json!(it.next().ok_or("--key needs a value")?),
                    other => return Err(format!("unknown flag {other}")),
                }
            }
            write_config(&cfg).map_err(|e| format!("write config: {e}"))?;
            Ok(cli_status())
        }
        Some("help") | Some("--help") | Some("-h") => Ok(cli_help()),
        Some(other) => Err(format!("unknown verb {other:?}\n\n{}", cli_help())),
    }
}

// ---------- wire loop ----------

fn handle(intents: &Intents, method: &str, params: &Value) -> Value {
    match method {
        "provider/models" => match provider_models(params) {
            Ok(r) => json!({"result": r}),
            Err(e) => json!({"error": e}),
        },
        "provider/chat" => match provider_chat(intents, params) {
            Ok(r) => json!({"result": r}),
            Err(e) => json!({"error": e}),
        },
        "provider/auth/refresh" => {
            cache_credential(params);
            let material = params.pointer("/credential/credential").cloned();
            match material {
                Some(m) => json!({"result": m}),
                None => json!({"error": err("invalid",
                    "refresh carried no credential material", false)}),
            }
        }
        "provider/auth/revoke" => match provider_revoke(params) {
            Ok(r) => json!({"result": r}),
            Err(e) => json!({"error": e}),
        },
        "provider/usage" => json!({"result": {"windows": []}}),
        "command/run" => {
            let argv: Vec<String> = params["argv"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
                .unwrap_or_default();
            // Bare `/<name>`: empty result — the host's provider-login
            // shortcut switches the session (subscription-plugin convention).
            if argv.is_empty() {
                return json!({"result": {}});
            }
            match run_cli(&argv) {
                Ok(text) => json!({"result": {"text": text}}),
                Err(e) => json!({"result": {"text": e}}),
            }
        }
        other => json!({"error": {"code": -32601,
            "message": format!("method not found: {other}")}}),
    }
}

fn sidecar() -> std::io::Result<()> {
    let intents = Intents::default();
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
        let method = v["method"].as_str().unwrap_or("");
        match v.get("id").cloned() {
            None if method == "plugin/shutdown" => return Ok(()),
            None => {} // notifications: never reply
            Some(id) => {
                let reply = if method == "plugin/manifest" {
                    json!({"id": id, "result": manifest()})
                } else {
                    let mut r = handle(&intents, method, &v["params"]);
                    r["id"] = id;
                    r
                };
                writeln!(out, "{reply}")?;
                out.flush()?;
            }
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `gray <name> <verb>` may or may not repeat the subcommand name.
    let args: Vec<String> = match args.first().map(String::as_str) {
        Some(NAME) => args[1..].to_vec(),
        _ => args,
    };
    match args.first().map(String::as_str) {
        Some("manifest") => println!("{}", manifest()),
        Some("sidecar") => {
            if let Err(e) = sidecar() {
                eprintln!("{NAME}-sub sidecar: {e}");
                std::process::exit(1);
            }
        }
        _ => {
            // Bare on a pipe = host spawn → sidecar; bare on a TTY = human → help.
            if args.is_empty() {
                if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
                    println!("{}", cli_help());
                    return;
                }
                if let Err(e) = sidecar() {
                    eprintln!("{NAME}-sub sidecar: {e}");
                    std::process::exit(1);
                }
                return;
            }
            match run_cli(&args) {
                Ok(text) => println!("{text}"),
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }
    }
}
