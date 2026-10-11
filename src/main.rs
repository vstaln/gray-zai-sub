//! zai-sub — Gray provider sidecar for Z.AI (GLM).
//!
//! Host speaks OpenAI-Responses to a per-turn loopback relay
//! (`acp_bridge::relay`); `openai_compat` translates to the upstream
//! OpenAI-compatible /chat/completions endpoint.
//!
//! Config: ~/.gray/zai/config.json  {"base_url","api_key"}
//! Env:    ZAI_BASE_URL (full …/v1 base) · ZAI_API_KEY

use std::io::{BufRead, Write};
use std::sync::Arc;

use acp_bridge::manifest as pmanifest;
use acp_bridge::relay::{self, Intents};
use openai_compat::{compat_turn, credential_key, probe_models, Compat};
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
    let home = std::env::var_os("GRAY_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".gray")))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(NAME)
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

/// Persist a host-delivered credential so `provider/chat` (which carries
/// none) can still authenticate. Never logs the value.
fn cache_credential(params: &Value) {
    if let Some(key) = credential_key(params) {
        let mut cfg = read_config();
        cfg["api_key"] = json!(key);
        let _ = write_config(&cfg);
    }
}

fn resolve() -> Compat {
    let file = read_config();
    let base = std::env::var(ENV_BASE).ok()
        .filter(|s| !s.is_empty())
        .or_else(|| file["base_url"].as_str().map(str::to_string))
        .unwrap_or_else(|| DEFAULT_BASE.to_string());
    let key = std::env::var(ENV_KEY).ok()
        .filter(|s| !s.is_empty())
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
    let cfg = Arc::new(resolve());
    let turn = compat_turn(cfg, None);
    let (relay_url, relay_token) = relay::mint_turn(intents, &model, turn, TOKEN_PREFIX)
        .map_err(|e| err("unavailable", e, true))?;
    Ok(json!({"relay_url": relay_url, "relay_token": relay_token}))
}

fn provider_revoke(_params: &Value) -> Result<Value, Value> {
    let mut cfg = read_config();
    cfg.as_object_mut().map(|m| m.remove("api_key"));
    let _ = write_config(&cfg);
    Ok(json!({"revoked": true}))
}

// ---------- command surface ----------

fn cli_help() -> String {
    format!(
        "{NAME}-sub — {DISPLAY} provider for gray\n\
         usage: {NAME}-sub manifest | sidecar | setup [--base URL] [--key K] | status | models\n\
         slash:  /{NAME} status | setup\n\
         config: ~/.gray/{NAME}/config.json · env {ENV_BASE} {ENV_KEY}"
    )
}

fn cli_status() -> String {
    let cfg = resolve();
    let key_src = cfg.api_key.is_some().then(key_source_label).unwrap_or_default();
    let no_key = format!("none — set ${ENV_KEY} or `{NAME}-sub setup --key K`");
    let probe = match probe_models(&cfg) {
        Ok(rows) => format!("{} models", rows.len()),
        Err(e) => format!("unreachable ({e})"),
    };
    format!(
        "{DISPLAY}\n  base: {}\n  key:  {}\n  probe: {probe}",
        cfg.upstream_base,
        if cfg.api_key.is_some() {
            key_src.as_str()
        } else {
            no_key.as_str()
        },
    )
}

/// Which rung of the credential ladder answered — shown in `status` so
/// "set" never leaves the user guessing where the key lives.
fn key_source_label() -> String {
    if std::env::var(ENV_KEY).ok().filter(|s| !s.is_empty()).is_some() {
        format!("env {ENV_KEY}")
    } else {
        "config api_key".to_string()
    }
}

fn run_cli(argv: &[String]) -> Result<String, String> {
    match argv.first().map(String::as_str) {
        Some("status") | None => Ok(cli_status()),
        Some("models") => match probe_models(&resolve()) {
            Ok(rows) => Ok(rows
                .iter()
                .filter_map(|m| m["id"].as_str())
                .collect::<Vec<_>>()
                .join("\n")),
            Err(e) => Err(e),
        },
        Some("setup") => {
            let mut cfg = read_config();
            let mut it = argv[1..].iter();
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--base" => cfg["base_url"] = json!(it.next().ok_or("--base needs a URL")?),
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
        "provider/refresh" => {
            cache_credential(params);
            json!({"result": {"refreshed": true}})
        }
        "provider/revoke" => match provider_revoke(params) {
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
