<div align="center">
  <img alt="gray-zai-sub" src="assets/icon.svg" width="120" height="120" />
  <h1>gray-zai-sub</h1>
  <p><strong>Z.AI's GLM coding models as a gray provider.</strong></p>
  <p>
    <a href="https://gray.alignment.id">Website</a> ·
    <a href="https://github.com/vstaln/gray-zai-sub">Source</a> ·
    <a href="https://github.com/vstaln/gray">gray</a>
  </p>
  <p>
    <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-1c1c20?style=flat-square&labelColor=0a0a0b" /></a>
    <a href="https://www.rust-lang.org"><img alt="Built with Rust" src="https://img.shields.io/badge/built%20with-rust-1c1c20?style=flat-square&labelColor=0a0a0b&logo=rust&logoColor=d4a373" /></a>
    <img alt="gray plugin" src="https://img.shields.io/badge/gray-plugin-1c1c20?style=flat-square&labelColor=0a0a0b&color=7aa2f7" />
  </p>
</div>

<br/>
```bash
gray plugin install zai
```

`gray-zai-sub` is a provider sidecar for [gray](https://github.com/vstaln/gray).
The host speaks the OpenAI Responses wire shape to a per-turn loopback
relay this plugin mints; the plugin translates each turn into
Z.AI's OpenAI-compatible `https://api.z.ai/api/coding/paas/v4` endpoint upstream.

That means streaming text, tool calls, and usage accounting all behave
like a first-class gray provider — while the wire stays one-shot and
stateless on the upstream side.

## Requirements

Requires a Z.AI API key.

Set `ZAI_API_KEY` or run `{bin} setup --key …`.

## Setup

```sh
zai-sub status          # shows configured/upstream state
zai-sub models          # live model list from the upstream
zai-sub setup --key …   # write ~/.gray/zai/config.json (0600)
```

Configuration resolves env-first, then `~/.gray/zai/config.json`:

| Setting | Env | Config key | Default |
|---|---|---|---|
| base URL | `ZAI_BASE_URL` | `base_url` | `https://api.z.ai/api/coding/paas/v4` |
| API key | `ZAI_API_KEY` | `api_key` | — |

## In gray

Select the provider by name (`zai`) with the `api_key`-style auth
method where required; the plugin reports models, auth state, and errors
through the normal provider ops (`provider/models`, `provider/chat`,
`provider/auth/*`).

## How it works

- `provider/chat` mints a **request-scoped loopback relay** (one response
  per turn) and returns its URL to the host.
- The host POSTs a Responses-shaped request; the plugin translates
  messages/tools/images into the OpenAI chat-completions shape, calls the
  upstream, and folds the result back into a Responses SSE stream
  (`response.created` → deltas → `response.completed` → `[DONE]`).
- Credentials travel in the provider `CredentialEnvelope`; secrets are
  never logged (relay URLs and tokens are redacted in debug output).

## Build from source

```sh
cargo build --release
gray plugin check ./target/release/zai-sub
```

## License

MIT
