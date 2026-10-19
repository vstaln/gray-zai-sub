#!/bin/sh
here="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
sys="$(uname -s)"
case "$sys" in
MINGW* | MSYS* | CYGWIN*) rel="x86_64-pc-windows-msvc/zai-sub.exe" ;;
Darwin*)
    case "$(uname -m)" in
    arm64) rel="aarch64-apple-darwin/zai-sub" ;;
    *) rel="x86_64-apple-darwin/zai-sub" ;;
    esac ;;
*)
    case "$(uname -m)" in
    aarch64 | arm64) rel="aarch64-unknown-linux-musl/zai-sub" ;;
    *) rel="x86_64-unknown-linux-musl/zai-sub" ;;
    esac ;;
esac
if [ -x "$here/$rel" ]; then exec "$here/$rel" "$@"; fi
for cand in "$here/target/release/zai-sub.exe" "$here/target/release/zai-sub" "$here/target/debug/zai-sub.exe" "$here/target/debug/zai-sub"; do
    if [ -x "$cand" ]; then exec "$cand" "$@"; fi
done
echo "zai-sub binary not found in $here" >&2
exit 1
