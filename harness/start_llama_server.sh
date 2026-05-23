#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_SERVER="${LLAMA_SERVER:-$REPO_ROOT/.local/llama.cpp/llama-b9266/llama-server}"
MODEL_REPO="${MODEL_REPO:-bartowski/Qwen_Qwen3.5-9B-GGUF:Q4_K_M}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.5-9b}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
LLAMA_CTX="${LLAMA_CTX:-8192}"
LLAMA_GPU_LAYERS="${LLAMA_GPU_LAYERS:-99}"
LLAMA_REASONING="${LLAMA_REASONING:-off}"
LLAMA_REASONING_FORMAT="${LLAMA_REASONING_FORMAT:-none}"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found or not executable: $LLAMA_SERVER" >&2
  exit 1
fi

exec "$LLAMA_SERVER" \
  -hf "$MODEL_REPO" \
  --alias "$MODEL_ALIAS" \
  --host "$LLAMA_HOST" \
  --port "$LLAMA_PORT" \
  -c "$LLAMA_CTX" \
  -ngl "$LLAMA_GPU_LAYERS" \
  --jinja \
  --reasoning "$LLAMA_REASONING" \
  --reasoning-format "$LLAMA_REASONING_FORMAT" \
  --no-mmproj \
  "$@"
