#!/usr/bin/env bash
# Downloads an embedded LLM (GGUF weights + tokenizer) for ai4jira.
# Uses curl (system trust store) to avoid Rust TLS / corporate-CA issues.
#
# Usage:
#   scripts/download-llm.sh [preset]
#
# Presets (must match src/core/config.rs::llm_preset):
#   ministral-3b   (default) small & fast, weak instruction-following
#   mistral-7b     Mistral-7B-Instruct v0.2 — stronger, same [INST] format
#
# The chosen preset's local tokenizer filename matches what the Rust config
# expects, so after downloading just run with the same LLM_PRESET, e.g.:
#   LLM_PRESET=mistral-7b cargo run
#
# You can still override any field via env vars (LLM_GGUF_REPO / LLM_GGUF_FILE /
# LLM_TOKENIZER_REPO / LLM_TOKENIZER_FILE). Set HF_TOKEN for gated repos.
set -euo pipefail

PRESET="${1:-${LLM_PRESET:-ministral-3b}}"

case "$PRESET" in
  mistral-7b|mistral-7b-instruct|mistral)
    DEF_GGUF_REPO="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    DEF_GGUF_FILE="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    DEF_TOK_REPO="mistralai/Mistral-7B-Instruct-v0.2"
    DEF_TOK_FILE="mistral-7b-instruct-v0.2-tokenizer.json"
    ;;
  ministral-3b|ministral|"")
    DEF_GGUF_REPO="QuantFactory/Ministral-3b-instruct-GGUF"
    DEF_GGUF_FILE="Ministral-3b-instruct.Q4_K_M.gguf"
    DEF_TOK_REPO="ministral/Ministral-3b-instruct"
    DEF_TOK_FILE="ministral-tokenizer.json"
    ;;
  *)
    echo "Unknown preset '$PRESET'. Known: ministral-3b, mistral-7b." >&2
    echo "(Or set LLM_GGUF_REPO/LLM_GGUF_FILE/LLM_TOKENIZER_REPO/LLM_TOKENIZER_FILE.)" >&2
    exit 2
    ;;
esac

DEST_DIR="${LLM_MODEL_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models}"
mkdir -p "$DEST_DIR"

GGUF_REPO="${LLM_GGUF_REPO:-$DEF_GGUF_REPO}"
GGUF_FILE="${LLM_GGUF_FILE:-$DEF_GGUF_FILE}"
TOK_REPO="${LLM_TOKENIZER_REPO:-$DEF_TOK_REPO}"
TOK_FILE="${LLM_TOKENIZER_FILE:-$DEF_TOK_FILE}"

GGUF_URL="https://huggingface.co/${GGUF_REPO}/resolve/main/${GGUF_FILE}"
TOK_URL="https://huggingface.co/${TOK_REPO}/resolve/main/tokenizer.json"

GGUF_DEST="${DEST_DIR}/${GGUF_FILE}"
TOK_DEST="${DEST_DIR}/${TOK_FILE}"

# Optional auth header for gated repositories.
AUTH_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

fetch() {
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then
    echo "Already present: $dest"
    return 0
  fi
  echo "Downloading $url -> $dest"
  curl -L --fail --progress-bar ${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"} -o "$dest" "$url"
}

echo "Preset: $PRESET"
fetch "$TOK_URL" "$TOK_DEST"
fetch "$GGUF_URL" "$GGUF_DEST"

echo "Done."
echo "  GGUF:      $GGUF_DEST"
echo "  Tokenizer: $TOK_DEST"
echo "Run with:  LLM_PRESET=$PRESET cargo run"
