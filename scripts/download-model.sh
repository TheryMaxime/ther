#!/usr/bin/env bash
# Downloads a Whisper ggml model for ai4jira.
# Default: ggml-base.en.bin (~148 MB) from the whisper.cpp Hugging Face repo.
set -euo pipefail

MODEL="${1:-base.en}"
DEST_DIR="${WHISPER_MODEL_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models}"
FILE="ggml-${MODEL}.bin"
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${FILE}"

mkdir -p "$DEST_DIR"
DEST="${DEST_DIR}/${FILE}"

if [ -f "$DEST" ]; then
  echo "Model already present: $DEST"
  exit 0
fi

echo "Downloading $FILE -> $DEST"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --progress-bar -o "$DEST" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$DEST" "$URL"
else
  echo "Error: need curl or wget to download the model." >&2
  exit 1
fi

echo "Done: $DEST"
