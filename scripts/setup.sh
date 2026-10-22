#!/bin/bash
# Rebuild this dictation stack on a fresh Apple Silicon Mac.
# Idempotent: re-running repairs a broken venv without re-downloading the model.
set -euo pipefail

VENV="$HOME/srota-venv"
MODELS="$HOME/models/asr-eval"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> checking prerequisites"
command -v ffmpeg >/dev/null || { echo "ffmpeg missing:  brew install ffmpeg"; exit 1; }
if ! command -v uv >/dev/null; then
  echo "uv missing:  brew install uv   (or: curl -LsSf https://astral.sh/uv/install.sh | sh)"
  exit 1
fi

echo "==> venv at $VENV"
[ -d "$VENV" ] || uv venv "$VENV" --python 3.12

echo "==> python deps"
uv pip install --python "$VENV/bin/python" -q \
  torch torchaudio transformers soundfile numpy \
  indic-transliteration

echo "==> model: Trelis/tara (~2.9GB) -> $MODELS/tara"
mkdir -p "$MODELS"
"$VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
dest = os.path.expanduser("~/models/asr-eval/tara")
if os.path.exists(os.path.join(dest, "config.json")):
    print("   already present, skipping")
else:
    snapshot_download("Trelis/tara", local_dir=dest)
    print("   downloaded")
PY

chmod +x "$HERE"/*.py "$HERE"/*.sh

echo "==> smoke test"
CLIP="$(mktemp -t smoke).webm"
say -v Lekha -o "${CLIP%.webm}.aiff" "test mera naam kashish hai"
ffmpeg -y -loglevel error -i "${CLIP%.webm}.aiff" -c:a libopus -b:a 32k "$CLIP"
"$VENV/bin/python" "$HERE/srota_client.py" "$CLIP"
rm -f "$CLIP" "${CLIP%.webm}.aiff"

cat <<EOF

Done.

  Wire into Hermes:
    hermes config set stt.provider command
    hermes config set stt.command "$HERE/srota_client.py {input_path} {output_path}"

  Bind a hotkey to Cmd+Shift+D (Karabiner), then dictate.
  Daemon sleeps after 15 min idle; it wakes on the next dictation (~7s once).
EOF
