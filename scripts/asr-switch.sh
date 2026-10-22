#!/bin/bash
# asr-switch.sh — point the dictation client at a different backend.
#
#   ./asr-switch.sh tara     Whisper large-v3 mixedcode  (:8766)  best accuracy — DEFAULT
#   ./asr-switch.sh apex     Apex, Latin natively         (:8767)  fastest, drops words
#   ./asr-switch.sh srota    original Qwen3-ASR 0.6B      (:8765)  old baseline
#   ./asr-switch.sh status   show target + daemon health
#
# Edits TARA_URL in the client by pattern (not line number — the file moves).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT="$HERE/srota_client.py"
[ -f "$CLIENT" ] || { echo "client not found at $CLIENT"; exit 1; }

case "${1:-status}" in
  tara|apex|srota)
    case "$1" in
      tara)  PORT=8766 ;; apex) PORT=8767 ;; srota) PORT=8765 ;;
    esac
    /usr/bin/sed -i '' -E "s#^TARA_URL = \"http://127\.0\.0\.1:[0-9]+\"#TARA_URL = \"http://127.0.0.1:${PORT}\"#" "$CLIENT"
    grep -n '^TARA_URL' "$CLIENT"
    echo "dictation -> :${PORT} ($1)"
    ;;
  status)
    grep -n '^TARA_URL' "$CLIENT"
    for p in 8765 8766 8767; do
      printf '  :%s ' "$p"
      curl -s -m 3 "http://127.0.0.1:$p/health" || echo -n "down"
      echo
    done
    ;;
  *) echo "usage: $0 {tara|apex|srota|status}"; exit 1 ;;
esac
