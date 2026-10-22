# Hinglish Dictation — local, private, Latin-script

Push-to-talk dictation for Hindi–English code-mixed speech on Apple Silicon.
Runs entirely on the machine, sleeps when idle, and writes **Latin letters**
(`mera naam kashish hai`) instead of Devanagari.

No API keys. No per-keystroke cost. An audio clip never leaves the laptop.

```
test मेरा नाम कशिश है   ->   test mera naam kashish hai
```

---

## Why this exists

macOS Dictation and Whisper both handle Hindi/English code-mixed speech badly or
write Hindi in Devanagari. For dictating into a chat box, Devanagari is useless:
you want `kal meeting mein deployment ki timeline finalize karni hai`, not
`कल मीटिंग में डिप्लॉयमेंट की टाइमलाइन फाइनलाइज़ करनी है`.

The fix is not a different model — it is a **script change**, applied after the
model has done the hard part. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for the measurements behind every choice here.

---

## Architecture

```
Right-Shift+Enter  →  ⌘⇧D
   │
   ├─ Hermes records mic → temp .webm (Opus)
   ├─ Hermes runs: srota_client.py <audio> <outfile>
   │
   ├─ CLIENT ──────────────────────────────────────────────
   │    1. GET :8766/health
   │       ├─ up      → 2
   │       └─ down    → spawn Tara daemon (~7s), back off to Srota if it fails
   │    2. POST /transcribe {"audio": "/abs/path.webm"}
   │
   ├─ DAEMON (Tara = Whisper large-v3, 1.55B, fp16 on MPS) ─
   │    1. ffmpeg → 16 kHz mono wav
   │    2. if > 28s: cut at the quietest 200ms window
   │    3. WhisperProcessor → mel features
   │    4. generate with forced prefix
   │       <|startoftranscript|><|hi|><|mixedcode|><|transcribe|><|notimestamps|>
   │    5. decode → mixed Devanagari + Latin
   │    6. torch.mps.empty_cache()
   │
   ├─ CLIENT post-processing (pure Python, deterministic) ──
   │    7. hinglish_romanize.py: Devanagari → Latin, English words untouched
   │    8. strip hesitation fillers
   │    9. write text to <outfile>
   │
   └─ Hermes inserts the text into the chat box
```

**Zero LLM tokens.** Transcription and romanization are local model work and
regex/lookup work respectively. Nothing calls an API, so dictation has no
per-use cost.

---

## Components

| File | Role |
| --- | --- |
| `scripts/srota_client.py` | Runs per dictation. Health-check, auto-launch, transcription, romanization, fallback. |
| `scripts/hinglish_asr_server.py` | Long-lived HTTP daemon on `127.0.0.1:8766`. Loads the model once, serves `/transcribe`, exits after 15 min idle. |
| `scripts/hinglish_romanize.py` | Devanagari → readable Latin. Deterministic, no model. |
| `scripts/asr-switch.sh` | Points the client at a different backend. |
| `scripts/setup.sh` | Rebuilds the whole thing on a fresh Mac. |

---

## Quickstart on a fresh machine

Requires macOS on Apple Silicon, Homebrew, `ffmpeg`, `uv`.

```bash
git clone git@github.com:3rectangles/<repo>.git hinglish-dictation
cd hinglish-dictation
./scripts/setup.sh          # venv, deps, downloads Trelis/tara (~2.9GB)

# smoke test — no Hermes needed
./scripts/srota_client.py ~/some_clip.webm
```

`setup.sh` is idempotent: re-running it repairs a broken venv without
re-downloading the model.

### Wire it into Hermes

```bash
hermes config set stt.provider command
hermes config set stt.command "$HOME/hinglish-dictation/scripts/srota_client.py {input_path} {output_path}"
```

### Hotkey (Karabiner)

`~/.config/karabiner/assets/complex_modifications/` — bind a key to `⌘⇧D`,
which is Hermes' dictation shortcut. Example rule:

```json
{
  "description": "Right-Shift + Enter -> Dictate (Cmd+Shift+D)",
  "manipulators": [{
    "type": "basic",
    "from": {"key_code": "return_or_enter", "modifiers": {"mandatory": ["right_shift"]}},
    "to": [{"key_code": "d", "modifiers": ["left_command", "left_shift"]}]
  }]
}
```

---

## Configuration

All via environment variables — no config file to lose.

| Variable | Default | Meaning |
| --- | --- | --- |
| `HINGLISH_SCRIPT` | `roman` | `roman` = Latin output, `devanagari` = raw model script |
| `HINGLISH_FILLERS` | `1` | `0` keeps `um`/`अ`/`आ` hesitations |
| `HINGLISH_IDLE_SECS` | `900` | Daemon exits after this long idle; `0` = never |
| `HINGLISH_MAX_SEC` | `28` | Chunk length ceiling (Whisper's window is 30s) |
| `HINGLISH_BACKEND` | `tara` | `tara`, `apex`, or `srota` |
| `HINGLISH_PORT` | `8766` | Daemon port |
| `SROTA_DEBUG_BACKEND` | — | Print which backend served the request to stderr |

---

## What it costs

| | Value |
| --- | --- |
| Warm dictation | **1.5 s** |
| Cold (wake from sleep + transcribe) | **7.1 s** |
| RAM awake | 3.8 GB |
| RAM asleep | 0 |
| Disk (model) | 2.9 GB |
| API cost per dictation | **0** |

Real memory is `footprint -p <pid>`, not `ps` — `ps` reports 0.29 GB because MPS
buffers live outside RSS.

---

## Backend comparison

Eight Hinglish clips, measured through the real dictation path (`.webm` Opus,
not lossless wav):

| Model | Accuracy | Latency | RAM | Script | Notes |
| --- | --- | --- | --- | --- | --- |
| **Tara** (`Trelis/tara`, Whisper large-v3 1.55B) | **6–8 / 8** | 1.5 s | 3.8 GB | Devanagari+Latin → romanized | **default** |
| Apex (`Oriserve/Whisper-Hindi2Hinglish-Apex`) | 4 / 8 | 1.3 s | 1.5 GB | Latin natively | drops leading words (`kal`, `कृपया`) |
| Srota (`moorlee/qwen3-asr-0.6b-hinglish`) | 2 / 8 | 1.3 s | 1.6 GB | mixed | hallucinated a whole sentence on one clip |

Tara's remaining misses are hearing, not script: `Kanban` → "plan one",
`Gurgaon` → "gurugaanv".

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `OSError: [Errno 48] Address already in use` | Old daemon still holds the port | `kill -9 $(lsof -nP -tiTCP:8766 -sTCP:LISTEN)` then relaunch |
| `ValueError: ... exceeds max_target_positions` | `max_new_tokens` > 443 | Whisper's cap is 448 minus the 5-token forced prefix; use 440 |
| Transcript ends mid-character with `�` | Token budget too small | Raise `max_new_tokens` |
| Long audio silently cut at 30 s | Whisper's encoder window | Already handled by `_split_at_silence`; check `HINGLISH_MAX_SEC` |
| `numpy._multiarray_umath` import error | Hermes injects its Python 3.14 `PYTHONPATH` | The client re-execs with `PYTHONPATH` stripped; keep that guard |
| Dictation hangs ~90 s, then fails | Backend crashed on startup | Client checks `proc.poll()` and fails fast; check the log |
| Memory never drops after long audio | MPS allocator cache | `torch.mps.empty_cache()` runs after every request |
| Daemon always resident | Idle sleep disabled | `HINGLISH_IDLE_SECS=900` |

Logs: `~/hinglish-tara.log`, `~/hinglish-apex.log`.

---

## Recreating the eval clips

The benchmark clips were not committed (audio, and they are synthetic anyway).
Regenerate equivalents with macOS TTS:

```bash
say -v Lekha -o clip.aiff "test mera naam kashish hai"
ffmpeg -y -i clip.aiff -ac 1 -ar 16000 clip.wav
ffmpeg -y -i clip.aiff -c:a libopus -b:a 32k clip.webm   # what the live path sends
```

Long clip: `ffmpeg -f concat -safe 0 -i list.txt -c:a libopus -b:a 32k long.webm`

Always test with **`.webm` Opus**, not `.wav` — lossless input hides real-world
digit errors. That mistake made Tara look wrong about a phone number for an
entire round of testing.
