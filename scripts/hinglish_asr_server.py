#!/usr/bin/env python3
"""Hinglish ASR server — Whisper-family backends for Hermes dictation.

Listens on 127.0.0.1:8766. POST {"audio": "/abs/path"} -> {"text": "..."}
GET /health -> {"status": "ready", "backend": "..."}

Env:
  HINGLISH_BACKEND   tara | apex        (default: tara)
  HINGLISH_PORT      default 8766
  HINGLISH_MODELS    root dir           (default ~/models/asr-eval)

Backends:
  tara  Trelis/tara        Whisper large-v3 1.55B, <|mixedcode|> mode -> Devanagari + Latin
  apex  Oriserve Apex      Whisper large-v3-turbo 809M -> romanized Latin Hinglish

This is intentionally a separate port from the Srota daemon (:8765) so both can
be A/B tested live without breaking the working dictation path.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.pop("PYTHONPATH", None)  # Hermes injects a py3.14 path; keep venv clean
sys.path[:] = [p for p in sys.path if "python3.14" not in p]

# Backend/port: argv wins, then env (launchd sets these), then defaults.
# argv is preferred so `pgrep -f "hinglish_asr_server.py tara"` can tell instances apart.
BACKEND = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HINGLISH_BACKEND", "tara")).lower()
PORT = int(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("HINGLISH_PORT", "8766"))
ROOT = os.path.expanduser(os.environ.get("HINGLISH_MODELS", "~/models/asr-eval"))
FFMPEG = "/opt/homebrew/bin/ffmpeg"

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

_lock = threading.Lock()
model = processor = tokenizer = None
_prefix = None
_model_lock = threading.Lock()


def _to_wav16k(src: str) -> str:
    """Any container -> 16 kHz mono wav (soundfile can't read webm/opus)."""
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", out],
        check=True,
    )
    return out


def _read_wav(path: str) -> np.ndarray:
    import wave

    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise RuntimeError(f"expected 16k mono, got {w.getframerate()}Hz {w.getnchannels()}ch")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def load():
    global model, processor, tokenizer, _prefix
    repo = os.path.join(ROOT, BACKEND)
    processor = WhisperProcessor.from_pretrained(repo)
    model = (
        WhisperForConditionalGeneration.from_pretrained(
            repo, dtype=torch.float16, low_cpu_mem_usage=True
        )
        .eval()
        .to("mps")
    )
    tokenizer = processor.tokenizer
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in
           ("<|startoftranscript|>", "<|hi|>", "<|en|>", "<|mixedcode|>",
            "<|transcribe|>", "<|notimestamps|>")}
    sot = ids["<|startoftranscript|>"]
    if BACKEND == "tara":
        prefix = [sot, ids["<|hi|>"], ids["<|mixedcode|>"], ids["<|transcribe|>"], ids["<|notimestamps|>"]]
    else:  # apex: emit romanized Latin Hinglish
        prefix = [sot, ids["<|en|>"], ids["<|transcribe|>"], ids["<|notimestamps|>"]]
    if any(i is None or i < 0 for i in prefix):
        raise RuntimeError(f"{BACKEND}: tokenizer missing required special tokens")
    _prefix = prefix


def _split_at_silence(audio: np.ndarray, sr: int = 16000, max_sec: float = 28.0, min_sec: float = None):
    """Cut long audio into <30s pieces at the quietest point, not blindly.

    Whisper's encoder is a hard 30s window (max_source_positions=1500 mel
    frames) and silently drops everything past it, so anything longer must be
    chunked. Splitting at the lowest-energy 200ms window inside [min_sec,
    max_sec] avoids cutting mid-word.
    """
    if min_sec is None:
        min_sec = max_sec * 0.6  # keep chunks full; cutting at 10s doubled the cuts
    max_len = int(max_sec * sr)
    min_len = int(min_sec * sr)
    win = int(0.20 * sr)
    hop = max(1, win // 2)
    chunks, i, n = [], 0, len(audio)
    while n - i > max_len:
        lo, hi = i + min_len, i + max_len
        seg = audio[lo:hi]
        best_j, best_e = 0, None
        for j in range(0, max(1, len(seg) - win), hop):
            e = float(np.abs(seg[j:j + win]).mean())
            if best_e is None or e < best_e:
                best_j, best_e = j, e
        cut = lo + best_j + win // 2
        chunks.append(audio[i:cut])
        i = cut
    chunks.append(audio[i:])
    return chunks


def _transcribe_chunk(audio: np.ndarray) -> str:
    feats = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(
        "mps", torch.float16
    )
    with _lock, torch.inference_mode():
        gen = model.generate(
            input_features=feats,
            decoder_input_ids=torch.tensor([_prefix], device="mps"),
            max_new_tokens=440,  # 448 max_target_positions minus the 5-token prefix; >443 raises
        )
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()


def transcribe(audio_path: str) -> str:
    wav = _to_wav16k(audio_path)
    try:
        audio = _read_wav(wav)
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass
    max_sec = float(os.environ.get("HINGLISH_MAX_SEC", "28"))
    parts = [_transcribe_chunk(c) for c in _split_at_silence(audio, max_sec=max_sec)]
    text = " ".join(p for p in parts if p)
    # Long audio inflates the MPS allocator cache (~4GB -> ~5.9GB) and it is
    # never returned on its own. Release it so a 16GB Mac doesn't stay pinned.
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    # Whisper appends a Devanagari danda; strip it so dictation stays clean.
    return text.rstrip("। .").strip()


# Idle sleep: the daemon costs ~3.5GB resident, and a 16GB Mac should not pay
# that for a dictation tool used a few times a day. After IDLE_SECS with no
# request it exits; the client relaunches it on the next dictation (~9s cold).
IDLE_SECS = float(os.environ.get("HINGLISH_IDLE_SECS", "900"))
_last_activity = time.time()


def _touch():
    global _last_activity
    _last_activity = time.time()


def _idle_watchdog():
    while True:
        time.sleep(15)
        idle = time.time() - _last_activity
        if IDLE_SECS > 0 and idle >= IDLE_SECS:
            print(f"[hinglish-asr] idle {idle:.0f}s >= {IDLE_SECS:.0f}s -> exiting", flush=True)
            os._exit(0)  # hard exit releases MPS memory immediately


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code: int, obj: dict):
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            _touch()
            self._send(200, {"status": "ready" if model is not None else "loading", "backend": BACKEND})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        _touch()
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            audio = body.get("audio")
            if not audio or not os.path.isabs(audio):
                self._send(400, {"error": "audio must be an absolute path"})
                return
            self._send(200, {"text": transcribe(audio)})
        except subprocess.CalledProcessError as e:
            self._send(500, {"error": f"ffmpeg failed: {e}"})
        except Exception as e:  # keep the daemon alive on odd input
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    load()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[hinglish-asr] backend={BACKEND} port={PORT} ready", flush=True)
    threading.Thread(target=_idle_watchdog, daemon=True).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
