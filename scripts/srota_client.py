#!/usr/bin/env python3
"""Hermes command-STT client — Tara primary, Srota fallback.

Usage:
    srota_client.py <input_audio> [<output_path>]

Prints the transcript to stdout, or writes it to <output_path>.

Backends (both speak the same JSON API):
  Tara   :8766  hinglish_asr_server.py  Whisper large-v3 <|mixedcode|>  ~2.2s
  Srota  :8765  srota_server.py         Qwen3-ASR 0.6B                 ~1.3s

On-demand: if a daemon isn't running, this client launches it in the background
and waits for the model to load. If Tara can't come up, it falls back to Srota
rather than failing the dictation.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Hermes injects its own Python 3.14 site-packages into PYTHONPATH, which makes
# this 3.12 venv import a foreign numpy and die on _multiarray_umath.
# Re-exec clean (ignoring all PYTHON* env vars) whenever PYTHONPATH is present.
if os.environ.get("PYTHONPATH") and not os.environ.get("SROTA_CLEAN_REEXEC"):
    os.environ["SROTA_CLEAN_REEXEC"] = "1"
    clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    os.execve(sys.executable, [sys.executable, "-E", os.path.abspath(__file__), *sys.argv[1:]], clean_env)

# Paths derive from this file's location so the repo works from any clone dir.
BASE = os.path.dirname(os.path.abspath(__file__))

TARA_URL = "http://127.0.0.1:8766"
SROTA_URL = "http://127.0.0.1:8765"

TARA_SCRIPT = os.path.join(BASE, "hinglish_asr_server.py")
SROTA_SCRIPT = os.path.join(BASE, "srota_server.py")

TARA_LOG = os.path.expanduser("~/hinglish-tara.log")
SROTA_LOG = os.path.expanduser("~/srota-server.log")
SROTA_ERR = os.path.expanduser("~/srota-server.err.log")

# Tara loads 1.55B params on MPS: ~15-25s. Srota ~15s.
STARTUP_TIMEOUT = 90

BACKENDS = [
    ("tara", TARA_URL, TARA_SCRIPT, {"HINGLISH_BACKEND": "tara", "HINGLISH_PORT": "8766"},
     TARA_LOG, TARA_LOG),
    ("srota", SROTA_URL, SROTA_SCRIPT, {}, SROTA_LOG, SROTA_ERR),
]

# "roman" (default) = Devanagari transliterated to Latin letters, English words
# left untouched. "devanagari" = the model's raw mixed script.
SCRIPT_MODE = os.environ.get("HINGLISH_SCRIPT", "roman").lower()


def to_output_script(text):
    if SCRIPT_MODE != "roman" or not text:
        return text
    try:
        from hinglish_romanize import romanize
        return romanize(text)
    except Exception:
        return text  # never break dictation over a cosmetic transliteration


def daemon_ready(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_running(url, script, env_extra, out_log, err_log):
    """True once `url` answers /health, launching `script` if needed.

    Fails fast if the child process exits instead of burning the full
    STARTUP_TIMEOUT — a crashed backend must hand over to the next one
    quickly, or dictation hangs for 90s.
    """
    if daemon_ready(url):
        return True
    # Strip PYTHONPATH so the daemon can't import Hermes' foreign site-packages.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(env_extra)
    with open(out_log, "ab") as out, open(err_log, "ab") as err:
        proc = subprocess.Popen(
            [sys.executable, "-E", script],
            stdout=out, stderr=err,
            env=env,
            start_new_session=True,
        )
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if daemon_ready(url):
            return True
        if proc.poll() is not None:
            return False  # backend died on startup (bad model path, OOM, ...)
        time.sleep(1)
    return False


def transcribe(url, audio):
    req = urllib.request.Request(
        url + "/",
        data=json.dumps({"audio": audio}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    audio = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.isabs(audio):
        audio = os.path.abspath(audio)  # daemons run ffmpeg from their own cwd

    errors = []
    for name, url, script, env_extra, out_log, err_log in BACKENDS:
        try:
            if not ensure_running(url, script, env_extra, out_log, err_log):
                errors.append(f"{name}: daemon did not become ready in {STARTUP_TIMEOUT}s (log: {out_log})")
                continue
        except Exception as e:
            errors.append(f"{name}: launch failed: {e}")
            continue

        try:
            result = transcribe(url, audio)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            errors.append(f"{name}: HTTP {e.code}: {body[:200]}")
            continue
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue

        if "error" in result:
            errors.append(f"{name}: {result['error'][:200]}")
            continue

        text = (result.get("text") or "").strip()
        if not text:
            errors.append(f"{name}: empty transcript")
            continue
        text = to_output_script(text)

        if os.environ.get("SROTA_DEBUG_BACKEND"):
            print(f"[stt backend: {name}]", file=sys.stderr)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)
        return

    print("STT ERROR — all backends failed:\n  " + "\n  ".join(errors), file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
