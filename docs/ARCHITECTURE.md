# Architecture & design decisions

Every choice below was made from a measurement, not from a model card. Numbers
are from an M5 MacBook Air, 16 GB unified memory, macOS 26.

---

## 1. The problem is script, not accuracy

Whisper transcribes Hinglish well when you force the language. It writes Hindi in
Devanagari because that is what `<|hi|>` means. For dictating into a chat box
that is the wrong output format — you want Latin letters.

The obvious fix is "use a model that outputs Latin". We tested that. The less
obvious fix is "keep the accurate model, change the script afterwards", and it
wins on quality. That is the design.

---

## 2. Model selection

Eight hand-written Hinglish clips: pure Hindi, pure English, code-mixed, a phone
number, technical nouns (`Kanban`, `deployment`), and politeness markers
(`कृपया`).

| Model | Params | Disk | RAM | Latency | RTF | Clean clips | Script |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tara `Trelis/tara` | 1.55B | 2.9 GB | 3.8 GB | 1.5 s | 0.56 | **6–8 / 8** | Devanagari + Latin |
| Apex `Oriserve/Whisper-Hindi2Hinglish-Apex` | 0.8B | 1.5 GB | 1.5 GB | 1.3 s | 0.32 | 4 / 8 | Latin only |
| Srota `moorlee/qwen3-asr-0.6b-hinglish` | 0.6B | 1.2 GB | 1.6 GB | 1.3 s | 0.32 | 2 / 8 | mixed |

**Tara wins on accuracy.** It is a Whisper large-v3 fine-tune with a
`<|mixedcode|>` token; the forced decode prefix is

```
<|startoftranscript|><|hi|><|mixedcode|><|transcribe|><|notimestamps|>
```

### Why Apex is not the default despite writing Latin natively

It is faster and half the memory, and it nails phone numbers (`9876543210`
straight to digits). But it silently *drops words*:

| Truth | Apex |
| --- | --- |
| कल meeting में deployment की timeline finalize करनी है | "Meeting mein ..." — `कल` gone |
| कृपया मेरे लिए एक coffee बना दीजिए | "Mere lie ek coffee bana dijie" — `कृपया` gone |
| Kanban board पर तीन tasks pending हैं | "Board par tin task spending hain" |

A dropped `कल` changes the meaning of a sentence. A romanization artifact
(`prate` for `पड़ते`) never does. **Word loss is worse than cosmetic ugliness**,
so Tara + post-processing became the default and Apex stayed as an option
(`HINGLISH_BACKEND=apex`, port 8767).

Srota was retired as primary: it hallucinated an entire sentence
("अब इस वर्कअप में आपका स्वागत है") on a clip that contained no such thing.

---

## 3. Devanagari → Latin (`hinglish_romanize.py`)

A translation table, not a model. `indic_transliteration` (ITRANS) does the
character mapping; four deterministic passes make the result readable rather
than scholarly.

`मेरा नाम काशिश है` → ITRANS → `merA nAma kAshisha hai` → after passes →
`mera naam kaashish hai`

| Pass | Rule | Example |
| --- | --- | --- |
| Artifacts | nukta `D.`/`.D` → `r`, `uu.n` → `oon`, `e.n` → `ein`, final `ii` → `i` | `paDate` → `padte`, `hU.n` → `hoon`, `hindI` → `hindi` |
| Final schwa deletion | drop word-final inherent `a` | `naama` → `naam`, `kAshisha` → `kaashish` |
| Vowel shortening | word-final `aa` → `a` | `meraa` → `mera`, `zyaadaa` → `zyaada` |
| Medial schwa | collapse `-aCa` → `Ca`, only if result ≥ 4 chars | `karana` → `karna`, `lagata` → `lagta` |

Two bugs worth remembering:

- **Greedy vowel rules.** `e\.?n\b` → `ein` (for `mein`) also rewrote `teen` as
  `teein`. Anchored with `(?<![aeiou])`.
- **Over-aggressive schwa.** Without the length guard, `bana` collapsed to `bna`.
  The guard is `len(result) >= 4`.

English words embedded in the sentence are never touched — only maximal
Devanagari runs are converted. `The production server crashed again` passes
through byte-identical.

Cost: 100 conversions in 0.001 s, import 0.06 s, zero tokens.

---

## 4. The 30-second cliff

Whisper's encoder window is fixed at 1500 mel frames = 30 s. Audio beyond that is
**silently discarded** — not truncated with an error, just dropped.

Proven: a 31.50 s clip and a 63.00 s clip returned *byte-identical* text, ending
mid-word.

Fix: `_split_at_silence()` cuts at the lowest-energy 200 ms window in the range
`[0.6 × max_sec, max_sec]`, so seams land between words instead of inside them.
Chunks are transcribed separately and joined.

A first attempt used `min_sec = 10 s` and made output *worse* — twice as many
cuts, and the joins lost words. Raising the floor to `0.6 × max_sec` fixed it.
Verified seamless: a 63 s clip produces 1003 characters, with chunk 0 ending
`...nine eight seven six` and chunk 1 starting `five four three two one zero`.

---

## 5. Token budget

`max_target_positions = 448` for Whisper, and the 5-token forced prefix counts
against it. Setting `max_new_tokens = 448` raises:

```
ValueError: Input length ... exceeds max_target_positions
```

Setting it too low (222, the original value) silently truncated long
transcripts mid-character, leaving a `\ufffd` at the end of the string. Working
value: **440**.

---

## 6. Memory

`ps -o rss=` reports **0.29 GB** for the daemon. That is wrong — MPS buffers are
allocated outside the process RSS. The real number is `phys_footprint` from
`footprint -p <pid>`: **~3.8 GB** idle.

Long audio inflated it to 5.9 GB and it never came back down, on a 16 GB machine
already swapping at 5.85 GB of 7. Fix: `torch.mps.empty_cache()` after every
request, in a `try/except` so a failure never breaks transcription. Settles at
3.5–3.9 GB.

Only one local model runs at a time — two concurrently triggers a Metal
`Compute error`. That is why Apex is stopped rather than kept warm.

---

## 7. Idle sleep

A dictation tool used a few times a day should not hold 3.8 GB permanently.

`_idle_watchdog()` runs on a daemon thread, ticks every 15 s, and calls
`os._exit(0)` once `now - last_activity >= HINGLISH_IDLE_SECS` (default 900 =
15 min). A hard exit is deliberate: it guarantees the MPS memory is returned
immediately instead of hoping the GC and allocator cooperate.

`/health` counts as activity, because the client polls it during cold start.

Wake-on-dictation needs no extra machinery — the client already spawns a dead
backend. Measured with a 25 s threshold: exited at 38 s idle (15 s granularity,
plus one health poll resetting the clock).

| Path | Time |
| --- | --- |
| Warm | **1.51 s** server-side, 1.61 s end-to-end |
| Cold from slept daemon | **7.1 s** |

Benchmark warm latency only after the third request. The first two after a cold
start run ~2× slower while MPS compiles kernels — that looked like a 3.7 s
regression that did not exist.

---

## 8. Failure handling

The original client had two bugs that made it hang:

1. If Tara was down it launched the *Srota* server, then polled port 8766 for the
   full 90 s timeout before giving up.
2. It never checked whether the child process had already died, so a crashed
   backend burned the entire timeout instead of failing over.

Both fixed: backends are declared as `(name, url, script, env, log, err)`
tuples, launch targets the matching port, and the wait loop calls `proc.poll()`
each second. A deterministically broken Tara now falls back to Srota in **8.8 s**
instead of 90 s.

Third bug, subtler: Hermes injects its own Python 3.14 `PYTHONPATH` into child
processes, which breaks numpy inside the 3.12 venv
(`_multiarray_umath` import error). The client detects it and re-execs itself
with `PYTHONPATH` stripped and `SROTA_CLEAN_REEXEC=1` set as a guard against
looping.

---

## 9. Filler removal

Hesitations are stripped deterministically, in two passes, because script tells
you which words are safe to remove:

- **Devanagari fillers, before transliteration**: `अ आ अम अम्म अं उम उम्म हम्म ओ ए हां`.
  Unambiguous — no Hindi word is spelled those ways.
- **Latin fillers, after**: `um umm uh hmm erm ah mm`.

A bare Latin `a` or `am` is deliberately **kept**: in English speech those are
the article and the verb. `I need a coffee and I am tired` must survive intact.
Hesitations in Hindi speech arrive as `अ`/`अम`, so they are still removed:

```
अ अ मेरा नाम काशिश है उम हाँ   ->   mera naam kaashish hai
```

Order matters: stripping must happen on Devanagari *before* transliteration,
because after romanization `अ` and `a` are indistinguishable.

---

## 10. Known limitations

- Romanization is rule-based, so rare consonant clusters still slip: `पड़ते` →
  `prate` (should be `padte`).
- Proper nouns are limited by the acoustic model, not the script: `Kanban` →
  "plan one", `Gurgaon` → "gurugaanv". A vocabulary bias would need
  `initial_prompt`, which is unavailable while the decoder prefix is forced.
- Phone numbers come out as words ("nine eight seven six") when spoken as
  digits. Apex returns numerals; Tara does not.
- No punctuation is inserted beyond what the model emits.
