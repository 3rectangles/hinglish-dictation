"""Devanagari -> readable Latin Hinglish, built for dictation output.

Apex-style romanization without Apex's accuracy cost: keep Tara's opinion of
what was said, only change the script it is written in.

ITRANS gives `merA nAma kAshisha hai`. Two fixes make it read like a human
typed it: expand ITRANS' case-coded long vowels, then delete the Hindi
word-final schwa (`naama` -> `naam`, `kAshisha` -> `kaashish`).
"""
import os
import re

from indic_transliteration import sanscript

# ITRANS is case-coded: capitals are not emphasis, they are vowel length.
_VOWELS = [
    ("RRi", "ri"), ("RRI", "ri"), ("LLi", "li"),
    ("ai", "ai"), ("au", "au"),
    ("A", "aa"), ("I", "ii"), ("U", "uu"),
    ("RRi", "ri"), ("M", "n"), ("H", "h"),
]
_CONSONANTS = [
    ("~N", "n"), ("JN", "n"), ("~n", "n"), ("N", "n"),
    ("chh", "chh"), ("Ch", "chh"), ("ch", "ch"),
    ("sh", "sh"), ("Sh", "sh"), ("S", "sh"),
    ("T", "t"), ("D", "d"), ("R", "r"), ("Dh", "dh"), ("Th", "th"),
    ("kh", "kh"), ("gh", "gh"), ("jh", "jh"), ("Th", "th"), ("Dh", "dh"),
    ("K", "kh"), ("G", "gh"), ("Ch", "chh"), ("J", "j"),
    ("P", "ph"), ("B", "bh"),
    ("x", "ksh"), ("GY", "gy"), ("j~n", "gy"),
]
_ORDER = sorted(_CONSONANTS + _VOWELS, key=lambda p: -len(p[0]))

_CONS = "bcdfghjklmnpqrstvwxyz"


_CONTROL = ".,!?;:।-"

# Hesitations Whisper emits as standalone words. Devanagari ones are stripped
# before transliteration because there they are unambiguous; the Latin list is
# applied after. A bare Latin "a"/"am" is deliberately KEPT — in English speech
# those are articles and verbs ("a coffee", "I am"), not filler.
_DEV_FILLERS = {
    "अ", "आ", "अम", "अम्म", "अं", "उम", "उम्म", "हम्म", "म्म", "ओ", "ए",
    "आअ", "अआ", "अह", "आह", "एं", "हां", "हाँ",
}
_LATIN_FILLERS = {
    "um", "umm", "ummm", "uh", "uhh", "uhhh", "uhm", "hmm", "hmmm",
    "mm", "mmm", "er", "erm", "ah", "ahh", "aah", "eh", "amm", "aamm",
}
_FILLERS_ON = os.environ.get("HINGLISH_FILLERS", "1") != "0"


def _strip_fillers(text: str, fillers: set) -> str:
    if not _FILLERS_ON:
        return text
    kept = [
        tok for tok in re.split(r"(\s+)", text)
        if tok.strip(_CONTROL).lower() not in fillers
    ]
    out = "".join(kept)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def _itrans_to_plain(s: str) -> str:
    for src, dst in _ORDER:
        s = s.replace(src, dst)
    s = s.replace("}", "").replace("{", "").replace("|", " ").replace("\\", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _drop_final_schwa(s: str) -> str:
    """Hindi deletes the inherent 'a' at a word's end: naama->naam."""
    return re.sub(rf"([{_CONS}])a\b", r"\1", s)


# ITRANS is a scholarly scheme; these are the artefacts it leaves that make
# dictation look machine-mangled. Ordered, longest/most specific first.
_ARTIFACTS = [
    (r"\.d", "r"), (r"\.D", "r"), (r"D\.", "r"),          # nukta: da -> ra
    (r"uu\.?n\b", "oon"),                                 # hoon
    (r"ii\.?n\b", "een"),
    (r"(?<![aeiou])e\.?n\b", "ein"),                       # mein (not "teen"->"teein")
    (r"\bmen\b", "mein"),
    (r"aa\.?n\b", "aan"),
    (r"uu(?=[nm])", "u"),                                 # jaauunga -> jaunga
    (r"ii\b", "i"),                                       # hindii -> hindi
    (r"jie\b", "jiye"),                                   # diijie -> dijiye
    (r"\bhaa\b", "hai"),
    (r"\bhai\b", "hai"),
]

# Vowel shortening: ITRANS writes आ as "aa" everywhere, but word-final "aa" is
# how no one types — meraa->mera, zyaadaa->zyaada, binaa->bina.
_SHORTEN = [
    (r"aa\b", "a"),
]

# Hindi collapses the medial schwa in -anaa/-ataa verb forms: karanaa->karna,
# lagataa->lagta. Applied only inside a word, after the final-aa shortening.
_SCHWA2 = [
    (r"([^aeiou\s])a([^aeiou\s])a\b", r"\1\2a"),
]


def _clean(s: str) -> str:
    for pat, rep in _ARTIFACTS:
        s = re.sub(pat, rep, s)
    for pat, rep in _SHORTEN:
        s = re.sub(pat, rep, s)

    # Medial schwa: only where the result is still a plausible word. Without
    # the length guard "bana" collapses to "bna" instead of staying "bana".
    def schwa(word):
        m = re.match(r"^(.+?)a([^aeiou\s])a$", word)
        if not m:
            return word
        cand = m.group(1) + m.group(2) + "a"
        return cand if len(cand) >= 4 else word

    s = " ".join(schwa(w) for w in s.split(" "))
    return re.sub(r"\.\s", " ", s)


def romanize(text: str) -> str:
    """Romanize only the Devanagari; leave existing Latin (English) untouched."""
    def conv(m):
        raw = sanscript.transliterate(m.group(0), sanscript.DEVANAGARI, sanscript.ITRANS)
        return _clean(_drop_final_schwa(_itrans_to_plain(raw)))

    # Convert maximal Devanagari runs in place, so English words and spacing
    # embedded in the sentence survive verbatim.
    text = _strip_fillers(text, _DEV_FILLERS)
    # ITRANS has no entry for the Devanagari letters used for English vowels
    # (ऑ as in "office"), so they leak through untransliterated.
    text = text.replace("ऑ", "ओ").replace("\u0972", "\u090D")
    out = re.sub(r"[\u0900-\u097F]+", conv, text)
    out = _strip_fillers(out, _LATIN_FILLERS)
    out = re.sub(r"\s+", " ", out).strip()
    # Tidy spacing before punctuation and collapse doubled dots from the danda.
    out = re.sub(r"\s+([,.!?])", r"\1", out)
    return out


if __name__ == "__main__":
    for t in [
        "test मेरा नाम काशिश है",
        "मेरा नाम काशिश है और मैं गुरुगोव में software engineer हूँ",
        "कृपया मेरे लिए एक coffee बना दीजिए बिना sugar के",
        "मुझे हिंदी में बात करना ज़्यादा natural लगता है but English words मिलाने पड़ते हैं",
    ]:
        print(f"{t}\n  -> {romanize(t)}\n")
