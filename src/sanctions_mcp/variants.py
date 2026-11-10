"""Labelled spelling variants of a listed name, the ways real screening inputs drift.

Each generator takes the words of a name and returns a new name, or None when it does not
apply. The matcher never sees which generator produced a query.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

from sanctions_mcp.normalize import (
    ARTICLES,
    CANONICAL,
    CONNECTORS,
    FRENCH_TITLE_WORDS,
    TITLES,
    VARIANT_GROUPS,
    fold_latin,
)

# Words every screener drops: an initial for one of them is not an initial for a name.
_PARTICLES = TITLES | ARTICLES | CONNECTORS | FRENCH_TITLE_WORDS


def _plain(word: str) -> str:
    return re.sub(r"[^a-z]", "", fold_latin(word).casefold())


# One orthographic change at a time, as seen across transliteration schemes.
ORTHOGRAPHIC_RULES: list[tuple[str, str]] = [
    (r"ou", "u"),
    (r"(?<=[^aeiou])u(?=[^aeiou])", "ou"),
    (r"ee", "i"),
    (r"(?<=[^aeiou])i(?=[^aeiou])", "ee"),
    (r"oo", "u"),
    (r"q", "k"),
    (r"(?<=[aeiou])k(?=[aeiou])", "q"),
    (r"dh", "z"),
    (r"(?<=[aeiou])z(?=[aeiou])", "dh"),
    (r"(?<=[aeiou])j", "dj"),
    (r"^j", "dj"),
    (r"kh", "h"),
    (r"sh", "ch"),
    (r"([bdfklmnrst])\1", r"\1"),
    (r"(?<=[aeiou])([bdlmnrst])(?=[aeiou])", r"\1\1"),
    (r"ei", "ai"),
    (r"ai", "ay"),
    (r"ay", "ai"),
    (r"(?<=[^aeiou])y(?=[^aeiou])", "i"),
    (r"eh$", "ih"),
    (r"ih$", "eh"),
    (r"a(?=[^aeiou]$)", "e"),
    (r"e(?=[^aeiou]$)", "a"),
    (r"ph", "f"),
    (r"w", "v"),
]
HONORIFICS = ["Sheikh", "Haji", "Mullah", "Dr", "Maulana", "Qari"]
KEYBOARD = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn", "k": "jiolm", "l": "kop",
    "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "wa", "r": "edft", "s": "awedxz",
    "t": "rfgy", "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}  # fmt: skip
ARTICLE_PREFIX = re.compile(r"^(al|el)-", re.IGNORECASE)  # Al-Nouri, never Aleksandr
GLUE_PREFIXES = ("abdul", "abdel", "abdur", "abu", "abou")


def _style(template: str, word: str) -> str:
    """Give `word` the letter case of `template` (UPPER, Title or lower)."""
    if template.isupper() and len(template) > 1:
        return word.upper()
    if template[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def transliteration(words: list[str], rng: random.Random) -> list[str] | None:
    swaps = []
    for i, w in enumerate(words):
        low = w.lower()
        canon = CANONICAL.get(low, low)
        group = [canon, *VARIANT_GROUPS.get(canon, ())]
        options = [g for g in group if g != low]
        if len(group) > 1 and options:
            swaps.append((i, options))
    if swaps and rng.random() < 0.5:
        i, options = rng.choice(swaps)
        out = list(words)
        out[i] = _style(words[i], rng.choice(options))
        return out
    edits = []
    for i, w in enumerate(words):
        if len(w) < 3:
            continue
        for pattern, repl in ORTHOGRAPHIC_RULES:
            if re.search(pattern, w.lower()):
                edits.append((i, pattern, repl))
    if not edits:
        return None
    i, pattern, repl = rng.choice(edits)
    low = words[i].lower()
    matches = list(re.finditer(pattern, low))
    m = rng.choice(matches)
    new = low[: m.start()] + m.expand(repl) + low[m.end() :]
    if new == low:
        return None
    out = list(words)
    out[i] = _style(words[i], new)
    return out


def order(words: list[str], rng: random.Random) -> list[str] | None:
    if len(words) < 2:
        return None
    last, rest = words[-1], words[:-1]
    if rng.random() < 0.5:
        return [last.upper() + ",", *rest]
    return [last, *rest]


def drop_middle(words: list[str], rng: random.Random) -> list[str] | None:
    if len(words) < 3:
        return None
    i = rng.randrange(1, len(words) - 1)
    return words[:i] + words[i + 1 :]


def particles(words: list[str], rng: random.Random) -> list[str] | None:
    with_article = [i for i, w in enumerate(words) if ARTICLE_PREFIX.match(w) and len(w) > 4]
    if with_article and rng.random() < 0.6:
        i = rng.choice(with_article)
        bare = ARTICLE_PREFIX.sub("", words[i])
        out = list(words)
        out[i] = _style(words[i], bare.lower()) if bare else words[i]
        return out
    if len(words) >= 2 and rng.random() < 0.5:
        return [words[0], "bin", *words[1:]]
    out = list(words)
    out[-1] = rng.choice(["Al-", "El-", "al "]) + words[-1]
    return out


def honorific(words: list[str], rng: random.Random) -> list[str] | None:
    return [rng.choice(HONORIFICS), *words]


def typo(words: list[str], rng: random.Random) -> list[str] | None:
    options = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not options:
        return None
    i = rng.choice(options)
    w = words[i]
    low = w.lower()
    pos = rng.randrange(1, len(low) - 1)
    kind = rng.choice(["sub", "del", "ins", "swap"])
    if kind == "sub" and low[pos] in KEYBOARD:
        new = low[:pos] + rng.choice(KEYBOARD[low[pos]]) + low[pos + 1 :]
    elif kind == "del":
        new = low[:pos] + low[pos + 1 :]
    elif kind == "ins":
        new = low[:pos] + low[pos] + low[pos:]
    else:
        new = low[: pos - 1] + low[pos] + low[pos - 1] + low[pos + 1 :]
    if new == low:
        return None
    out = list(words)
    out[i] = _style(w, new)
    return out


def spacing(words: list[str], rng: random.Random) -> list[str] | None:
    """Spacing and hyphen changes in compound names, the way they drift in real records:
    Abdul Rahman / Abdulrahman, Abu Bakr / Abubakr, Saif Ullah / Saifullah,
    Nur al-Din / Nuruddin, Al-Tikriti / Al Tikriti."""
    options: list[list[str]] = []
    for i, w in enumerate(words):
        low = w.lower()
        for prefix in GLUE_PREFIXES:  # split a glued compound
            if low.startswith(prefix) and len(low) > len(prefix) + 3:
                options.append(
                    [*words[:i], w[: len(prefix)], _style(w, low[len(prefix) :]), *words[i + 1 :]]
                )
                break
        for suffix in ("ullah", "uddin", "eddine"):
            if low.endswith(suffix) and len(low) > len(suffix) + 2:
                head = w[: -len(suffix)]
                tail = {"ullah": "Ullah", "uddin": "ud Din", "eddine": "ed Dine"}[suffix]
                options.append([*words[:i], head, *tail.split(), *words[i + 1 :]])
        if "-" in w:
            options.append([*words[:i], *w.split("-"), *words[i + 1 :]])
        if i + 1 < len(words):  # glue a compound together
            nxt = words[i + 1]
            if low in ("abdul", "abdel", "abd", "abu", "abou") or nxt.lower() in ("ullah", "allah"):
                options.append([*words[:i], w + nxt.lower(), *words[i + 2 :]])
    return rng.choice(options) if options else None


def initials(words: list[str], rng: random.Random) -> list[str] | None:
    options = [
        i
        for i, w in enumerate(words[:-1])
        if len(w) >= 3 and w[0].isalpha() and _plain(w) not in _PARTICLES
    ]
    if len(words) < 3 or not options:
        return None
    i = rng.choice(options)
    out = list(words)
    out[i] = words[i][0].upper() + "."
    return out


GENERATORS: dict[str, Callable[[list[str], random.Random], list[str] | None]] = {
    "transliteration": transliteration,
    "drop_middle": drop_middle,
    "particles": particles,
    "honorific": honorific,
    "typo": typo,
    "spacing": spacing,
    "initials": initials,
}


def make_variant(name: str, kind: str, rng: random.Random) -> str | None:
    words = name.split()
    out = GENERATORS[kind](words, rng)
    if not out:
        return None
    text = " ".join(out).replace(" ,", ",")
    return text if text != name else None
