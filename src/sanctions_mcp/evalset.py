"""Build the screening eval set from the watchlist, deterministically from one seed.

Positive queries name a listed person in a way the list does not record:
    alias          a real alias held out of the index (a spelling variant of a kept name)
    cross_script   an Arabic script name, with every Arabic script name held out
    <variant>      a generated variant of the primary name (see variants.py)
Clean queries are names of nobody on the list, built from real name parts (the given
names of one listed person and the family name of another from the same origin group), so
every alert on them is a false alert. A clean name is never a listed name, a positive query
or another clean name in disguise (see `name_keys`), because no name only screener can tell
such namesakes apart. Nor is it a listed name with its middle names left out.

Entities are split into train, dev and test by a hash of their id, and each split's queries
are built from that split's people only. The index, the word weights and the candidates
searched while training still cover the whole list, as they do in production.
"""

from __future__ import annotations

import difflib
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sanctions_mcp.data import Entity
from sanctions_mcp.normalize import (
    ARTICLES,
    CONNECTORS,
    FRENCH_TITLE_WORDS,
    KNOWN_GIVEN_NAMES,
    TITLES,
    Token,
    fold_latin,
    has_arabic,
    is_arabic_particle,
    neutral_key,
    tokenize,
)
from sanctions_mcp.variants import GENERATORS, make_variant

SPLITS = ("train", "dev", "test")
# The 22 Arab League members, plus Iran, Afghanistan and Pakistan. A person listed under any
# of them, or with any Arabic script name, is in the "mena_south_asia" group. The group pairs
# clean names and slices the results; the matcher never sees it.
MENA_SOUTH_ASIA = frozenset(
    [
        "ae",
        "af",
        "bh",
        "dz",
        "eg",
        "iq",
        "ir",
        "jo",
        "kw",
        "lb",
        "ly",
        "ma",
        "mr",
        "om",
        "pk",
        "ps",
        "qa",
        "sa",
        "sd",
        "so",
        "sy",
        "tn",
        "ye",
        "km",
        "dj",
    ]
)
ALIAS_SIMILARITY = 0.6  # aliases closer than this to a kept name count as spelling variants


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    target: str | None  # entity id, or None for a clean name
    kind: str
    group: str  # "mena_south_asia" or "other"
    split: str


@dataclass
class EvalSet:
    seed: int
    queries: list[Query]
    held_out: dict[str, list[str]] = field(default_factory=dict)  # entity id -> names

    def by_split(self, split: str) -> list[Query]:
        return [q for q in self.queries if q.split == split]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": self.seed,
            "held_out": self.held_out,
            "queries": [asdict(q) for q in self.queries],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EvalSet:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `sanctions build-evalset` first.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            seed=payload["seed"],
            queries=[Query(**q) for q in payload["queries"]],
            held_out=payload["held_out"],
        )


def split_of(entity_id: str, seed: int) -> str:
    bucket = int(hashlib.sha1(f"{seed}:{entity_id}".encode()).hexdigest(), 16) % 10
    return "train" if bucket < 6 else ("dev" if bucket < 8 else "test")


def origin_group(entity: Entity) -> str:
    if set(entity.countries) & MENA_SOUTH_ASIA or any(has_arabic(n) for n in entity.names):
        return "mena_south_asia"
    return "other"


def _eligible(entity: Entity) -> bool:
    return _is_latin(entity.name) and len(tokenize(entity.name)) >= 2


def _word_forms(t: Token, *, arabic: bool) -> list[str]:
    """The ways one word can equal another word to a name only screener.

    Well known names are compared as names (Hamid is not Ahmad). Other Latin words are equal
    when they share a consonant key and the vowel pattern behind it (Fadel and Fadil).
    Against Arabic script, where short vowels are not written, a shared key is enough."""
    if t.folded in KNOWN_GIVEN_NAMES or not t.skeleton:
        return [t.folded]
    glued = t.glued or (False,) * len(t.skeletons)
    if arabic or t.from_arabic:
        return [k for k, g in zip(t.skeletons, glued, strict=True) if k and not g]
    shapes = t.shapes or t.skeletons
    return [f"{k}:{s}" for k, s, g in zip(t.skeletons, shapes, glued, strict=True) if k and not g]


def name_key(name: str) -> str:
    """The main key of a name: one form per word, sorted, so word order, titles, particles
    and spelling are set aside. Names with an Arabic script word are keyed by consonants."""
    tokens = tokenize(name)
    arabic = any(t.from_arabic for t in tokens)
    words = sorted(_word_forms(t, arabic=arabic)[0] for t in tokens)
    return ("A|" if arabic else "L|") + " ".join(words)


def name_keys(name: str) -> set[str]:
    """Every key a name can take, reading each word every way it can be read, both as Latin
    and against Arabic script. A clean name that shares any of these with a listed name is a
    namesake to the matcher, so it is not used."""
    tokens = tokenize(name)
    keys = {name_key(name)}
    for prefix, arabic in (("L|", False), ("A|", True)):
        for combo in itertools.product(*(_word_forms(t, arabic=arabic) for t in tokens)):
            keys.add(prefix + " ".join(sorted(combo)))
    return keys


def listed_keys(name: str, cap: int = 64) -> set[str]:
    """The keys of a listed name under every reading of its words (as Latin, or by
    consonants when it has an Arabic script word), so a clean name that meets it on a second
    reading (Reza for رضا, Ilyas for الياس) is not used. Names whose readings multiply past
    `cap` keep their main key only."""
    tokens = tokenize(name)
    arabic = any(t.from_arabic for t in tokens)
    forms = [_word_forms(t, arabic=arabic) for t in tokens]
    keys = {name_key(name)}
    if math.prod(len(f) for f in forms) <= cap:
        prefix = "A|" if arabic else "L|"
        keys |= {prefix + " ".join(sorted(c)) for c in itertools.product(*forms)}
    return keys


def arabic_view_keys(name: str, cap: int = 64) -> set[str]:
    """The keys of a name as an Arabic script query sees it: common names by name, every
    other word by its consonants only. An Arabic script clean name that shares one with any
    listed name, in either script, is a namesake to the matcher."""
    forms = [_word_forms(t, arabic=True) for t in tokenize(name)]
    if not forms or math.prod(len(f) for f in forms) > cap:
        return set()
    return {"A|" + " ".join(sorted(c)) for c in itertools.product(*forms)}


def _partial_keys(name: str) -> set[str]:
    """Keys of the shorter names a clean name could be cut from: the first one or two words
    and the last word of a Latin name of three words or more ("Karim Al-Nouri" from "Karim
    Fadel Al-Nouri")."""
    tokens = tokenize(name)
    if len(tokens) < 3 or any(t.from_arabic for t in tokens):
        return set()
    keys = set()
    for k in (1, 2):
        part = tokens[:k] + tokens[-1:]
        keys.add("L|" + " ".join(sorted(_word_forms(t, arabic=False)[0] for t in part)))
    return keys


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, neutral_key(a), neutral_key(b)).ratio()


def _is_latin(name: str) -> bool:
    return not re.search(r"[^\x00-\x7f]", fold_latin(name))


def _alias_query(entity: Entity, rng: random.Random) -> str | None:
    """A Latin alias close to another kept name. An alias with a twin that differs only in
    case, accents, punctuation or word order is never used, because hiding the alias would
    leave the twin in the index for the exact baseline to find."""
    counts = Counter(neutral_key(n) for n in entity.names)
    latin = [n for n in entity.names if _is_latin(n)]
    options = []
    for alias in latin[1:]:
        if counts[neutral_key(alias)] > 1:
            continue
        others = [n for n in latin if n != alias]
        if others and max(_similar(alias, o) for o in others) >= ALIAS_SIMILARITY:
            options.append(alias)
    return rng.choice(options) if options else None


_NOT_GIVEN_NAMES = TITLES | ARTICLES | CONNECTORS | FRENCH_TITLE_WORDS


def _family_word(entity: Entity) -> str:
    return entity.name.split()[-1]


def _clean_name(a: Entity, b: Entity, rng: random.Random) -> str | None:
    """One or two given names of `a`, titles and particles skipped, and the family name of
    `b`."""
    given = [
        w
        for w in a.name.split()[:-1]
        if re.sub(r"[^a-z]", "", fold_latin(w).casefold()) not in _NOT_GIVEN_NAMES
    ][: rng.choice([1, 2])]
    if not given:
        return None
    return " ".join([*given, _family_word(b)])


def _arabic_name(entity: Entity) -> str | None:
    """The person's first name written only in Arabic script, if any."""
    return next((n for n in entity.names if has_arabic(n) and not re.search("[A-Za-z]", n)), None)


def _arabic_clean_name(a: str, b: str, rng: random.Random) -> str | None:
    """One or two given names from Arabic script name `a` (titles and connectors skipped)
    and the last word of Arabic script name `b`."""
    given = [w for w in a.split()[:-1] if not is_arabic_particle(w)][: rng.choice([1, 2])]
    family = b.split()[-1:]
    if not given or not family or is_arabic_particle(family[0]):
        return None
    return " ".join([*given, *family])


def build(
    entities: Sequence[Entity],
    *,
    seed: int = 13,
    entities_per_split: dict[str, int] | None = None,
    clean_per_split: dict[str, int] | None = None,
    arabic_clean_per_split: dict[str, int] | None = None,
    variants_per_entity: int = 2,
) -> EvalSet:
    rng = random.Random(seed)
    per_split = entities_per_split or {"train": 4000, "dev": 2000, "test": 2000}
    clean_n = clean_per_split or {"train": 6000, "dev": 5000, "test": 5000}
    # Clean names in Arabic script too, so a scorer pays for false alerts on Arabic script
    # queries (it would otherwise learn to lift every Arabic script query for free).
    arabic_clean_n = arabic_clean_per_split or {"train": 1000, "dev": 1000, "test": 1000}

    pools: dict[str, list[Entity]] = {s: [] for s in SPLITS}
    for e in sorted(entities, key=lambda e: hashlib.sha1(f"{seed}/{e.id}".encode()).hexdigest()):
        if _eligible(e):
            pools[split_of(e.id, seed)].append(e)

    queries: list[Query] = []
    held_out: dict[str, list[str]] = {}
    counter = 0

    def add(text: str, target: Entity | None, kind: str, group: str, split: str) -> None:
        nonlocal counter
        counter += 1
        queries.append(
            Query(f"q{counter:06d}", text, target.id if target else None, kind, group, split)
        )

    for split in SPLITS:
        for e in pools[split][: per_split[split]]:
            group = origin_group(e)
            hidden: set[str] = set()
            alias = _alias_query(e, rng)
            if alias:
                hidden.add(alias)
                add(alias, e, "alias", group, split)
            arabic = [n for n in e.names if has_arabic(n)]
            if arabic and any(not has_arabic(n) for n in e.names):
                hidden.update(arabic)
                add(rng.choice(arabic), e, "cross_script", group, split)
            if hidden:
                held_out[e.id] = sorted(hidden)
            kept_keys = {neutral_key(n) for n in e.names if n not in hidden}
            kinds = list(GENERATORS)
            rng.shuffle(kinds)
            made = 0
            for kind in kinds:
                if made >= variants_per_entity:
                    break
                variant = make_variant(e.name, kind, rng)
                if variant and neutral_key(variant) not in kept_keys:
                    add(variant, e, kind, group, split)
                    made += 1

    # Clean names come after every positive, so none can be a listed name, a positive query
    # in any split, or an earlier clean name under another spelling.
    used = {k for e in entities for n in e.names for k in listed_keys(n)}
    used |= {k for e in entities for n in e.names for k in _partial_keys(n)}
    for q in queries:
        used |= name_keys(q.text)
    arabic_listed: set[str] | None = None
    for split in SPLITS:
        donors = pools[split]
        by_group: dict[str, list[Entity]] = defaultdict(list)
        for d in donors:
            by_group[origin_group(d)].append(d)
        made = attempts = 0
        while made < clean_n[split] and attempts < clean_n[split] * 20 and donors:
            attempts += 1
            a = rng.choice(donors)
            group = origin_group(a)
            family = name_key(_family_word(a))
            sample = rng.sample(by_group[group], min(len(by_group[group]), 25))
            pool = [d for d in sample if d.id != a.id and name_key(_family_word(d)) != family]
            if not pool:
                continue
            name = _clean_name(a, rng.choice(pool), rng)
            if not name or len(tokenize(name)) < 2:
                continue
            keys = name_keys(name)
            if keys & used:
                continue
            used |= keys
            add(name, None, "clean", group, split)
            made += 1

        if arabic_listed is None:  # built once, only when Arabic script clean names are made
            arabic_listed = {k for e in entities for n in e.names for k in arabic_view_keys(n)}
        arabic = [(d, n) for d in donors if (n := _arabic_name(d))]
        made = misses = 0
        # Stop after a long run of misses: a small list runs out of new combinations.
        while made < arabic_clean_n[split] and misses < 2000 and len(arabic) >= 2:
            a, a_name = rng.choice(arabic)
            family = name_key(a_name.split()[-1])
            sample = rng.sample(arabic, min(len(arabic), 25))
            pool = [(d, n) for d, n in sample if d.id != a.id and name_key(n.split()[-1]) != family]
            name = _arabic_clean_name(a_name, rng.choice(pool)[1], rng) if pool else None
            keys = name_keys(name) if name and len(tokenize(name)) >= 2 else set()
            if not keys or keys & used or keys & arabic_listed:
                misses += 1
                continue
            misses = 0
            used |= keys
            add(name, None, "clean", origin_group(a), split)
            made += 1
    return EvalSet(seed=seed, queries=queries, held_out=held_out)


def kinds(queries: Iterable[Query]) -> list[str]:
    return sorted({q.kind for q in queries})
