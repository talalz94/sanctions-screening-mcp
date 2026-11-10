"""Pair features, token alignment and the two scorers (fixed rules, learned weights).

The learned scorer is a logistic regression over a handful of named features, so every
score can be traced back to weights a reviewer can read.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rapidfuzz.distance import JaroWinkler

from sanctions_mcp.normalize import ARABIC_SPELLED_NAMES, KNOWN_GIVEN_NAMES, Token, latin_token

STRONG = 0.92
WEAK = 0.80
DISTINCT_NAME_CAP = 0.90  # two different well known names are never a strong match
# Across scripts a shared key of two consonants or more is a full match (the other guards
# in _cross_script_similarity keep distinct names apart); one shared consonant is weak.
CROSS_SCRIPT_SHORT_KEY = {2: 1.0, 1: 0.85}

FEATURES = (
    "query_coverage",
    "name_coverage",
    "query_unmatched",
    "name_unmatched",
    "strong_matches",
    "weakest_match",
    "rarest_match",
    "length_gap",
    "single_token",
    "cross_script",
    "evidence",
)


def _key_floor(length: int) -> float:
    return 0.97 if length >= 4 else 0.94 if length == 3 else 0.88


def token_similarity(q: Token, c: Token) -> float:
    if len(q.folded) == 1 or len(c.folded) == 1:  # an initial matches what it can stand for
        return 0.9 if (q.folded in c.initials or c.folded in q.initials) else 0.0
    # Two well known names that differ (Hasan and Husain, Bashir and Bashar) are compared as
    # written and can be a near miss at most, never lifted by a shared consonant key.
    distinct_names = (
        q.folded != c.folded and q.folded in KNOWN_GIVEN_NAMES and c.folded in KNOWN_GIVEN_NAMES
    )
    if q.from_arabic or c.from_arabic:
        return _cross_script_similarity(q, c, distinct_names)
    folded = JaroWinkler.normalized_similarity(q.folded, c.folded)
    if distinct_names:
        return min(folded, DISTINCT_NAME_CAP)
    # A shared key lifts the score only when the vowel pattern behind it matches too, so
    # Kareem meets Karim but Akram does not; a key read off a glued "al" lifts it less.
    floor = 0.0
    for i, key in enumerate(q.skeletons):
        if not key or key not in c.skeletons:
            continue
        j = c.skeletons.index(key)
        if _at(q.shapes, i) != _at(c.shapes, j):
            continue
        level = _key_floor(len(key))
        if _at(q.glued, i) or _at(c.glued, j):
            level = min(level, _key_floor(2))
        floor = max(floor, level)
    return max(folded, floor)


def _ends_in_i(token: Token) -> bool:
    """A final long i: written ي or ی in Arabic script (Amiri, Karimi, Ali), and i, y or ee
    in Latin. Arabic script writes long vowels, so this is one vowel both sides show."""
    if token.from_arabic:
        return token.folded.endswith("y")
    return token.folded.endswith(("i", "y", "ee"))


def _cross_script_similarity(q: Token, c: Token, distinct_names: bool) -> float:
    """At least one word is in Arabic script, so short vowels are missing on that side and
    the consonant keys carry the comparison, with four guards against a false 1.0."""
    if distinct_names:
        return min(JaroWinkler.normalized_similarity(q.folded, c.folded), DISTINCT_NAME_CAP)
    if q.folded == c.folded and set(q.skeletons) & set(c.skeletons):
        return 1.0  # the same word, such as one known name in both scripts
    if q.from_arabic and c.from_arabic:
        # Within Arabic script a name keeps its letters, long vowels included, so two
        # different spellings are two different names (سلمان Salman, سليمان Sulaiman).
        return _key_miss(q, c)
    if q.from_arabic != c.from_arabic:
        lat, ar = (c, q) if q.from_arabic else (q, c)
        # 1. An Arabic word recognised as a known name is compared as that name's Latin
        #    spelling, vowel pattern included (Makhmoud meets محمود, Mahmoudi does not).
        if ar.folded in KNOWN_GIVEN_NAMES:
            return token_similarity(lat, latin_token(ar.folded))
        # 2. A known name whose Arabic spelling is known is not an unrecognised Arabic word
        #    (Hammad, written حماد, is not حميد).
        if lat.folded in ARABIC_SPELLED_NAMES:
            return _key_miss(q, c)
    q_keys = {
        k: g for k, g in zip(q.skeletons, q.glued or (False,) * len(q.skeletons), strict=True) if k
    }
    c_keys = {
        k: g for k, g in zip(c.skeletons, c.glued or (False,) * len(c.skeletons), strict=True) if k
    }
    if not q_keys or not c_keys:  # a vowel or digit only word has no key to compare
        if q.folded == c.folded:
            return 1.0
        return 0.85 * JaroWinkler.normalized_similarity(q.folded, c.folded)
    shared = q_keys.keys() & c_keys.keys()
    # 3. A final long i shows on both sides, so Amir is not امیری (Amiri).
    if not shared or _ends_in_i(q) != _ends_in_i(c):
        return _key_miss(q, c)
    level = 0.0
    for key in shared:
        # 4. One consonant says little (Ayyub and أبو share only b).
        step = 1.0 if len(key) >= 3 else CROSS_SCRIPT_SHORT_KEY.get(len(key), 0.85)
        if q_keys[key] or c_keys[key]:  # read off a glued "al"
            step = min(step, _key_floor(2))
        level = max(level, step)
    return level


def _key_miss(q: Token, c: Token) -> float:
    q_keys = [k for k in q.skeletons if k] or [q.folded]
    c_keys = [k for k in c.skeletons if k] or [c.folded]
    return 0.85 * max(JaroWinkler.normalized_similarity(a, b) for a in q_keys for b in c_keys)


def _at(values: tuple, i: int):
    return values[i] if i < len(values) else None


@dataclass(frozen=True)
class Alignment:
    pairs: tuple[tuple[int, int, float], ...]  # (query token, name token, similarity)


def align(query: Sequence[Token], name: Sequence[Token], memo: dict | None = None) -> Alignment:
    """Greedy one to one alignment by similarity, ignoring word order.

    `memo` caches similarities for one query across many listed names: the same listed word
    is the same Token object in every name that has it."""
    scored = []
    for i, q in enumerate(query):
        for j, c in enumerate(name):
            if memo is None:
                sim = token_similarity(q, c)
            else:
                key = (i, id(c))
                sim = memo.get(key)
                if sim is None:
                    sim = memo[key] = token_similarity(q, c)
            scored.append((sim, i, j))
    scored.sort(reverse=True)
    used_q: set[int] = set()
    used_c: set[int] = set()
    pairs = []
    for sim, i, j in scored:
        if i in used_q or j in used_c:
            continue
        used_q.add(i)
        used_c.add(j)
        pairs.append((i, j, sim))
    return Alignment(tuple(pairs))


def features(
    query: Sequence[Token],
    name: Sequence[Token],
    q_weights: Sequence[float],
    n_weights: Sequence[float],
    max_idf: float,
    memo: dict | None = None,
) -> np.ndarray:
    alignment = align(query, name, memo)
    q_sim = [0.0] * len(query)
    n_sim = [0.0] * len(name)
    for i, j, sim in alignment.pairs:
        q_sim[i] = sim
        n_sim[j] = sim
    q_total = sum(q_weights) or 1.0
    n_total = sum(n_weights) or 1.0
    strong = [w for w, s in zip(q_weights, q_sim, strict=True) if s >= STRONG]
    matched = [s for s in q_sim if s >= WEAK]
    # An Arabic script query against a name with no Arabic script word. A Latin query against
    # an Arabic script alias is not counted, so the weight never lifts those pairs.
    cross = any(t.from_arabic for t in query) and not any(t.from_arabic for t in name)
    return np.array(
        [
            sum(w * s for w, s in zip(q_weights, q_sim, strict=True)) / q_total,
            sum(w * s for w, s in zip(n_weights, n_sim, strict=True)) / n_total,
            sum(s < WEAK for s in q_sim) / len(query),
            sum(s < WEAK for s in n_sim) / len(name),
            min(len(strong), 4) / 4,
            min(matched) if matched else 0.0,
            (max(strong) / max_idf) if strong else 0.0,
            min(abs(len(query) - len(name)), 3) / 3,
            float(len(query) == 1 or len(name) == 1),
            float(cross),
            min(sum(strong) / max_idf, 3.0) / 3.0,
        ],
        dtype=np.float64,
    )


class RuleScorer:
    """Fixed weights: the average of both coverages, minus a penalty for unmatched words."""

    name = "rules"

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        cov = 0.5 * x[:, 0] + 0.5 * x[:, 1]
        return np.clip(cov - 0.1 * x[:, 3], 0.0, 1.0)


class LogisticScorer:
    """Logistic regression on standardised features, fitted on queries of the train split
    only, on rows that mirror the screening decision (see `matcher.training_pairs`)."""

    name = "learned"

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        coef: np.ndarray,
        intercept: float,
        alert_threshold: float | None = None,
    ):
        self.mean, self.scale, self.coef, self.intercept = mean, scale, coef, intercept
        self.alert_threshold = alert_threshold  # set on dev clean names by `sanctions train`

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, *, c: float = 1.0) -> LogisticScorer:
        from sklearn.linear_model import LogisticRegression

        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        model = LogisticRegression(C=c, max_iter=2000)
        model.fit((x - mean) / scale, y)
        return cls(mean, scale, model.coef_[0], float(model.intercept_[0]))

    def score(self, x: np.ndarray) -> np.ndarray:
        # A plain row sum rather than a BLAS product: a row gets the same bits whatever
        # batch it is scored in, so screen() and explain() always agree.
        z = (((np.atleast_2d(x) - self.mean) / self.scale) * self.coef).sum(axis=1)
        z = z + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def weights(self) -> dict[str, float]:
        """Weight per feature, in standard deviations of that feature."""
        return dict(zip(FEATURES, (float(w) for w in self.coef), strict=True))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "features": list(FEATURES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "alert_threshold": self.alert_threshold,
        }
        tmp = path.with_name(path.name + ".tmp")  # written whole, then swapped in
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> LogisticScorer:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `sanctions train` first.")
        p = json.loads(path.read_text(encoding="utf-8"))
        if p["features"] != list(FEATURES):
            raise ValueError(f"{path} was trained on different features; retrain it")
        if p.get("alert_threshold") is None:
            raise ValueError(f"{path} has no calibrated alert threshold; run `sanctions train`")
        return cls(
            np.array(p["mean"]),
            np.array(p["scale"]),
            np.array(p["coef"]),
            p["intercept"],
            p["alert_threshold"],
        )
