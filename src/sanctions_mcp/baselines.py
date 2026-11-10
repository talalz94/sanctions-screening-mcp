"""Baselines a screening team would start from, using no name knowledge at all."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from sanctions_mcp.data import Entity
from sanctions_mcp.normalize import neutral_key


class ExactMatcher:
    """Alerts only when the name equals a listed name, ignoring case, accents,
    punctuation and word order."""

    name = "exact"

    def __init__(self, entities: Sequence[Entity], exclude: dict[str, list[str]] | None = None):
        hidden = {eid: set(names) for eid, names in (exclude or {}).items()}
        self._by_key: dict[str, set[str]] = defaultdict(set)
        for e in entities:
            for n in e.names:
                if n not in hidden.get(e.id, ()):
                    self._by_key[neutral_key(n)].add(e.id)

    def scores(self, text: str, top_k: int | None = None) -> dict[str, float]:
        return dict.fromkeys(sorted(self._by_key.get(neutral_key(text), ()))[:top_k], 1.0)

    def scores_many(self, texts: Sequence[str]) -> list[dict[str, float]]:
        return [self.scores(t) for t in texts]


class TfidfMatcher:
    """Character n-gram TF-IDF with cosine similarity, a common fuzzy matching setup.
    With `rescore`, the top candidates are rescored with RapidFuzz token sort ratio.

    Token sort, not token set: token set ratio scores 100 whenever one name's words are a
    subset of the other's ("Muhammad Ali" against "Muhammad Ali Hassan"), so at a strict
    false alert budget it alerts on almost nothing."""

    def __init__(
        self,
        entities: Sequence[Entity],
        exclude: dict[str, list[str]] | None = None,
        *,
        rescore: bool = False,
        shortlist: int = 200,
    ):
        hidden = {eid: set(names) for eid, names in (exclude or {}).items()}
        self.name = "tfidf-rapidfuzz" if rescore else "tfidf"
        self.rescore = rescore
        self.shortlist = shortlist
        self._texts: list[str] = []
        self._owners: list[str] = []
        for e in entities:
            for n in e.names:
                if n not in hidden.get(e.id, ()):
                    self._texts.append(n)
                    self._owners.append(e.id)
        self._keys = [neutral_key(n) for n in self._texts]  # what RapidFuzz compares
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), lowercase=True, strip_accents="unicode",
            sublinear_tf=True, dtype=np.float32,
        )  # fmt: skip
        self._matrix = self._vectorizer.fit_transform(self._texts).T.tocsr()

    def _rank(self, sims: np.ndarray, text: str, top_k: int | None) -> dict[str, float]:
        k = min(self.shortlist, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        best: dict[str, float] = {}
        # Compared the way the exact baseline compares names: case, accents, punctuation and
        # apostrophes ignored, so "Al-Nouri" and "al Nouri" are the same words.
        key = neutral_key(text) if self.rescore else ""
        for i in top:
            score = float(sims[i])
            if score <= 0:
                continue
            if self.rescore:
                score = fuzz.token_sort_ratio(key, self._keys[i]) / 100.0
            eid = self._owners[i]
            if score > best.get(eid, -1.0):
                best[eid] = score
        return dict(sorted(best.items(), key=lambda kv: -kv[1])[:top_k])

    def scores_many(self, texts: Sequence[str], batch: int = 128) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            sims = (self._vectorizer.transform(chunk) @ self._matrix).toarray()
            out.extend(self._rank(row, text, None) for row, text in zip(sims, chunk, strict=True))
        return out

    def scores(self, text: str, top_k: int | None = None) -> dict[str, float]:
        sims = (self._vectorizer.transform([text]) @ self._matrix).toarray()[0]
        return self._rank(sims, text, top_k)
