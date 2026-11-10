"""Watchlist index and candidate search.

Every listed name is tokenized once. Each word is indexed under its skeleton keys (exact)
and, for keys of four letters or more, under every key with one letter deleted (fuzzy).
Two words meet if their skeletons are equal or one edit apart, which covers most typos
and spelling drift without comparing a query against every listed name.

Candidates are ranked by the rarity of the query words they share. An exact key match
counts fully and a fuzzy one counts 0.4, so a name that matches every word exactly is
never crowded out by thousands of names that match loosely. The cut off counts people,
not names, so one person with many similar aliases cannot take every place.

An initial ("M.") retrieves nothing on its own. It adds a little weight to names that
other query words already found and that have a word starting with that letter.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from sanctions_mcp.data import Entity
from sanctions_mcp.normalize import Token, tokenize

FUZZY_WEIGHT = 0.4
MIN_FUZZY_KEY = 4
ROWS_PER_ENTITY = 5  # an entity's best few names are scored, not all of its aliases
ROW_MARGIN = 1.0  # ... and only names within this much retrieval weight of its best one


@dataclass(frozen=True, slots=True)
class IndexedName:
    entity_id: str
    text: str
    tokens: tuple[Token, ...]


def is_initial(token: Token) -> bool:
    return len(token.folded) == 1


def exact_keys(token: Token) -> set[str]:
    keys = {s for s in token.skeletons if s}
    if not keys and token.folded:
        keys.add("#" + token.folded)  # initials and vowel only words
    return keys


def fuzzy_keys(token: Token) -> set[str]:
    keys: set[str] = set()
    for skel in token.skeletons:
        if len(skel) >= MIN_FUZZY_KEY:
            keys |= {skel[:i] + skel[i + 1 :] for i in range(len(skel))}
    return keys


class WatchlistIndex:
    def __init__(
        self,
        entities: Sequence[Entity],
        *,
        exclude: Mapping[str, Iterable[str]] | None = None,
    ):
        hidden = {eid: set(names) for eid, names in (exclude or {}).items()}
        self.entities = {e.id: e for e in entities}
        self.names: list[IndexedName] = []
        self.listed_names = 0  # names on the list, before identical spellings are merged
        self._initials: list[str] = []  # per row: every letter one of its words can start with
        self._exact: dict[str, list[int]] = defaultdict(list)
        self._fuzzy: dict[str, list[int]] = defaultdict(list)
        letter_df: dict[str, int] = defaultdict(int)
        for e in entities:
            seen: set[tuple[Token, ...]] = set()
            for text in e.names:
                if text in hidden.get(e.id, ()):
                    continue
                tokens = tuple(tokenize(text))
                if not tokens:
                    continue
                self.listed_names += 1
                if tokens in seen:  # the same name once normalised: it would score the same
                    continue
                seen.add(tokens)
                idx = len(self.names)
                self.names.append(IndexedName(e.id, text, tokens))
                initials = "".join(sorted({ch for t in tokens for ch in t.initials}))
                self._initials.append(initials)
                for ch in initials:
                    letter_df[ch] += 1
                exact: set[str] = set()
                fuzzy: set[str] = set()
                for t in tokens:
                    exact |= exact_keys(t)
                    fuzzy |= exact_keys(t) | fuzzy_keys(t)
                for key in exact:
                    self._exact[key].append(idx)
                for key in fuzzy:
                    self._fuzzy[key].append(idx)
        n = len(self.names)
        self.max_idf = math.log(n + 1) + 1.0
        self._letter_idf = {ch: self._idf_of(count) for ch, count in letter_df.items()}
        self._idf_cache: dict[frozenset[str], float] = {}

    def _idf_of(self, df: int) -> float:
        return math.log((len(self.names) + 1) / (df + 1)) + 1.0

    def idf(self, token: Token) -> float:
        """Rarity of a word: high for rare names, low for Muhammad or Ali.

        It counts the names that share any of the word's keys, so a word with two keys
        (Latin "Rashid": rshd and rsd) weighs the same as its Arabic script spelling with
        one. An initial counts every name with a word starting with that letter."""
        if is_initial(token):
            return self._letter_idf.get(token.folded, self.max_idf)
        keys = frozenset(exact_keys(token))
        cached = self._idf_cache.get(keys)
        if cached is None:
            postings = [self._exact.get(k, ()) for k in keys]
            df = len(postings[0]) if len(postings) == 1 else len(set().union(*postings))
            cached = self._idf_of(df) if df else self.max_idf
            self._idf_cache[keys] = cached
        return cached

    def candidates(self, tokens: Sequence[Token], limit: int = 200) -> list[int]:
        """Rows of the `limit` best people for this name, each person's best rows first.

        When the people at the cut match every query word exactly, everyone tied with them is
        kept too (up to twice the limit), so a common name never loses a full match to the
        order of the list. Below that tier, ties break on the entity id."""
        words = [t for t in tokens if not is_initial(t)]
        if not words:
            return []
        weights: dict[int, float] = defaultdict(float)
        for token in words:
            w = self.idf(token)
            exact_hits: set[int] = set()
            for key in exact_keys(token):
                exact_hits.update(self._exact.get(key, ()))
            fuzzy_hits: set[int] = set()
            for key in exact_keys(token) | fuzzy_keys(token):
                fuzzy_hits.update(self._fuzzy.get(key, ()))
            for idx in exact_hits:
                weights[idx] += w
            for idx in fuzzy_hits - exact_hits:
                weights[idx] += FUZZY_WEIGHT * w
        for token in tokens:
            if is_initial(token):
                w = self.idf(token)
                for idx in weights:
                    if token.folded in self._initials[idx]:
                        weights[idx] += w
        n_query = len(tokens)
        for idx in weights:  # prefer names of a similar length when weights tie
            weights[idx] -= 0.01 * abs(len(self.names[idx].tokens) - n_query)

        # Rank people by their best row, never by hash order, so the same query always gets
        # the same candidates.
        best: dict[str, float] = {}
        rows_by_entity: dict[str, list[int]] = defaultdict(list)
        for idx, w in weights.items():
            eid = self.names[idx].entity_id
            rows_by_entity[eid].append(idx)
            if w > best.get(eid, -math.inf):
                best[eid] = w
        ranked = heapq.nsmallest(2 * limit, best, key=lambda e: (-best[e], e))
        if len(ranked) > limit:
            cut = best[ranked[limit - 1]] - 1e-9
            end = limit
            full = sum(self.idf(t) for t in tokens)  # every word matched exactly, same length
            if cut >= full - 1e-6:
                while end < len(ranked) and best[ranked[end]] >= cut:
                    end += 1
            ranked = ranked[:end]
        out: list[int] = []
        for eid in ranked:
            rows = sorted(rows_by_entity[eid], key=lambda i: (-weights[i], i))
            top = weights[rows[0]]
            out.extend(i for i in rows[:ROWS_PER_ENTITY] if weights[i] >= top - ROW_MARGIN)
        return out

    def __len__(self) -> int:
        return len(self.names)
