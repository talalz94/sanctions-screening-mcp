"""The screening matcher: tokenize, find candidates, score each listed name, keep the best
name per entity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sanctions_mcp.index import WatchlistIndex
from sanctions_mcp.normalize import Token, tokenize
from sanctions_mcp.scoring import FEATURES, WEAK, LogisticScorer, RuleScorer, align, features


@dataclass(frozen=True)
class Match:
    entity_id: str
    score: float
    matched_name: str


class Matcher:
    def __init__(
        self,
        index: WatchlistIndex,
        scorer: RuleScorer | LogisticScorer,
        *,
        candidate_limit: int = 200,
    ):
        self.index = index
        self.scorer = scorer
        self.candidate_limit = candidate_limit
        self.name = f"matcher-{scorer.name}"
        self._weights = [[index.idf(t) for t in n.tokens] for n in index.names]

    def _features(
        self, tokens: Sequence[Token], q_weights: Sequence[float], idx: int, memo: dict
    ) -> np.ndarray:
        name = self.index.names[idx]
        weights = self._weights[idx]
        return features(tokens, name.tokens, q_weights, weights, self.index.max_idf, memo)

    def candidate_features(
        self, tokens: Sequence[Token], limit: int | None = None
    ) -> tuple[list[int], np.ndarray]:
        cands = self.index.candidates(tokens, limit or self.candidate_limit)
        if not cands:
            return [], np.zeros((0, len(FEATURES)))
        q_weights = [self.index.idf(t) for t in tokens]
        memo: dict = {}
        return cands, np.stack([self._features(tokens, q_weights, i, memo) for i in cands])

    def scores(self, text: str, top_k: int | None = None) -> dict[str, float]:
        """Best score per candidate entity (all of them by default, for evaluation)."""
        return {m.entity_id: m.score for m in self.screen(text, top_k=top_k)}

    def _best_rows(self, tokens: Sequence[Token]) -> dict[str, tuple[float, int]]:
        """Score every candidate row and keep each entity's best (score, row). On a tie the
        row that comes first among the candidates wins, in screen() and explain() alike."""
        cands, x = self.candidate_features(tokens)
        if not cands:
            return {}
        best: dict[str, tuple[float, int]] = {}
        for idx, score in zip(cands, self.scorer.score(x), strict=True):
            eid = self.index.names[idx].entity_id
            if eid not in best or score > best[eid][0]:
                best[eid] = (float(score), idx)
        return best

    def screen(self, text: str, top_k: int | None = 10) -> list[Match]:
        best = self._best_rows(tokenize(text))
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:top_k]
        return [Match(eid, score, self.index.names[idx].text) for eid, (score, idx) in ranked]

    def candidate_entities(self, text: str) -> set[str]:
        tokens = tokenize(text)
        return {
            self.index.names[i].entity_id
            for i in self.index.candidates(tokens, self.candidate_limit)
        }

    def explain(self, text: str, entity_id: str) -> dict:
        """Why `text` scores what it does against one listed entity: the name screening
        picked, the word by word alignment and every feature value.

        It scores the same candidate rows as screen(), so the score and matched name are
        always the ones screen_party reported. An entity that was not among the candidates
        scores 0.0, as it does in screening."""
        tokens = tokenize(text)
        has_names = any(n.entity_id == entity_id for n in self.index.names)
        if not tokens or not has_names:
            return _no_comparison(entity_id, "no comparable names")
        rows = [
            i
            for i in self.index.candidates(tokens, self.candidate_limit)
            if self.index.names[i].entity_id == entity_id
        ]
        if not rows:
            reason = f"not among the top {self.candidate_limit} candidates for this name"
            return _no_comparison(entity_id, reason)
        q_weights = [self.index.idf(t) for t in tokens]
        memo: dict = {}
        x = np.stack([self._features(tokens, q_weights, i, memo) for i in rows])
        s = self.scorer.score(x)
        k = int(np.argmax(s))  # the first best row, as in screen()
        name = self.index.names[rows[k]]
        pairs = sorted(align(tokens, name.tokens).pairs)
        q_sim = {i: sim for i, _, sim in pairs}
        n_sim = {j: sim for _, j, sim in pairs}
        return {
            "entity_id": entity_id,
            "score": float(s[k]),
            "retrieved": True,
            "matched_name": name.text,
            "alignment": [
                {
                    "query": tokens[i].folded,
                    "listed": name.tokens[j].folded,
                    "similarity": round(sim, 3),
                    "matched": sim >= WEAK,
                }
                for i, j, sim in pairs
            ],
            # The same rule as the query_unmatched and name_unmatched features.
            "unmatched_query_words": [
                t.folded for i, t in enumerate(tokens) if q_sim.get(i, 0.0) < WEAK
            ],
            "unmatched_listed_words": [
                t.folded for j, t in enumerate(name.tokens) if n_sim.get(j, 0.0) < WEAK
            ],
            "features": {f: round(float(v), 4) for f, v in zip(FEATURES, x[k], strict=True)},
            "reason": None,
        }


def _no_comparison(entity_id: str, reason: str) -> dict:
    return {
        "entity_id": entity_id,
        "score": 0.0,
        "retrieved": False,
        "matched_name": None,
        "alignment": [],
        "unmatched_query_words": [],
        "unmatched_listed_words": [],
        "features": {},
        "reason": reason,
    }


def training_pairs(
    matcher: Matcher, queries: Sequence, *, limit: int | None = None, top_k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Training rows that mirror the screening decision, built from the same candidate rows
    that screening scores (`matcher.candidate_limit` unless `limit` is given).

    Positive: for a query that names a listed person, that person's best scoring name.
    Negative: for a clean query, its top `top_k` candidates under the fixed rule scorer,
    the names most likely to raise false alerts.

    Other candidates of a positive query are left out on purpose: many are namesakes with
    the same name as the target, and labelling them negative would teach the model noise.
    """
    rule = RuleScorer()
    xs, ys = [], []
    for q in queries:
        cands, x = matcher.candidate_features(tokenize(q.text), limit)
        if not cands:
            continue
        s = rule.score(x)
        if q.target is None:
            order = np.argsort(-s, kind="stable")[:top_k]
            xs.append(x[order])
            ys.extend([False] * len(order))
            continue
        mine = [i for i, c in enumerate(cands) if matcher.index.names[c].entity_id == q.target]
        if mine:
            best = max(mine, key=lambda i: s[i])
            xs.append(x[best : best + 1])
            ys.append(True)
    if not xs:
        return np.zeros((0, len(FEATURES))), np.zeros(0, dtype=bool)
    return np.vstack(xs), np.array(ys, dtype=bool)
