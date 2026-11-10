"""Screening service: the operations the MCP server exposes, with no MCP code in them.

Keeping this layer plain Python means the same calls can be tested directly, served over
MCP, or called by the triage agent without a server in between.

Customer identifiers (birth year and country) are read in one place, here, by
`clean_birth_year` and `clean_country`. The agreement flags, the triage agent's clearing
rule and the CLI all use them, so a value that is not recognised counts as missing
everywhere and never as a disagreement.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sanctions_mcp.data import (
    SNAPSHOT,
    Entity,
    describe_list,
    load_watchlist,
    snapshot_path,
)
from sanctions_mcp.index import WatchlistIndex
from sanctions_mcp.matcher import Matcher
from sanctions_mcp.scoring import WEAK, LogisticScorer, RuleScorer

MAX_NAME_LENGTH = 200
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
EARLIEST_BIRTH_YEAR = 1900
SCORERS = ("learned", "rules")
# A listed person's id as OpenSanctions writes them: letters, digits and . _ : - only. The
# MCP server accepts no other id, and the triage agent puts no other id into a prompt, so
# an id read from a list file can never carry text such as instructions.
ENTITY_ID_MAX_LENGTH = 80
ENTITY_ID_PATTERN = rf"^[A-Za-z0-9._:-]{{1,{ENTITY_ID_MAX_LENGTH}}}$"
_ENTITY_ID = re.compile(ENTITY_ID_PATTERN)

# Every officially assigned ISO 3166-1 alpha-2 code, lowercase as OpenSanctions writes them.
_ISO_CODES = """
ad ae af ag ai al am ao aq ar as at au aw ax az ba bb bd be bf bg bh bi bj bl bm bn bo bq br
bs bt bv bw by bz ca cc cd cf cg ch ci ck cl cm cn co cr cu cv cw cx cy cz de dj dk dm do dz
ec ee eg eh er es et fi fj fk fm fo fr ga gb gd ge gf gg gh gi gl gm gn gp gq gr gs gt gu gw
gy hk hm hn hr ht hu id ie il im in io iq ir is it je jm jo jp ke kg kh ki km kn kp kr kw ky
kz la lb lc li lk lr ls lt lu lv ly ma mc md me mf mg mh mk ml mm mn mo mp mq mr ms mt mu mv
mw mx my mz na nc ne nf ng ni nl no np nr nu nz om pa pe pf pg ph pk pl pm pn pr ps pt pw py
qa re ro rs ru rw sa sb sc sd se sg sh si sj sk sl sm sn so sr ss st sv sx sy sz tc td tf tg
th tj tk tl tm tn to tr tt tv tw tz ua ug um us uy uz va vc ve vg vi vn vu wf ws ye yt za zm
zw
"""
ISO_COUNTRIES = frozenset(_ISO_CODES.split())
# Codes people often write instead of the ISO one: UK for gb, EL (the EU's code) for gr.
COUNTRY_ALIASES = {"uk": "gb", "el": "gr"}


class ScreeningError(ValueError):
    """A request the service cannot answer, with a message safe to show the caller."""


class CalibrationError(RuntimeError):
    """No alert threshold was calibrated for the list that is loaded. Alert flags from an
    uncalibrated threshold would mean nothing, so nothing that shows them may start."""


@dataclass(frozen=True)
class Candidate:
    entity_id: str
    listed_name: str
    matched_name: str
    score: float
    alert: bool
    birth_dates: tuple[str, ...]
    countries: tuple[str, ...]
    birth_year_agrees: bool | None
    country_agrees: bool | None


def clean_name(name: str) -> str:
    text = _CONTROL.sub(" ", name or "").strip()
    if not text:
        raise ScreeningError("name is empty")
    if len(text) > MAX_NAME_LENGTH:
        raise ScreeningError(f"name is longer than {MAX_NAME_LENGTH} characters")
    return " ".join(text.split())


def valid_entity_id(entity_id: object) -> bool:
    """True for an id that matches ENTITY_ID_PATTERN."""
    return isinstance(entity_id, str) and _ENTITY_ID.fullmatch(entity_id) is not None


def clean_country(value: object) -> str | None:
    """A country as a lowercase ISO 3166-1 alpha-2 code, or None when it is missing or not
    a code in ISO_COUNTRIES. Spaces and case are ignored, and UK and EL map to gb and gr.
    A value that is not recognised (IRQ, Iraq, ZZ) counts as missing."""
    if not isinstance(value, str):
        return None
    code = value.strip().lower()
    code = COUNTRY_ALIASES.get(code, code)
    return code if code in ISO_COUNTRIES else None


def clean_birth_year(value: object) -> int | None:
    """A birth year from 1900 to this year, or None when it is missing or not such a year.
    A four digit string is read as a year; anything else (70, 1970.5, 2100) counts as
    missing."""
    if isinstance(value, str):
        text = value.strip()
        if not (len(text) == 4 and text.isdigit()):
            return None
        value = int(text)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if EARLIEST_BIRTH_YEAR <= value <= date.today().year else None


def birth_year_agrees(birth_year: object, dates: Sequence[str]) -> bool | None:
    """Whether the customer's birth year is one of the listed birth years. None when either
    side has no usable year."""
    year = clean_birth_year(birth_year)
    listed = {d[:4] for d in dates if d[:4].isdigit()}
    if year is None or not listed:
        return None
    return str(year) in listed


def country_agrees(country: object, countries: Sequence[str]) -> bool | None:
    """Whether the customer's country is one of the listed countries. None when either
    side has no recognised ISO code (a listed code such as suhh, the Soviet Union, is not
    compared)."""
    code = clean_country(country)
    # OpenSanctions also lists territories such as ua-cri (Crimea): compare their country.
    parents = [str(c).strip().split("-")[0] for c in countries]
    listed = {c for c in map(clean_country, parents) if c}
    if code is None or not listed:
        return None
    if code in listed:
        return True
    if len(listed) < len(set(parents)):  # some listed code could not be read
        return None
    return False


def model_path(data_dir: Path, scorer: str = "learned") -> Path:
    """Where `sanctions train` saves the learned scorer (scorer.json) and the rule scorer's
    calibrated threshold (rules.json)."""
    if scorer not in SCORERS:
        raise ValueError(f"scorer {scorer!r} is not one of {SCORERS}")
    name = "scorer.json" if scorer == "learned" else "rules.json"
    return data_dir / "models" / SNAPSHOT / name


def describe_calibration(calibration: dict) -> str:
    """One line on where an alert threshold comes from, for list_sources and the CLI."""
    return (
        f"calibrated by `sanctions train`: at most {calibration['budget_per_1000']} false "
        f"alerts per 1,000 clean names of the {calibration['split']} split "
        f"({calibration['clean_names']:,} names), scored against the full list"
    )


def load_calibrated_scorer(
    data_dir: Path, list_sha256: str, scorer: str = "learned"
) -> tuple[RuleScorer | LogisticScorer, float, str]:
    """The scorer and the alert threshold `sanctions train` calibrated for this exact list
    file, with a line on where the threshold comes from. Raises CalibrationError when there
    is none: no model yet, a model from before thresholds were calibrated on the full list,
    or a model calibrated on another list file."""
    path = model_path(data_dir, scorer)
    if not path.exists():
        raise CalibrationError(f"{path.as_posix()} not found. Run `sanctions train` first.")
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise TypeError("not a JSON object")
    except (OSError, ValueError, TypeError) as err:
        raise CalibrationError(
            f"{path.as_posix()} could not be read ({err}). Run `sanctions train` again."
        ) from None
    calibration = saved.get("calibration")
    threshold = saved.get("alert_threshold")
    if not isinstance(calibration, dict) or not isinstance(threshold, int | float):
        raise CalibrationError(
            f"{path.as_posix()} has no alert threshold calibrated on the full list. "
            "Run `sanctions train` again."
        )
    if saved.get("list_sha256") != list_sha256:
        raise CalibrationError(
            f"The alert threshold in {path.as_posix()} was calibrated on a different list "
            "file. Run `sanctions train` again."
        )
    try:
        model = LogisticScorer.load(path) if scorer == "learned" else RuleScorer()
    except ValueError as err:  # trained on other features, or no threshold
        raise CalibrationError(str(err)) from None
    return model, float(threshold), describe_calibration(calibration)


class ScreeningService:
    def __init__(
        self,
        entities: Sequence[Entity],
        *,
        alert_threshold: float,
        scorer: RuleScorer | LogisticScorer | None = None,
        threshold_source: str = "given by the caller",
        dataset: dict | None = None,
    ):
        """`alert_threshold` has no default on purpose: a threshold that was not calibrated
        for this scorer and list would make every alert flag meaningless."""
        self.entities = {e.id: e for e in entities}
        self.matcher = Matcher(WatchlistIndex(entities), scorer or RuleScorer())
        self.alert_threshold = alert_threshold
        self.threshold_source = threshold_source
        self.dataset = dataset

    @classmethod
    def from_data_dir(cls, data_dir: Path, *, scorer: str = "learned") -> ScreeningService:
        """Load the list and the alert threshold `sanctions train` calibrated for it on the
        full list index, the index this service screens against. Fails closed with
        CalibrationError when there is no such threshold for this exact list file."""
        path = snapshot_path(data_dir)
        entities = load_watchlist(path)
        dataset = describe_list(path, len(entities))
        model, threshold, source = load_calibrated_scorer(data_dir, dataset["sha256"], scorer)
        return cls(
            entities,
            scorer=model,
            alert_threshold=threshold,
            threshold_source=source,
            dataset=dataset,
        )

    def screen(
        self,
        name: str,
        *,
        birth_year: int | None = None,
        country: str | None = None,
        limit: int = 10,
    ) -> list[Candidate]:
        text = clean_name(name)
        out = []
        for m in self.matcher.screen(text, top_k=max(1, min(limit, 25))):
            e = self.entities[m.entity_id]
            out.append(
                Candidate(
                    entity_id=e.id,
                    listed_name=e.name,
                    matched_name=m.matched_name,
                    score=round(m.score, 4),
                    alert=m.score >= self.alert_threshold,
                    birth_dates=e.birth_dates,
                    countries=e.countries,
                    birth_year_agrees=birth_year_agrees(birth_year, e.birth_dates),
                    country_agrees=country_agrees(country, e.countries),
                )
            )
        return out

    def entity(self, entity_id: str) -> Entity:
        try:
            return self.entities[entity_id]
        except KeyError:
            raise ScreeningError(f"no listed person with id {entity_id!r}") from None

    def explain(self, name: str, entity_id: str) -> dict:
        """The matcher's explanation of one score, with the alert flag screening would give.

        The alert is set only when the person is among the candidates screening scores for
        this name (`retrieved`) and the score reaches the threshold. Every key below is
        always present, so the MCP server and the triage agent see the same fields."""
        self.entity(entity_id)
        text = clean_name(name)
        why = dict(self.matcher.explain(text, entity_id))
        retrieved = why.get("retrieved")
        if retrieved is None:  # a matcher that does not say: check the candidates here
            retrieved = entity_id in self.matcher.candidate_entities(text)
        why["retrieved"] = bool(retrieved)
        if not why["retrieved"] and not why.get("reason"):
            limit = self.matcher.candidate_limit
            why["reason"] = f"not among the top {limit} candidates for this name"
        why.setdefault("reason", None)
        why["score"] = float(why.get("score", 0.0))
        why.setdefault("matched_name", None)
        why["alignment"] = [
            {**a, "matched": bool(a.get("matched", a["similarity"] >= WEAK))}
            for a in why.get("alignment", [])
        ]
        why.setdefault("unmatched_query_words", [])
        why.setdefault("unmatched_listed_words", [])
        why.setdefault("features", {})
        why["alert"] = why["retrieved"] and why["score"] >= self.alert_threshold
        why["alert_threshold"] = self.alert_threshold
        return why

    def sources(self) -> dict:
        """The list loaded and the threshold in use. The publisher, snapshot and license
        are named only when the loaded file is the pinned OpenSanctions snapshot."""
        dataset = self.dataset or {}
        pinned = dataset.get("snapshot") is not None
        lists = Counter(ds for e in self.entities.values() for ds in e.datasets)
        return {
            "publisher": "OpenSanctions" if pinned else "unknown (not the pinned snapshot)",
            "collection": dataset.get("collection"),
            "snapshot": dataset.get("snapshot"),
            "license": dataset.get("license") or "unknown: not the pinned OpenSanctions snapshot",
            "list_sha256": dataset.get("sha256"),
            "people": len(self.entities),
            "names_indexed": len(self.matcher.index),
            "scorer": self.matcher.scorer.name,
            "alert_threshold": self.alert_threshold,
            "alert_threshold_source": self.threshold_source,
            "largest_lists": [{"name": n, "people": c} for n, c in lists.most_common(15)],
        }
