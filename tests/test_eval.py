import hashlib

import httpx
import pytest

from sanctions_mcp.baselines import ExactMatcher, TfidfMatcher
from sanctions_mcp.data import ChecksumError, download, load_watchlist, snapshot_path
from sanctions_mcp.evalset import EvalSet, build, name_key, name_keys, split_of
from sanctions_mcp.evaluate import SystemRun, latency_indices, run_system, summarize, threshold_for
from sanctions_mcp.normalize import has_arabic, neutral_key, tokenize


@pytest.fixture(scope="module")
def evalset(entities):
    return build(
        entities,
        seed=3,
        entities_per_split={"train": 50, "dev": 50, "test": 50},
        clean_per_split={"train": 20, "dev": 20, "test": 20},
    )


def test_splits_are_by_entity(evalset):
    for q in evalset.queries:
        if q.target:
            assert split_of(q.target, 3) == q.split


def test_clean_names_are_never_listed_namesakes(entities, evalset):
    listed = {name_key(n) for e in entities for n in e.names}
    clean = [q for q in evalset.queries if q.kind == "clean"]
    assert clean and all(not name_keys(q.text) & listed for q in clean)
    assert all(len(tokenize(q.text)) >= 2 for q in clean)


def test_clean_names_never_repeat_a_positive_or_each_other(evalset):
    positive = set().union(*(name_keys(q.text) for q in evalset.queries if q.target))
    seen: set[str] = set()
    for q in evalset.queries:
        if q.kind == "clean":
            keys = name_keys(q.text)
            assert not keys & positive, q.text
            assert not keys & seen, q.text
            seen |= keys


def test_clean_names_are_not_a_listed_name_cut_short(entities, evalset):
    cut = {
        name_key(" ".join(e.name.split()[:k] + e.name.split()[-1:]))
        for e in entities
        for k in (1, 2)
        if len(e.name.split()) >= 3
    }
    assert all(name_key(q.text) not in cut for q in evalset.queries if q.kind == "clean")


def test_skeleton_only_collisions_are_allowed_as_clean_names():
    # Hamid Ali is a different person from a listed Ahmad Ali, even though the two share
    # consonant keys, so the eval may use it to count false alerts.
    assert not name_keys("Hamid Ali") & {name_key("Ahmad Ali")}
    assert name_keys("Kareem Ali") & {name_key("Karim Ali")}


def test_alias_queries_leave_no_same_string_twin_in_the_index(entities, evalset):
    by_id = {e.id: e for e in entities}
    for q in evalset.queries:
        if q.kind == "alias":
            kept = [n for n in by_id[q.target].names if n not in evalset.held_out[q.target]]
            assert neutral_key(q.text) not in {neutral_key(n) for n in kept}, q.text


def test_cross_script_queries_hold_out_every_arabic_name(entities):
    from sanctions_mcp.data import Entity

    twice = Entity(
        "FX-201", "Person", "Hamid Karim Saidi", ("حامد كريم سعيدي", "حامد كريم السعيدي"),
        (), ("iq",), ("test",),
    )  # fmt: skip
    es = build(
        [*entities, twice],
        seed=3,
        entities_per_split={"train": 50, "dev": 50, "test": 50},
        clean_per_split={"train": 5, "dev": 5, "test": 5},
    )
    by_id = {e.id: e for e in [*entities, twice]}
    cross = [q for q in es.queries if q.kind == "cross_script"]
    assert any(q.target == "FX-201" for q in cross)
    for q in cross:
        assert has_arabic(q.text)
        arabic = {n for n in by_id[q.target].names if has_arabic(n)}
        assert arabic <= set(es.held_out[q.target])


def test_listed_names_block_clean_names_on_every_reading():
    from sanctions_mcp.evalset import listed_keys

    assert name_keys("Ali Reza") & listed_keys("علي رضا")
    assert name_keys("Ilyas Khan") & listed_keys("الياس خان")


def test_eval_set_round_trip(evalset, tmp_path):
    evalset.save(tmp_path / "es.json")
    assert EvalSet.load(tmp_path / "es.json").queries == evalset.queries


def test_threshold_allows_at_most_the_budget():
    clean = [0.9, 0.8, 0.7] + [0.1] * 997
    t = threshold_for(clean, 1)
    assert sum(s >= t for s in clean) <= 1
    assert threshold_for(clean, 5) <= t


def test_summary_applies_dev_thresholds_to_test():
    from sanctions_mcp.evalset import Query

    dev = SystemRun("x", "dev", {"p1": 0.9}, {f"c{i}": 0.1 for i in range(1000)})
    test = SystemRun("x", "test", {"p2": 0.95, "p3": 0.05}, {"c": 0.2}, latency_ms=[1.0, 2.0])
    queries = [
        Query("p2", "a", "E1", "alias", "other", "test"),
        Query("p3", "b", "E2", "typo", "other", "test"),
        Query("c", "c", None, "clean", "other", "test"),
    ]
    s = summarize(dev, test, queries)
    point = next(p for p in s["operating_points"] if p["budget_per_1000"] == 5)
    assert point["test_recall"] == 0.5
    assert s["recall_by_kind"]["alias"]["recall"] == 1.0
    assert s["false_alerts_per_1000_by_group"]["other"]["rate"] == 1000.0


def test_baselines_and_matcher_run_through_the_harness(entities, evalset, matcher):
    queries = evalset.by_split("test")
    timed = len(latency_indices(len(queries), 5))
    for system in (ExactMatcher(entities), TfidfMatcher(entities), matcher):
        run = run_system(system, queries, "test", latency_sample=5)
        assert set(run.target_score) | set(run.max_score) == {q.qid for q in queries}
        assert len(run.latency_ms) == timed  # the same sample for every system


def test_latency_sample_spans_positives_and_clean_names():
    assert latency_indices(10, 5) == [0, 2, 4, 6, 8]
    assert latency_indices(3, 5) == [0, 1, 2]


def test_fuzzy_baseline_ignores_punctuation(entities):
    fuzzy = TfidfMatcher(entities, rescore=True)
    a = fuzzy.scores("Karim Fadel Al-Nouri")
    b = fuzzy.scores("Karim Fadel al Nouri")
    assert a["FX-001"] == b["FX-001"] == 1.0


def test_watchlist_loader_reads_people_only(entities):
    assert len(entities) == 12
    karim = next(e for e in entities if e.id == "FX-001")
    assert karim.names[0] == "Karim Fadel Al-Nouri" and len(karim.names) == 4
    assert karim.birth_dates == ("1970-03-01",) and karim.countries == ("iq",)


def test_download_verifies_the_checksum(tmp_path):
    body = b"id,schema,name\n"
    good = hashlib.sha256(body).hexdigest()
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    )
    path = download(tmp_path, client=client, url="https://x.test/f.csv", sha256=good)
    assert path == snapshot_path(tmp_path) and path.read_bytes() == body
    with pytest.raises(ChecksumError):
        download(tmp_path, client=client, url="https://x.test/f.csv", sha256="0" * 64, force=True)


def test_missing_snapshot_points_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="sanctions download"):
        load_watchlist(snapshot_path(tmp_path))


def test_arabic_script_clean_names_skip_titles_and_connectors():
    import random

    from sanctions_mcp.evalset import _arabic_clean_name

    for seed in range(20):
        name = _arabic_clean_name(
            "شيخ كريم بن فاضل النوري", "يوسف عبد الرحمن حداد", random.Random(seed)
        )
        assert name and has_arabic(name) and name.endswith("حداد")
        assert "شيخ" not in name and "بن" not in name.split()


def test_arabic_script_clean_names_meet_latin_listed_namesakes():
    from sanctions_mcp.evalset import arabic_view_keys

    assert name_keys("حسين الحمداني") & arabic_view_keys("Hossein Hamadani")
    assert name_keys("عبد خليل") & arabic_view_keys("Abdul Khalil")
