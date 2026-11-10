import numpy as np
import pytest

from sanctions_mcp.data import Entity
from sanctions_mcp.index import WatchlistIndex
from sanctions_mcp.matcher import Matcher, training_pairs
from sanctions_mcp.normalize import tokenize
from sanctions_mcp.scoring import (
    FEATURES,
    STRONG,
    WEAK,
    LogisticScorer,
    RuleScorer,
    token_similarity,
)


def sim(a: str, b: str) -> float:
    return token_similarity(tokenize(a)[0], tokenize(b)[0])


def person(eid: str, name: str, *aliases: str) -> Entity:
    return Entity(eid, "Person", name, aliases, (), (), ("test",))


def top(matcher, name):
    results = matcher.screen(name, top_k=3)
    return results[0].entity_id if results else None


@pytest.mark.parametrize(
    ("query", "entity"),
    [
        ("Kareem Fadel Nouri", "FX-001"),  # transliteration and dropped article
        ("كريم فاضل النوري", "FX-001"),  # Arabic script
        ("Youssef Abdel Rahman Hadad", "FX-002"),
        ("Haji Saifulla Zarin Durani", "FX-003"),  # title and spelling
        ("Сидоренков Иван", "FX-004"),  # Cyrillic, surname first
        ("Nuruddin Hamdani Qudsi", "FX-006"),
        ("Tarik Mahmud Jaberi", "FX-008"),
        ("A. Fawzi Barakeh", "FX-009"),  # initial
    ],
)
def test_variants_find_the_listed_person(matcher, query, entity):
    assert top(matcher, query) == entity


def test_unrelated_name_scores_low(matcher):
    results = matcher.screen("Grace Wanjiru Otieno", top_k=1)
    assert not results or results[0].score < 0.6


def test_organizations_and_vessels_are_not_indexed(entities):
    assert all(e.schema == "Person" for e in entities)


def test_exact_key_matches_rank_above_fuzzy_ones(entities):
    index = WatchlistIndex(entities)
    first = index.candidates(tokenize("Ahmad Salim Qasimov"))[0]
    assert index.names[first].entity_id == "FX-007"


def test_held_out_names_are_not_indexed(entities):
    hidden = {"FX-001": ["كريم فاضل النوري"]}
    index = WatchlistIndex(entities, exclude=hidden)
    assert "كريم فاضل النوري" not in {n.text for n in index.names}


def test_initials_and_cross_script_similarity():
    a, b = tokenize("A. Fawzi")[0], tokenize("Abdullah")[0]
    assert token_similarity(a, b) == 0.9
    lat, ara = tokenize("Karim")[0], tokenize("كريم")[0]
    assert token_similarity(lat, ara) == 1.0


def test_explain_shows_the_alignment(matcher):
    why = matcher.explain("Kareem Fadel Nouri", "FX-001")
    assert why["score"] > 0.8
    assert why["matched_name"] in {"Karim Fadel Al-Nouri", "كريم فاضل النوري"}
    assert {a["query"] for a in why["alignment"]} == {"karim", "fadel", "nouri"}
    assert set(why["features"]) == set(FEATURES)
    assert why["retrieved"] and why["reason"] is None


def test_explain_reports_what_screening_scored(entities):
    tight = Matcher(WatchlistIndex(entities), RuleScorer(), candidate_limit=2)
    for query in ["Karim Nouri", "Muhammad Ali", "Tarek Jaberi", "Hussein Mansour"]:
        for match in tight.screen(query, top_k=None):
            why = tight.explain(query, match.entity_id)
            assert why["score"] == pytest.approx(match.score)
            assert why["matched_name"] == match.matched_name
    missing = tight.explain("Karim Nouri", "FX-005")
    assert missing["score"] == 0.0 and not missing["retrieved"]
    assert "not among" in missing["reason"]
    assert tight.explain("...", "FX-001")["reason"] == "no comparable names"


def test_explain_lists_words_below_the_match_threshold(matcher):
    why = matcher.explain("Tarik Mahmud Smith", "FX-008")
    assert why["unmatched_query_words"] == ["smith"]
    share = len(why["unmatched_query_words"]) / 3
    assert share == pytest.approx(why["features"]["query_unmatched"], abs=1e-4)
    assert [a["query"] for a in why["alignment"] if not a["matched"]] == ["smith"]


def test_initials_match_every_spelling_both_ways():
    for initial, name in [
        ("O.", "Osama"),
        ("O.", "Usama"),
        ("G.", "Gamal"),
        ("B.", "Brahim"),
        ("O.", "عمر"),
        ("I.", "عيسى"),
    ]:
        assert sim(initial, name) == 0.9, (initial, name)
        assert sim(name, initial) == 0.9, (name, initial)
    assert sim("Z.", "Osama") == 0.0


def test_a_shared_key_does_not_make_different_names_strong():
    for a, b in [
        ("Muhammad", "Mahmud"),
        ("Hasan", "Husain"),
        ("Ahmad", "Hamid"),
        ("Hamid", "Hamad"),
        ("Karim", "Akram"),
        ("Salim", "Aslam"),
        ("Bashir", "Bashar"),
        ("Mahmood", "محمد"),
    ]:
        assert sim(a, b) < STRONG, (a, b)
    assert sim("Ahmad", "Hamid") < WEAK
    for a, b in [("Fadel", "Fadil"), ("Kassem", "Qasim"), ("Mansour", "Mansur")]:
        assert sim(a, b) >= STRONG, (a, b)


def test_glued_al_is_weaker_evidence_than_a_real_key():
    assert WEAK <= sim("Almatyouti", "Matyouti") < STRONG
    assert sim("Alberto", "Berto") < STRONG


def test_words_without_consonants_need_the_same_spelling():
    assert sim("Awa", "Ewa") < WEAK
    assert sim("Aye", "Oo") == 0.0
    assert sim("Ewa", "آية") < WEAK
    assert sim("Aya", "آية") == 1.0


def test_candidates_count_people_not_rows():
    crowd = person("A-CROWD", "Muhammad Ali Bx0", *(f"Muhammad Ali Bx{i}" for i in range(1, 260)))
    target = person("Z-TARGET", "Muhammad Ali Zafar")
    index = WatchlistIndex([crowd, target])
    found = {index.names[i].entity_id for i in index.candidates(tokenize("Muhammad Ali"), 200)}
    assert found == {"A-CROWD", "Z-TARGET"}


def test_full_matches_are_never_cut_by_list_order():
    twins = [person(f"E{i}", "Muhammad Ali") for i in range(3)]
    partial = [person(f"P{i}", "Muhammad Ali Khan") for i in range(3)]
    index = WatchlistIndex(twins + partial)
    full = {index.names[i].entity_id for i in index.candidates(tokenize("Muhammad Ali"), 2)}
    assert full == {"E0", "E1", "E2"}  # everyone tied on a full match is kept
    loose = {index.names[i].entity_id for i in index.candidates(tokenize("Muhammad Hasan"), 2)}
    assert len(loose) == 2  # a tie on a partial match is cut at the limit


def test_an_initial_weighs_little_and_helps_the_name_it_abbreviates():
    index = WatchlistIndex(
        [person("E1", "Imam Ali"), person("E2", "Muhammad Ali Khan"), person("E3", "Ma Lin")]
    )
    assert index.idf(tokenize("M.")[0]) <= index.idf(tokenize("Muhammad")[0])
    first = index.candidates(tokenize("M. Ali Khan"))[0]
    assert index.names[first].entity_id == "E2"
    assert index.candidates(tokenize("M."), 10) == []


def test_rule_scorer_prefers_full_coverage():
    full = np.zeros(len(FEATURES))
    full[[0, 1]] = 1.0
    half = np.zeros(len(FEATURES))
    half[[0, 1]] = 0.5
    s = RuleScorer().score(np.stack([full, half]))
    assert s[0] > s[1]


def test_learned_scorer_round_trip(matcher, tmp_path):
    from sanctions_mcp.evalset import Query

    queries = [
        Query("q1", "Kareem Nouri", "FX-001", "transliteration", "mena_south_asia", "train"),
        Query("q2", "Tarek Jaberi", "FX-008", "transliteration", "mena_south_asia", "train"),
        Query("q3", "Karim Haddad", None, "clean", "mena_south_asia", "train"),
    ]
    x, y = training_pairs(matcher, queries)
    assert x.shape[1] == len(FEATURES) and y.any() and not y.all()
    scorer = LogisticScorer.fit(x, y)
    scorer.save(tmp_path / "uncalibrated.json")
    with pytest.raises(ValueError, match="alert threshold"):
        LogisticScorer.load(tmp_path / "uncalibrated.json")
    scorer.alert_threshold = 0.5
    scorer.save(tmp_path / "scorer.json")
    loaded = LogisticScorer.load(tmp_path / "scorer.json")
    assert np.allclose(loaded.score(x), scorer.score(x))
    assert set(loaded.weights()) == set(FEATURES)


def test_learned_scores_do_not_depend_on_the_batch():
    rng = np.random.default_rng(0)
    x = rng.random((33, len(FEATURES)))
    scorer = LogisticScorer(
        x.mean(axis=0), x.std(axis=0) + 0.1, rng.normal(size=len(FEATURES)), 0.3
    )
    whole = scorer.score(x)
    assert all(scorer.score(x[i : i + 1])[0] == whole[i] for i in range(len(x)))
    assert all(scorer.score(x[i : i + 5])[0] == whole[i] for i in range(len(x) - 5))


def test_cross_script_guards_against_a_false_perfect_match():
    for latin, arabic in [
        ("Amir", "امیری"),  # Amiri: the final long i shows in Arabic script
        ("Umar", "عمري"),
        ("Karim", "کریمی"),
        ("Ayyub", "أبو"),  # one shared consonant says little
        ("Hammad", "حميد"),  # Hammad is written حماد, so حميد is another name
        ("Alireza", "رياض"),  # only the glued al reading meets it
        ("Mahmoudi", "محمد"),
    ]:
        assert sim(latin, arabic) < STRONG, (latin, arabic)
    for latin, arabic in [
        ("Makhmoud", "محمود"),
        ("Ali", "علي"),
        ("Karimi", "کریمی"),
        ("Musa", "موسى"),
        ("Nur", "نور"),
        ("Kareem", "كريم"),
    ]:
        assert sim(latin, arabic) >= STRONG, (latin, arabic)


def test_within_arabic_script_a_different_spelling_is_a_different_name():
    assert sim("سلمان", "سليمان") < STRONG  # Salman, Sulaiman
    assert sim("أسامة", "عصام") < STRONG  # Usama, Isam
    assert sim("محمد", "محمد") == 1.0
    assert sim("إياد", "اياد") == 1.0  # the hamza under alef is often left out
