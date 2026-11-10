"""Command line tests on the fictional list. No network and no model calls: downloads are
replaced by stubs and the agent commands get a scripted model."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from sanctions_mcp import cli
from sanctions_mcp.agent.models import ToolCall, Turn
from sanctions_mcp.agent.scenarios import Scenario
from sanctions_mcp.agent.triage import Alert
from sanctions_mcp.cli import NO_CASE_RECORDS, build_parser, main
from sanctions_mcp.data import (
    SHA256,
    SNAPSHOT,
    ChecksumError,
    describe_list,
    load_watchlist,
    sha256_of,
    snapshot_path,
)
from sanctions_mcp.evalset import EvalSet, build
from sanctions_mcp.evaluate import SLICE_BUDGET, threshold_for
from sanctions_mcp.index import WatchlistIndex
from sanctions_mcp.matcher import Matcher
from sanctions_mcp.normalize import tokenize
from sanctions_mcp.report import END, START, SYSTEMS, render
from sanctions_mcp.scoring import LogisticScorer, RuleScorer
from sanctions_mcp.service import ScreeningService, model_path
from sanctions_mcp.settings import LLMSettings

SPLITS = ("train", "dev", "test")
FIXTURE_LIST = snapshot_path(Path(__file__).parent / "fixtures" / "data")


@pytest.fixture(autouse=True)
def work_dir(tmp_path, monkeypatch):
    """Every command runs in a scratch folder, so a default path such as results/ or
    README.md can never reach the repo's own files."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


@pytest.fixture
def small_evalset(monkeypatch):
    """The CLI builds a small eval set, so the fictional list needs seconds, not minutes."""

    def small(entities, seed):
        return build(
            entities,
            seed=seed,
            entities_per_split=dict.fromkeys(SPLITS, 50),
            clean_per_split=dict.fromkeys(SPLITS, 20),
        )

    monkeypatch.setattr(cli, "build", small)


@pytest.fixture
def trained(data_dir, small_evalset):
    assert main(["--data-dir", str(data_dir), "train"]) == 0
    return data_dir


@pytest.fixture
def out(tmp_path):
    """A results folder and a README of the test's own. `results_flags` goes to eval, and
    `flags` (--results-dir and --readme) to reproduce and report."""
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\n{END}\n", encoding="utf-8")
    results = tmp_path / "results"
    results_flags = ["--results-dir", str(results)]
    return SimpleNamespace(
        results=results,
        readme=readme,
        results_flags=results_flags,
        flags=[*results_flags, "--readme", str(readme)],
    )


def evalset_file(data_dir: Path) -> Path:
    return cli.evalset_path(SimpleNamespace(data_dir=data_dir))


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from keys(v)


# ---------------------------------------------------------------- screening and thresholds


def test_screen_without_a_model_prints_scores_only(data_dir, capsys):
    assert main(["--data-dir", str(data_dir), "screen", "Kareem Fadil Nouri", "--top", "1"]) == 0
    captured = capsys.readouterr()
    assert "Karim Fadel Al-Nouri" in captured.out and "FX-001" in captured.out
    assert "ALERT" not in captured.out
    assert "alerts need" in captured.err and "sanctions train" in captured.err
    assert "from the rules scorer (no learned model loads)" in captured.err


def test_screen_without_a_threshold_names_the_scorer_behind_the_scores(trained, entities, capsys):
    path = model_path(trained, "learned")
    saved = read(path)
    path.write_text(json.dumps({k: v for k, v in saved.items() if k != "calibration"}), "utf-8")
    capsys.readouterr()
    argv = ["--data-dir", str(trained), "screen", "Kareem Nouri", "--top", "1"]
    assert main(argv) == 0
    captured = capsys.readouterr()
    # The learned model still loads, so it scores; only the alert flags are left out.
    assert "Scores only, from the learned scorer, with no alert flags" in captured.err
    top = Matcher(WatchlistIndex(entities), LogisticScorer.load(path)).screen("Kareem Nouri", 1)
    first = captured.out.splitlines()[0]
    assert first.startswith(f"{top[0].score:.3f}  ") and first.endswith(f"[{top[0].entity_id}]")
    assert "ALERT" not in captured.out
    assert main([*argv, "--scorer", "rules"]) == 0  # rules.json still holds a threshold
    assert "Alert threshold" in capsys.readouterr().err


def test_train_calibrates_the_served_thresholds_on_the_full_list(trained, entities):
    learned, rules = read(model_path(trained, "learned")), read(model_path(trained, "rules"))
    es = EvalSet.load(evalset_file(trained))
    dev_clean = [q.text for q in es.by_split("dev") if q.target is None]

    def threshold(index, scorer):
        matcher = Matcher(index, scorer)
        clean = [max(matcher.scores(t).values(), default=0.0) for t in dev_clean]
        return threshold_for(clean, SLICE_BUDGET)

    scorer = LogisticScorer.load(model_path(trained, "learned"))
    full, held_out = WatchlistIndex(entities), WatchlistIndex(entities, exclude=es.held_out)
    assert learned["alert_threshold"] == threshold(full, scorer)
    assert learned["calibration"]["held_out_index_threshold"] == threshold(held_out, scorer)
    assert rules["alert_threshold"] == threshold(full, RuleScorer())
    for saved in (learned, rules):
        assert saved["list_sha256"] == sha256_of(snapshot_path(trained))
        assert saved["evalset_sha256"] == sha256_of(evalset_file(trained))
        calibration = saved["calibration"]
        assert (calibration["budget_per_1000"], calibration["split"]) == (SLICE_BUDGET, "dev")
        assert calibration["index"] == "full list"
        assert calibration["clean_names"] == len(dev_clean)


def test_screen_shows_alert_flags_once_trained(trained, capsys):
    assert main(["--data-dir", str(trained), "screen", "Kareem Fadil Nouri", "--top", "2"]) == 0
    captured = capsys.readouterr()
    assert "ALERT" in captured.out and "Alert threshold" in captured.err
    assert main(["--data-dir", str(trained), "screen", "Kareem Nouri", "--scorer", "rules"]) == 0
    assert "rules scorer" in capsys.readouterr().err


def test_serve_and_the_agent_commands_fail_closed_without_a_model(data_dir, capsys):
    for command in (
        ["serve"],
        ["triage", "Kareem Nouri", "--entity-id", "FX-001"],
        ["triage-eval", "--yes"],
    ):
        assert main(["--data-dir", str(data_dir), *command]) == 1, command
        assert "Run `sanctions train` first" in capsys.readouterr().err


def test_a_threshold_that_does_not_fit_this_list_is_refused(trained, capsys):
    path = model_path(trained, "learned")
    saved = read(path)
    path.write_text(json.dumps({**saved, "list_sha256": "0" * 64}), encoding="utf-8")
    assert main(["--data-dir", str(trained), "serve"]) == 1
    assert "different list file" in capsys.readouterr().err
    path.write_text(json.dumps({k: v for k, v in saved.items() if k != "calibration"}), "utf-8")
    assert main(["--data-dir", str(trained), "serve"]) == 1
    assert "sanctions train" in capsys.readouterr().err
    assert main(["--data-dir", str(trained), "screen", "Kareem Fadil Nouri"]) == 0
    assert "ALERT" not in capsys.readouterr().out


def test_list_sources_names_the_scorer_and_where_its_threshold_comes_from(trained):
    learned = ScreeningService.from_data_dir(trained).sources()
    rules = ScreeningService.from_data_dir(trained, scorer="rules").sources()
    assert (learned["scorer"], rules["scorer"]) == ("learned", "rules")
    for s in (learned, rules):
        source = s["alert_threshold_source"]
        assert "calibrated by `sanctions train`" in source and "dev split" in source
        assert "full list" in source
        assert s["snapshot"] is None and s["license"].startswith("unknown")


# ---------------------------------------------------------------- results and the README


def test_eval_and_report_write_the_readme(data_dir, small_evalset, out):
    results, readme, flags = out.results, out.readme, out.flags
    assert main(["--data-dir", str(data_dir), "eval", "--system", "exact", *out.results_flags]) == 0
    saved = read(results / "exact.json")
    assert saved["command"] == "uv run sanctions eval --system exact" and saved["split"] == "test"
    assert saved["dataset"] == {
        "source": "unverified list file",
        "collection": None,
        "snapshot": None,
        "license": None,
        "sha256": sha256_of(snapshot_path(data_dir)),
        "people": 12,
    }
    assert saved["evalset"] == {"seed": cli.SEED, "sha256": sha256_of(evalset_file(data_dir))}
    # Nothing else in the results folder is read: agent summaries, the list counts, notes.
    (results / "agent-dev.json").write_text('{"model": "x"}', encoding="utf-8")
    (results / "agent").mkdir()
    (results / "agent" / "dev.json").write_text("{}", encoding="utf-8")
    assert main(["--data-dir", str(data_dir), "stats", "--results-dir", str(results)]) == 0
    assert main(["--data-dir", str(data_dir), "report", *flags]) == 0
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n") and "| Exact match |" in text
    assert "List file (not the pinned snapshot): 12 people from 1 source list;" in text
    # The fictional list is never called the snapshot.
    assert "Snapshot `" not in text and "OpenSanctions snapshot `" not in text
    assert "a list file not verified as the pinned OpenSanctions snapshot" in text
    n = saved["latency_ms"]["n"]
    assert f"on the same {n:,} test queries for every system" in text
    assert "Arab League, Iran, Afghanistan, Pakistan" in text


def result_row(system: str, n: int | None = 1000) -> dict:
    """A saved result as save_result writes it, measured on the pinned snapshot."""
    points = [
        {
            "budget_per_1000": b,
            "threshold": 0.9,
            "test_recall": 0.5,
            "test_false_alerts_per_1000": 4,
        }
        for b in (1, 5, 10)
    ]
    groups = ("mena_south_asia", "other")
    return {
        "system": system,
        "label": SYSTEMS[system],
        "command": f"uv run sanctions eval --system {system}",
        "split": "test",
        "api_cost_usd_per_1k_screenings": 0.0,
        "dataset": {"source": "OpenSanctions", "snapshot": SNAPSHOT, "sha256": SHA256},
        "evalset": {"seed": 13, "sha256": "e" * 64},
        "operating_points": points,
        "recall_by_kind": {"alias": {"n": 10, "recall": 0.5}},
        "recall_by_group": {g: {"n": 5, "recall": 0.5} for g in groups},
        "false_alerts_per_1000_by_group": {g: {"n": 50, "rate": 4.0} for g in groups},
        "candidate_recall": None,
        "latency_ms": {"p50": 1.0, "p95": 2.0, **({"n": n} if n else {})},
        "counts": {"test_positive": 10, "test_clean": 100, "dev_positive": 10, "dev_clean": 100},
        "environment": {"python": "3.12", "os": "os", "cpu": "cpu", "logical_cpus": 8},
        "created_utc": "2026-11-10T00:00:00+00:00",
    }


def test_report_names_the_pinned_snapshot_and_the_latency_sample():
    results = {s: result_row(s) for s in ("exact", "tfidf")}
    stats = {
        "snapshot": SNAPSHOT,
        "list_sha256": SHA256,
        "people": 41227,
        "source_lists": 77,
        "names_indexed": 172001,
    }
    text = render(results, stats)
    assert text.startswith(
        f"Snapshot `{SNAPSHOT}`: 41,227 people from 77 source lists; 172,001 names indexed.\n"
    )
    assert f"OpenSanctions snapshot `{SNAPSHOT}`" in text
    assert "on the same 1,000 test queries for every system" in text
    assert render(results).startswith("**Test split**")  # without stats.json, no line
    older = {s: result_row(s, n=None) for s in ("exact", "tfidf")}  # saved before n was
    assert "Latency is one screening at a time, end to end.\n" in render(older)


def test_fixture_results_never_replace_the_published_ones(
    data_dir, small_evalset, work_dir, out, capsys
):
    for command in (["eval", "--system", "exact"], ["reproduce", "--systems", "exact"]):
        assert main(["--data-dir", str(data_dir), *command]) == 1
        assert "--results-dir" in capsys.readouterr().err
    assert not (work_dir / "results").exists() and not evalset_file(data_dir).exists()
    assert main(["--data-dir", str(data_dir), "eval", "--system", "exact", *out.results_flags]) == 0
    assert main(["report", *out.results_flags]) == 1  # no --readme
    assert "--readme" in capsys.readouterr().err
    assert main(["--data-dir", str(data_dir), "stats"]) == 0
    assert "Not saved" in capsys.readouterr().err and not (work_dir / "results").exists()


def test_report_refuses_rows_from_different_eval_sets(data_dir, small_evalset, out, capsys):
    for system in ("exact", "tfidf"):
        argv = ["--data-dir", str(data_dir), "eval", "--system", system, *out.results_flags]
        assert main(argv) == 0
    path = out.results / "tfidf.json"
    path.write_text(json.dumps({**read(path), "evalset": {"seed": 1, "sha256": "0"}}), "utf-8")
    assert main(["report", *out.flags]) == 1
    assert "not all measured on the eval set" in capsys.readouterr().err


def test_report_refuses_list_counts_from_another_list(data_dir, small_evalset, out, capsys):
    assert main(["--data-dir", str(data_dir), "eval", "--system", "exact", *out.results_flags]) == 0
    assert main(["--data-dir", str(data_dir), "stats", *out.results_flags]) == 0
    stats = out.results / "stats.json"
    stats.write_text(json.dumps({**read(stats), "list_sha256": "0" * 64}), encoding="utf-8")
    assert main(["report", *out.flags]) == 1
    assert "stats.json was counted on a different list file" in capsys.readouterr().err


def test_eval_takes_no_readme():
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(["eval", "--system", "exact", "--readme", "README.md"])
    assert stop.value.code == 2
    for command in (["reproduce"], ["report"]):
        assert build_parser().parse_args([*command, "--readme", "x.md"]).readme == Path("x.md")


def test_a_readme_without_the_markers_is_one_clear_error(data_dir, small_evalset, out, capsys):
    assert main(["--data-dir", str(data_dir), "eval", "--system", "exact", *out.results_flags]) == 0
    capsys.readouterr()
    plain = out.readme.with_name("plain.md")
    plain.write_text("no markers here\n", encoding="utf-8")
    assert main(["report", *out.results_flags, "--readme", str(plain)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1
    assert f"needs the {START} and {END} markers" in err
    assert plain.read_text(encoding="utf-8") == "no markers here\n"


@pytest.mark.parametrize("readme", ["missing.md", "plain.md"])
def test_reproduce_checks_the_readme_before_any_slow_work(
    data_dir, small_evalset, out, readme, capsys
):
    path = out.readme.with_name(readme)
    if readme == "plain.md":
        path.write_text("no markers here\n", encoding="utf-8")
    argv = ["--data-dir", str(data_dir), "reproduce", *out.results_flags, "--readme", str(path)]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1
    assert ("not found" if readme == "missing.md" else "markers") in err
    assert not evalset_file(data_dir).exists() and not out.results.exists()


def test_reproduce_rebuilds_the_eval_set_and_records_it(data_dir, small_evalset, out):
    results, readme, flags = out.results, out.readme, out.flags
    stale = evalset_file(data_dir)
    EvalSet(seed=cli.SEED, queries=[]).save(stale)  # an eval set left behind by older code
    stale_sha = sha256_of(stale)
    argv = ["--data-dir", str(data_dir), "reproduce", "--systems", "exact", "matcher-learned"]
    assert main([*argv, *flags]) == 0
    es_sha = sha256_of(stale)
    assert es_sha != stale_sha and EvalSet.load(stale).queries
    assert read(model_path(data_dir, "learned"))["evalset_sha256"] == es_sha
    for system in ("exact", "matcher-learned"):
        assert read(results / f"{system}.json")["evalset"]["sha256"] == es_sha
    stats = read(results / "stats.json")
    assert stats["people"] == 12 and stats["eval_set"]["sha256"] == es_sha
    text = readme.read_text(encoding="utf-8")
    assert "| This matcher, learned weights |" in text and "12 people from 1 source list" in text


def test_eval_retrains_a_model_trained_on_another_eval_set(trained, out, capsys):
    path = model_path(trained, "learned")
    saved = read(path)
    path.write_text(json.dumps({**saved, "evalset_sha256": "0" * 64}), encoding="utf-8")
    argv = ["--data-dir", str(trained), "eval", "--system", "matcher-learned", *out.results_flags]
    assert main(argv) == 0
    assert "trained on another list file or eval set" in capsys.readouterr().err
    assert read(path)["evalset_sha256"] == saved["evalset_sha256"]


def test_eval_trains_a_model_when_there_is_none(data_dir, small_evalset, out, capsys):
    argv = ["--data-dir", str(data_dir), "eval", "--system", "matcher-learned", *out.results_flags]
    assert main(argv) == 0
    assert "No model trained on this list and eval set yet" in capsys.readouterr().err
    model = read(model_path(data_dir, "learned"))
    assert model["list_sha256"] == sha256_of(snapshot_path(data_dir))


def test_a_changed_list_file_rebuilds_the_eval_set_and_retrains(trained, out, capsys):
    """Every input of a result must come from the list file loaded now. The edit below
    changes the file but not the people in it, so the eval set built again is the same:
    only the list fingerprints can tell the saved files are stale."""
    capsys.readouterr()
    list_file = snapshot_path(trained)
    es_sha = sha256_of(evalset_file(trained))
    text = list_file.read_text(encoding="utf-8").replace("2026-11-01T00:", "2026-11-02T00:")
    list_file.write_text(text, encoding="utf-8", newline="\n")
    new_sha = sha256_of(list_file)
    # Until it is built again, the saved eval set is not counted as this list's.
    assert main(["--data-dir", str(trained), "stats"]) == 0
    assert "eval_set" not in json.loads(capsys.readouterr().out)

    argv = ["--data-dir", str(trained), "eval", "--system", "matcher-learned", *out.results_flags]
    assert main(argv) == 0
    err = capsys.readouterr().err
    assert "The saved eval set was built from another list file" in err
    assert "trained on another list file or eval set" in err
    assert sha256_of(evalset_file(trained)) == es_sha  # the same queries, now on record
    model = read(model_path(trained, "learned"))
    assert (model["list_sha256"], model["evalset_sha256"]) == (new_sha, es_sha)
    result = read(out.results / "matcher-learned.json")
    assert (result["dataset"]["sha256"], result["evalset"]["sha256"]) == (new_sha, es_sha)
    assert main(["--data-dir", str(trained), "stats"]) == 0
    assert json.loads(capsys.readouterr().out)["eval_set"]["sha256"] == es_sha

    assert main(argv) == 0  # nothing is stale any more
    err = capsys.readouterr().err
    assert "Building the eval set" not in err and "Training" not in err


def test_an_eval_set_with_no_record_of_its_list_is_built_again(trained, capsys):
    record = evalset_file(trained).with_suffix(".list_sha256")
    record.unlink()  # as an eval set saved by older code
    capsys.readouterr()
    assert main(["--data-dir", str(trained), "train"]) == 0
    assert "does not record which list file it came from" in capsys.readouterr().err
    assert record.read_text(encoding="utf-8").strip() == sha256_of(snapshot_path(trained))


def test_eval_on_the_dev_split_never_runs_test(data_dir, small_evalset, out, monkeypatch, capsys):
    results, flags = out.results, out.flags
    splits = []
    real = cli.run_system

    def run_system(system, queries, split, **kwargs):
        splits.append(split)
        return real(system, queries, split, **kwargs)

    monkeypatch.setattr(cli, "run_system", run_system)
    argv = ["--data-dir", str(data_dir), "eval", "--system", "exact", "--split", "dev"]
    assert main([*argv, *out.results_flags]) == 0
    assert splits == ["dev"]
    saved = read(results / "dev" / "exact.json")
    assert saved["split"] == "dev" and saved["command"].endswith("--split dev")
    assert not any(k.startswith("test_") for k in keys(saved))
    assert "dev_recall" in saved["operating_points"][0]
    assert "dev recall" in capsys.readouterr().err
    assert not (results / "exact.json").exists()
    assert main(["report", *flags]) == 1  # a dev run never reaches the README


# ---------------------------------------------------------------- stats


def test_stats_counts_the_list_without_building_the_eval_set(data_dir, entities, out, capsys):
    results = out.results
    assert main(["--data-dir", str(data_dir), "stats"]) == 0
    captured = capsys.readouterr()
    stats = json.loads(captured.out)
    index = WatchlistIndex(entities)
    assert (stats["people"], stats["source_lists"], stats["snapshot"]) == (12, 1, None)
    assert stats["names_indexed"] == len(index) and stats["names_listed"] == index.listed_names
    assert stats["names_including_aliases"] == sum(len(e.names) for e in entities)
    assert stats["list_sha256"] == sha256_of(snapshot_path(data_dir))
    assert "eval_set" not in stats and not (data_dir / "evalsets").exists()
    assert "Not saved" in captured.err
    assert main(["--data-dir", str(data_dir), "stats", "--results-dir", str(results)]) == 0
    saved = read(results / "stats.json")
    assert saved.pop("created_utc") and saved == stats


def test_names_the_index_cannot_read_are_counted_apart(data_dir, capsys):
    unreadable = "カリム・ファデル"
    path = snapshot_path(data_dir)
    text = path.read_text(encoding="utf-8")
    alias = "Kareem Fadil Al-Nuri;"
    path.write_text(text.replace(alias, f"{alias}{unreadable};"), encoding="utf-8", newline="\n")
    assert main(["--data-dir", str(data_dir), "stats"]) == 0
    stats = json.loads(capsys.readouterr().out)
    index = WatchlistIndex(load_watchlist(path))
    assert stats["names_indexed"] == len(index) <= stats["names_listed"] == index.listed_names
    if not tokenize(unreadable):  # a script the normaliser does not read gives no words
        assert stats["names_including_aliases"] == stats["names_listed"] + 1


# ---------------------------------------------------------------- download


def test_download_needs_consent_when_not_interactive(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["--data-dir", str(tmp_path), "download"]) == 2
    out = capsys.readouterr().out
    assert "Run again with --yes" in out and "Re-run" not in out


@pytest.mark.parametrize(
    ("error", "says"),
    [
        (
            httpx.HTTPStatusError(
                "gone", request=httpx.Request("GET", "https://x.test"), response=httpx.Response(404)
            ),
            ["HTTP 404", "no longer public", "set SNAPSHOT and SHA256"],
        ),
        (ChecksumError("bad"), ["does not match the pinned SHA-256", "set SNAPSHOT and SHA256"]),
        (httpx.ConnectError("no route"), ["network connection"]),
    ],
)
def test_download_failures_end_with_one_clear_line(tmp_path, monkeypatch, capsys, error, says):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "download", fail)
    assert main(["--data-dir", str(tmp_path), "download", "--yes"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and err.count("\n") == 1 and SNAPSHOT in err
    assert all(s in err for s in says), err


def test_download_keeps_a_file_that_is_not_the_snapshot(data_dir, capsys):
    before = snapshot_path(data_dir).read_bytes()
    assert main(["--data-dir", str(data_dir), "download", "--yes"]) == 1
    assert "--force" in capsys.readouterr().err
    assert snapshot_path(data_dir).read_bytes() == before


# ---------------------------------------------------------------- the agent commands


class ScriptedModel:
    name = "scripted"

    def __init__(self, *turns, scripts=None):
        self.turns, self.scripts, self.sent, self.sessions = turns, scripts, [], 0

    def start(self, system, tools):
        self.sessions += 1
        script = self.scripts[self.sessions - 1] if self.scripts else self.turns
        return ScriptedSession(list(script), self.sent)


class ScriptedSession:
    def __init__(self, turns, sent):
        self.turns, self.sent = turns, sent

    def _next(self):
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn

    def send(self, text):
        self.sent.append(text)
        return self._next()

    def send_tool_results(self, results):
        self.sent.append(results)
        return self._next()


def look_then(disposition):
    look = Turn("", [ToolCall("t1", "get_entity", {"entity_id": "FX-001"})], "tool_use", 50, 10)
    args = {
        "disposition": disposition,
        "reasons": [{"claim": "born 1970", "evidence": "get_entity.birth_dates"}],
        "injection_seen": False,
    }
    return [look, Turn("", [ToolCall("s1", "submit_disposition", args)], "tool_use", 100, 20)]


@pytest.fixture
def agent(entities, monkeypatch):
    """Point the agent commands at a scripted model and an in memory service, which
    describes the fictional list as ScreeningService.from_data_dir would."""
    service = ScreeningService(
        entities,
        alert_threshold=0.6,
        threshold_source="set by the test",
        dataset=describe_list(FIXTURE_LIST, len(entities)),
    )
    models = []

    def use(model):
        models.append(model)
        return model

    monkeypatch.setattr(cli, "_agent_setup", lambda args: (LLMSettings(), service, models[-1]))
    return use


TRIAGE = ["triage", "Kareem Fadil Nouri", "--entity-id", "FX-001"]


def test_triage_says_plainly_that_it_keeps_no_case_records(data_dir, agent, monkeypatch, capsys):
    model = agent(ScriptedModel(*look_then("likely_match")))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["--data-dir", str(data_dir), *TRIAGE, "--country", "IQ"]) == 0
    out = capsys.readouterr().out
    assert '"final": "likely_match"' in out and NO_CASE_RECORDS in out
    assert "recorded locally" not in out.lower()
    assert '"country": "iq"' in model.sent[0]  # the alert carries the country as compared


def test_triage_asks_until_the_decision_is_one_it_knows(data_dir, agent, monkeypatch, capsys):
    agent(ScriptedModel(*look_then("likely_match")))
    answers = iter(["aprove", "reject"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main(["--data-dir", str(data_dir), *TRIAGE]) == 0
    out = capsys.readouterr().out
    assert "Please type approve, reject or skip." in out
    assert f"Analyst decision: reject. {NO_CASE_RECORDS}" in out


def test_triage_checks_the_alert_before_any_model_call(data_dir, agent, capsys):
    model = agent(ScriptedModel())
    argv = ["--data-dir", str(data_dir), "triage", "Kareem Nouri", "--entity-id", "FX-999"]
    assert main(argv) == 1
    assert "no listed person" in capsys.readouterr().err and model.sessions == 0


@pytest.mark.parametrize(
    "flag",
    [["--country", "IRQ"], ["--country", "Iraq"], ["--birth-year", "70"], ["--birth-year", "2999"]],
)
def test_triage_rejects_identifiers_it_cannot_compare(flag, capsys):
    with pytest.raises(SystemExit) as stop:
        main([*TRIAGE, *flag])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert "ISO 3166-1 country code" in err or "4 digit birth year" in err


def test_triage_reads_identifiers_the_way_the_service_does():
    args = build_parser().parse_args([*TRIAGE, "--country", " UK ", "--birth-year", "1970"])
    assert (args.country, args.birth_year) == ("gb", 1970)


def test_triage_eval_results_stay_out_of_the_published_ones(
    data_dir, agent, work_dir, monkeypatch, capsys
):
    # The README report reads only results/<system>.json, never results/agent/.
    assert Path("results") / "agent" == cli.DEFAULT_AGENT_RESULTS_DIR
    model = agent(ScriptedModel(*look_then("likely_match")))
    scripted_alerts(monkeypatch)
    assert main(["--data-dir", str(data_dir), "triage-eval", "--yes"]) == 1  # not the snapshot
    assert "--results-dir" in capsys.readouterr().err
    assert model.sessions == 0 and not (work_dir / "results").exists()


def scripted_alerts(monkeypatch):
    alerts = [
        Scenario(
            "match_agree", Alert("A0001", "Kareem Fadil Nouri", 1970, "iq", "FX-001", 1.0), "match"
        ),
        Scenario(
            "false_positive",
            Alert("A0002", "Kareem Fadil Nouri", 1999, "gb", "FX-001", 1.0),
            "not_match",
        ),
    ]
    monkeypatch.setattr(
        "sanctions_mcp.agent.scenarios.build_scenarios", lambda service, es, split, n: alerts
    )


def test_triage_eval_saves_every_alert_as_it_finishes(
    data_dir, agent, small_evalset, tmp_path, monkeypatch, capsys
):
    agent(ScriptedModel(*look_then("likely_false_positive")))
    scripted_alerts(monkeypatch)
    summary_dir = tmp_path / "agent-results"
    argv = ["--data-dir", str(data_dir), "triage-eval", "--yes", "--limit", "5"]
    assert main([*argv, "--results-dir", str(summary_dir)]) == 0
    captured = capsys.readouterr()
    assert "Only 2 suitable alerts" in captured.err
    assert "about $" in captured.out and "at most about $" in captured.out
    assert "up to" not in captured.out
    summary = read(summary_dir / "dev.json")
    assert summary["n"] == 2 and summary["complete"] is True
    assert summary["screening"]["scorer"] == "rules"
    # The clear on the real match was turned into escalate by the code rule; the false
    # positive was cleared. Both outcomes are right.
    assert summary["accuracy_final"] == 1.0 and summary["policy_overrides"] == 0.5
    rows_file = data_dir / "agent-runs" / "dev.jsonl"
    rows = [json.loads(line) for line in rows_file.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["match_agree", "false_positive"]
    assert [r["final"] for r in rows] == ["escalate", "likely_false_positive"]
    assert rows[0]["alert"]["entity_id"] == "FX-001" and rows[0]["end_reason"] == "submitted"


def test_triage_eval_with_no_suitable_alerts_runs_nothing(
    data_dir, agent, small_evalset, tmp_path, monkeypatch, capsys
):
    model = agent(ScriptedModel())
    monkeypatch.setattr(
        "sanctions_mcp.agent.scenarios.build_scenarios", lambda service, es, split, n: []
    )
    earlier = data_dir / "agent-runs" / "dev.jsonl"  # the rows of an earlier paid run
    earlier.parent.mkdir(parents=True)
    earlier.write_text('{"alert_id": "A0001"}\n', encoding="utf-8")
    summary_dir = tmp_path / "agent-results"
    argv = ["--data-dir", str(data_dir), "triage-eval", "--yes", "--results-dir", str(summary_dir)]
    assert main(argv) == 1
    assert "error: no suitable alerts in the dev split" in capsys.readouterr().err
    assert earlier.read_text(encoding="utf-8") == '{"alert_id": "A0001"}\n'
    assert not summary_dir.exists() and model.sessions == 0


def test_end_of_input_at_a_prompt_counts_as_no(
    data_dir, agent, small_evalset, tmp_path, monkeypatch, capsys
):
    def end_of_input(prompt=""):
        raise EOFError

    def no_download(*args, **kwargs):
        raise AssertionError("nothing should be downloaded")

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", end_of_input)
    monkeypatch.setattr(cli, "download", no_download)
    assert main(["--data-dir", str(tmp_path / "empty"), "download"]) == 1
    assert "Cancelled. Nothing was downloaded." in capsys.readouterr().out

    model = agent(ScriptedModel(*look_then("likely_match")))
    scripted_alerts(monkeypatch)
    summary_dir = tmp_path / "agent-results"
    argv = ["--data-dir", str(data_dir), "triage-eval", "--results-dir", str(summary_dir)]
    assert main(argv) == 1
    assert "Not run. Nothing was sent to the model." in capsys.readouterr().out
    assert model.sessions == 0 and not summary_dir.exists()


def test_an_interrupted_triage_eval_keeps_what_was_paid_for(
    data_dir, agent, small_evalset, tmp_path, monkeypatch
):
    agent(ScriptedModel(scripts=[look_then("likely_match"), [KeyboardInterrupt()]]))
    scripted_alerts(monkeypatch)
    summary_dir = tmp_path / "agent-results"
    argv = ["--data-dir", str(data_dir), "triage-eval", "--yes", "--results-dir", str(summary_dir)]
    assert main(argv) == 130
    summary = read(summary_dir / "dev.json")
    assert summary["n"] == 1 and summary["complete"] is False
    rows = (data_dir / "agent-runs" / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1


def test_a_triage_eval_stopped_before_any_alert_leaves_the_last_run_alone(
    data_dir, agent, small_evalset, tmp_path, monkeypatch
):
    agent(ScriptedModel(scripts=[[KeyboardInterrupt()]]))
    scripted_alerts(monkeypatch)
    earlier = data_dir / "agent-runs" / "dev.jsonl"  # the rows of an earlier paid run
    earlier.parent.mkdir(parents=True)
    earlier.write_text('{"alert_id": "A0001"}\n', encoding="utf-8")
    summary_dir = tmp_path / "agent-results"
    summary_dir.mkdir()
    (summary_dir / "dev.json").write_text('{"n": 30}\n', encoding="utf-8")
    argv = ["--data-dir", str(data_dir), "triage-eval", "--yes", "--results-dir", str(summary_dir)]
    assert main(argv) == 130
    assert earlier.read_text(encoding="utf-8") == '{"alert_id": "A0001"}\n'
    assert (summary_dir / "dev.json").read_text(encoding="utf-8") == '{"n": 30}\n'
    assert not (data_dir / "agent-runs" / "dev.jsonl.part").exists()


def test_screen_checks_top_and_the_name(data_dir, capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--data-dir", str(data_dir), "screen", "Karim Nouri", "--top", "0"])
    assert stop.value.code == 2
    assert main(["--data-dir", str(data_dir), "screen", "   "]) == 1
    assert "error:" in capsys.readouterr().err


def test_a_damaged_model_file_is_a_calibration_error(trained):
    from sanctions_mcp.data import sha256_of, snapshot_path
    from sanctions_mcp.service import CalibrationError, load_calibrated_scorer, model_path

    path = model_path(trained)
    path.write_text('{"features": [', encoding="utf-8")  # cut off mid write
    list_sha = sha256_of(snapshot_path(trained))
    with pytest.raises(CalibrationError, match="could not be read"):
        load_calibrated_scorer(trained, list_sha)
