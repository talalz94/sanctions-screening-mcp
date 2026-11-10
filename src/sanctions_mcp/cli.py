"""Command line entry point.

sanctions download          fetch the pinned OpenSanctions snapshot (asks you to accept its license)
sanctions build-evalset     generate the eval set from the snapshot (deterministic)
sanctions train             fit the learned scorer, then calibrate the alert thresholds
sanctions eval --system X   evaluate one system (test split, or --split dev) and save the result
sanctions reproduce         rebuild the eval set, train, evaluate every system, rewrite the README
sanctions report            rewrite the README results from saved files
sanctions screen "NAME"     screen one name against the full list
sanctions stats             counts behind the data card, saved to results/stats.json
sanctions serve             run the MCP server (stdio, or --http for streamable HTTP)
sanctions triage NAME ...   let the agent investigate one alert and propose a disposition
sanctions triage-eval       run the agent over scripted alerts (calls the configured model)

Anything that shows an alert flag (serve, triage, triage-eval, and screen when it can)
uses a threshold `sanctions train` calibrated for the exact list file loaded. Without one,
serve and the agent commands stop with a message, and screen prints scores only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import httpx

from sanctions_mcp.baselines import ExactMatcher, TfidfMatcher
from sanctions_mcp.data import (
    NOTICE,
    SNAPSHOT,
    ChecksumError,
    NotPinnedSnapshotError,
    default_data_dir,
    describe_list,
    download,
    load_watchlist,
    sha256_of,
    snapshot_path,
)
from sanctions_mcp.evalset import EvalSet, build
from sanctions_mcp.evaluate import SLICE_BUDGET, run_system, summarize, threshold_for
from sanctions_mcp.index import WatchlistIndex
from sanctions_mcp.matcher import Matcher, training_pairs
from sanctions_mcp.normalize import has_arabic
from sanctions_mcp.report import (
    SYSTEMS,
    ReadmeError,
    ResultsMismatchError,
    check_readme,
    is_pinned,
    load_results,
    load_stats,
    render,
    save_result,
    save_stats,
    update_readme,
)
from sanctions_mcp.scoring import LogisticScorer, RuleScorer
from sanctions_mcp.service import (
    EARLIEST_BIRTH_YEAR,
    SCORERS,
    CalibrationError,
    ScreeningError,
    ScreeningService,
    clean_birth_year,
    clean_country,
    clean_name,
    load_calibrated_scorer,
    model_path,
)
from sanctions_mcp.settings import SettingsError, load_settings

SEED = 13
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_AGENT_RESULTS_DIR = DEFAULT_RESULTS_DIR / "agent"  # the README report never reads it
DEFAULT_README = Path("README.md")
MOVE_ON = (
    "To go on, set SNAPSHOT and SHA256 in src/sanctions_mcp/data.py to a current snapshot; "
    "the numbers will then differ from the README."
)
NO_CASE_RECORDS = (
    "This tool stores no case records: record the analyst decision in your case system."
)


class CliError(Exception):
    """A command that cannot run as asked, with a message that says what to do instead."""


def _data_dir(args) -> Path:
    return args.data_dir or default_data_dir()


def evalset_path(args) -> Path:
    return _data_dir(args) / "evalsets" / SNAPSHOT / f"seed-{SEED}.json"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)  # stdout stays clean for JSON and MCP stdio


def _ask(prompt: str) -> str:
    """The answer typed, in lowercase. When input ends instead (Ctrl+D, Ctrl+Z, a closed
    stdin) the answer is empty, which every prompt here reads as no."""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        print()  # end the prompt line
        return ""


def _load_list(args) -> tuple[list, dict]:
    """The people on the list, and what the list file is (see data.describe_list)."""
    path = snapshot_path(_data_dir(args))
    entities = load_watchlist(path)
    return entities, describe_list(path, len(entities))


def _evalset_list_path(args) -> Path:
    """Beside the eval set: the SHA-256 of the list file it was built from."""
    return evalset_path(args).with_suffix(".list_sha256")


def _evalset_list_sha(args) -> str | None:
    path = _evalset_list_path(args)
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def _save_evalset(args, es: EvalSet, dataset: dict) -> Path:
    """Save the eval set with the SHA-256 of the list file it was built from. The old
    record goes first, so an eval set cut off while saving never passes for a good one."""
    path, list_record = evalset_path(args), _evalset_list_path(args)
    list_record.unlink(missing_ok=True)
    es.save(path)
    list_record.write_text(dataset["sha256"] + "\n", encoding="utf-8", newline="\n")
    return path


def _evalset(args, entities, dataset: dict, *, rebuild: bool = False) -> tuple[EvalSet, str]:
    """The eval set and the SHA-256 of its file, which models and results record. It is
    built again when asked, when it is missing, and when it was not built from this exact
    list file."""
    path = evalset_path(args)
    built_from = _evalset_list_sha(args)
    if not rebuild and path.exists() and built_from != dataset["sha256"]:
        why = (
            "does not record which list file it came from"
            if built_from is None
            else "was built from another list file"
        )
        _log(f"The saved eval set {why}. Building it again ...")
        rebuild = True
    if rebuild or not path.exists():
        _log(f"Building the eval set (seed {SEED}) ...")
        _save_evalset(args, build(entities, seed=SEED), dataset)
    return EvalSet.load(path), sha256_of(path)


NOT_PINNED_LIST = "the loaded list is not the pinned OpenSanctions snapshot"


def _output(args, name: str, default: Path, *, pinned: bool, reason: str, flags: str) -> Path:
    """An output path. The defaults hold the published numbers, so they are used only for
    results measured on the pinned snapshot; anything else needs a path given explicitly
    (`flags` names the options to pass)."""
    given = getattr(args, name)
    if given is not None:
        return given
    if not pinned:
        raise CliError(
            f"{reason}, so these results must not replace the published ones in "
            f"{default.as_posix()}. Pass {flags} to write them somewhere else."
        )
    return default


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: dict) -> None:
    """Write to a temporary file next to the target, then swap it in, so an interrupted
    write never leaves half a file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def cmd_download(args) -> int:
    print(NOTICE)
    if not args.yes:
        if not sys.stdin.isatty():
            print("Not an interactive terminal. Run again with --yes to accept the notice.")
            return 2
        if _ask("Type yes to download: ") != "yes":
            print("Cancelled. Nothing was downloaded.")
            return 1
    try:
        path = download(_data_dir(args), force=args.force)
    except httpx.HTTPStatusError as err:
        _log(
            f"error: OpenSanctions snapshot {SNAPSHOT} could not be downloaded (HTTP "
            f"{err.response.status_code}). A 401, 403 or 404 usually means this pinned "
            f"snapshot is no longer public. {MOVE_ON}"
        )
        return 1
    except httpx.HTTPError as err:
        _log(
            f"error: OpenSanctions snapshot {SNAPSHOT} could not be downloaded "
            f"({type(err).__name__}: {err}). Check the network connection and try again."
        )
        return 1
    except ChecksumError:
        _log(
            f"error: the file downloaded for OpenSanctions snapshot {SNAPSHOT} does not match "
            f"the pinned SHA-256, so it was not kept. {MOVE_ON}"
        )
        return 1
    print(f"Done. Verified snapshot at {path.as_posix()}")
    return 0


def cmd_build_evalset(args) -> int:
    entities, dataset = _load_list(args)
    es = build(entities, seed=SEED)
    path = _save_evalset(args, es, dataset)
    counts = Counter((q.split, q.kind) for q in es.queries)
    _log(f"Saved {len(es.queries):,} queries to {path.as_posix()}")
    for split in ("train", "dev", "test"):
        parts = ", ".join(f"{k} {n}" for (s, k), n in sorted(counts.items()) if s == split)
        _log(f"  {split}: {parts}")
    return 0


def _clean_threshold(matcher: Matcher, texts: Sequence[str]) -> float:
    """The threshold that lets at most SLICE_BUDGET of every 1,000 clean names alert."""
    return threshold_for(
        [max(matcher.scores(t).values(), default=0.0) for t in texts], SLICE_BUDGET
    )


def _train(args, entities, dataset: dict, es: EvalSet, es_sha: str) -> None:
    """Fit the learned scorer, then calibrate the alert thresholds the server, `screen` and
    the triage agent use, and save them with the list and eval set they came from."""
    start = time.perf_counter()
    held_out_index = WatchlistIndex(entities, exclude=es.held_out)
    x, y = training_pairs(Matcher(held_out_index, RuleScorer()), es.by_split("train"))
    scorer = LogisticScorer.fit(x, y)
    fitted = time.perf_counter() - start
    dev_clean = [q.text for q in es.by_split("dev") if q.target is None]
    if not dev_clean:
        raise CliError("the eval set has no clean names in the dev split to set a threshold on")
    # The model is fitted with held out names kept out of the index, as the eval needs. The
    # server, `screen` and the triage agent search the full list, so their thresholds are
    # set there: the same dev clean names (none of them is any listed name, held out or
    # not), scored against the full index. The held out threshold is logged for comparison.
    held_out_t = _clean_threshold(Matcher(held_out_index, scorer), dev_clean)
    full_index = WatchlistIndex(entities)
    scorer.alert_threshold = _clean_threshold(Matcher(full_index, scorer), dev_clean)
    rules_t = _clean_threshold(Matcher(full_index, RuleScorer()), dev_clean)
    calibration = {
        "budget_per_1000": SLICE_BUDGET,
        "split": "dev",
        "clean_names": len(dev_clean),
        "index": "full list",
    }
    record = {"list_sha256": dataset["sha256"], "evalset_sha256": es_sha}
    learned_path = model_path(_data_dir(args), "learned")
    scorer.save(learned_path)
    saved = _read_json(learned_path)
    saved.update(
        scorer="learned",
        **record,
        calibration={**calibration, "held_out_index_threshold": held_out_t},
    )
    _write_json(learned_path, saved)
    rules_path = model_path(_data_dir(args), "rules")
    _write_json(
        rules_path,
        {"scorer": "rules", "alert_threshold": rules_t, **record, "calibration": calibration},
    )
    _log(
        f"Trained on {len(y):,} pairs ({int(y.sum()):,} positive) in {fitted:.0f} s. "
        f"Saved {learned_path.as_posix()}"
    )
    _log(
        f"Alert thresholds for {SLICE_BUDGET} false alerts per 1,000 dev clean names "
        f"({len(dev_clean):,}) on the full list: learned {scorer.alert_threshold:.4f} (on the "
        f"eval's held out index it would be {held_out_t:.4f}), rules {rules_t:.4f} "
        f"(saved {rules_path.as_posix()})"
    )
    for name, w in sorted(scorer.weights().items(), key=lambda kv: -abs(kv[1])):
        _log(f"  {name:<16} {w:+.3f}")


def cmd_train(args) -> int:
    entities, dataset = _load_list(args)
    es, es_sha = _evalset(args, entities, dataset)
    _train(args, entities, dataset, es, es_sha)
    return 0


def _build_system(name: str, args, entities, dataset: dict, es: EvalSet, es_sha: str):
    if name == "exact":
        return ExactMatcher(entities, exclude=es.held_out)
    if name in ("tfidf", "tfidf-rapidfuzz"):
        return TfidfMatcher(entities, exclude=es.held_out, rescore=name == "tfidf-rapidfuzz")
    index = WatchlistIndex(entities, exclude=es.held_out)
    if name == "matcher-rules":
        return Matcher(index, RuleScorer())
    path = model_path(_data_dir(args), "learned")
    saved = _read_json(path)
    if not saved:
        _log("No model trained on this list and eval set yet. Training one ...")
        _train(args, entities, dataset, es, es_sha)
    elif saved.get("list_sha256") != dataset["sha256"] or saved.get("evalset_sha256") != es_sha:
        _log("The saved model was trained on another list file or eval set. Training it again ...")
        _train(args, entities, dataset, es, es_sha)
    return Matcher(index, LogisticScorer.load(path))


def _as_dev(value):
    """A dev only summary. evaluate.summarize was given the dev run in place of the test
    run, so its test_ keys hold dev numbers: they are renamed dev_."""
    if isinstance(value, dict):
        return {
            ("dev_" + k[5:] if k.startswith("test_") else k): _as_dev(v) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_as_dev(v) for v in value]
    return value


def _evaluate(
    name: str,
    args,
    entities,
    dataset: dict,
    es: EvalSet,
    es_sha: str,
    results_dir: Path,
    split: str,
) -> None:
    """Evaluate one system. Thresholds always come from dev. With split "test" the result
    covers the test split and goes to `results_dir`; with "dev" every number is from dev,
    the test split is never run, and the result goes to `results_dir/dev`, which the README
    report never reads."""
    system = _build_system(name, args, entities, dataset, es, es_sha)
    dev = es.by_split("dev")

    def progress(done: int, total: int) -> None:
        _log(f"  {name}: {done:,}/{total:,}")

    dev_run = run_system(system, dev, "dev", on_progress=progress)
    if split == "dev":
        summary = _as_dev(summarize(dev_run, dev_run, dev))
        out_dir = results_dir / "dev"
    else:
        test = es.by_split("test")
        summary = summarize(dev_run, run_system(system, test, "test", on_progress=progress), test)
        out_dir = results_dir
    extra = {}
    if isinstance(getattr(system, "scorer", None), LogisticScorer):
        extra["weights"] = system.scorer.weights()
    path = save_result(
        name,
        summary,
        out_dir,
        extra,
        dataset=dataset,
        evalset={"seed": SEED, "sha256": es_sha},
        split=split,
    )
    points = ", ".join(
        f"@{p['budget_per_1000']}: {100 * p[f'{split}_recall']:.1f}%"
        for p in summary["operating_points"]
    )
    _log(f"{name:<16} {split} recall {points}   saved {path.as_posix()}")


def cmd_eval(args) -> int:
    entities, dataset = _load_list(args)
    results_dir = _output(
        args,
        "results_dir",
        DEFAULT_RESULTS_DIR,
        pinned=dataset["snapshot"] is not None,
        reason=NOT_PINNED_LIST,
        flags="--results-dir",
    )
    es, es_sha = _evalset(args, entities, dataset)
    _evaluate(args.system, args, entities, dataset, es, es_sha, results_dir, args.split)
    return 0


def cmd_reproduce(args) -> int:
    entities, dataset = _load_list(args)
    pinned = dataset["snapshot"] is not None
    both = "--results-dir and --readme"
    results_dir = _output(
        args, "results_dir", DEFAULT_RESULTS_DIR, pinned=pinned, reason=NOT_PINNED_LIST, flags=both
    )
    readme = _output(
        args, "readme", DEFAULT_README, pinned=pinned, reason=NOT_PINNED_LIST, flags=both
    )
    check_readme(readme)  # now, not after the slow work below
    # Always rebuilt: the build is deterministic, so unchanged code gives the same file,
    # and changed eval set code can never leave stale queries behind.
    es, es_sha = _evalset(args, entities, dataset, rebuild=True)
    _train(args, entities, dataset, es, es_sha)
    for name in args.systems:
        _evaluate(name, args, entities, dataset, es, es_sha, results_dir, "test")
    path = save_stats(list_stats(args, entities, dataset), results_dir)
    _log(f"Saved {path.as_posix()}")
    return _report(args, results_dir, readme)


def _report(args, results_dir: Path, readme: Path | None = None) -> int:
    results = load_results(results_dir)
    if not results:
        print(f"No results in {results_dir.as_posix()}. Run `sanctions reproduce` first.")
        return 1
    if readme is None:
        readme = _output(
            args,
            "readme",
            DEFAULT_README,
            pinned=all(is_pinned(r) for r in results.values()),
            reason=f"the results in {results_dir.as_posix()} were not all measured on the "
            "pinned OpenSanctions snapshot",
            flags="--readme",
        )
    changed = update_readme(readme, render(results, load_stats(results_dir)))
    print(f"{readme.as_posix()} {'updated' if changed else 'already up to date'}")
    return 0


def cmd_report(args) -> int:
    return _report(args, args.results_dir or DEFAULT_RESULTS_DIR)


def _scores_only_scorer(args) -> RuleScorer | LogisticScorer:
    """The scorer for scores without alert flags: the learned model when one was asked
    for and its file still loads (it only lacks a threshold calibrated on this list),
    otherwise the rule scorer."""
    if args.scorer == "learned":
        try:
            return LogisticScorer.load(model_path(_data_dir(args), "learned"))
        except (OSError, ValueError, KeyError):  # missing, unreadable or another model
            pass
    return RuleScorer()


def cmd_screen(args) -> int:
    entities, dataset = _load_list(args)
    try:
        scorer, threshold, source = load_calibrated_scorer(
            _data_dir(args), dataset["sha256"], args.scorer
        )
        _log(f"Alert threshold {threshold:.4f}, {args.scorer} scorer, {source}.")
    except CalibrationError:
        scorer, threshold = _scores_only_scorer(args), None
        instead = "" if scorer.name == args.scorer else f" (no {args.scorer} model loads)"
        _log(
            f"Scores only, from the {scorer.name} scorer{instead}, with no alert flags: alerts "
            "need a threshold calibrated on this list. Run `sanctions train` first."
        )
    matcher = Matcher(WatchlistIndex(entities), scorer)
    by_id = {e.id: e for e in entities}
    name = clean_name(args.name)  # an empty or overlong name stops here, as in the service
    matches = matcher.screen(name, top_k=args.top)
    if not matches:
        print("No candidates.")
    for m in matches:
        e = by_id[m.entity_id]
        flag = "" if threshold is None else ("ALERT  " if m.score >= threshold else "       ")
        print(f"{m.score:.3f}  {flag}{e.name}  (matched on: {m.matched_name})  [{m.entity_id}]")
        born = ", ".join(e.birth_dates) or "n/a"
        countries = ", ".join(e.countries) or "n/a"
        print(f"       born {born}; countries {countries}")
        if args.explain:
            alignment = matcher.explain(name, m.entity_id).get("alignment", [])
            print("       " + json.dumps(alignment, ensure_ascii=False))
    return 0


def list_stats(args, entities, dataset: dict) -> dict:
    """Counts of the list itself, plus counts of the eval set when one built from this
    list file is already saved. The eval set is never built here, so the list counts never
    wait for it."""
    index = WatchlistIndex(entities)  # the full index the server and `screen` search
    stats = {
        "snapshot": dataset["snapshot"],  # None unless the file is the pinned snapshot
        "list_sha256": dataset["sha256"],
        "people": len(entities),
        "source_lists": len({ds for e in entities for ds in e.datasets}),
        # Every distinct name string, including names that give no words to index.
        "names_including_aliases": sum(len(e.names) for e in entities),
        # Names with at least one word to index, before one person's names that are
        # identical once normalised are merged.
        "names_listed": index.listed_names,
        # Rows in the index after that merge: the count list_sources reports.
        "names_indexed": len(index),
        "people_with_an_arabic_script_name": sum(
            any(has_arabic(n) for n in e.names) for e in entities
        ),
        "people_with_a_birth_date": sum(bool(e.birth_dates) for e in entities),
    }
    path = evalset_path(args)
    if path.exists() and _evalset_list_sha(args) == dataset["sha256"]:
        es = EvalSet.load(path)
        stats["eval_set"] = {
            "sha256": sha256_of(path),
            "names_held_out_of_the_index": sum(len(v) for v in es.held_out.values()),
            "queries": {
                split: dict(sorted(Counter(q.kind for q in es.by_split(split)).items()))
                for split in ("train", "dev", "test")
            },
        }
    return stats


def cmd_stats(args) -> int:
    entities, dataset = _load_list(args)
    stats = list_stats(args, entities, dataset)
    print(json.dumps(stats, indent=2))
    if args.results_dir is None and dataset["snapshot"] is None:
        _log(
            f"Not saved: {NOT_PINNED_LIST}. Pass --results-dir to save these counts somewhere "
            "other than results/."
        )
        return 0
    path = save_stats(stats, args.results_dir or DEFAULT_RESULTS_DIR)
    _log(f"Saved {path.as_posix()}")
    return 0


def cmd_serve(args) -> int:
    from sanctions_mcp.server import build_server

    service = ScreeningService.from_data_dir(_data_dir(args), scorer=args.scorer)
    _log(
        f"Serving {len(service.entities):,} people with the {args.scorer} scorer, alert "
        f"threshold {service.alert_threshold:.4f}, {service.threshold_source}."
    )
    server = build_server(service)
    if args.http:
        # Stateless JSON responses: the shape a hosted runtime such as AgentCore expects.
        server.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            stateless_http=True,
            json_response=True,
        )
    else:
        server.run()
    return 0


def _agent_setup(args):
    from sanctions_mcp.agent.models import model_from_settings

    settings = load_settings(args.settings)
    service = ScreeningService.from_data_dir(_data_dir(args))
    source = args.settings.as_posix() if args.settings else "defaults and environment"
    _log(f"Model: {settings.provider} {settings.model} (settings: {source}).")
    return settings, service, model_from_settings(settings)


def _ask_decision() -> str:
    while True:
        answer = _ask("Analyst decision (approve / reject / skip): ")
        if answer in ("approve", "reject", "skip", ""):
            return answer or "skip"  # no answer, or input ended: skip
        print("Please type approve, reject or skip.")


def cmd_triage(args) -> int:
    from sanctions_mcp.agent.triage import Alert, triage

    _, service, model = _agent_setup(args)
    # Checks the name and the entity id before any model call, and gives the score that
    # screening gives this person, even when they are outside the top candidates.
    why = service.explain(args.name, args.entity_id)
    name = clean_name(args.name)
    alert = Alert("manual", name, args.birth_year, args.country, args.entity_id, why["score"])
    result = triage(alert, model, service)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    if result.overridden:
        print(f"Note: the clearing rule changed {result.proposed} to {result.final}.")
    if not result.submitted:
        print(
            f"Note: the model submitted no disposition ({result.end_reason}), so the alert "
            "is escalated."
        )
    if sys.stdin.isatty():
        print(f"Analyst decision: {_ask_decision()}. {NO_CASE_RECORDS}")
    else:
        print(NO_CASE_RECORDS)
    return 0


def cmd_triage_eval(args) -> int:
    from sanctions_mcp.agent.evaluate import cost_estimate, run, summarize
    from sanctions_mcp.agent.scenarios import build_scenarios
    from sanctions_mcp.agent.triage import MAX_TURNS

    settings, service, model = _agent_setup(args)
    summary_dir = _output(
        args,
        "results_dir",
        DEFAULT_AGENT_RESULTS_DIR,
        pinned=(service.dataset or {}).get("snapshot") is not None,
        reason=NOT_PINNED_LIST,
        flags="--results-dir",
    )
    es, _ = _evalset(args, list(service.entities.values()), service.dataset)
    scenarios = build_scenarios(service, es, split=args.split, n=args.limit)
    n = len(scenarios)
    if n == 0:
        # Before the prompt and before the rows file is opened, so the files of an earlier
        # run stay as they are.
        raise CliError(f"no suitable alerts in the {args.split} split, so nothing was run")
    if n < args.limit:
        _log(f"Only {n} suitable alerts in the {args.split} split (asked for {args.limit}).")
    typical, worst = cost_estimate(settings)
    fallbacks = getattr(model, "fallbacks", "off")
    print(f"{n} alerts on {model.name}, effort {settings.effort}, refusal fallback {fallbacks}.")
    print(
        f"Cost at the prices in settings: about ${n * typical:.2f} for a typical run, and at "
        f"most about ${n * worst:.2f} if every alert used all {MAX_TURNS} model calls at "
        f"max_tokens {settings.max_tokens:,}."
    )
    if getattr(model, "use_fallbacks", False):
        print(
            "Both figures leave out refusal fallbacks: a declined turn that a fallback model "
            "answers is billed twice, the second time at that model's own rates."
        )
    if not args.yes and (not sys.stdin.isatty() or _ask("Type yes to run: ") != "yes"):
        print("Not run. Nothing was sent to the model.")
        return 1

    sources = service.sources()
    screening = {
        k: sources[k]
        for k in ("snapshot", "list_sha256", "scorer", "alert_threshold", "alert_threshold_source")
    }
    summary_path = summary_dir / f"{args.split}.json"
    # One row per alert, written as each alert finishes. Rows quote names from the list, so
    # they stay in the data dir, which git ignores, never in results/.
    rows_path = _data_dir(args) / "agent-runs" / f"{args.split}.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    done: list = []

    def record(s, r) -> None:
        done.append((s, r))
        row = {"kind": s.kind, "truth": s.truth, "alert": asdict(s.alert), **r.to_dict()}
        rows.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.flush()
        _log(
            f"  {r.alert_id} {s.kind:<18} proposed {r.proposed:<22} final {r.final:<22} "
            f"{r.end_reason}"
        )

    # Rows go to a .part file first: the files of an earlier run are replaced only once an
    # alert of this run has finished, so a run that fails at once leaves them as they were.
    part_path = rows_path.with_name(rows_path.name + ".part")
    rows = part_path.open("w", encoding="utf-8", newline="\n")
    try:
        run(scenarios, model, service, on_result=record)
    finally:
        rows.close()
        if done:
            # Also on an interrupt or a crash: the alerts already paid for are kept.
            part_path.replace(rows_path)
            summary = summarize(
                [s for s, _ in done],
                [r for _, r in done],
                settings,
                fallbacks=fallbacks,
                screening=screening,
            )
            summary["complete"] = len(done) == n
            _write_json(summary_path, summary)
            _log(
                f"Saved {summary_path.as_posix()}, and one row per alert in {rows_path.as_posix()}"
            )
        else:
            part_path.unlink(missing_ok=True)
            _log("No alert finished, so the files of any earlier run are left as they were.")
    headline = (
        "accuracy_final",
        "accuracy_proposed",
        "no_submission",
        "proposed_clear_on_real_match",
        "final_clear_on_real_match",
        "final_clear_on_real_match_n",
        "injection_followed",
        "errors",
        "cost_usd_per_1000_alerts",
    )
    print(json.dumps({k: summary[k] for k in headline}, indent=2))
    return 0


def _country_arg(text: str) -> str:
    code = clean_country(text)
    if code is None:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a 2 letter ISO 3166-1 country code, such as iq or gb"
        )
    return code


def _birth_year_arg(text: str) -> int:
    year = clean_birth_year(text)
    if year is None:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a 4 digit birth year from {EARLIEST_BIRTH_YEAR} to this year"
        )
    return year


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of 1 or more")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanctions", description="Sanctions name screening.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="default: $SANCTIONS_DATA_DIR or ./data. Keep any other data dir outside the "
        "repo: it holds list data that must never be committed.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the pinned OpenSanctions snapshot")
    p.add_argument("--yes", action="store_true", help="accept the data notice without a prompt")
    p.add_argument(
        "--force",
        action="store_true",
        help="download again, replacing the file at the snapshot path even when it is not "
        "the pinned snapshot",
    )
    p.set_defaults(func=cmd_download)

    sub.add_parser("build-evalset", help="generate the eval set").set_defaults(
        func=cmd_build_evalset
    )
    sub.add_parser(
        "train", help="fit the learned scorer and calibrate the alert thresholds"
    ).set_defaults(func=cmd_train)

    def results_dir_option(p, text="default: results/, allowed only for the pinned snapshot"):
        p.add_argument("--results-dir", type=Path, default=None, help=text)

    def readme_option(p):
        p.add_argument(
            "--readme",
            type=Path,
            default=None,
            help="default: README.md, allowed only for the pinned snapshot. It needs the "
            "results markers.",
        )

    p = sub.add_parser("eval", help="evaluate one system")
    p.add_argument("--system", choices=list(SYSTEMS), required=True)
    p.add_argument(
        "--split",
        choices=["dev", "test"],
        default="test",
        help="test (default) reports the test split; dev reports dev only, never runs test, "
        "and saves to <results-dir>/dev/",
    )
    results_dir_option(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("reproduce", help="train, evaluate every system, rewrite the README")
    p.add_argument("--systems", nargs="+", choices=list(SYSTEMS), default=list(SYSTEMS))
    results_dir_option(p)
    readme_option(p)
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("report", help="rewrite the README results")
    results_dir_option(
        p,
        "where the saved results are read from (default: results/). Results not all measured "
        "on the pinned snapshot need --readme.",
    )
    readme_option(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("screen", help="screen one name against the full list")
    p.add_argument("name")
    p.add_argument("--top", type=_positive_int, default=5)
    p.add_argument("--explain", action="store_true")
    p.add_argument("--scorer", choices=SCORERS, default="learned")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("stats", help="counts behind the data card, saved to stats.json")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="where stats.json goes (default: results/, used only for the pinned snapshot; "
        "for any other list the counts are printed but not saved)",
    )
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("serve", help="run the MCP server (stdio by default)")
    p.add_argument("--http", action="store_true", help="stateless streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--scorer", choices=SCORERS, default="learned")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("triage", help="let the agent investigate one alert")
    p.add_argument("name")
    p.add_argument("--entity-id", required=True)
    p.add_argument("--birth-year", type=_birth_year_arg, help="4 digit year, e.g. 1970")
    p.add_argument("--country", type=_country_arg, help="ISO 3166-1 alpha-2 code, e.g. iq")
    p.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="settings file (default: settings.toml in the working folder, if there is one)",
    )
    p.set_defaults(func=cmd_triage)

    p = sub.add_parser("triage-eval", help="run the agent over scripted alerts")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--limit", type=_positive_int, default=30, help="number of alerts (default 30)")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    p.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="settings file (default: settings.toml in the working folder, if there is one)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="where the run summary goes (default: results/agent/, used only for the pinned "
        "snapshot)",
    )
    p.set_defaults(func=cmd_triage_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (
        FileNotFoundError,
        CalibrationError,
        ScreeningError,
        SettingsError,
        NotPinnedSnapshotError,
        ResultsMismatchError,
        ReadmeError,
        CliError,
    ) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
