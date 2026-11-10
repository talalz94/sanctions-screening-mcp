"""Save eval results as JSON and render them into the README between two markers."""

from __future__ import annotations

import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

from sanctions_mcp import __version__
from sanctions_mcp.data import SHA256, SNAPSHOT

START = "<!-- results:start -->"
END = "<!-- results:end -->"
API_COST_PER_1K = 0.0  # screening runs locally; the agent slice adds model costs

SYSTEMS: dict[str, str] = {
    "exact": "Exact match",
    "tfidf": "Fuzzy, character n-gram TF-IDF",
    "tfidf-rapidfuzz": "Fuzzy, TF-IDF then RapidFuzz token sort",
    "matcher-rules": "This matcher, fixed weights",
    "matcher-learned": "This matcher, learned weights",
}
KIND_LABELS = {
    "alias": "Real alias held out of the list",
    "cross_script": "Arabic script name, Latin entries only",
    "transliteration": "Generated: transliteration",
    "typo": "Generated: typo",
    "drop_middle": "Generated: middle name dropped",
    "initials": "Generated: initial for a name",
    "particles": "Generated: al, bin added or removed",
    "honorific": "Generated: title added",
    "spacing": "Generated: compound spacing",
}
GROUP_LABELS = {"mena_south_asia": "Arab League, Iran, Afghanistan, Pakistan", "other": "Other"}
STATS_FILE = "stats.json"


class ResultsMismatchError(ValueError):
    """The saved results were not all measured on the same list and eval set."""


class ReadmeError(ValueError):
    """The README to update is missing, or has no place marked for the results."""


def eval_command(system: str, split: str = "test") -> str:
    command = f"uv run sanctions eval --system {system}"
    return command if split == "test" else f"{command} --split {split}"


def _cpu() -> str:
    try:
        if platform.system() == "Windows":
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _created_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def save_result(
    system: str,
    summary: dict,
    results_dir: Path,
    extra: dict | None = None,
    *,
    dataset: dict,
    evalset: dict,
    split: str = "test",
) -> Path:
    """Write one system's result. `dataset` describes the list file that was loaded (see
    `data.describe_list`) and `evalset` the eval set it was measured on (seed, sha256)."""
    payload = {
        "system": system,
        "label": SYSTEMS[system],
        "command": eval_command(system, split),
        "split": split,
        "api_cost_usd_per_1k_screenings": API_COST_PER_1K,
        "dataset": dataset,
        "evalset": evalset,
        **summary,
        **(extra or {}),
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "cpu": _cpu(),
            "logical_cpus": os.cpu_count(),
        },
        "version": __version__,
        "created_utc": _created_utc(),
    }
    path = results_dir / f"{system}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def save_stats(stats: dict, results_dir: Path) -> Path:
    """Write the list counts of `sanctions stats` to `<results_dir>/stats.json`."""
    path = results_dir / STATS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**stats, "created_utc": _created_utc()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def load_stats(results_dir: Path) -> dict | None:
    """The saved list counts, or None when `sanctions stats` has not saved any here."""
    path = results_dir / STATS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_results(results_dir: Path) -> dict[str, dict]:
    """The saved test split result of each screening system.

    Only the `<system>.json` files that save_result writes are read, so anything else in
    the folder (stats.json, agent runs, notes, dev split runs in a subfolder) can never
    break the report or reach the results tables."""
    out = {}
    for system in SYSTEMS:
        path = results_dir / f"{system}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("split", "test") == "test":
            out[system] = data
    return out


def is_pinned(result: dict) -> bool:
    """True when a result was measured on the pinned OpenSanctions snapshot."""
    return (result.get("dataset") or {}).get("sha256") == SHA256


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def _point(r: dict, budget: int) -> dict:
    return next(p for p in r["operating_points"] if p["budget_per_1000"] == budget)


def _check_same(rows: list[dict]) -> None:
    """Every row of one table must come from the same list file and the same eval set."""
    checks = {
        "the list file": lambda r: (r.get("dataset") or {}).get("sha256"),
        "the eval set": lambda r: (r.get("evalset") or {}).get("sha256"),
        "the query counts": lambda r: r.get("counts"),
    }
    for what, value in checks.items():
        seen = {json.dumps(value(r), sort_keys=True) for r in rows}
        if len(seen) > 1:
            systems = ", ".join(r["system"] for r in rows)
            raise ResultsMismatchError(
                f"The saved results ({systems}) were not all measured on {what}. Run "
                "`sanctions reproduce` so every system is measured on the same data."
            )


def stats_line(stats: dict | None, first: dict) -> str | None:
    """One line on the list behind the results, from stats.json. None when there are no
    saved counts, or when the results do not record which list file they came from."""
    list_sha = (first.get("dataset") or {}).get("sha256")
    if not stats or list_sha is None:
        return None
    if stats.get("list_sha256") != list_sha:
        raise ResultsMismatchError(
            f"{STATS_FILE} was counted on a different list file than the results. Run "
            "`sanctions stats` or `sanctions reproduce` again."
        )
    snapshot = stats.get("snapshot")
    source = f"Snapshot `{snapshot}`" if snapshot else "List file (not the pinned snapshot)"
    lists = _count(stats["source_lists"], "source list")
    names = _count(stats["names_indexed"], "name")
    return f"{source}: {stats['people']:,} people from {lists}; {names} indexed."


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}" + ("" if n == 1 else "s")


def _latency_note(rows: list[dict]) -> str:
    sizes = {r["latency_ms"].get("n") for r in rows}
    note = "Latency is one screening at a time, end to end"
    if len(sizes) == 1 and None not in sizes:
        return f"{note}, on the same {sizes.pop():,} test queries for every system."
    return note + "."


def render(results: dict[str, dict], stats: dict | None = None) -> str:
    """The README results block. `stats` is the saved stats.json, if any; its line is left
    out when it is missing."""
    rows = [results[s] for s in SYSTEMS if s in results]
    if not rows:
        return "No results yet. Run `uv run sanctions reproduce`.\n"
    _check_same(rows)
    first = rows[0]
    counts = first["counts"]
    data_line = stats_line(stats, first)
    out = [data_line, ""] if data_line else []
    out += [
        f"**Test split**: {counts['test_positive']:,} queries that name a listed person and "
        f"{counts['test_clean']:,} clean names. Recall is the share of the named persons that "
        "alert, at thresholds set on the dev split to allow 1, 5 or 10 false alerts per 1,000 "
        "clean names.",
        "",
        "| System | Recall at 1 | Recall at 5 | Recall at 10 | Test false alerts per 1,000 "
        "(threshold for 5) | Candidate recall | p50 ms | p95 ms | API cost per 1k screenings "
        "| Reproduce |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lat = r["latency_ms"]
        out.append(
            f"| {r['label']} | {_pct(_point(r, 1)['test_recall'])} "
            f"| {_pct(_point(r, 5)['test_recall'])} | {_pct(_point(r, 10)['test_recall'])} "
            f"| {_point(r, 5)['test_false_alerts_per_1000']:.1f} "
            f"| {_pct(r.get('candidate_recall'))} | {lat['p50']:.1f} | {lat['p95']:.1f} "
            f"| ${r['api_cost_usd_per_1k_screenings']:.2f} | `{r['command']}` |"
        )

    kinds = [k for k in KIND_LABELS if k in first["recall_by_kind"]]
    out += [
        "",
        "**Recall by kind of query**, test split, at the threshold for 5 false alerts per 1,000:",
        "",
        "| Query | n | " + " | ".join(r["label"] for r in rows) + " |",
        "|---|---:|" + "---:|" * len(rows),
    ]
    for k in kinds:
        n = first["recall_by_kind"][k]["n"]
        cells = " | ".join(_pct(r["recall_by_kind"][k]["recall"]) for r in rows)
        out.append(f"| {KIND_LABELS[k]} | {n:,} | {cells} |")

    groups = [g for g in GROUP_LABELS if g in first["false_alerts_per_1000_by_group"]]
    out += [
        "",
        "**By name origin**, test split, same threshold. A listed person is in the first group "
        "when any of their countries is one of the 22 Arab League members, Iran, Afghanistan "
        "or Pakistan, or when any of their names is in Arabic script; everyone else is Other:",
        "",
        "| System | "
        + " | ".join(f"Recall, {GROUP_LABELS[g]}" for g in groups)
        + " | "
        + " | ".join(f"False alerts per 1,000, {GROUP_LABELS[g]}" for g in groups)
        + " |",
        "|---|" + "---:|" * (2 * len(groups)),
    ]
    for r in rows:
        rec = " | ".join(_pct(r["recall_by_group"][g]["recall"]) for g in groups)
        fa = " | ".join(f"{r['false_alerts_per_1000_by_group'][g]['rate']:.1f}" for g in groups)
        out.append(f"| {r['label']} | {rec} | {fa} |")

    learned = results.get("matcher-learned", {}).get("weights")
    if learned:
        out += [
            "",
            "**Learned weights** (logistic regression on standardised features, trained on the "
            "train split only):",
            "",
            "| Feature | Weight |",
            "|---|---:|",
        ]
        for name, w in sorted(learned.items(), key=lambda kv: -abs(kv[1])):
            out.append(f"| `{name}` | {w:+.2f} |")

    env = max(rows, key=lambda r: r["created_utc"])
    e = env["environment"]
    if is_pinned(first):
        source = f"OpenSanctions snapshot `{SNAPSHOT}`"
    else:
        source = "a list file not verified as the pinned OpenSanctions snapshot"
    out += [
        "",
        f"Measured on {e['cpu']} ({e['logical_cpus']} logical CPUs, CPU only), Python "
        f"{e['python']}, {env['created_utc'][:10]} UTC, {source}. {_latency_note(rows)}",
        "",
        "Generated by `uv run sanctions report` from `results/`. Do not edit by hand.",
    ]
    return "\n".join(out) + "\n"


def check_readme(readme: Path) -> str:
    """The README text, once it is known to have the two markers, in order. Raises
    ReadmeError otherwise, so `reproduce` can check before any of its slow work."""
    if not readme.is_file():
        raise ReadmeError(f"{readme.as_posix()} not found")
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(START), text.find(END)
    if start == -1 or end == -1 or end < start:
        raise ReadmeError(f"{readme.as_posix()} needs the {START} and {END} markers, in that order")
    return text


def update_readme(readme: Path, block: str) -> bool:
    text = check_readme(readme)
    start, end = text.find(START), text.find(END)
    updated = text[: start + len(START)] + "\n" + block + text[end:]
    if updated == text:
        return False
    readme.write_text(updated, encoding="utf-8", newline="\n")
    return True
