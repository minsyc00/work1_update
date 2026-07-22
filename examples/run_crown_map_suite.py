from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from statistics import mean, median
from time import perf_counter
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples" / "run_crown_mcpp_experiment.py"


def _map_paths() -> List[Path]:
    return sorted(
        path
        for path in (ROOT / "maps").glob("*/*.json")
        if path.name != "crown_fleet_profile.json"
    )


def _parse_cli_json(stdout: str) -> Dict[str, Any]:
    start = stdout.find("{")
    if start < 0:
        raise ValueError("CROWN CLI produced no JSON report")
    value = json.loads(stdout[start:])
    if not isinstance(value, dict):
        raise ValueError("CROWN CLI report must be a JSON object")
    return value


def _run_once(
    map_path: Path,
    outputs_root: Path,
    timeout_sec: float,
    lns_seconds: float | None,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(RUNNER),
        "--map",
        str(map_path),
        "--outputs-root",
        str(outputs_root),
        "--no-render",
    ]
    if lns_seconds is not None:
        command.extend(["--lns-seconds", str(lns_seconds)])
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-crown-suite")
    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "map": map_path.parent.name,
            "status": "timeout",
            "wall_runtime_sec": perf_counter() - started,
            "timeout_sec": timeout_sec,
            "stdout_tail": (error.stdout or "")[-2000:],
            "stderr_tail": (error.stderr or "")[-2000:],
        }
    result: Dict[str, Any] = {
        "map": map_path.parent.name,
        "returncode": completed.returncode,
        "wall_runtime_sec": perf_counter() - started,
        "stderr_tail": completed.stderr[-2000:],
    }
    if completed.returncode != 0:
        failure_report = None
        try:
            failure_report = _parse_cli_json(completed.stdout)
        except (ValueError, json.JSONDecodeError):
            pass
        result.update({"status": "failed", "stdout_tail": completed.stdout[-2000:]})
        if failure_report is not None:
            result["failure_report"] = failure_report
        return result
    try:
        report = _parse_cli_json(completed.stdout)
    except (ValueError, json.JSONDecodeError) as error:
        result.update(
            {
                "status": "invalid_report",
                "error": str(error),
                "stdout_tail": completed.stdout[-2000:],
            }
        )
        return result
    coverage = report.get("coverage_fraction")
    target = report.get("coverage_target")
    valid = (
        coverage is not None
        and target is not None
        and float(coverage) + 1.0e-9 >= float(target)
        and report.get("continuous_conflict_validated") is True
    )
    result.update(
        {
            "status": "passed" if valid else "invalid_solution",
            "report": report,
        }
    )
    return result


def _numeric_values(results: List[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for result in results:
        report = result.get("report") or result.get("failure_report") or {}
        value = report.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _aggregate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, float | None], List[Dict[str, Any]]] = {}
    for result in results:
        key = (result["map"], result.get("lns_time_budget_sec"))
        groups.setdefault(key, []).append(result)
    aggregates = []
    for (map_name, budget), group in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1] or -1.0)
    ):
        passed = [result for result in group if result["status"] == "passed"]
        wall_times = [float(result["wall_runtime_sec"]) for result in group]
        first_feasible = _numeric_values(passed, "end_to_end_first_feasible_sec")
        peak_rss = _numeric_values(group, "peak_rss_mb")
        makespans = _numeric_values(passed, "makespan")
        gaps = _numeric_values(passed, "optimality_gap")
        aggregates.append(
            {
                "map": map_name,
                "lns_time_budget_sec": budget,
                "runs": len(group),
                "passed": len(passed),
                "success_rate": len(passed) / len(group),
                "wall_runtime_sec_mean": mean(wall_times),
                "wall_runtime_sec_median": median(wall_times),
                "end_to_end_first_feasible_sec_mean": (
                    mean(first_feasible) if first_feasible else None
                ),
                "peak_rss_mb_max": max(peak_rss) if peak_rss else None,
                "makespan_mean": mean(makespans) if makespans else None,
                "optimality_gap_mean": mean(gaps) if gaps else None,
            }
        )
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible end-to-end CROWN regressions over bundled maps."
    )
    parser.add_argument(
        "--maps",
        nargs="*",
        default=None,
        help="Map directory names; omitted means every bundled map.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--lns-seconds", type=float, default=None)
    parser.add_argument(
        "--lns-budgets",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Run a budget sweep (seconds). Cannot be combined with "
            "--lns-seconds; omitted uses each map profile's own budget."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=ROOT / "outputs" / "crown_map_suite",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "crown_map_suite_report.json",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be positive")
    if args.lns_seconds is not None and args.lns_budgets is not None:
        parser.error("--lns-seconds and --lns-budgets are mutually exclusive")
    if args.lns_seconds is not None and args.lns_seconds <= 0.0:
        parser.error("--lns-seconds must be positive")
    if args.lns_budgets is not None and any(
        value <= 0.0 for value in args.lns_budgets
    ):
        parser.error("every --lns-budgets value must be positive")
    selected = set(args.maps or ())
    paths = [
        path for path in _map_paths() if not selected or path.parent.name in selected
    ]
    unknown = selected.difference(path.parent.name for path in paths)
    if unknown:
        parser.error("unknown map directories: " + ",".join(sorted(unknown)))
    args.outputs_root.mkdir(parents=True, exist_ok=True)
    results = []
    suite_started = perf_counter()
    budgets = (
        list(args.lns_budgets)
        if args.lns_budgets is not None
        else [args.lns_seconds]
    )
    for map_path in paths:
        for budget in budgets:
            budget_dir = (
                "profile_budget"
                if budget is None
                else "lns_" + str(float(budget)).replace(".", "p") + "s"
            )
            for repeat in range(args.repeats):
                output_dir = args.outputs_root / f"repeat_{repeat:02d}"
                if len(budgets) > 1:
                    output_dir = args.outputs_root / budget_dir / f"repeat_{repeat:02d}"
                result = _run_once(
                    map_path,
                    output_dir,
                    args.timeout_seconds,
                    budget,
                )
                result["repeat"] = repeat
                result["lns_time_budget_sec"] = budget
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {
        "suite": "CROWN bundled-map end-to-end regression",
        "runtime_sec": perf_counter() - suite_started,
        "map_count": len(paths),
        "repeat_count": args.repeats,
        "budget_count": len(budgets),
        "lns_budgets_sec": budgets,
        "passed": sum(result["status"] == "passed" for result in results),
        "failed": sum(result["status"] != "passed" for result in results),
        "aggregates": _aggregate_results(results),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
