"""Router eval: how well the supervisor splits questions into categories.

Run from the project root:
    python -m evals.eval_router            run every question once
    python -m evals.eval_router --runs 3   run each question 3 times, also reports stability

Only the router is called (split_question), not the whole graph, so the MCP Server and
the paper search APIs are not needed. Categories are checked in code; sub-questions are
printed for the wrong cases but not scored.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from agents.supervisor import CATEGORY_TO_NODE, ROUTE_FAIL_REASON, split_question


CASES_PATH = Path(__file__).parent / "router_cases.jsonl"
CATEGORIES = list(CATEGORY_TO_NODE)


def load_cases() -> list[dict]:
    lines = CASES_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def run_case(case: dict, runs: int) -> dict:
    """Route one question `runs` times and record every result."""
    results = []
    for _ in range(runs):
        tasks, reason, _ = await split_question(case["question"])
        results.append({
            "categories": sorted(t["category"] for t in tasks),
            "tasks": tasks,
            "failed": reason == ROUTE_FAIL_REASON,
        })
    return {**case, "expected": sorted(case["expected"]), "results": results}


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "-"


def report(outcomes: list[dict], runs: int) -> None:
    all_runs = [(o, r) for o in outcomes for r in o["results"]]
    correct = [(o, r) for o, r in all_runs if r["categories"] == o["expected"]]

    print(f"\n{'=' * 20} Router eval: {len(outcomes)} questions x {runs} run(s) {'=' * 20}")
    print(f"Exact match (category set): {pct(len(correct), len(all_runs))}")
    for label, want_multi in (("  single-task questions", False), ("  multi-task questions ", True)):
        group = [(o, r) for o, r in all_runs if (len(o["expected"]) > 1) == want_multi]
        print(f"{label}: {pct(sum(r['categories'] == o['expected'] for o, r in group), len(group))}")
    failed = sum(r["failed"] for _, r in all_runs)
    if failed:
        print(f"Routing failed (fell back to other): {failed}")
    if runs > 1:
        stable = sum(len({tuple(r["categories"]) for r in o["results"]}) == 1 for o in outcomes)
        print(f"Stable (same categories in every run): {pct(stable, len(outcomes))}")

    # Each category is scored on its own: was it chosen when needed (recall), and needed when chosen (precision)
    print(f"\n{'category':<10}{'precision':>12}{'recall':>12}{'support':>10}")
    for category in CATEGORIES:
        tp = sum(category in r["categories"] and category in o["expected"] for o, r in all_runs)
        fp = sum(category in r["categories"] and category not in o["expected"] for o, r in all_runs)
        fn = sum(category not in r["categories"] and category in o["expected"] for o, r in all_runs)
        precision = f"{tp / (tp + fp):.0%}" if tp + fp else "-"
        recall = f"{tp / (tp + fn):.0%}" if tp + fn else "-"
        print(f"{category:<10}{precision:>12}{recall:>12}{tp + fn:>10}")

    wrong = [o for o in outcomes if any(r["categories"] != o["expected"] for r in o["results"])]
    if not wrong:
        return
    print(f"\n{'=' * 20} Wrong cases ({len(wrong)}) {'=' * 20}")
    for o in wrong:
        print(f"\nQ: {o['question']}")
        print(f"   expected: {o['expected']}")
        if o.get("note"):
            print(f"   note:     {o['note']}")
        for i, r in enumerate(o["results"], 1):
            mark = "ok" if r["categories"] == o["expected"] else "WRONG"
            print(f"   run {i} [{mark}]: " + "; ".join(f"[{t['category']}] {t['sub_question']}" for t in r["tasks"]))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1, help="times to route each question (default 1)")
    args = parser.parse_args()

    cases = load_cases()
    started = time.monotonic()
    outcomes = []
    # One question at a time: a local Ollama model handles requests one by one anyway
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['question']}")
        outcomes.append(await run_case(case, args.runs))
    report(outcomes, args.runs)
    print(f"\nTook {time.monotonic() - started:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
