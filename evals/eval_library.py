"""Library Agent eval: does each question go to the right source?

Run from the project root:
    python -m evals.eval_library            run every question once
    python -m evals.eval_library --runs 3   run each question 3 times

The source is checked from the subgraph's state, not from the wording of the answer:
which local papers the resolver found, and whether the documents came from the online search.
A case passes when exactly the expected library papers were found and the online search was used
exactly when the paper is not in the library. Online cases need network access.
Questions that describe a paper without naming it are expected to fail in this version:
the resolver only matches titles and aliases.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, cast

from agents.library import library_graph

CASES_PATH = Path(__file__).parent / "library_cases.jsonl"


def load_cases() -> list[dict]:
    lines = CASES_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def run_once(question: str) -> dict:
    started = time.monotonic()
    result = await library_graph.ainvoke(cast(Any, {"query": question}))
    return {
        "papers": {paper["title"] for paper in result["local_paper"]["papers"]},
        "online": result["source"] == "external",
        "tokens": result["tokens"],
        "seconds": time.monotonic() - started,
    }


def passed(case: dict, run: dict) -> bool:
    return run["papers"] == set(case["papers"]) and run["online"] == case["online"]


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="times to ask each question (default 1)"
    )
    args = parser.parse_args()

    cases = load_cases()
    outcomes = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['question']}")
        runs = []
        for _ in range(args.runs):
            try:
                runs.append(await run_once(case["question"]))
            except Exception as e:
                print(f"   Library lookup failed: {e}")
                runs.append(
                    {"papers": set(), "online": False, "tokens": 0, "seconds": 0.0}
                )
        outcomes.append((case, runs))

    all_runs = [(case, run) for case, runs in outcomes for run in runs]
    print(
        f"\n{'=' * 20} Library eval: {len(cases)} questions x {args.runs} run(s) {'=' * 20}"
    )
    print(f"Right source: {sum(passed(c, r) for c, r in all_runs)}/{len(all_runs)}")
    for label, want_online in (("  in library    ", False), ("  not in library", True)):
        group = [(c, r) for c, r in all_runs if c["online"] == want_online]
        print(f"{label}: {sum(passed(c, r) for c, r in group)}/{len(group)}")
    print(
        f"Average per question: {sum(r['seconds'] for _, r in all_runs) / len(all_runs):.1f}s, "
        f"{sum(r['tokens'] for _, r in all_runs) / len(all_runs):,.0f} tokens"
    )

    wrong = [
        (case, runs)
        for case, runs in outcomes
        if not all(passed(case, run) for run in runs)
    ]
    if wrong:
        print(f"\n{'=' * 20} Wrong cases ({len(wrong)}) {'=' * 20}")
    for case, runs in wrong:
        print(f"\nQ: {case['question']}")
        print(f"   expected: papers={case['papers']} online={case['online']}")
        if case.get("note"):
            print(f"   note:     {case['note']}")
        for i, run in enumerate(runs, 1):
            mark = "ok" if passed(case, run) else "WRONG"
            print(
                f"   run {i} [{mark}]: papers={sorted(run['papers'])} online={run['online']}"
            )


if __name__ == "__main__":
    asyncio.run(main())
