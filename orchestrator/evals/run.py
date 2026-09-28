"""Score the agent on evals/cases.yaml, once per model.

    cd orchestrator
    .venv/Scripts/python -m evals.run --model gpt-6-astra --model grok-4.7

Each case runs the real agent (runner, graph, prompts, tools) against the
stand-in pipeline in evals/pipeline.py. Only the model changes between runs,
so the numbers compare models on this project's actual job: turning a
sentence into the right program, and how long that takes.

Writes evals/results/<model>.json per model, then rebuilds evals/RESULTS.md
from every results file there, so models can be run separately.
"""

from __future__ import annotations

import os

# Before `config` is imported: it reads the environment once, and these win
# over .env. The limits are pinned so a result does not depend on whoever's
# .env ran it: the turn deadline and step cap are the documented ones
# (.env.example), and a single model call gets a minute, so a slow answer is
# measured rather than cut off. Speed is reported on its own.
os.environ.setdefault("LOG_LEVEL", "WARNING")
LIMITS = {"ORCH_TURN_TIMEOUT_S": "180", "ORCH_MAX_STEPS": "10",
          "ORCH_LLM_TIMEOUT_S": "60"}
os.environ.update(LIMITS)

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import yaml  # noqa: E402

import agent.graph  # noqa: E402
from agent.llm import build_model  # noqa: E402
from agent.runner import AgentRunner, Attachment  # noqa: E402
from config import CFG, key_for, provider_of  # noqa: E402
from evals.pipeline import EvalPipeline  # noqa: E402
from memory.store import PhraseMemory  # noqa: E402
from tools.perception import Perception  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PHOTO = HERE / "fixtures" / "operator.jpg"


# 1. Scoring -----------------------------------------------------------------

def _subject_text(spec: dict[str, Any]) -> str:
    return json.dumps(spec.get("subject", {})).lower()


def _detects(spec: dict[str, Any]) -> list[str]:
    return [d.lower() for d in spec.get("subject", {}).get("detect", [])]


def _excluded(node: Any) -> list[str]:
    """Every phrase under any `exclude`, including a trigger's `other`."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "exclude" and isinstance(value, list):
                found += [str(v).lower() for v in value]
            else:
                found += _excluded(value)
    elif isinstance(node, list):
        for item in node:
            found += _excluded(item)
    return found


#: Params perception fills in when they are left out, so leaving one out is
#: the same program. Mirrors the defaults in perception/behaviors/.
DEFAULTS = {("pose_trigger", "gesture"): "hand_raised",
            ("privacy", "mode"): "blur"}


def check(expect: dict[str, Any], pipeline: EvalPipeline,
          turns: list[Any]) -> list[str]:
    """What is wrong with the outcome. Empty means the case passed.

    kind        a behaviour of this kind is running; the checks below look
                only at behaviours of this kind when it is given
    detect      one of them detects a phrase containing any of these words
    detect_all  every one of these words is in some detect phrase
    mentions    every one of these words is somewhere in a subject selector
    excludes    every one of these words is in some `exclude` list
    trigger     a watch trigger of this type is set
    params      one of them has exactly these params (a param left out
                counts as its default, since perception fills it in)
    deg         a pan_to within 5 degrees of this (negative is left)
    total       exactly this many behaviours are running
    absent      no running behaviour detects something that is one of these
                (a detect phrase whose last word starts with it)
    model       the detector was switched to this model
    nothing     nothing was ever installed
    reply       the final reply contains any of these words
    """
    problems: list[str] = []
    specs = list(pipeline.specs.values())
    kind = expect.get("kind")
    mine = [s for s in specs if s.get("kind") == kind] if kind else specs
    running = sorted(s.get("kind", "?") for s in specs) or "nothing"

    for i, turn in enumerate(turns, 1):
        if not turn.ok:
            problems.append(f"turn {i} failed: {turn.reply[:120]}")

    if kind and not mine:
        problems.append(f"no {kind} running (running: {running})")
    if "detect" in expect and mine and not any(
            w in d for s in mine for d in _detects(s) for w in expect["detect"]):
        problems.append(f"detects {[_detects(s) for s in mine]}, "
                        f"wanted one of {expect['detect']}")
    if "detect_all" in expect and mine:
        phrases = [d for s in mine for d in _detects(s)]
        missing = [w for w in expect["detect_all"]
                   if not any(w in d for d in phrases)]
        if missing:
            problems.append(f"nothing detects {missing} (detects {phrases})")
    if "mentions" in expect and mine:
        missing = [w for w in expect["mentions"]
                   if not any(w in _subject_text(s) for s in mine)]
        if missing:
            problems.append(f"subject never mentions {missing}")
    if "excludes" in expect and mine:
        excluded = [e for s in mine for e in _excluded(s)]
        missing = [w for w in expect["excludes"]
                   if not any(w in e for e in excluded)]
        if missing:
            problems.append(f"no exclusion for {missing} (excludes {excluded})")
    if "trigger" in expect and mine:
        types = [t.get("type") for s in mine
                 for t in s.get("params", {}).get("triggers", [])]
        if expect["trigger"] not in types:
            problems.append(f"triggers {types}, wanted {expect['trigger']}")
    for key, want in expect.get("params", {}).items():
        got = [s.get("params", {}).get(key, DEFAULTS.get((s.get("kind"), key)))
               for s in mine]
        if want not in got:
            problems.append(f"params.{key} is {got}, wanted {want!r}")
    if "deg" in expect and mine:
        degs = [s.get("params", {}).get("deg") for s in mine]
        if not any(isinstance(d, (int, float)) and abs(d - expect["deg"]) <= 5
                   for d in degs):
            problems.append(f"pan_to deg {degs}, wanted {expect['deg']}")
    if "total" in expect and len(specs) != expect["total"]:
        problems.append(f"{len(specs)} running ({running}), "
                        f"wanted {expect['total']}")
    for word in expect.get("absent", []):
        # About, not mentioning: "pencil eraser" is an eraser. A phrase is
        # about its last noun, the same rule the stand-in probe scores by.
        about = [d for s in specs for d in _detects(s)
                 if d.split() and d.split()[-1].startswith(word)]
        if about:
            problems.append(f"still running something about {word!r}: {about}")
    if "model" in expect:
        call = pipeline.last("POST", "/model")
        got = ((call or {}).get("json") or {}).get("name")
        if (got or "").lower() != expect["model"]:
            problems.append(f"model switched to {got!r}, wanted {expect['model']!r}")
    if expect.get("nothing") and pipeline.installs:
        problems.append(f"installed {pipeline.installs} behaviour(s) "
                        f"for something that needed none ({running})")
    if "reply" in expect and turns:
        reply = turns[-1].reply.lower()
        if not any(w.lower() in reply for w in expect["reply"]):
            problems.append(f"reply {turns[-1].reply[:120]!r} mentions none "
                            f"of {expect['reply']}")
    return problems


# 2. Running -----------------------------------------------------------------

def _network_fault(problem: str) -> bool:
    lowered = problem.lower()
    return "model call failed" in lowered and any(
        sign in lowered for sign in ("ssl", "connection error", "connection reset"))


def use_model(name: str) -> None:
    """Point the agent at one model, for both text and image turns."""
    provider = provider_of(name)
    model = build_model(provider, name, key_for(provider))
    agent.graph.chat_model = lambda *_a, **_k: model
    agent.graph.vision_model = lambda *_a, **_k: model


async def run_case(case: dict[str, Any], scratch: Path) -> dict[str, Any]:
    pipeline = EvalPipeline(case.get("scene"), case.get("count", 1))
    perception = Perception(transport=pipeline.transport())
    memory = PhraseMemory(uri=str(scratch / case["id"]), embed_model="")
    runner = AgentRunner(perception, memory)
    says = case["say"] if isinstance(case["say"], list) else [case["say"]]
    turns = []
    started = time.perf_counter()
    try:
        for i, text in enumerate(says):
            photos = ([Attachment("operator.jpg", "image/jpeg", PHOTO.read_bytes())]
                      if case.get("photo") and i == 0 else None)
            turns.append(await runner.user_turn(text, photos))
    finally:
        await perception.aclose()
    problems = check(case.get("expect", {}), pipeline, turns)
    return {
        "id": case["id"],
        "passed": not problems,
        "problems": problems,
        "seconds": [round(t.seconds, 2) for t in turns],
        "wall_seconds": round(time.perf_counter() - started, 2),
        "replies": [t.reply for t in turns],
        "installed": list(pipeline.specs.values()),
        "probes": pipeline.paths("POST").count("/probe"),
        "refusals": pipeline.refusals,
    }


async def run_model(name: str, cases: list[dict[str, Any]], jobs: int,
                    repeat: int) -> list[dict[str, Any]]:
    use_model(name)
    gate = asyncio.Semaphore(jobs)
    scratch = Path(tempfile.mkdtemp(prefix="retask-evals-"))

    async def one(case: dict[str, Any], attempt: int) -> dict[str, Any]:
        async with gate:
            for tries in range(1, 4):
                try:
                    result = await run_case(case, scratch / f"{attempt}-{tries}")
                except Exception as exc:                # noqa: BLE001
                    result = {"id": case["id"], "passed": False,
                              "problems": [f"crashed: {type(exc).__name__}: {exc}"],
                              "seconds": [], "wall_seconds": 0.0, "replies": [],
                              "installed": [], "probes": 0, "refusals": 0}
                # A dropped connection says nothing about the model, so it
                # gets another go. A slow or wrong answer does not.
                if not any(_network_fault(p) for p in result["problems"]):
                    break
            result["attempt"], result["tries"] = attempt, tries
            mark = "pass" if result["passed"] else "FAIL"
            print(f"  {name:>14} {mark} {case['id']:<24} "
                  f"{sum(result['seconds']):6.1f}s  "
                  f"{'; '.join(result['problems'])[:110]}", flush=True)
            return result

    return list(await asyncio.gather(*(
        one(case, attempt) for attempt in range(repeat) for case in cases)))


# 3. Reporting ---------------------------------------------------------------

def timed_out(result: dict[str, Any]) -> bool:
    return any("timed out" in p for p in result["problems"])


def summarize(name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [s for r in results for s in r["seconds"]]
    return {
        "model": name,
        "passed": sum(r["passed"] for r in results),
        "runs": len(results),
        "wrong": sum(1 for r in results if not r["passed"] and not timed_out(r)),
        "timeouts": sum(1 for r in results if timed_out(r)),
        "slowest_turn_s": max(turns) if turns else None,
        "median_turn_s": round(statistics.median(turns), 1) if turns else None,
        "p90_turn_s": (round(statistics.quantiles(turns, n=10)[-1], 1)
                       if len(turns) >= 10 else None),
        "refusals": sum(r["refusals"] for r in results),
        "self_corrected": sum(1 for r in results if r["refusals"] and r["passed"]),
        "network_retries": sum(r.get("tries", 1) - 1 for r in results),
    }


def write_report() -> Path:
    """RESULTS.md from every model's results file."""
    runs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(RESULTS.glob("*.json"))]
    lines = [
        "# Eval results",
        "",
        "Generated by `python -m evals.run`. Each case runs the real agent "
        "against the stand-in pipeline in `evals/pipeline.py`; see "
        "`evals/README.md` for what is measured and what is not.",
        "",
        "| Model | Passed | Wrong | Timed out | Median turn | p90 turn | Slowest | Refused calls | Fixed after a refusal |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        s = run["summary"]
        pct = 100 * s["passed"] / s["runs"] if s["runs"] else 0
        lines.append(
            f"| `{s['model']}` | {s['passed']}/{s['runs']} ({pct:.0f}%) "
            f"| {s['wrong']} | {s['timeouts']} "
            f"| {s['median_turn_s']} s | {s['p90_turn_s'] or '–'} s "
            f"| {s['slowest_turn_s']} s "
            f"| {s['refusals']} | {s['self_corrected']} |")
    limits = runs[0]["limits"] if runs else LIMITS
    per_case = sorted({len(run["results"]) // max(len({r["id"] for r in run["results"]}), 1)
                       for run in runs})
    times = " or ".join(str(n) for n in per_case) or "0"
    lines += ["", f"Run on {', '.join(sorted({r['date'] for r in runs}))}, "
                  f"{times} runs per case, with a {limits['ORCH_TURN_TIMEOUT_S']} s "
                  f"turn deadline, {limits['ORCH_MAX_STEPS']} model steps and "
                  f"{limits['ORCH_LLM_TIMEOUT_S']} s per model call."]

    ids = list(dict.fromkeys(r["id"] for run in runs for r in run["results"]))
    lines += ["", "## Per case", "",
              "| Case | " + " | ".join(f"`{run['summary']['model']}`" for run in runs) + " |",
              "|---|" + "---|" * len(runs)]
    for case_id in ids:
        cells = []
        for run in runs:
            mine = [r for r in run["results"] if r["id"] == case_id]
            if not mine:
                cells.append("–")
                continue
            passed = sum(r["passed"] for r in mine)
            secs = sum(sum(r["seconds"]) for r in mine) / len(mine)
            mark = "pass" if passed == len(mine) else (
                f"{passed}/{len(mine)}" if passed else
                "**timeout**" if all(timed_out(r) for r in mine) else "**fail**")
            cells.append(f"{mark} · {secs:.1f} s")
        lines.append(f"| {case_id} | " + " | ".join(cells) + " |")

    failures = [(run["summary"]["model"], r) for run in runs
                for r in run["results"] if not r["passed"]]
    if failures:
        lines += ["", "## Failures", ""]
        for model, r in failures:
            lines.append(f"- `{model}` · **{r['id']}**: {'; '.join(r['problems'])}")

    out = HERE / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", action="append",
                        help="model to score; repeat for several (default: ORCH_MODEL)")
    parser.add_argument("--case", action="append", help="only these case ids")
    parser.add_argument("--jobs", type=int, default=3, help="cases run at once")
    parser.add_argument("--repeat", type=int, default=1,
                        help="runs per case; models are not deterministic")
    args = parser.parse_args()

    cases = yaml.safe_load((HERE / "cases.yaml").read_text(encoding="utf-8"))
    if args.case:
        cases = [c for c in cases if c["id"] in set(args.case)]
    RESULTS.mkdir(exist_ok=True)

    asyncio.run(run_all(args.model or [CFG.model], cases, args.jobs,
                        args.repeat, merge=bool(args.case)))
    print(f"report: {write_report()}")


async def run_all(models: list[str], cases: list[dict[str, Any]], jobs: int,
                  repeat: int, merge: bool = False) -> None:
    """Every model in one event loop. langchain-openai caches its HTTP
    client per address, so a second `asyncio.run` for another Grok model
    reuses a client bound to a closed loop and fails with "Event loop is
    closed" — a harness fault that would be scored against the model."""
    for name in models:
        print(f"{name}: {len(cases)} cases x {repeat}", flush=True)
        results = await run_model(name, cases, jobs, repeat)
        path = RESULTS / f"{name}.json"
        if merge and path.exists():
            # Re-running some cases replaces just those in the last full run.
            rerun = {c["id"] for c in cases}
            kept = [r for r in json.loads(path.read_text(encoding="utf-8"))["results"]
                    if r["id"] not in rerun]
            results = kept + results
        summary = summarize(name, results)
        path.write_text(json.dumps({
            "date": time.strftime("%Y-%m-%d"), "summary": summary,
            "limits": LIMITS, "results": results,
        }, indent=2), encoding="utf-8")
        print(f"{name}: {summary['passed']}/{summary['runs']} passed, "
              f"median turn {summary['median_turn_s']} s", flush=True)


if __name__ == "__main__":
    main()
