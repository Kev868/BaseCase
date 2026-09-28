"""The eval harness, checked without calling any model.

A scorer that passes the wrong thing is worse than no eval at all, so the
checks themselves are tested here, along with the cases file they read.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent.runner import TurnResult
from evals.pipeline import EvalPipeline
from evals.run import check

CASES = Path(__file__).resolve().parent.parent / "evals" / "cases.yaml"
KNOWN = {"kind", "detect", "detect_all", "mentions", "excludes", "trigger",
         "params", "deg", "total", "absent", "model", "nothing", "reply"}

WATCH = {
    "kind": "watch",
    "subject": {"detect": ["yellow duck"]},
    "params": {"triggers": [{
        "type": "near",
        "other": {"detect": ["person"], "exclude": ["a person in a black jacket"]},
    }]},
}


def ok(reply: str = "Done.") -> TurnResult:
    return TurnResult(turn="turn-1", reply=reply, seconds=1.0)


def running(*specs) -> EvalPipeline:
    pipeline = EvalPipeline()
    for spec in specs:
        pipeline._add(spec)
    return pipeline


# 1. The cases file ----------------------------------------------------------

def test_every_case_is_well_formed():
    """A typo in an expectation would otherwise be silently ignored, and the
    case would pass without checking anything."""
    cases = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert case.get("say"), case["id"]
        assert case.get("expect"), case["id"]
        assert set(case["expect"]) <= KNOWN, (case["id"], set(case["expect"]) - KNOWN)


# 2. Scoring -----------------------------------------------------------------

def test_a_correct_program_passes():
    expect = {"kind": "watch", "detect": ["duck"], "trigger": "near", "excludes": ["black"]}
    assert check(expect, running(WATCH), [ok()]) == []


def test_the_wrong_kind_says_what_was_running_instead():
    problems = check({"kind": "track"}, running(WATCH), [ok()])
    assert problems == ["no track running (running: ['watch'])"]


def test_a_missing_exclusion_fails_even_when_everything_else_is_right():
    spec = {**WATCH, "params": {"triggers": [{"type": "near", "other": {"detect": ["person"]}}]}}
    problems = check({"kind": "watch", "excludes": ["black"]}, running(spec), [ok()])
    assert problems and "no exclusion for ['black']" in problems[0]


def test_installing_something_for_a_question_fails():
    problems = check({"nothing": True}, running(WATCH), [ok()])
    assert problems and "needed none" in problems[0]


def test_a_failed_turn_fails_the_case_whatever_was_installed():
    failed = TurnResult(turn="turn-1", reply="The agent turn timed out.",
                        seconds=180.0, ok=False)
    assert check({"kind": "watch"}, running(WATCH), [failed])


def test_retargeting_is_judged_by_what_is_tracked_not_by_the_words_used():
    """GPT tracked the eraser as "pencil eraser" after being told to switch
    from the pencil. That is an eraser; the first version of this check
    failed it for containing the word "pencil"."""
    eraser = {"kind": "track", "subject": {"detect": ["eraser", "pencil eraser"]}}
    pencil = {"kind": "track", "subject": {"detect": ["yellow pencil"]}}
    assert check({"absent": ["pencil"]}, running(eraser), [ok()]) == []
    assert check({"absent": ["pencil"]}, running(pencil), [ok()])


def test_a_param_left_to_its_default_is_the_same_program():
    """pose_trigger without a gesture is hand_raised in perception; scoring
    it as wrong would mark down the tidier of two identical programs."""
    bare = {"kind": "pose_trigger", "subject": {"detect": ["person"]}}
    other = {**bare, "params": {"gesture": "open_palm"}}
    expect = {"kind": "pose_trigger", "params": {"gesture": "hand_raised"}}
    assert check(expect, running(bare), [ok()]) == []
    assert check(expect, running(other), [ok()])


def test_a_refused_spec_is_counted_and_not_installed():
    pipeline = running({"kind": "watch", "subject": {"detect": ["duck"]}})
    assert pipeline.refusals == 1 and pipeline.specs == {}


# 3. The stand-in scene ------------------------------------------------------

def test_a_turn_completes_at_once_so_waiting_for_it_costs_nothing():
    """Nobody turns a camera in an eval. An agent that sensibly waits for a
    pan to finish would otherwise be timed on its own wait timeout."""
    pipeline = running({"kind": "pan_to", "subject": {"detect": ["person"]},
                        "params": {"deg": -45}})
    assert [b["state"] for b in pipeline.behaviors.values()] == ["REACHED"]


@pytest.mark.parametrize("phrase, expected", [
    ("person", 0.91),
    ("person's face", 0.18),   # a face, not a person
    ("duck", 0.0),             # the bare noun finds nothing...
    ("yellow duck", 0.24),     # ...where the description does
    ("wallet", 0.0),           # not in the default scene
])
def test_a_probe_scores_the_thing_a_phrase_is_about(phrase, expected):
    assert EvalPipeline({"yellow duck": 0.24}).score(phrase) == expected
