"""The stand-in pipeline the evals run against.

`tests/fake_pipeline.py`, plus what a scored run needs on top:

* a scene, so probing a wording returns a believable score instead of the
  test fake's all-zeroes (which would leave every model unable to find
  anything, and the eval measuring nothing);
* the full spec of everything installed, so the result can be checked
  against what the case expected, not just its kind;
* counts of installs and refusals, to see whether a model fixes its own
  rejected calls.

`/snapshot` answers 404, so no camera frame goes to the model: the turn is
decided on the words and the tool results alone, the same for every model.
"""

from __future__ import annotations

from typing import Any

import httpx

from tests.fake_pipeline import FakePipeline

#: What is "in front of the camera" in every case, as probe confidences. Low
#: scores for faces and plain "duck" are measured behaviour of the real
#: detector (see documents/SELECTORS.md), kept so wording choices matter.
DEFAULT_SCENE: dict[str, float] = {
    "person": 0.91, "people": 0.88, "laptop": 0.82, "chair": 0.70,
    "mug": 0.66, "cup": 0.60, "phone": 0.55, "notebook": 0.52,
    "desk": 0.50, "table": 0.50, "person's face": 0.18, "face": 0.12,
    "duck": 0.0,
}

#: What `/describe` reports: things on the table.
SCENE_OBJECTS: list[dict[str, Any]] = [
    {"label": "laptop", "where": "centre", "size": "large", "conf": 0.82},
    {"label": "mug", "where": "right", "size": "small", "conf": 0.66},
    {"label": "notebook", "where": "lower left", "size": "medium", "conf": 0.52},
]


class EvalPipeline(FakePipeline):
    def __init__(self, scene: dict[str, float] | None = None,
                 people: int = 1) -> None:
        super().__init__()
        self.visible = {**DEFAULT_SCENE, **(scene or {})}
        #: behaviour id -> the spec exactly as the agent sent it.
        self.specs: dict[str, dict[str, Any]] = {}
        self.installs = 0
        self.refusals = 0

        self.count_answer = people
        self.tracks = [
            {"track_id": i + 1, "label": "person", "conf": 0.9,
             "cx": -0.5 + i * 0.5, "cy": 0.0, "area": 0.12, "attributes": {}}
            for i in range(people)
        ]
        self.scene_objects = [
            *SCENE_OBJECTS,
            {"label": "person", "where": "centre", "size": "large",
             "conf": 0.91, "count": people},
        ]

    def _route(self, method: str, path: str, body: Any,
               request: httpx.Request) -> httpx.Response:
        if path == "/snapshot":
            return httpx.Response(404, json={"detail": "no camera in evals"})
        if path == "/behaviors" and method == "DELETE":
            self.specs.clear()
        return super()._route(method, path, body, request)

    def _add(self, body: Any) -> httpx.Response:
        response = super()._add(body)
        if response.status_code == 201:
            behavior_id = response.json()["id"]
            self.installs += 1
            self.specs[behavior_id] = body
            if body.get("kind") == "pan_to":
                # The operator follows the arrow at once. Nobody turns a
                # camera here, so an agent that waits for the turn would
                # otherwise wait out its whole timeout, and be timed on that.
                self.behaviors[behavior_id]["state"] = "REACHED"
        else:
            self.refusals += 1
        return response

    def _drop(self, behavior_id: str) -> httpx.Response:
        response = super()._drop(behavior_id)
        if response.status_code == 202:
            self.specs.pop(behavior_id, None)
        return response

    def _probe(self, body: dict[str, Any]) -> httpx.Response:
        for phrase in body.get("phrases") or []:
            self.probe_scores[phrase] = self.score(phrase)
        return super()._probe(body)

    def score(self, phrase: str) -> float:
        """What the detector would give this wording.

        A phrase is scored by the thing it is about, which in English is the
        last noun: "person's face" is a face, not a person, and "yellow duck"
        finds the duck where plain "duck" does not. So of the visible things
        the phrase names, the one ending latest wins, the longest on a tie.
        """
        lowered = phrase.lower()
        named = [(lowered.rfind(thing) + len(thing), len(thing), score)
                 for thing, score in self.visible.items() if thing in lowered]
        return max(named)[2] if named else 0.0
