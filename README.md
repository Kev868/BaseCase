# Retask

**Type a sentence, and one webcam becomes a different device.** "Highlight
everyone" makes it a people counter. "Blur everyone's face except mine" makes it
a privacy camera. "Watch my phone and tell me if it disappears" makes it a
sentinel. No retraining, no new code: an agent turns the sentence into a typed
program, and a real-time vision pipeline runs that program on every frame.

Built at Hack the North 2026 by Qinkai and Kevin. The version we presented is
tagged [`hackathon-demo`](https://github.com/Kev868/BaseCase/tree/hackathon-demo);
[AFTER-THE-HACKATHON.md](AFTER-THE-HACKATHON.md) covers what changed since.

<!-- Add a demo video or GIF here before sharing: it is the fastest way to show what this does. -->

## What it can become

| Say | It becomes |
|---|---|
| "Highlight everyone" | People counter |
| "Track the pencil" … "now the eraser" | Follow-cam that retargets instantly |
| "Watch the yellow duck, ignore anyone in a black jacket" | Sentinel with an exclusion rule |
| "Blur everyone's face except mine" + a photo | Privacy camera |
| "Count people crossing this line" | Traffic counter |
| "Tell me when someone raises their hand" | Gesture trigger |
| "What's on the table right now?" | Scene Q&A |
| "Switch your model to the COCO detector" | Model swap, live |

Every one of these is a combination of four primitives (selectors, behaviours,
triggers and events) rather than a feature of its own.

## How it works

The idea is **two clocks**. The agent thinks in seconds; the camera acts every
frame. Most "LLM + camera" projects put the model inside the frame loop, which
runs at a few frames per second and stops when the network does. Here the agent
runs once per instruction, compiles it into a behaviour spec, and gets out of
the way. The pipeline then detects, tracks, checks attributes, evaluates
triggers and renders on every frame with no model call at all, and wakes the
agent only when something it asked about happens.

```
 operator ──"watch my phone"──▶ ORCHESTRATOR (:8000)          agent, seconds
                                 ground → think ⇄ tools → reply
                                        │  validated behaviour specs (HTTP)
                                        ▼                ▲ events (WebSocket)
 webcam ──▶ PERCEPTION (:8001)                              pipeline, every frame
            detect → track → attributes → behaviours → render ──▶ MJPEG feed
                                        │
 CONSOLE (:3000) ◀── feed, state, trace, events ─────────────┘
```

Design points worth reading the code for:

- **A compiler, not a chatbot.** English is the source language, a behaviour
  spec is the intermediate representation ([linker/schemas.py](linker/schemas.py),
  shared by both services), and the pipeline is the runtime. Invalid specs are
  rejected with a specific reason ("label(s) ['duck'] are not in the coco
  vocabulary") that the agent reads and fixes on its own.
- **One pass serves every behaviour.** Detection runs once per frame on the
  union of all behaviours' prompts, and CLIP appearance scores are cached per
  track and refreshed about once a second, so two behaviours watching people
  cost one forward pass.
- **Single writer.** Only the frame loop changes pipeline state. API calls
  enqueue operations, and a new configuration is swapped in between two frames,
  keeping the tracker so that starting a privacy blur never makes the follow-cam
  lose its target ([perception/runtime/loop.py](perception/runtime/loop.py)).
- **The agent measures before it commits.** Before thinking, it is told what is
  running, which detector is live and which phrasings have worked before. It can
  probe candidate wordings against the live frame, because open-vocabulary
  detection is phrase-sensitive: "yellow duck" found the duck in 12 of 12
  frames where "duck" found it in 0 ([orchestrator/agent/graph.py](orchestrator/agent/graph.py)).
- **Degrade, never crash.** A model that cannot run on this machine reports
  itself unavailable and the service still boots. A behaviour whose model is
  missing pauses with a reason and resumes when the model returns.

## Measured, not guessed

- **Detector size.** `yoloe-11s-seg` runs at 38 ms/frame against 62 ms for the
  large model, and the large one was better calibrated but not more sensitive
  on the hard objects. Small won.
- **Tracker gates.** Open-vocabulary detections score 0.15–0.30 where COCO
  detections score 0.8+, and ByteTrack's gates, built for COCO, silently dropped
  them: detected every frame, tracked for none. First fixed by lowering the
  tracker's activation gate, then properly by a monotonic confidence remap into
  the range the tracker expects, restored on the way out.
- **Webcam capture.** Setting the pixel format before the resolution silently
  gave 10 fps on a Logitech C920; resolution first gave 30.5 fps. DirectShow
  opens the camera in 2.35 s where OpenCV's Windows default (MSMF) took 32.9 s.

The full story, including the bug that looked like ten bugs, is in
[PROJECT.md](PROJECT.md).

## Run it

Needs Python 3.14 (what it was built and tested on), Node 20.19+, a webcam and
an OpenAI API key. An NVIDIA GPU is recommended; CPU works but is slow. The
first perception start downloads about 1.2 GB of weights.

```bash
# 1. Perception, :8001. Reads webcam 0; set VIDEO_SOURCE to a file or stream URL instead.
cd perception
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
.venv/bin/python main.py

# 2. Orchestrator, :8000
cd orchestrator
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env                          # then set OPENAI_API_KEY
.venv/bin/python main.py

# 3. Console, :3000
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>. Perception also serves a bare debug page at
<http://localhost:8001/>.

### Tests

```bash
cd perception   && .venv/bin/python -m pytest   # 456 tests
cd orchestrator && .venv/bin/python -m pytest   # 163 tests
cd frontend     && npm run typecheck
```

No test opens a camera, loads real weights or reaches the network. The
perception suite drives the real frame loop, worker and behaviour code with a
fake capture, a fake detector and a hash-based stand-in for CLIP; the agent is
tested against a scripted model.

## Layout

```
perception/     the real-time pipeline (FastAPI, OpenCV, Ultralytics, supervision)
orchestrator/   the agent (LangGraph), its tools, phrase memory and event handling
frontend/       the operator console (React, TypeScript, Vite)
linker/         the shared contract both services import
PROJECT.md      design, decisions, what broke, and what is still wrong
docs/notes/     working notes from the build
```

## Team

- **Qinkai** ([@batteryspecial](https://github.com/batteryspecial)): the
  perception pipeline: detection, tracking, attribute scoring and calibration,
  the behaviour engine, the model zoo and rendering.
- **Kevin** ([@Kev868](https://github.com/Kev868)): the orchestrator (agent
  loop, tools, phrase memory, event handling), the operator console, camera
  selection and capture tuning, and the diagnosis and first fix of the tracker
  dropping open-vocabulary detections.

## Limitations

- **A switch takes seconds, not frames.** One measured end-to-end turn ("watch
  the walnut"), from sentence to installed behaviour, took 8.3 s. Installed
  behaviours then run at frame rate with no model in the loop.
- **Wording matters.** Open-vocabulary detection is only as good as the phrase.
  The agent probes and remembers what works, but an object it has no words for
  stays hard to find.
- **Built for a demo on a trusted network.** Both services accept requests from
  any origin and have no authentication. Do not expose them beyond a LAN you
  control.
- **Private on screen, not in transit.** The agent sends camera snapshots to its
  model provider, so a blurred feed does not mean no raw image leaves the
  machine.
- Several capabilities are unit-tested but not yet proven on recorded footage;
  [PROJECT.md](PROJECT.md) says which.
