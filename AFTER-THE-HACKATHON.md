# After the hackathon

Short version: Qinkai and I built this at Hack the North 2026. The version we
presented is tagged
[`hackathon-demo`](https://github.com/Kev868/BaseCase/tree/hackathon-demo).
Everything after that tag is me going back through the project afterwards, and
[this diff](https://github.com/Kev868/BaseCase/compare/hackathon-demo...main)
shows every line that changed.

## What we built together

Retask turns a sentence into a running camera program. Say "watch my phone and
tell me if it disappears" and the webcam becomes a sentinel. Say "blur
everyone's face except mine" and it becomes a privacy camera. No retraining,
and no new code per feature.

We split it roughly down the middle:

- **Qinkai** built perception, the part that runs on every frame: detection
  (YOLOE), tracking (ByteTrack), the CLIP checks that tell "a person" apart
  from "the person in the red hoodie", the behaviours (track, watch, count,
  blur, gestures, the keyboard guide), the model zoo, all the drawing on the
  video, and the shared contract both services use.
- **I** built the agent side and the console: the LangGraph agent that turns a
  sentence into a behaviour spec (and fixes its own spec when perception
  rejects it), its tools, the phrase memory, the event watcher that wakes it up
  when something happens, and the React console you drive it all from. On the
  perception side I did camera selection and capture tuning, and I tracked
  down why the tracker was quietly dropping open-vocabulary detections, which
  Qinkai then fixed properly.

The commit history keeps who wrote what, commit by commit.
[PROJECT.md](PROJECT.md) is the long write-up: design decisions, what broke,
and what's still wrong.

## What I changed after

Once the pressure was off I went back and reviewed the whole thing properly. I
did this pass with help from an AI coding assistant. Nothing about what Retask
does or how it's built changed; this is bug fixes, speed, and making it easier
for someone new to run and read.

### Two real bugs

**The privacy camera could end up blurring your own face.** "Blur everyone
except me" works by matching the people on screen against your photo. The
matcher also compared against people who had already *left* the frame, because
their data stays cached for a while. So if you walked behind something long
enough to get a new tracking id, the old copy of you was still in the cache,
the two looked identical, and the matcher refused to pick either one. You got
blurred, and the follow-cam lost you the same way. Now it only compares against
who's actually on screen.

**Some questions got treated as commands.** The agent has a safety net: if you
give it an ongoing instruction ("watch my phone") and it finishes without
setting anything up, it gets pushed to set something up. That net was
triggered by keywords, and "show" and "tell me if" were on the list, so "show
me what's on the table" had its correct answer thrown away and replaced with an
error. Questions are now recognised and left alone. While I was in there:
"count people crossing the door" never triggered the net at all, because the
pattern matched "cross" but not "crossing". Fixed too.

Both fixes have tests that fail on the old code.

### Speed

- **Gesture models run side by side.** With several gesture behaviours on at
  once, their models used to run one after another inside the frame loop, and
  three at once dropped us from 30 to about 16 fps. Now they run in parallel,
  so a frame costs the slowest model instead of all of them added up. This is
  tested with stand-in models; I haven't re-measured the fps on the GPU yet.
- **Video encoding moved off the server's main loop.** Drawing the overlays and
  compressing each frame takes about 16 ms with a blur on, and it ran on the
  same loop that answers API calls and pushes live updates, so everything else
  waited behind it roughly a quarter of the time. It now runs on a background
  thread, and several viewers share one encode.
- **The console re-renders 10 times a second, not 30.** It was redrawing the
  whole page on every camera frame.
- **The agent's pre-flight checks run at the same time.** Before it thinks, it
  looks up what's running, the pipeline's health, the model list and its phrase
  memory. Those ran one after another on every single turn; now they run in
  parallel.
- **Counting doesn't hammer the GPU.** Asking for a count of something the
  camera isn't already tracking could run an extra detector pass on every frame
  of the counting window. It now takes about five samples. That spare detector
  also got a lock, so two requests can't step on each other.

### Easier to run and read

- A fresh clone couldn't be set up: the first step says to copy
  `.env.example`, and a `.gitignore` rule was hiding that file. Fixed.
- Rewrote the README for someone seeing the project for the first time: what it
  does, how it works, how to run it, and who built what.
- Moved our hackathon planning notes into [docs/notes/](docs/notes/) so the top
  level is just the project.
- Two tests failed instead of skipping when an optional package wasn't
  installed. Fixed, along with a missing test dependency.

### Grok, and measuring the agent

The agent used to be OpenAI-only. It now runs on Grok too: set
`ORCH_PROVIDER=xai` and an xAI key, and it's the same code pointed at xAI's
API, since xAI speaks the same protocol.

That raised the obvious question of which model is actually better at this
job, so I built an eval ([orchestrator/evals/](orchestrator/evals/)). It's 24
instructions covering everything the demo does, including questions that
shouldn't set anything up and one thing it can't do. Each one runs through the
real agent against a stand-in pipeline, and gets checked on whether the right
program ended up running. Two runs each:

| Model | Got it right | Typical time per instruction |
|---|---|---|
| GPT-6-Astra (what we demoed on) | 48/48 | 4.8 s |
| Grok 4.7 | 48/48 | 9.2 s |
| Grok 4.20, non-reasoning | 47/48 | 1.9 s |

Both reasoning models were perfect, so the difference is speed. The fast
Grok was about 2.5 times quicker than GPT, but once it replied "Cleared all
behaviors" without actually clearing anything. Full breakdown in
[RESULTS.md](orchestrator/evals/RESULTS.md).

Building it also caught mistakes in the eval itself. My first scorer failed
GPT for tracking a "pencil eraser" after being told to switch from the pencil
to the eraser (it's still an eraser), and the stand-in pipeline made models
that waited for the camera to finish turning look slow. Both are fixed, with
tests. The honest caveat: this measures turning English into the right
program, not whether the camera then sees the thing. That part is tested
separately, in perception.

Tests went from 452 to 456 in perception and from 149 to 187 in the
orchestrator, all passing.
