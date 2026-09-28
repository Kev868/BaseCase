# Agent evals

How often does the agent turn a sentence into the right program, and how long
does it take? This runs the real agent (the runner, the LangGraph loop, the
prompts, the tools) on the instructions in [cases.yaml](cases.yaml) and
checks what ends up running. Only the model changes between runs, so the
numbers compare models on this project's actual job.

Latest numbers: [RESULTS.md](RESULTS.md).

## Run it

```bash
cd orchestrator
.venv/Scripts/python -m evals.run --model gpt-6-astra --model grok-4.7
.venv/Scripts/python -m evals.run --model grok-4.7 --case blur-except-me --repeat 5
```

It needs the key for each model's provider in `orchestrator/.env`
(`OPENAI_API_KEY` for GPT models, `XAI_API_KEY` for Grok). No camera, GPU or
running pipeline is needed. Each model writes `results/<model>.json`, and
`RESULTS.md` is rebuilt from every file there.

## What a case checks

A case is a sentence, or a short conversation, and what must be true once the
agent has finished. For example:

```yaml
- id: watch-duck-authorised
  say: Watch the yellow duck and tell me if anyone approaches, but people in black jackets are allowed
  expect: {kind: watch, detect: [duck], trigger: near, excludes: [black]}
```

passes only if a `watch` behaviour is running on something named like a duck,
with a `near` trigger, and "black" appears in an exclusion. The full list of
checks is in the docstring of `check()` in [run.py](run.py). Every turn must
also finish without an error.

The 24 cases cover every capability in the demo list, plus questions that
must *not* install anything ("show me what's on the desk") and a request the
pipeline cannot serve (the keyboard guide).

## What it does not measure

The model talks to a stand-in pipeline ([pipeline.py](pipeline.py)), not a
camera, so:

- **This is the compile step only.** A correct program that the detector then
  fails to see still passes. Detection quality is perception's business and
  is tested there, against recorded clips.
- **Wording probes return fixed scores.** They follow the real detector's
  measured habits: "person's face" scores low, and "yellow duck" finds the
  duck where plain "duck" does not. But they are a model of the scene, not
  one.
- **No camera frame is sent**, so every model decides on the words and tool
  results alone, the same for all of them.
- **Models are not deterministic.** One run per case is a snapshot; use
  `--repeat` before reading much into a single pass or fail.

A case that fails on a dropped connection is retried, since that says nothing
about the model. A slow answer is not: the agent runs with the same model
timeout it has in production (`ORCH_LLM_TIMEOUT_S`), so a model too slow for
that fails here too.
