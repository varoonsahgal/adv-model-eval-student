# 🔭 Module 01: Modern AI Evaluation Foundations

> **Lab:** [`labs/1_TraceBasedEvalSetup_STUDENT.ipynb`](../../labs/1_TraceBasedEvalSetup_STUDENT.ipynb) · **Time:** 20–25 min core lab + 25–40 min extended · **Tools:** Langfuse (free cloud tier), Python

![Lab 01 roadmap](images/lab-roadmap.png)

---

## 🧭 Overview

This module lays the foundation for the rest of the course. You'll learn **why** the metrics you already know (accuracy, F1, BLEU, "it reads well") aren't enough for LLM applications, **what** to measure instead, and **how** to capture the evidence you need to measure it: **traces**.

By the end you will have instrumented a small Q&A assistant with **Langfuse**, read its traces back like an automated evaluation job would, attached human, rule-based and LLM-judge **scores** to those traces, and used a trace to find the *root cause* of a confidently wrong answer.

Everything later in the course builds on this:

| Later module | What it reuses from Module 01 |
|---|---|
| 02 Failure analysis | Turning observed failures into a taxonomy |
| 03 Structured outputs | Rule-based evaluators and pass/fail scores |
| 04 / 04b RAG | Retriever + generation observations, faithfulness vs. correctness |
| 05 Generative outputs | LLM-as-judge rubrics and calibration |
| 06 / 06b Agents | Trace trees, spans, scoring individual steps |
| 07 / 07b Safety | Behavior-match checks, guardrails as observations |
| 08 Pipelines | Scores aggregated into release gates |

---

## 💡 Why This Matters

A model can score 92% on your test set and still:

- 🗣️ write a **fluent, confident answer that is wrong**,
- 📚 cite a **deprecated document** perfectly faithfully,
- 🐢 take **1.4 seconds** when your SLA is 1.2,
- 💸 use **5× more output tokens** than it needs, on your most common query type,
- 🧨 describe **irreversible actions** without checking what the user meant.

None of those show up in an accuracy number. Several don't even show up if you read the final answer: you need to see *inside* the request.

> 🧠 **The core idea of this course:** surface metrics tell you *what* a system produced. Modern evaluation also asks *how* it got there, *whether* that process was sound, and *what it cost*. Traces are how you see the "how".

Real-world consequences are not hypothetical. Teams discover in production that a chatbot invented a refund policy, an agent booked something it wasn't asked to, or a "cheap" feature costs thousands per day in tokens. Every one of those is cheaper to catch with an evaluation suite than with an incident review.

---

## 🎯 Learning Objectives

After this module you will be able to:

1. **Explain** why accuracy and fluency are insufficient for LLM systems, with concrete failure examples.
2. **Describe** the evaluation dimensions used throughout the course: correctness, faithfulness, robustness, safety, system-level behavior, and efficiency (latency/cost).
3. **Map** evaluation approaches to system types: structured outputs, RAG, free-form generation, and agents.
4. **Instrument** an LLM application with Langfuse (SDK v4): traces, spans, generations, tags, sessions, and metadata.
5. **Read traces back programmatically** using the current Observations (v2) and Scores (v3) APIs, and handle asynchronous ingestion.
6. **Design** measurable evaluation criteria with a rubric, scale, and method.
7. **Attach scores** (numeric, boolean, categorical) to traces and to individual observations.
8. **Choose** between rule-based, LLM-as-judge, and human evaluators, and explain why judges must be calibrated.
9. **Use a trace to root-cause** a bad answer in a multi-step (RAG) pipeline.

---

## 📋 Prerequisites

- 🐍 Comfortable Python: functions, dicts, loops, `with` blocks.
- 📏 Classic ML evaluation vocabulary: accuracy, precision, recall.
- 🤖 What an LLM API call is: you send messages, you get text plus token usage back.
- 🔑 A free **Langfuse Cloud** account (you'll create it in Task 1). No credit card. An OpenAI key is **optional** (only Task 11 uses it, and it has an offline fallback).

---

## 🧩 Key Concepts

### 1. 📉 Why traditional metrics fall short

Traditional evaluation assumes a **single correct answer** you can compare against. LLM systems break that assumption in four ways:

| Problem | What it looks like | Why accuracy misses it |
|---|---|---|
| **Many valid answers** | Five different, equally good summaries | Exact-match punishes valid alternatives |
| **Fluency masks errors** | A beautifully written answer with a wrong number | Fluency scores reward the writing, not the facts |
| **Context sensitivity** | Safe on one phrasing, unsafe on a paraphrase | Averages smooth over the dangerous edge case |
| **Hidden intermediate steps** | Right tool, wrong argument; right doc type, stale version | Only the final output is scored |

> ⚠️ **Fluent ≠ correct.** Language models are trained to produce plausible text. Plausibility is *exactly* the property that makes their mistakes hard to spot.

### 2. 🧭 The evaluation dimensions

![One system, many dimensions](images/eval-dimensions-radar.png)

Think of every system as a profile across several axes rather than a single score:

- ✅ **Correctness:** does the output match expected behavior? For structured outputs: right fields and values. For text: factually right and on-question.
- 📚 **Faithfulness (grounding):** are the claims traceable to the provided sources? *Different from correctness:* an answer can faithfully repeat a wrong document, or be correct by luck without support.
- 🔀 **Robustness:** does behavior stay stable under typos, paraphrases, and messy inputs?
- 🛡️ **Safety:** does it avoid harmful content, refuse what it should, and resist manipulation?
- 🧭 **System-level behavior:** for pipelines and agents, did it take the right steps in the right order? Did it ask for clarification when needed?
- ⏱️💸 **Efficiency:** latency (user experience, SLAs) and tokens/cost (budget). These are evaluation dimensions, not just accounting.

> 💡 **Insight:** you'll see in Lab 01 that the `ambiguous_question` answer is *correct* about how to delete files, yet it **fails** clarification (it guessed) and **fails** the token-ratio check (5.25× output/input). Correctness alone would have passed it.

### 3. 🗺️ Evaluation stacks by system type

The dimensions are universal; *how* you measure them depends on what the system does. This is the map for the whole course:

| System type | Typical checks | Module |
|---|---|---|
| **Structured outputs** (JSON, forms, extraction) | Schema validation → field-level scoring → cross-field consistency → robustness | 03 |
| **RAG** (retrieve, then generate) | Retrieval Recall/Precision/nDCG → faithfulness (NLI, judges) → correctness → stress tests | 04, 04b |
| **Generative text** (summaries, analysis) | Claim-level factuality → completeness → numeric checks → rubric-based LLM judge | 05 |
| **Agents** (tools, multi-step, side effects) | Tool selection → input validation → process constraints → **end-state** → reliability (pass^k) | 06, 06b |
| **Any of the above, under attack or noise** | Perturbations, safety layers, guardrails, prompt-injection metrics | 07, 07b |
| **All of it, continuously** | CI → pre-prod → production stages, gates, regression detection | 08 |

### 4. 🧰 The tooling landscape (2026)

| Category | Examples | What it gives you |
|---|---|---|
| **Tracing / observability** | **Langfuse** (open source), LangSmith, Arize Phoenix, Datadog LLM Observability | Capture every request as a tree of steps with timing, tokens and cost |
| **Evaluation frameworks** | RAGAS, DeepEval, OpenAI Evals, promptfoo, Inspect | Pre-built metrics, judges, dataset runners |
| **Standards** | OpenTelemetry **GenAI semantic conventions** | Common span names (`chat`, `execute_tool`, `invoke_agent`) so traces work across tools |

This course uses **Langfuse** because it's open source, has a free cloud tier, can be self-hosted, and its v4 Python SDK is built on OpenTelemetry. The *concepts* (traces, observations, scores) transfer to every tool above.

### 5. 🌳 Traces, observations and the Langfuse data model

![Anatomy of a trace](images/trace-anatomy.png)

| Term | Meaning | In the lab |
|---|---|---|
| **Trace** | Everything that happened for **one** user request | `qa-assistant-simple_factual` |
| **Observation** | Any step inside a trace. Each has an `id` and a `parent_observation_id` | the span and the generation |
| **Span** | A generic unit of work (a function, a pipeline stage) | the root `qa-assistant-…` span |
| **Generation** | An **LLM call**: carries `model`, token **usage**, and **cost** | `llm-completion` |
| **Retriever / Tool / Agent / Evaluator / Guardrail** | Specialised observation types that make the UI readable | `search-docs` retriever (Task 10), `judge-…` evaluator (Task 11) |
| **Session** | A group of related traces | your `RUN_ID`: one notebook run |
| **Tags / metadata** | Labels for filtering and slicing | `["module-01-lab", "simple_factual"]` |
| **Score** | An evaluation result attached to a trace *or* a specific observation | `correctness = 1.0` |

**Three SDK v4 rules that trip people up** (all used in Task 3):

1. `langfuse.start_as_current_observation(as_type=..., name=...)` opens an observation; anything opened inside its `with` block becomes its **child**. That parent pointer *is* the tree.
2. **Trace-level** attributes (name, tags, session, metadata) are set with `propagate_attributes(...)`, which wraps the root observation. Passing `metadata=` to a span keeps it on that span only.
3. **The root observation *is* the request.** Its input, output and latency are what you read back as "the trace's".

**Two read-side realities** (Task 5):

- ⏳ **Ingestion is asynchronous.** After `langfuse.flush()` the data takes a few seconds to become queryable. Querying immediately is the #1 cause of errors, so the lab wraps every read in a `wait_for(...)` retry helper.
- 🔄 **Use the current read APIs.** Read observations with `langfuse.api.observations.get_many(...)` (Observations API **v2**, and ask for the field groups you need: `io`, `usage`, `model`, …) and scores with `langfuse.api.scores_v3.get_many_v3(...)`. The older `api.trace.list/get` and `api.scores.get_many` endpoints return **HTTP 410** for Langfuse organizations created on or after Sept 16, 2026. A "trace" is rebuilt from its observations.

![Output-only vs trace-based evaluation](images/output-vs-trace.png)

### 6. 📝 Designing evaluation criteria

Deciding **what** to measure is often harder than measuring it. A criterion is only useful if two different reviewers would give the same score. In Task 6 you'll write each criterion with five fields:

| Field | Purpose | Example |
|---|---|---|
| `name` | becomes the score name in Langfuse | `destructive_action_safety` |
| `description` | the question it answers | "Does the answer warn before irreversible actions?" |
| `scale` | `binary`, `1-5`, or `categorical` | `binary` |
| `method` | `rule`, `llm_judge`, or `human` | `rule` |
| `rubric` | what each score means: this makes it **repeatable** | "1 = warns before every destructive step; 0 = otherwise" |

> ✍️ **Rule of thumb:** if you can't write the rubric line, the criterion isn't measurable yet. And aim for at least one criterion that uses **trace data** (latency, tokens) and not just the text.

### 7. 🏷️ Scores: attaching evidence to traces

![Score types](images/score-types.png)

`langfuse.create_score(trace_id=..., name=..., value=..., data_type=..., comment=..., observation_id=...)`

- **NUMERIC** (`0.7`) for ratings you'll average and trend.
- **BOOLEAN** (`0`/`1`) for pass/fail checks; read back as `True`/`False` via the v3 API.
- **CATEGORICAL** (`"assumed_intent"`) for labels you'll count and group.
- `comment` explains **why**: invaluable when someone audits the score later.
- `observation_id` attaches the score to a specific **step**, so the UI shows *where* the failure is (Task 10).
- Every score records its **source**: `API` (from code), `ANNOTATION` (a human in the UI), `EVAL` (Langfuse-managed evaluators). Keeping sources separate lets you compare humans with automation.

### 8. ⚖️ Three kinds of evaluators

![Use the cheapest evaluator that works](images/evaluator-ladder.png)

| Evaluator | Best for | Watch out for |
|---|---|---|
| 🧮 **Rule-based** (Task 8) | Facts about the trace: latency ≤ SLA, output/input ratio, "warning present when deleting" | Brittle on nuance. A regex that decides "did it refuse?" is easy to fool |
| 🤖 **LLM-as-judge** (Task 11) | Subjective criteria: clarification, helpfulness, tone | It's another LLM: cost, latency, bias. **Never trust a judge you haven't calibrated** |
| 🧑‍⚖️ **Human** (Task 7) | Ground truth for calibration; ambiguous cases | Slow and expensive. Sample, don't label 100% |

**Calibration** means comparing the judge's scores with yours on the same items and measuring agreement. In Task 11 the default *simulated* judge only checks whether an answer ends with "?", and you'll see it disagree with you. That's the point.

### 9. 📊 Aggregating scores without fooling yourself

In Task 9 you'll build a **trace × score** matrix. Three traps:

1. **Sparse matrices:** not every criterion applies to every trace. "Average of everything" mixes different sets of criteria per trace. Track **score coverage** as its own metric.
2. **Mixed scales:** averaging a manual `0.7` helpfulness with binary rule checks hides more than it reveals. Group by dimension first.
3. **One number for management:** usually a mistake. Report the weakest dimension and the worst trace, not just the mean.

### 10. 🕵️ Root cause in multi-step pipelines

![Root-causing a fluent, faithful, wrong answer](images/rag-root-cause.png)

In Task 10 the same question runs through a healthy and a buggy RAG pipeline. Both answers are fluent and **faithful to their context**. Only one is correct. The trace shows why: the buggy retriever ranked a **deprecated v1 document** first. The generator did its job perfectly.

- **Faithful ≠ correct.** Faithfulness asks "did the answer follow the context?" Correctness asks "is it true?"
- **Score the step, not just the trace.** A `retrieval_freshness` score on the **retriever observation** tells the next engineer *where* to look, and it needs **no ground-truth answer**, so it can run on 100% of production traffic.

### 11. 💰 Cost and latency as evaluation dimensions

Langfuse computes cost from `model` + token `usage`, so you never send a price. Per-request costs look tiny (fractions of a cent). Multiply by a million requests a month and the **output/input token ratio** of your most common query type becomes a budget line. Latency belongs to the user experience: which *step* is slow matters as much as the total.

---

## 🧪 Example Walkthrough

Here is the heart of Task 3: one traced LLM interaction, in the shape every later lab reuses.

```python
from langfuse import get_client, propagate_attributes
langfuse = get_client()                      # reads LANGFUSE_* env vars

with propagate_attributes(                   # ① trace-level attributes
    trace_name="qa-assistant-simple_factual",
    session_id=RUN_ID,                        #    groups this notebook run
    tags=["module-01-lab", "simple_factual"],
    metadata={"expected_behavior": "answer"},
):
    with langfuse.start_as_current_observation(   # ② root span = the request
        as_type="span", name="qa-assistant-simple_factual",
        input="What is the maximum file upload size?",
    ) as root:
        with langfuse.start_as_current_observation(   # ③ child generation = the LLM call
            as_type="generation", name="llm-completion", model="gpt-4o-mini",
            input=[{"role": "user", "content": "What is the maximum file upload size?"}],
        ) as gen:
            answer = "The maximum file upload size is 50 MB for free-tier users..."
            gen.update(output=answer, usage_details={"input": 12, "output": 24})
        root.update(output=answer)            # ④ request-level output

langfuse.flush()                              # ⑤ send now (notebooks exit early!)
```

Then, as an evaluation job would (Tasks 5–8):

```python
obs = langfuse.api.observations.get_many(trace_id=tid, fields="core,basic,io,usage,model,metrics")
# ...wait_for() retries until ingestion catches up...
langfuse.create_score(trace_id=tid, name="rule_latency_sla", value=1, data_type="BOOLEAN",
                      comment="LLM latency 0.80s vs SLA 1.2s")
```

What you'll actually see in the lab (verified on Langfuse 4.x):

| Trace | Latency | Out/In tokens | Rule results |
|---|---|---|---|
| simple_factual | 0.80 s | 2.0× | ✅ all pass |
| multi_step_reasoning | 1.42 s | 1.7× | ❌ latency SLA |
| ambiguous_question | 1.10 s | **5.2×** | ❌ token ratio, ❌ behavior match (should clarify) |
| out_of_scope | 0.60 s | 2.8× | ✅ all pass |

---

## ⚠️ Common Mistakes or Misunderstandings

| ❌ Mistake | ✅ Instead |
|---|---|
| "The answer reads well, so it's fine." | Fluency is the property LLMs are optimized for. Check facts, grounding, and behavior. |
| Querying Langfuse right after `flush()` and concluding tracing is broken | Ingestion is async: poll with a timeout (`wait_for`). |
| Using `langfuse.trace(...)` or `api.trace.list()` from old tutorials | SDK v4 uses `start_as_current_observation` + `propagate_attributes`; read via Observations v2 / Scores v3. |
| Passing `metadata=` to a span and expecting it on the trace | Trace-level attributes go in `propagate_attributes(...)`. |
| Swapping public and secret keys, or using the wrong region | `pk-lf-…` is public, `sk-lf-…` is secret; US vs EU hosts are different. |
| Treating "faithful" as "correct" | A faithful answer to a stale document is still wrong. |
| Trusting an LLM judge because it gave a confident reason | Calibrate against your own labels; version your judge prompts. |
| Averaging every score into one number | Report per dimension; track coverage; highlight the weakest trace. |
| Committing API keys to a shared notebook | Use Colab **Secrets** or a hidden prompt; rotate any leaked key. |

---

## 🚀 Before You Start the Lab

1. 🌍 Pick a Langfuse Cloud **region** (US `us.cloud.langfuse.com` or EU `cloud.langfuse.com`) and **remember it**.
2. 🏢 Create an **organization** and a **project** (e.g. `trace-eval-lab`).
3. 🔑 Create API keys (**Settings → API Keys**). Copy the public `pk-lf-…` and secret `sk-lf-…` keys. The secret is shown **once**.
4. 🔐 In Colab, open the 🔑 **Secrets** panel and add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` with *Notebook access* on. (Optional: `OPENAI_API_KEY` for Task 11.)
5. 🧠 Keep these questions in mind while you work:
   - What does the trace tell me that the answer text doesn't?
   - Which step would I score if this answer were wrong?
   - Would two reviewers agree with my rubric?

> 💰 **Cost:** the Langfuse Hobby tier is free and more than enough for this course. The lab uses **simulated** LLM responses, so no model costs unless you opt into the real judge.

---

## 🔑 Key Takeaways

1. 🔭 **Traces show what outputs hide:** per-step latency, tokens, cost, model, and every intermediate input and output.
2. 🧭 **Evaluate along several dimensions.** Correctness, faithfulness, robustness, safety, system behavior and efficiency can disagree, and the weakest one is what users experience.
3. 🏷️ **Scores belong next to the evidence:** on the trace, and on the specific **observation** when you know where the failure is.
4. ⚖️ **Use the cheapest evaluator that works:** rules for facts, LLM judges for nuance, humans to calibrate both.
5. 📚 **Faithful ≠ correct, fluent ≠ correct.** Only the trace revealed the deprecated document.
6. ⏳ **Plan for async ingestion and use current APIs** (Observations v2, Scores v3) in every evaluation pipeline.

---

## 🧠 Self-Check

<details>
<summary>1. Why does the lab wrap the LLM generation in a span instead of making the generation the root?</summary>

So that retrieval, validation, or a second LLM call can slot in as siblings under the same root later (as in Task 10) without changing the trace structure. The root observation represents the whole request.
</details>

<details>
<summary>2. The rule `rule_latency_sla` fails <code>multi_step_reasoning</code>. Is that a quality problem? Who owns it?</summary>

It's a user-experience/SLA problem rather than an answer-quality problem, often owned by the platform or infrastructure team. It's still an evaluation result: it belongs on the dashboard next to quality scores.
</details>

<details>
<summary>3. Why doesn't <code>retrieval_freshness</code> need a ground-truth answer?</summary>

It checks a property of the retrieved documents (their `status`/`version` metadata), not the answer. That makes it cheap enough to run on all production traffic.
</details>

<details>
<summary>4. Your LLM judge and you disagree on 3 of 5 items. What are the two possible explanations, and how do you tell them apart?</summary>

Either the judge is wrong, or your rubric is ambiguous. Read the judge's reasoning (stored as the score comment): if its reasoning is a defensible reading of your rubric, tighten the rubric; if not, fix the judge prompt or model.
</details>

<details>
<summary>5. What HTTP status tells you you're calling a legacy Langfuse read endpoint?</summary>

**410** `LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION`. Switch to `api.observations.get_many` and `api.scores_v3.get_many_v3`.
</details>

---

## 📖 Glossary

**Trace** · one request, end to end. **Observation** · any step in a trace (span, generation, retriever, tool, agent, evaluator, guardrail). **Generation** · an LLM call with model, usage and cost. **Session** · a group of traces. **Score** · an evaluation result on a trace or observation. **Flush** · send buffered telemetry now. **Ingestion delay** · time before written data is queryable. **Faithfulness** · claims supported by the provided context. **Calibration** · measuring an automated evaluator's agreement with trusted (human) labels.
