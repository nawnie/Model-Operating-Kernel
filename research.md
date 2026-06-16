# MOK Dataset Archive — Phase 1 Research Notes

Status: Phase 1 source map  
Project target: Model Operating Kernel dataset archive for a logic-first, tool-native researcher/coding model  
Date: 2026-06-16

## Core training position

MOK should be trained to prefer **procedural intelligence over memorized content**.

The target behavior is:

```text
unknown or unstable fact
-> recognize uncertainty
-> choose the right tool or retrieval path
-> inspect evidence
-> build a compact source map
-> verify claims
-> answer with boundaries
-> write durable learning notes/cards back to RAG when useful
```

This archive should not become a broad subject-matter pretraining pile. Direct content knowledge should be intentionally sparse. The model should learn how to find, verify, and apply information from tools, source files, traces, search maps, tests, and retrieval cards.

## MOK repo anchors

MOK is already a good host for this training direction because the repo defines it as a local-first runtime coordination layer, not an in-model MoE. It registers experts, selects routes, manages VRAM pressure, invokes local or HTTP-backed models, records traces, and feeds evaluation data back into routing improvements.

Project source:

- https://github.com/nawnie/Model-Operating-Kernel/blob/main/README.md

Current runtime features that matter for the dataset archive:

- expert registry and lifecycle state
- VRAM budget accounting and idle-expert eviction
- rule-based R0 routing plus learned-router scaffolding
- circuit breakers
- mock, HTTP, Ollama, and llama.cpp-style backends
- GGUF metadata inspection
- JSONL trace logging
- oracle scoring
- training-pair export
- smoke evaluation harnesses

Relevant source files:

- `src/mok/telemetry/events.py` defines `TraceEvent` fields such as prompt, route expert, route confidence, route reason, called experts, evictions, success, error type, token count, metadata, and latency.
- `src/mok/evaluation/oracle.py` defines oracle scoring, regret metrics, and a harness for writing oracle-score JSONL records.
- `src/mok/evaluation/export.py` joins runtime JSONL traces with oracle scores to create router training pairs.
- `evaluation/mok_core_smoke.jsonl` already tests the behavior family this archive should strengthen: project-file safety, claim verification, tool-result skepticism, clarification, current-information lookup, and test/surface inspection.

## Atlas LoRA precedent

Atlas LoRA is the closest training-policy precedent found so far. Its README states the central question as whether a lightweight LoRA adapter can improve use of structured retrieval context. It tests lane/card behavior rather than flat text dumping.

Project source:

- https://github.com/nawnie/atlas-lora-adapter/blob/main/README.md

Key Atlas lessons to carry into MOK:

```text
Knowledge = Atlas cards, lanes, and source records
Behavior = adapter reading discipline
Proof = saved evaluation outputs
```

MOK should generalize this idea from retrieval-card reading into full operating-kernel behavior:

```text
Knowledge = RAG cards, source maps, repo files, tool outputs, traces, eval results
Behavior = route, retrieve, inspect, verify, patch, test, cite, remember
Proof = executable tests, oracle scores, trace replay, source-bound answers
```

Atlas-specific behaviors to preserve:

- choose the correct lane/card
- ignore distractor context
- follow source hierarchy
- avoid unsupported exact identifiers
- off-ramp when evidence is missing
- answer from selected evidence instead of guessing
- preserve evidence-bound wording and avoid broad claims from narrow evals

## Fable 5 status

No separate Fable 5 repository was found under the connected GitHub search terms `fable`, `fable5`, or `Fable-5` for user `nawnie` during this pass. Treat Fable 5 as an unresolved source gap until a repo path, local archive, or file name is supplied.

Phase 1 should not block on that. The archive can proceed from MOK + Atlas + public primary sources, then add Fable-specific notes when located.

## Dataset family map

### 01 — Logic and reasoning

Purpose: teach constraint tracking, contradiction detection, proof checking, assumption management, and concise reasoning summaries.

Good source patterns:

- synthetic constraint-satisfaction tasks with deterministic validators
- symbolic logic and counterexample tasks
- proof sketch / proof check pairs
- math reasoning tasks used as structure examples, not bulk content memorization
- formal theorem proving examples where the checker is the reward signal

Primary references:

- MATH introduces 12,500 competition math problems with step-by-step solutions: https://arxiv.org/abs/2103.03874
- miniF2F provides 488 formal Olympiad-level theorem statements across Lean, Metamath, Isabelle, and HOL Light: https://arxiv.org/abs/2109.00110
- Tree of Thoughts frames deliberate search over candidate reasoning paths: https://arxiv.org/abs/2305.10601

MOK inclusion rule: prefer generated, validated reasoning tasks and proof-checkable tasks over scraped solution dumps. Do not train long hidden chain-of-thought transcripts as final-answer style. Train compact reasoning summaries, verification notes, and decision records.

### 02 — Tool use and function calling

Purpose: teach when to use tools, how to select tools, how to form tool calls, how to inspect observations, and how to recover when tools fail.

Good source patterns:

- function-call records with exact schema matching
- single-tool, multi-tool, parallel-tool, and irrelevant-tool cases
- API-documentation reading records
- tool failure and retry traces
- tool-result skepticism records
- examples where no tool is needed

Primary references:

- ReAct interleaves reasoning and acting so actions can gather external information: https://arxiv.org/abs/2210.03629
- ToolLLM / ToolBench builds tool-use SFT data over 16,000+ real-world APIs and includes solution-path annotations: https://arxiv.org/abs/2307.16789
- ToolBench repository and data notes: https://github.com/OpenBMB/ToolBench
- Gorilla / APIBench focuses on generating accurate API calls and adapting to changing documentation with retrieval: https://arxiv.org/abs/2305.15334
- Berkeley Function Calling Leaderboard evaluates AST matching, execution, relevance detection, type matching, and hallucinated parameters: https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html

MOK inclusion rule: train tool discipline around MOK's actual tool inventory and runtime traces. Public tool-use datasets are best used for schema inspiration, validators, and evaluation patterns unless license and provenance are fully audited.

### 03 — Research workflows

Purpose: teach search planning, source triage, claim/evidence matrices, citation discipline, contradiction handling, and research notes that are not chat logs.

Good source patterns:

- user question -> query plan -> source map -> evidence matrix -> grounded answer
- current-information questions requiring fresh lookup
- niche-term questions requiring discovery
- conflicting-source reconciliation
- primary-source preference and source hierarchy
- quote-limited summarization
- explicit `found X / did not find Y` reports

Primary references:

- GAIA tests real-world questions needing reasoning, multimodality, browsing, and tool-use proficiency: https://arxiv.org/abs/2311.12983
- GAIA Hugging Face organization page: https://huggingface.co/gaia-benchmark
- WebArena provides long-horizon web tasks across realistic websites and external knowledge bases: https://arxiv.org/abs/2307.13854
- AgentBench evaluates LLMs as agents across interactive environments and highlights failures in long-term reasoning and decision-making: https://arxiv.org/abs/2308.03688
- MLAgentBench evaluates agents doing ML experimentation with file reads/writes, code execution, and output inspection: https://arxiv.org/abs/2310.03302

MOK inclusion rule: train the workflow, not the answer cache. Keep source references and evidence boundaries. Do not store broad factual answers unless they are attached to provenance and freshness metadata.

### 04 — Coding workflows

Purpose: teach repository inspection, bug reproduction, patch planning, minimal edits, testing, rollback conditions, and honest dev reports.

Good source patterns:

- repo state + issue + files inspected + tests run + patch summary + result
- failing test interpretation
- dependency/environment diagnosis
- code review records
- PR summaries
- tool logs with explicit failure handling

Primary references:

- SWE-bench uses real GitHub issues and corresponding PRs and asks models to edit codebases to resolve issues: https://arxiv.org/abs/2310.06770
- SWE-bench official leaderboard and family pages: https://www.swebench.com/
- SWE-bench-Live and SWE-bench++ style work should be considered for contamination-resistant, generated, executable coding tasks: https://arxiv.org/abs/2505.23419 and https://arxiv.org/abs/2512.17419

MOK inclusion rule: generate MOK-native coding traces from this repo first. Prefer executable tests and minimal diffs over natural-language code Q&A. Public coding benchmarks should be held out or used only after contamination policy is clear.

### 05 — Retrieval / Atlas / Cartographer / RAG behavior

Purpose: teach the model to treat retrieval as structured evidence, not a dump of text.

Good source patterns:

- lane selection
- card selection
- card rejection with distractors
- source hierarchy following
- compact evidence pack reading
- unsupported evidence off-ramp
- search result -> Cartographer map -> RAG note/card
- retrieval test question after write-back

Primary references:

- RAG combines parametric generation with non-parametric retrieval memory for knowledge-intensive tasks and addresses provenance/update problems: https://arxiv.org/abs/2005.11401
- Self-RAG trains adaptive retrieve/generate/critique behavior with reflection tokens: https://arxiv.org/abs/2310.11511
- RAGTruth annotates hallucinations in retrieval-augmented generation responses: https://arxiv.org/abs/2401.00396
- Atlas LoRA local precedent: https://github.com/nawnie/atlas-lora-adapter

MOK inclusion rule: every retrieval record should include source IDs, selected evidence, rejected distractors, unsupported claims to avoid, and off-ramp criteria.

### 06 — Verification and source boundaries

Purpose: teach MOK to verify before final answers and to state evidence boundaries clearly.

Good source patterns:

- fact decomposition into atomic claims
- source support/refute/not-enough-information labels
- source conflict records
- stale knowledge correction
- quote compliance and citation discipline
- high-stakes caution records

Primary references:

- FActScore evaluates long-form factuality by decomposing generations into atomic facts supported by reliable sources: https://arxiv.org/abs/2305.14251
- FEVEROUS verifies claims against unstructured text and structured tables: https://arxiv.org/abs/2106.05707
- RAGTruth covers unsupported or contradictory claims in RAG outputs: https://arxiv.org/abs/2401.00396

MOK inclusion rule: train explicit uncertainty and refusal/off-ramp behavior when evidence is absent. Penalize confident unsupported claims, fake citations, and overgeneralization from narrow evidence.

### 07 — Memory write-back and learning behavior

Purpose: teach positive curiosity with disciplined persistence.

Target behavior:

```text
New or unknown term detected
-> express learning-ready stance
-> search/retrieve before claiming
-> build a source map
-> write a durable note/card with provenance
-> tag freshness and staleness policy
-> generate a retrieval test question
```

Good source patterns:

- unknown-item detection
- learning trigger classification
- search plan
- Cartographer output
- RAG write-back note
- retrieval test and answer-after-retrieval
- stale-info expiration policy

Primary references:

- Reflexion stores verbal reflections in episodic memory to improve later decisions without weight updates: https://arxiv.org/abs/2303.11366
- Self-RAG provides adaptive retrieval and critique behavior: https://arxiv.org/abs/2310.11511

MOK inclusion rule: reward curiosity only when it triggers verification and useful memory. Do not reward fake enthusiasm, noisy memory writes, or unsupported summaries.

### 08 — Router/orchestration/runtime traces

Purpose: train MOK's router and orchestration layer from its own traces.

Good source patterns:

- prompt -> route decision -> expert -> outcome
- VRAM pressure -> eviction decision -> result
- circuit breaker events
- backend failure -> fallback
- oracle score -> improved routing pair
- trace replay records
- multi-expert decomposition

MOK-native sources:

- `src/mok/routing/router.py`
- `src/mok/telemetry/events.py`
- `src/mok/evaluation/oracle.py`
- `src/mok/evaluation/export.py`
- `evaluation/mok_core_smoke.jsonl`

MOK inclusion rule: this should be one of the highest-value local datasets because it teaches the operating kernel how to improve itself without pretending it has universal knowledge.

### 09 — Failure recovery and off-ramps

Purpose: teach graceful failure, retries, fallback, and honest reporting.

Good source patterns:

- failed search -> query reformulation
- missing evidence -> off-ramp
- bad tool output -> verify with second source or local check
- failed test -> report failure and next patch target
- partial success reports
- contradiction found -> state conflict and source priority

Primary references:

- ReAct emphasizes acting to gather information and update plans: https://arxiv.org/abs/2210.03629
- Reflexion emphasizes feedback-driven improvement via memory: https://arxiv.org/abs/2303.11366
- Agent safety benchmarks are relevant for avoiding harmful or unsafe agent behavior: https://arxiv.org/abs/2412.14470

MOK inclusion rule: reward `I found X but not Y`, `test failed because`, and `evidence is insufficient` over bluffing.

### 10 — Safety, privacy, and project discipline

Purpose: keep local-first MOK useful without leaking private data, damaging projects, or overstepping tool boundaries.

Good source patterns:

- local repo safety checks
- secret redaction
- permission boundaries
- benchmark contamination flags
- license/provenance checks
- high-stakes domain caution
- user-specific/private-source distinction

MOK inclusion rule: generated traces and private data stay out of version control. Commit only manifests, schemas, validators, sample records, and research notes unless data is explicitly public and licensed.

## Training mix draft

Initial mix for first experiments:

| Family | Weight |
|---|---:|
| Tool use and routing | 20% |
| Research workflow | 15% |
| Coding workflow | 15% |
| Verification/off-ramp | 15% |
| Atlas/Cartographer/RAG | 15% |
| Logic/reasoning | 10% |
| MOK runtime/router traces | 5% |
| Positive curiosity/learning behavior | 5% |

Adjustment rule: if the model memorizes instead of retrieving, increase Atlas/RAG, verification, and tool-use records. If it over-searches simple stable tasks, add no-tool-needed contrast examples.

## Phase 1 data inclusion policy

Include:

- MOK-native traces after redaction
- MOK smoke/eval scenarios
- synthetic tasks with validators
- source-backed research maps
- code tasks with executable tests
- tool-call records with exact schemas
- retrieval-card records with selected and rejected evidence
- preference pairs showing good vs bad verification behavior

Exclude:

- private traces without redaction
- secrets, tokens, keys, local absolute paths that reveal sensitive data
- benchmark test answers intended for held-out evaluation
- copyrighted corpora without clear license
- broad subject-matter dumps
- chat-log transcripts
- hidden chain-of-thought dumps as target style
- unsupported exact IDs or source claims
- generic inspirational slop without operational behavior

## Open gaps for Phase 1B

- Locate Fable 5 source material or confirm it is unavailable.
- Identify the exact Cartographer output schema and expected RAG card fields.
- Decide whether `datasets_archive/` stores only manifests/samples or can store small generated public examples.
- Add license notes per public dataset before any raw data is imported.
- Define a redaction standard for MOK traces.
- Define a contamination split: training examples, dev examples, held-out eval examples.

## Phase 2 handoff

Phase 2 should create stable JSON schemas before generating volume. The first schemas should cover:

- base MOK record
- tool-use trace record
- Atlas/retrieval record
- memory write-back record
- coding workflow record
- verifier/off-ramp record

Only after schemas and validators exist should larger synthetic generation begin.
