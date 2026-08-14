# MOK Dataset Archive

This directory is the source-map, manifest, schema, sample, generator, validator, and QC scaffold for the Model Operating Kernel dataset archive.

MOK's training direction is **behavior-first**:

- train reasoning, tool choice, routing, verification, coding workflow, research workflow, retrieval discipline, and learning/write-back habits;
- avoid broad factual memorization as the main capability target;
- make tools, repo inspection, tests, Atlas/Cartographer/RAG retrieval, and source verification the default path for unstable or unknown facts;
- commit only manifests, schemas, source policy, small samples, generators, validators, eval harnesses, and QC reports unless a dataset is explicitly public, licensed, and safe to version;
- keep generated traces, private training data, local datasets, and model assets out of version control.

The archive is designed to make MOK wiser at **finding, checking, applying, and remembering** information rather than merely stuffing subject facts into weights.

## Operating pattern

```text
unknown / unstable / niche information
-> detect uncertainty
-> choose tool or retrieval path
-> inspect source material
-> build Cartographer-style source map
-> select Atlas/RAG evidence cards
-> verify claims
-> answer with evidence boundaries
-> write durable learning note/card when useful
-> test retrieval later
```

## Directory plan

```text
datasets_archive/
  README.md
  manifest.json
  licenses_and_sources/
    source_policy.md
  schemas/
    README.md
    mok_record.schema.json
    trace_record.schema.json
    tool_call_record.schema.json
    verifier_record.schema.json
    retrieval_record.schema.json
    memory_writeback_record.schema.json
    coding_workflow_record.schema.json
  00_seed_policy/
  01_logic_reasoning/
  02_tool_use/
  03_research_workflows/
  04_coding_workflows/
  05_retrieval_atlas_cartographer/
  06_verification_and_source_boundaries/
  07_memory_writeback_and_learning/
  08_router_orchestration/
  09_failure_recovery/
  10_safety_privacy_project_discipline/
  11_eval_sets/
  12_reward_model_pairs/
  13_synthetic_generation_recipes/
  14_qc_reports/
```

Phase 1 creates the research map and policy scaffold. Phase 2 should add schemas before any large generation run.

## Dataset families

| ID | Family | Behavioral target |
|---|---|---|
| `01_logic_reasoning` | Logic and reasoning | Constraint tracking, contradiction detection, proof checking, assumption control, concise reasoning summaries |
| `02_tool_use` | Tool use and routing | Choose the right tool, call it correctly, inspect observations, retry intelligently, avoid unnecessary tool use |
| `03_research_workflows` | Research workflows | Search planning, source triage, evidence matrices, gap analysis, citation discipline |
| `04_coding_workflows` | Coding workflows | Inspect repo state, reproduce bugs, patch minimally, run tests, report results honestly |
| `05_retrieval_atlas_cartographer` | Atlas / Cartographer / RAG | Select lanes/cards, reject distractors, map sources, answer from evidence, off-ramp when unsupported |
| `06_verification_and_source_boundaries` | Verification and source boundaries | Verify claims, handle conflicts, state uncertainty, avoid fake citations |
| `07_memory_writeback_and_learning` | Memory write-back and learning | Detect unknowns, research them, write durable RAG notes, include staleness policy |
| `08_router_orchestration` | MOK runtime/router/oracle traces | Use MOK traces, route decisions, oracle scores, fallback events, and VRAM pressure records |
| `09_failure_recovery` | Failure recovery and off-ramps | Recover from failed tools/tests/searches and report partial findings honestly |
| `10_safety_privacy_project_discipline` | Safety, privacy, and project discipline | Redact secrets, respect local project boundaries, avoid contamination and unsafe automation |

## First training mix recommendation

| Family | Initial weight |
|---|---:|
| Tool use and routing | 20% |
| Research workflow | 15% |
| Coding workflow | 15% |
| Verification/off-ramp | 15% |
| Atlas/Cartographer/RAG | 15% |
| Logic/reasoning | 10% |
| MOK runtime/router traces | 5% |
| Positive curiosity / learning behavior | 5% |

Adjustment rule: if the model memorizes instead of retrieving, increase Atlas/RAG, verification, and tool-use records. If it over-searches simple stable tasks, add no-tool-needed contrast examples.

## Phase 1 deliverables

Phase 1 is complete when the repo contains:

- `research.md` with source-backed notes, not chat logs;
- `datasets_archive/README.md`;
- `datasets_archive/manifest.json`;
- `datasets_archive/licenses_and_sources/source_policy.md`;
- a clear source inclusion/exclusion policy;
- an initial dataset family map;
- a list of unresolved source gaps for Phase 1B.

## Phase 2 handoff

Phase 2 should create JSON schemas and validators for the core record types before scaling generation:

- base MOK record;
- tool-use trace record;
- retrieval/Atlas record;
- memory write-back record;
- coding workflow record;
- verifier/off-ramp record;
- router/orchestration trace record.

No large dataset generation should start until schemas and validators exist. Otherwise the archive risks becoming a swamp of noisy synthetic data wearing a fake mustache and pretending to be a curriculum.
