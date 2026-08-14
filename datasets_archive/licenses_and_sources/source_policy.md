# MOK Dataset Source Policy

This policy controls what can enter the MOK dataset archive and what must stay out.

The archive's purpose is to train MOK's **operating behavior**: reasoning, research, tool use, coding workflow, retrieval discipline, verification, routing, memory write-back, and failure recovery. The archive is not a general subject-matter pretraining dump.

## Source classes

### 1. MOK-native runtime traces

Examples:

- JSONL trace events from MOK runtime;
- route decisions;
- oracle scores;
- training-pair exports;
- smoke/eval results;
- backend failure/fallback traces;
- VRAM pressure and eviction records.

Status: highest value, but privacy-sensitive.

Rules:

- redact private prompts, secrets, local absolute paths, tokens, credentials, and personal data;
- keep full private traces out of version control;
- commit only tiny synthetic or redacted samples;
- prefer normalized trace records with clear source IDs, timestamps, privacy status, and validator output.

### 2. MOK-native synthetic records

Examples:

- generated tool-selection tasks;
- generated route/oracle examples;
- generated coding workflow records;
- generated verification/off-ramp pairs;
- generated Cartographer maps and RAG write-back notes.

Status: preferred for scalable training if validators are strong.

Rules:

- every generated record must pass JSON schema validation;
- every source-backed record must include source references;
- every retrieval record must include selected evidence and rejected distractors when applicable;
- every coding record should include test or validation outcome;
- every memory write-back record should include staleness policy and a retrieval test question;
- remove chat-log residue, previous drafts, assistant-facing TODOs, fake citations, and unsupported exact IDs.

### 3. Atlas / Cartographer / RAG project records

Examples:

- Atlas lane/card selection examples;
- compact evidence packs;
- source hierarchy examples;
- distractor-card rejection tasks;
- Cartographer source maps;
- RAG write-back notes.

Status: design-aligned and high value.

Rules:

- preserve the Atlas principle: knowledge lives in cards/sources, behavior lives in the adapter/model;
- do not train the model to memorize card content when retrieval can supply it;
- train selected-card discipline, off-ramps, and unsupported-claim avoidance;
- use local project data only after ownership, privacy, and license status are clear.

### 4. Public benchmark or research datasets

Examples:

- ToolBench / ToolLLM-style tool-use records;
- BFCL-style function calling evals;
- SWE-bench-style coding tasks;
- GAIA / WebArena / AgentBench-style long-horizon agent tasks;
- RAGTruth / FActScore / FEVEROUS-style verification records;
- MATH / miniF2F / theorem-proving-style reasoning tasks.

Status: useful for source patterns, schema design, validators, evals, and small licensed samples.

Rules:

- audit license and provenance before importing raw data;
- avoid training on held-out eval/test answers;
- prefer generating MOK-native analogues instead of copying benchmark records;
- keep public benchmark tasks primarily as eval inspiration unless license and contamination policy are explicit;
- preserve source URL, license, split, and transformation notes in the manifest.

### 5. Web or documentation-derived records

Examples:

- API documentation reading tasks;
- source hierarchy decisions;
- query reformulation tasks;
- current-information lookup tasks;
- claim/evidence matrices.

Status: useful, but high risk for staleness and copyright leakage.

Rules:

- store compact notes and claim/evidence matrices, not copied pages;
- include fetch date and freshness policy;
- cite primary sources when possible;
- do not include long verbatim passages;
- mark records that require future refresh.

## Required manifest metadata

Every dataset shard or sample file should eventually have a manifest entry with:

```json
{
  "id": "string",
  "family": "string",
  "path": "string",
  "status": "planned | sample | generated | validated | deprecated",
  "source_name": "string",
  "source_url": "string or null",
  "license": "string or unknown",
  "provenance_status": "owned | generated | public-audited | public-unaudited | private-redacted",
  "privacy_status": "public | redacted | private-do-not-commit",
  "contamination_split": "train | dev | eval | holdout | unknown",
  "created_by": "human | generator | trace_export | mixed",
  "validator": "string or null",
  "notes": "string"
}
```

## Inclusion checklist

A record may enter the archive when it satisfies the relevant checks:

- [ ] trains a behavior MOK needs;
- [ ] has clear source/provenance metadata;
- [ ] has license status recorded;
- [ ] has privacy status recorded;
- [ ] has contamination split recorded;
- [ ] avoids hidden chain-of-thought target text;
- [ ] avoids unsupported exact claims;
- [ ] avoids fake citations;
- [ ] avoids private data leakage;
- [ ] passes its JSON schema;
- [ ] passes family-specific validators;
- [ ] has a reason to exist beyond generic instruction tuning.

## Exclusion checklist

Do not include:

- secrets, keys, credentials, tokens, or auth headers;
- private user data without redaction;
- unredacted local paths that expose sensitive context;
- broad factual dumps detached from source and freshness metadata;
- copyrighted long-form copied text without permission;
- benchmark test/holdout answers used for evaluation;
- chat logs as training records;
- previous draft remnants, assistant-facing TODOs, or prompts to the assistant;
- unsupported source claims;
- fake source IDs or fake citations;
- hidden chain-of-thought dumps as target answer style;
- noisy memory writes that would pollute RAG.

## Redaction standard for MOK traces

Before any trace becomes a dataset record:

1. Replace usernames, emails, local machine names, and absolute local paths with stable placeholders.
2. Remove credentials, keys, cookies, bearer tokens, and private URLs.
3. Remove or summarize sensitive file contents.
4. Keep the behavioral structure: user goal, route, tool decision, observation type, verification step, result, failure mode.
5. Add `privacy_status: redacted` and a redaction note.
6. Run a second pass that searches for common secret patterns.

## Contamination policy

Use three buckets:

- `train`: records safe for training;
- `dev`: records for tuning generation and validators;
- `holdout`: records reserved for evaluation only.

Public benchmark examples should default to `holdout` or `dev` until a deliberate import decision is made. MOK-native generated analogues may enter `train` after validation.

## Preferred archive shape

Commit:

- manifests;
- schemas;
- tiny samples;
- source policies;
- generators;
- validators;
- eval runners;
- QC reports.

Do not commit:

- large generated corpora;
- private raw traces;
- local datasets;
- model checkpoints;
- adapters;
- downloaded benchmark mirrors unless explicitly approved.

## Phase 1 unresolved source gap

Fable 5 training material was requested as a precedent, but no separate Fable 5 repository was found during the connected GitHub search pass. Add its repo/path or archive reference here when located, then update `research.md` and `manifest.json`.
