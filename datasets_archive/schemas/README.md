# Dataset Schema Handoff

Phase 2 should create JSON schemas before generating dataset volume.

The archive should never scale synthetic data until records have stable shapes and validators. Otherwise, malformed examples will quietly train sloppy behavior.

## Required schemas for Phase 2

### `mok_record.schema.json`

Base envelope shared by all dataset families.

Required concerns:

- stable `id`;
- `dataset_family`;
- `task_type`;
- `source_refs`;
- `license`;
- `privacy_status`;
- `contamination_split`;
- `quality_tags`;
- `validator_status`.

### `trace_record.schema.json`

MOK runtime/router trace format.

Required concerns:

- request prompt or redacted prompt;
- route expert;
- route confidence;
- route reason;
- router tier;
- called experts;
- fallback chain;
- success/failure;
- error type;
- oracle score when available.

### `tool_call_record.schema.json`

Tool-use and function-calling format.

Required concerns:

- tool inventory;
- user goal;
- selected tool;
- tool call arguments;
- observations;
- decision points;
- verification step;
- what not to do;
- reward signals.

### `retrieval_record.schema.json`

Atlas / Cartographer / RAG evidence format.

Required concerns:

- question;
- lane candidates;
- selected lane;
- card candidates;
- selected cards;
- rejected distractors;
- source hierarchy;
- answer from cards;
- unsupported claims to avoid;
- off-ramp flag.

### `memory_writeback_record.schema.json`

Curiosity and RAG write-back format.

Required concerns:

- unknown item detected;
- learning trigger;
- search plan;
- Cartographer output;
- RAG write-back note;
- source references;
- staleness policy;
- retrieval test question;
- answer after retrieval.

### `coding_workflow_record.schema.json`

Research/coding model workflow format.

Required concerns:

- repo state summary;
- task;
- files inspected;
- tests run;
- patch plan;
- code change summary;
- test result;
- rollback condition;
- final developer note.

### `verifier_record.schema.json`

Verification, source-boundary, and off-ramp format.

Required concerns:

- claim;
- source candidates;
- source hierarchy;
- evidence snippets or references;
- support/refute/not-enough-info label;
- confidence;
- unsupported claims to avoid;
- final answer boundary.

## Validator requirements

Every schema should be paired with a validator that checks:

- valid JSON;
- required fields;
- known dataset family ID;
- source/provenance metadata;
- privacy status;
- contamination split;
- no fake citations;
- no chat-log residue;
- no assistant-facing TODOs;
- no hidden chain-of-thought target text;
- no unsupported exact IDs;
- family-specific quality rules.

## Phase 2 first step

Create `mok_record.schema.json` first, then make the family-specific schemas extend its metadata conventions.
