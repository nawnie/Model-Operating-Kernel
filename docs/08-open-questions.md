# Open Questions

## Repo reality

- Where is the actual MoK implementation repo, if it already exists locally?
- Which of the source-PDF assumptions are already implemented versus still aspirational?

## Base-model decision

- Does `Qwen2.5-7B-Instruct` still win after quantized evaluation on the real 16 GB target?
- Are licensing or distribution constraints strong enough to change that choice?

## Coordinator

- What exact model should serve as the always-resident core coordinator?
- Should it share the same base family as the text experts or remain separate?

## Routing

- What is the final `mok.route.v1` schema?
- Which requests should escalate from `R2` to `R1` or coordinator planning?

## Serving stack

- Is `vLLM` enough for adapter hot-swap and concurrency on the target hardware?
- Does the project need `LoRAX` or another multi-adapter server earlier than expected?

## Data and evaluation

- What minimum trace volume is realistic before training `R2`?
- How will private or local-only trace data be versioned safely?
- What exact held-out suites will be treated as merge-blocking?

## Memory and cost

- What context-length shapes are most representative for the target workload?
- How aggressive should prefetch be before it starts hurting the landing zone?
