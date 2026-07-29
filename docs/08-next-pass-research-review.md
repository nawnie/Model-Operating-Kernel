# Next-Pass Research Review

**Added:** 2026-07-29  
**Status:** Review queue  
**Scope:** Public-safe repository alignment, runtime research, and evaluation planning

## Why this note exists

A GitReverse pass reconstructed this repository as a local-first Python control layer that registers experts, routes requests, manages VRAM pressure, invokes local or HTTP-backed models, records traces, evaluates routing, and exposes a small terminal companion.

That is an accurate description of the current public surface. It does not fully communicate the coordinator and consultation behavior that already exists in the code, and some older planning documents now lag behind the runnable slice.

This file records questions for the next deliberate review. It is not a claim that every item below is implemented.

## Current public signal

GitReverse correctly identified:

- explicit expert registration and selection
- rule-based routing with room for learned routing
- VRAM-aware loading and eviction
- local and HTTP-backed inference
- trace collection and evaluation support
- command-line and companion lifecycle controls

The public story is strongest when MoK is described as a model control plane or kernel, not as an in-model Mixture-of-Experts implementation.

## Documentation questions for the next pass

1. **Coordinator visibility**  
   Decide whether the README should describe the implemented consultation loop: focused expert calls, challenges to vague or overconfident replies, bounded follow-ups, multi-expert comparison, confidence gates, and final synthesis by MoK.

2. **Works now versus research direction**  
   Separate each major feature into:
   - works now
   - scaffolded or partially implemented
   - research direction
   - intentionally private or out of scope

3. **Roadmap drift**  
   Review the 14-day and 24-week plans against the current package. Several items originally written as future work now have runnable code, while real-hardware validation and learned-policy promotion remain open.

4. **Public expert taxonomy**  
   Decide whether Mind/Body/Soul is a stable public taxonomy or internal research language. Do not promote it in the README until configurations, traces, and evaluation fixtures support the terms consistently.

5. **Decision vocabulary**  
   Reconcile the documented routing stages with the code-level decision branches and trace events. Keep one canonical vocabulary across training records, telemetry, tests, and public diagrams.

6. **RNV1 relationship**  
   If MoK is linked to RNV1, describe the relationship at a high level only. Preserve ownership and disclosure boundaries around private datasets, hardware integration, partner material, and core RNV1 research.

7. **Base-model roles**  
   Resolve the apparent difference between the current small coordinator default and older shared-base research candidates. Clearly distinguish the resident coordinator, helper experts, adapter base, and on-demand models.

## Research and evaluation queue

### Consultation quality

- Replace or validate the current finding-length heuristic used to choose between experts.
- Test disagreement handling when the more verbose expert is wrong.
- Measure whether challenge and follow-up turns improve correctness enough to justify their latency.
- Add adversarial cases for vague, malformed, copied, unsupported, and confidently incorrect expert replies.
- Verify that MoK synthesis never silently presents raw expert output as its own conclusion.

### Confidence and abstention

- Calibrate confidence thresholds on held-out tasks rather than intuition alone.
- Measure false-confidence and unnecessary-abstention rates separately.
- Test maximum-iteration behavior and explicit uncertainty reporting.
- Preserve the rule that unavailable resources must not become invented certainty.

### Routing promotion

- Keep R0 as the measurable heuristic baseline.
- Promote a learned router only when it wins on routing quality, regret, calibration, and important slices.
- Record route quality and resource cost together; a correct route that repeatedly causes OOM is not a successful route.
- Define regression gates before training R2 or later routing policies.

### Consumer-hardware validation

On the actual 16 GB target, record:

- cold and warm load latency
- observed peak VRAM
- adapter attach and detach time
- eviction correctness
- long-context behavior
- OOM recovery
- backend timeout and circuit-breaker behavior
- trace completeness after partial failures

### Trace and privacy review

- Freeze required trace fields before using traces as training data.
- Add schema-version handling and replay tests.
- Decide what prompt, expert-output, path, and user-context data must be redacted.
- Keep generated traces, private datasets, weights, and local model paths out of version control.

## Next-pass acceptance checklist

- [ ] README distinguishes the runnable slice from the full research program.
- [ ] Coordinator and consultation behavior are described only to the level verified by tests.
- [ ] Roadmap and kickoff documents are marked complete, active, deferred, or superseded.
- [ ] One vocabulary is used for routing, checking, gating, confidence, and synthesis.
- [ ] Learned components have written heuristic baselines and promotion gates.
- [ ] Real 16 GB hardware measurements are captured before performance claims are expanded.
- [ ] Public RNV1 references remain high-level and respect IP boundaries.
- [ ] GitReverse is rerun after documentation changes as a repository-intent regression check.

## Expected GitReverse signal after the next pass

A future reconstruction should still identify a local-first, VRAM-aware model kernel. It should additionally recognize that MoK is a resident coordinator that decides when to answer directly, retrieve memory, consult one or more experts, challenge weak replies, stop when confidence is sufficient, and synthesize the final response.

It should not imply that all learned routing, adapter training, memory prediction, or embodied-system work is finished.

## Review sources

- [GitReverse reconstruction](https://www.gitreverse.com/nawnie/Model-Operating-Kernel)
- [Repository README](https://github.com/nawnie/Model-Operating-Kernel/blob/main/README.md)
- [Project overview](https://github.com/nawnie/Model-Operating-Kernel/blob/main/docs/01-project-overview.md)
- [24-week roadmap](https://github.com/nawnie/Model-Operating-Kernel/blob/main/docs/05-roadmap-24-weeks.md)
- [14-day kickoff](https://github.com/nawnie/Model-Operating-Kernel/blob/main/docs/07-kickoff-14-days.md)
- [Consultation engine](https://github.com/nawnie/Model-Operating-Kernel/blob/main/src/mok/orchestration/consultation.py)
- [Decision loop](https://github.com/nawnie/Model-Operating-Kernel/blob/main/src/mok/routing/decision_loop.py)
