# 14-Day Kickoff

## Day 1-2

- implement `src/mok/models/backends.py`
- add `MockBackend`
- add an `HTTPBackend` stub

Done when:

- backend unit tests pass

## Day 3

- implement `src/mok/orchestration/runtime.py`

Done when:

- the documented prompt -> route -> load -> answer loop runs end-to-end

## Day 4

- write `configs/example_experts.json`
- finish the first runtime test coverage pass

Done when:

- the E1 definition of done is green

## Day 5

- add `src/mok/telemetry/events.py`
- wire trace emission into every request

Done when:

- each request emits one valid structured trace

## Day 6

- add `R0` rules routing in `src/mok/routing/router.py`

Done when:

- example expert routes behave correctly for keyword, regex, and modality cases

## Day 7

- build the oracle-eval harness skeleton

Done when:

- the harness runs against mocks and reports regret sanity checks

## Day 8

- run the shared-base bake-off
- write the decision memo

Done when:

- one base model is chosen and justified

## Day 9

- benchmark LoRA hot-swap on the chosen base

Done when:

- attach latency and concurrency behavior are measured on the target runtime

## Day 10

- build the routing dataset script

Done when:

- cold-start data splits exist and are documented

## Day 11

- launch the first coder-adapter QLoRA run
- generate teacher-labeled route data

Done when:

- training is active and the initial label corpus is landing

## Day 12

- write the `R2` training script end-to-end

Done when:

- it trains on the cold-start dataset without shape or export failures

## Day 13

- train and calibrate `R2`
- run the first real route benchmark

Done when:

- first learned-router F1 and regret numbers exist

## Day 14

- evaluate coder adapter vs base
- write up week-two results
- revise dates against reality

Done when:

- there is a clear go/no-go readout for the first adapter lane and the roadmap has been adjusted honestly
