# 24-Week Roadmap

## Weeks 1-4: Runtime truth first

### E1

- implement backends
- implement the end-to-end runtime loop
- add example expert config
- get tests green around the mock path

Gate:

- the documented loop returns the expected response shape end-to-end

### E2

- wire two real experts
- land trace logging for every request
- compact logs into queryable local artifacts

Gate:

- real experts answer through the loop
- traces are queryable in DuckDB or equivalent local analysis flow

## Weeks 5-10: Routing and first base decisions

### T1a

- ship `R0` rules routing
- stand up the oracle route-eval harness
- measure first regret numbers

Gate:

- routing baseline is measured, not guessed

### T2a

- run the shared-base bake-off
- choose the base model
- train coder and instruct first
- benchmark LoRA hot-swap behavior on the real target

Gate:

- base choice is written down
- adapter attach behavior is measured on the target runtime

### T1b

- train and calibrate `R2`
- deploy it in shadow mode against `R0`

Gate:

- `R2` beats `R0` on F1 and regret
- calibration is acceptable

## Weeks 11-16: Cost and coordination

### E3

- add install-time VRAM profiling
- feed the cost model into budget decisions
- add long-context stress coverage

Gate:

- long-context scenarios avoid OOM
- predicted vs observed peak memory is close enough to trust

### T3

- freeze the route schema
- generate verified teacher traces
- train coordinator with SFT then DPO
- harden constrained decoding

Gate:

- coordinator outputs are valid
- merged answers beat or complement solo-model baselines

## Weeks 17-20: Memory policy and cost-aware routing

### T4

- add next-expert prediction
- add prefetch simulation and replay evaluation
- replace FIFO only if replay and real-session results support it

Gate:

- cold-miss rate drops on realistic sessions

### T5

- train the `R3` cost-aware route policy
- use off-policy evaluation before promotion

Gate:

- measured reward win over `R2`
- no slice regression

## Weeks 21-24: Full system proof

### E4/E5

- add vision as an on-demand full model
- complete the initial expert roster
- run the full benchmark and ablation report

Gate:

- MoK beats simple baselines end-to-end
- status of the main hypotheses is documented clearly

## Hard sequencing rule

Do not move a training phase ahead of its trace-producing engineering dependency. Do not declare a training phase complete unless it beats the heuristic it replaced.
