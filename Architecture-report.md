# Multi-Model Orchestration Architectures 

**Key idea:** Orchestration frameworks decompose a high-level task into subtasks, route each to an appropriate model or “agent,” pass state between them, and handle failures. In practice this involves four primitives: **decomposition** (breaking down the task), **routing** (selecting which model handles each subtask), **state management** (sharing context between steps), and **recovery** (error handling). For example, Augment’s *Multi-Agent Orchestration* guide emphasizes that a coordination layer *“decomposes complex tasks into subtasks, routes each subtask to a specialized agent, maintains shared state across agent boundaries, and recovers from failures at every handoff point”*. In practice, this often looks like a directed graph or pipeline of LLM calls (sometimes with non-LLM tools mixed in). The state can be managed via shared memory or “blackboard” structures, contextual tokens, or explicit state objects (e.g. LangGraph’s typed state schema). Empirical analyses show that **routing latency is cheap** (often <50ms) compared to LLM inference (seconds), so coordination overhead is typically dominated by model calls. 

In concrete systems, several patterns have emerged:  
- **Controller/Worker architecture:** e.g. *Agentic Lybic* (ArXiv ’23) uses a global “Controller” for overall planning, “Managers” for decomposition, specialized “Workers” for execution, and an “Evaluator” for quality assessment. Unlike static pipelines, Agentic Lybic can trigger adaptive re-planning if earlier outputs are poor.  
- **Graph-based routing:** frameworks like *LangGraph* represent the workflow as a graph where nodes are tasks and edges encode data flow. Routing functions (often simple predicates on the state) direct which node to execute next. LangGraph’s state graph example shows fields like `full_plan` (the decomposition) and `next` (the next subtask) that drive routing. CrewAI is another toolkit with similar orchestrator-worker patterns.  
- **Pattern-based coordination:** companies like Intent (“Context Engine”) and Anthropic advocate patterns like “orchestrator-workers” (hub-and-spoke) vs. peer-to-peer vs. hierarchical. For example, hub-and-spoke (central coordinator) ensures global state consistency but is a single point of failure; mesh (peer-to-peer) increases complexity but has no central bottleneck; hierarchical (multi-level) partitions context by sub-team. Hybrid strategies can adapt topologies per task (one benchmark found switching between parallel and hierarchical patterns gave ~23% gains).  

**State handoff:** Shared state between models is critical. Common approaches include blackboard/shared-memory (all models read/write a common workspace) or explicit message passing (each agent returns outputs that are fed into the next). LangGraph’s `StateGraph` provides typed fields (e.g. `full_plan`, `next`) so each model writes its results in a structured way. High-level workflows often require “living specifications” (durable artifacts, e.g. a file or database of the plan) so that even if the LLM context resets, the overall plan persists. In summary, current systems tend to use DAG or graph workflows with explicit data flow between LLMs, and emphasize logging the state transitions for traceability. (Failures are then recovered via retries or replanning – e.g., Anthropic suggests restarting the context with a fresh LLM agent if coherence is lost.)

**Existing systems:**  Many toolkits embody these ideas. In addition to LangGraph and CrewAI, others include AutoGen (from Microsoft Research), Reflexion, and custom orchestration layers built on LangChain or LlamaIndex. Recently Augment (2026) published a practical guide to multi-agent orchestration. The open-source *AdaptOrch* benchmark measured coordination overheads and validated hybrid topologies. 

# Mixture-of-Experts vs. Runtime Model Routing 

**In-model MoE (Mixture-of-Experts):**  A classic MoE model (e.g. a Switch Transformer) has *learned expert parameters* within a single model. At each layer or token, a gating network routes parts of the input to different expert sub-networks. This routing is fixed by the model’s training: adding a new expert requires retraining the model.  

**Runtime model routing:**  By contrast, runtime routing (sometimes called “model-of-models”) picks *entire models* at inference time. A separate router (could be a simple classifier or a small LLM) inspects the prompt and chooses which pretrained expert model (or ensemble of models) to invoke. This routing is **dynamic** and does not require retraining the models themselves. Because routing occurs at the granularity of whole calls (or sub-tasks), it’s more flexible: you can add new models or tools without modifying the router’s internal architecture. 

**Key differences:**  MoE’s gating inside a single model is optimized jointly during training (often via sparsely-activated gradients), whereas runtime routing treats each model as a black-box expert. MoE can achieve inference speedups and parameter efficiency in theory, but it requires specialized architectures and is typically server-only. Runtime routing can mix heterogeneous models (e.g. different sizes or modalities) as long as the router understands them. If the term *“MoK”* refers to any specific approach, we did not find a clear reference. In general, MoE (in-model) differs fundamentally from external routing: MoE’s experts share hidden representations and are co-trained, while runtime routing is a coordination of separately trained models. Agent-based routing (like the multi-agent orchestration above) is a form of runtime routing where each “expert” is effectively a specialized agent (possibly with tools). Tool orchestration (e.g. ReAct prompting) is another form: the router decides whether to invoke an LLM’s internal knowledge or an external tool for a given sub-task. In summary, runtime routing gives more flexibility at the cost of requiring a good router model or heuristic, whereas MoE relies on internal training to route optimally.  

# Shared-Base LoRA Serving 

Large models can be adapted to many tasks by learning *LoRA* (Low-Rank Adaptation) adapters. To serve many tasks cheaply, one can load *one base model* in GPU memory and dynamically attach different LoRA adapters per request. Recent systems demonstrate this is practical: e.g. HuggingFace’s TGI multi-LoRA and Facebook’s LoRAX all use one base with many adapters. The key challenge is that naive quantization often prevents weight-sharing, but specialized solutions exist:

- **Specialized quantization:** The LoRA-Inlaid method (NeurIPS 2024) uses a custom GPTQ algorithm (MLGPTQ) so that *one* quantized base model can support many LoRA adapters. This lets multiple low-rank adapters share the same compressed weight bank, greatly reducing GPU memory. 

- **Multi-tenant serving:** Systems like *Punica* introduce custom CUDA kernels so a GPU holds one copy of the base weights while applying many LoRA deltas per batch. Punica reports up to 12× throughput gains and only ~2 ms overhead relative to a single LoRA, by batching the LoRA updates efficiently. Hugging Face’s blog on “Multi-LoRA” likewise notes that you “deploy the base model once and load many adapters… effectively like having multiple fine-tuned models in one”. In practice, one keeps only the most-used adapters in GPU memory and offloads the rest to CPU RAM (LRU eviction), as in Spheron’s architecture.  

- **Tool support:** Frameworks such as vLLM and LoRAX explicitly support multi-LoRA serving. For example, LoRAX can load adapters “on-the-fly without blocking concurrent requests” and will pack requests for different adapters into mixed batches to maintain throughput. In short, multi-adapter serving is not only viable, it’s actively used in production. The memory cost per adapter is very low (e.g. ~60 MB for a rank-16 adapter on an 8B model), so one can host dozens. The consensus is that with the right infrastructure (custom quantization and batching kernels), a single quantized base model can efficiently serve many LoRA adapters.

# Adapter Hot-Swapping (vLLM / LoRAX) 

When dynamically loading LoRA adapters at runtime, there is some latency and overhead:

- **Attach latency:** Both vLLM and LoRAX perform just-in-time loading of adapter weights. This means the *first request* that uses a newly loaded adapter will be slower, since the system must copy the adapter to GPU and compile any necessary kernels. In vLLM, users have observed that the first LoRA call can drop throughput dramatically (e.g. 27 tok/s → 5 tok/s) until a “warm-up” run is done. The vLLM team notes explicitly: “The slow first request… is expected. The initial request triggers loading and optimization of the LoRA weights… Subsequent requests are much faster because the adapter is already loaded and optimized”. The remedy is simply to warm-up each adapter (e.g. a dummy inference) before actual use.  

- **Concurrency:** Both vLLM and LoRAX support concurrent inference with multiple adapters. vLLM’s batching logic can merge requests using different LoRAs into one batch (as long as `--max-loras` is set high). This incurs only a small overhead compared to homogenous batches: throughput/latency are “slightly worse” when mixing adapters, but much better than running separate model instances. LoRAX similarly uses *heterogeneous batching* to pack multi-adapter requests, keeping throughput nearly constant as more adapters are active. In short, a multi-tenant setup (many LoRAs) is well-supported: both systems can interleave different adapters in GPU without blocking each other.  

- **VRAM overhead:** The GPU memory is dominated by the base model and any KV cache, not the adapters. As one example, serving 8 LoRA adapters of rank-16 alongside an Llama-3.1-8B base (16 GB FP16) consumed only ~0.5 GB more on GPU, with the other 92 adapters kept in CPU RAM. In general, each rank-16 adapter adds only ~60 MB, so a 16 GB GPU can easily hold dozens before the base model is full. Both vLLM and LoRAX support evicting least-used adapters to CPU if needed. The empirical guidance is to set a `--max-loras` and `--max-cpu-loras` based on adapter rank (e.g. rank 16 = ~60 MB). 

- **Failure modes:** If the GPU runs out of memory, these systems will typically raise an OOM error or fail to load a new adapter. In practice one prevents this by tuning `--max-loras` and `--max-cpu-loras` or by offloading adapters to CPU. Neither vLLM nor LoRAX will silently drop requests – the adapter must be present (or error reported). One should design a scheduler to evict infrequently used adapters or increase hardware if eviction is too slow. In summary, dynamic LoRA hot-swapping is well-engineered: first-call latency is expected but short-lived; concurrency is supported with minor overhead; and VRAM costs are modest thanks to offloading.

# Base Model Bake-Off (on 16 GB hardware) 

Recent benchmarks compare Qwen-2.5-7B-Instruct, Llama-3.1-8B-Instruct, and Mistral-7B-v0.3 (all quantized) on generation tasks. AscentCore (Apr 2026) found that **Mistral-7B** achieves the highest raw quality (e.g. ROUGE) on diverse tasks, **Qwen-2.5-7B** offers the best quality-to-speed ratio, and **Llama-3.1-8B** has the strongest adherence to structured outputs (e.g. JSON schema). For example, they report Mistral topping the ROUGE-L scores, Qwen the fastest tokens/sec per quality, and Llama the highest schema-compliance rate. In terms of throughput, both Qwen-2.5-7B and Llama-3.1-8B achieve similar speed once quantized (both run at tens to hundreds of tokens/sec on a 16 GB GPU).  

**Context length:** Qwen-2.5-7B-Instruct supports up to 131,072 tokens (with ~8K output). Meta’s Llama-3.1-8B is also advertised for 128K context (although leveraging it requires correct RoPE implementation). By contrast, Mistral-7B uses a **sliding-window attention** (4K windows) to reach an “up to 32K” context; in practice it focuses on the most recent ~4K tokens. Thus, for extremely long-session use (>>8K tokens), Qwen or Llama-3.1 are better suited.  

**Resource fit:** All three 7–8B models fit on a single 16 GB GPU when quantized to 8-bit or 4-bit precision. Qwen-2.5-7B (FP16) is ~5–6 GB of weights, Llama-3.1-8B ~7–8 GB (FP16), Mistral-7B (FP16) ~7 GB. In practice, with careful offloading of non-critical buffers, all can run inference with up to an 8K or higher context on a 16 GB card. Spheron’s guide notes that, as a rule of thumb, one should budget ~1.3×–1.5× the model weight size for a moderate-context deployment. For example, a 16 GB base plus 4 GB of KV cache and overhead is ~20 GB; on 16 GB hardware one may need context reduction or offloading. In summary, Mistral-7B wins on quality, Qwen-7B on efficiency, Llama-8B on reliability, but all are feasible choices on 16 GB GPUs depending on the task.

# Router Model Design 

A router is typically a lightweight model that classifies each query into one of several target models (or a probability distribution over them). Key considerations include training signal and confidence. Some approaches train routers via supervised learning on labeled query-model pairs, others via contrastive or RL objectives.

- **Contrastive routing (cost-aware):** Chen *et al.* (2024) introduced a *Cost-Sensitive Contrastive Routing* (CSCR) method. They train an encoder to map prompts (or “logit footprints”) into an embedding space where proximity predicts which model will give high-quality output. Crucially, they explicitly incorporate model inference cost: the contrastive loss is weighted so that cheaper models are favored if they provide sufficient quality. This aligns the router’s decisions with end objectives (quality vs. latency). 

- **Surrogate/threshold gating:** Another paradigm is to train a small surrogate for the expensive model and use a *confidence threshold* to decide when to call the LLM. The TracER system (ICLR 2024) does this for classification: it records LLM outputs on many examples, trains a tiny surrogate model on those “traces,” and at inference time only invokes the LLM if the surrogate’s predicted label agrees with it with high confidence. In their words, “the surrogate is activated only when its agreement with the LLM exceeds a user-specified threshold”. This reduces calls without sacrificing accuracy. More generally, one can calibrate any router’s confidence scores (e.g. by temperature scaling) and set a rejection threshold so that only highly certain routes are taken. 

- **Classifier architecture:** Routers are often very small (a few MLP layers) because the input (query features or embeddings) is low-dimensional. They may be **multi-label** if a query could be handled by multiple models simultaneously (e.g. an ensemble), or multi-class if exclusive. In practice, most work treats it as multi-class (choose one model). Embedding-based methods (e.g. LLM embeddings of the query fed into an MLP) are common. Calibration (via Platt scaling or isotonic regression) can ensure the model’s confidence matches true accuracy. Cost-awareness can be built in by subtracting a cost term from a model’s score (essentially a utility = quality – λ·cost function) during selection.  

- **Empirical metrics:** It’s important to evaluate a router by **regret** (how much worse it does compared to an oracle who always picks the best model). Wen *et al.* (2025) frame routing as “minimizing decision-making regret” and learn a policy directly to do so. They show that optimizing regret (rather than separate accuracy and cost regressors) yields better end-to-end decisions. In practice, one often computes **oracle routing accuracy** (the fraction of times the router picks the same model that an “oracle” would) and **utility regret** (difference in utility). Calibration can be measured via expected calibration error (ECE) on held-out data. Mixed-task benchmarks (see Evaluation below) can test router robustness across domains.  

In summary, router design often balances simplicity with expressiveness: small classifiers on query features, potentially augmented with embedding encoders. Techniques like contrastive learning allow routers to encode cost-quality tradeoffs, while gating strategies ensure LLMs are only used when needed (reducing cost). 

# Offline Bandits / RL for Routing 

Rather than training a router in a purely supervised way, one can view the model-selection problem as a contextual bandit or RL: each query is a “context,” the chosen model is an “action,” and a reward reflects the quality-versus-cost outcome. **Offline** (logged) bandit learning can use historical data of (query, model used, outcome) to learn an improved policy. 

One sophisticated recent approach is *Rollout Routing Replay (R3)*. This was proposed in the context of MoE model training, but the idea is applicable: it records actual routing decisions (a distribution over experts) from inference and “replays” them during training so that the training-time router matches inference-time behavior. In MoE RL, R3 was shown to stabilize training by aligning training and inference routing distributions. For a simple routing policy, one can also apply counterfactual policy evaluation or inverse propensity scoring on logged data to estimate the value of alternative models. However, these methods typically require large volumes of logged data (each query evaluated by multiple models or with known selection probabilities) to avoid bias. 

**Practicality:** The question is whether such complexity is worth it. If you have a rich log of past queries and model outcomes, offline bandit methods (like learning a weighted classification to minimize regret) can yield better routers. But if data is scarce, simpler supervised or contrastive learning (as above) may suffice. In short, R3-like techniques can improve a router if you have the data and ML expertise, but they are complex and data-hungry. Without that, using a well-calibrated classifier or heuristic (e.g. prompt style or token length) is often adequate. 

# Coordinator Model Design 

Beyond routing, a *coordinator* sometimes refers to a model or policy that assembles the overall plan (ordering of subtasks) and merges outputs. Recent work demonstrates the power of a small coordinator with a structured output format. For example, the **Trinity** system (ArXiv ’25) uses a 0.6B “coordinator” LLM with a tiny (∼20K-parameter) output head to orchestrate multiple LLM agents. At each turn, Trinity encodes the conversation and hidden-state signals, and its head selects *which* agent to call and *what role* (Thinker, Worker, Verifier) it should play. Key insights from Trinity:  

- **Hidden-state encoding:** Even a compact LLM’s hidden representations can capture rich context. Trinity uses a lightweight head on top of hidden states (pre- and post-generation) to make routing decisions. This suggests one can “compress” information into hidden vectors and let a small head decode the plan.  

- **Structured roles:** Trinity’s coordinator outputs are structured as (agent, role) tokens, ensuring valid plans. The roles (e.g. “Think”, “Work”, “Verify”) break tasks into stages. Enforcing such structure (via output constraints or a carefully designed prompt) helps keep plans stable during training and inference.  

- **Training method:** Training the coordinator is nontrivial because its action space is high-dimensional and each evaluation requires running multiple models. Trinity found that **evolutionary strategies** (specifically separable CMA-ES) worked better than policy-gradient RL or imitation learning in their limited-budget setting. In other words, they treated it as a black-box optimization and got state-of-art results.  

From these insights, a practical approach is: use a small LLM (sub-1B parameters) as coordinator, train it to output valid routes (perhaps in JSON or another schema), and consider derivative-free or offline methods if RL gradients are too noisy. Providing structured supervision (via examples or rules) is crucial to “teach” the coordinator what constitutes a valid plan.  

# Route Schema Design 

Defining a stable routing schema (e.g. `mok.route.v1`) is essential so that training data, routers, and evaluation all agree on the format. While we did not find a public reference for `mok.route.v1`, general best practices are:

- Use a **versioned, explicit format** (e.g. a JSON schema with a `"version": "mok.route.v1"` field).  
- Include fields like `query_id`, `timestamp`, `selected_model`, `model_confidence` (if applicable), and any routing rationale or metadata. For multi-step plans, include an array of steps with (model, input, output).  
- Make sure the schema is *forward-compatible*: if you add new fields later (e.g. context embeddings or cost estimates), old training code should simply ignore unknown fields.  
- Validate at inference time that the router’s output conforms (e.g. using JSON schema tools), to catch errors early.  

Because this is largely a design choice, no specific citation applies. The key is consistency across training and inference – once you define `mok.route.v1`, do not change its semantics (only patch carefully).

# Expert ABI / Backend Contract 

An “expert ABI” is a uniform interface for calling any model (local, HTTP API, adapter, multimodal). The goal is to treat all experts interchangeably. A practical design is to mimic well-known LLM inference APIs (like OpenAI’s) but generalized:

- **Function signature:**  e.g. `generate(model_name, prompt, parameters) → {text, tokens, metadata}`. Here `model_name` could encode adapter or variant. `parameters` include things like `temperature`, `max_tokens`, etc. The return should include the generated text, logprobs or token stream, and timing info. 

- **Local models:**  A local expert (e.g. a HuggingFace pipeline or Triton-served model) implements this function call, perhaps via an in-process call or via a local HTTP server. Tools like vLLM already implement an OpenAI-compatible `/v1/completions` endpoint for local models, which can serve adapters via an extra parameter. 

- **HTTP models:**  For remote models (e.g. a hosted API), you can wrap them so that calling `generate("modelX", ...)` under the hood sends an HTTP request. The unified interface means the router code doesn’t care if it’s local or remote. 

- **Multimodal experts:**  For vision or audio experts, extend the interface with appropriate fields. For instance, `generate("vision-model", {image: <binary>})` or a specialized `classify(model, image)` call that returns labels. The ABI should allow passing non-text modalities (e.g. images as byte arrays) alongside text. 

In effect, the cleanest design is a minimal adapter pattern: each expert (LLM or tool) implements the same method signature and returns outputs in a consistent JSON format. Systems like MLflow or TorchServe do similar model-as-a-service wrapping. Unfortunately, we found no single published standard. Most engineering teams simply adopt or adapt existing APIs (e.g. huggingface transformers’ `model.generate()`, OpenAI’s API, or Triton’s HTTP interface) and enforce a common input/output schema in their router code.

# VRAM Estimation & Memory Scheduling 

Accurately predicting peak GPU memory usage is crucial. VRAM usage during inference breaks down into four main components:

1. **Model weights:** Linear in parameter count and precision (e.g. 8B params × 2 bytes = 16 GB for FP16).  
2. **KV cache:** Grows with context length and batch size. Formula: *cache_per_token = 2 × (num_layers) × (KV_heads) × (head_dim) × (bytes)*. For example, Llama-3.1-70B (GQA with 8 KV heads) uses ~0.31 MB per token in BF16; at 128K tokens that’s ~40 GB of KV data. Even smaller models use a few MB per token.  
3. **Activation memory:** Temporary scratch space during generation. Typically ~5–10% of total for inference.  
4. **Framework overhead:** CUDA context, buffers, misc (often 0.5–2 GB). Also fragmentation overhead (~10–15%) should be budgeted.

A practical rule (from Spheron) is to multiply the **model weight size** by ~1.3–1.5× for moderate concurrency and context, and by ~1.5–2× for heavy use or very long context. For example, a 16 GB FP16 model with modest batch might require ~20–24 GB total in inference (some of which can be offloaded). 

**Context scheduling:** Tools like vLLM’s *PagedAttention* reduce waste by allocating KV blocks on demand. By default, most frameworks pre-allocate max-length buffers (wasting 60–80%). PagedAttention instead creates KV pages as tokens are generated, allowing 2–4× more concurrent requests on a given GPU. For long-context runs, you may also quantize the KV cache (FP16→FP8) to halve its size. 

**Memory planning checklist:** Account for *all* components. Spheron summarizes:  
- Model weights (params×bytes).  
- KV cache per-token × max context × concurrent requests.  
- Activations (~5–10%).  
- Framework overhead and allocator fragmentation (~10%).  
- Headroom for bursts: use the higher end of estimates.  

If this sum exceeds GPU VRAM, options include: reducing batch/context length, using tensor parallelism (split model across GPUs), or CPU offloading (slower, typically only for dev/debug). In production, one would tune these parameters or invest in a larger GPU.

# Learned Cache Eviction / Prefetch 

Managing the KV cache efficiently can greatly extend context and concurrency. Beyond heuristics, **learned caching** approaches have recently emerged:

- **KV eviction as RL:** The KVP (KV Cache Eviction) framework (ArXiv ’26) formulates eviction as a reinforcement-learning ranking problem. A lightweight RL agent learns to score each token in the KV cache by predicted future utility. Tokens with lowest scores are evicted when memory is needed. Crucially, KVP uses a global “eviction error” reward that makes its scoring robust across all cache budgets. In practice, this means the policy learns which historical tokens are least likely to be needed. They train a distinct policy per attention head, and the result is a fine-grained eviction order that “minimizes degradation” under any budget. This data-driven eviction can adapt to actual usage patterns better than simple recency-based heuristics. 

- **Prefix caching (LPC):** For conversational LLMs, one often repeats prefixes of text. Learned Prefix Caching (NeurIPS 2025) trains a tiny model to predict whether a new prompt continues the last conversation. It then caches KV blocks of the prefix between sessions. LPC achieved much higher cache hit rates than LRU: up to 47% reduction in required cache size for the same hit rate. This speeds up “prefilling” for ongoing dialogs and cuts first-token latency by 42–75% on some workloads.  

- **Other memory hierarchy:** Some systems (IceCache, MagicPig, etc.) implement multi-tier cache: they offload less-used KV entries to CPU or disk. A learned eviction score (like KVP’s lowest-ranked tokens) provides a natural signal for *which* entries to spill to slower memory. Prefetching (loading likely-needed blocks in advance) is less studied, but one could imagine a predictor that, given the current buffer, preloads the next KV page if the sequence is expected to continue. 

In practice, these are active research areas. A production router might log KV usage patterns (e.g. token timestamp of last use) to train or tune such policies. At minimum, enabling PagedAttention (as above) and carefully setting max cache pages are common optimizations. 

# Trace Logging Schema 

Effective routing and policy learning requires rich logs. A recommended logging schema includes:

- **Prompt and metadata:** The input prompt (or its embedding), a timestamp, session ID, and any context info.  
- **Router decision:** Which model(s)/adapter(s) were selected, with confidence scores or probability.  
- **Model invocation:** For each step/agent call, log the model name, input text, and raw output text. Also log resource metrics (latency, tokens generated).  
- **Outcome:** If there’s a downstream label or success criteria, record it (e.g. human rating or automated metric). For tool use or code, log success/failure flags.  
- **State changes:** If using a stateful orchestrator, log any updates to the shared state after each step.  
- **Error events:** Any failures (timeout, OOM, hallucination detected) should be logged with details. 

This enables training of routers/coordinators: you have (prompt → model → result) tuples plus observed reward. For memory policies, one could log KV cache entries and when they were evicted or reused. (For example, KVP logs token access patterns as “experience” for the RL agent.) In short, collect data at every decision point. While no specific standard was found, many systems adapt logging from ML frameworks (e.g. TensorBoard-like tables) or ELK/Graylog pipelines.  

# Evaluation Design 

To evaluate the routing architecture end-to-end, one should use a mix of synthetic and real tasks with known “oracles”:

- **Oracle routing:** Compute the *oracle utility* by running all candidate models on each prompt and picking the best answer post-hoc. Then measure the router’s **regret** (how much utility it lost relative to oracle) or **oracle accuracy** (fraction of times it matched the oracle model). Wen *et al.* explicitly learn by minimizing regret. Regret metrics (absolute or relative) are very informative for routing. 

- **Calibration metrics:** For probabilistic routers, measure calibration (e.g. expected calibration error) so that confidence outputs match actual success rates. Also measure “top-k” accuracy if the router can propose k models.  

- **Mixed-task benchmarks:** Construct a benchmark that mixes tasks of different modalities or domains (e.g. Q&A, summarization, math, coding, RAG). This tests whether the router generalizes. Use standardized datasets for each task. Evaluate end-to-end success (e.g. accuracy on math problems, correctness of code outputs) under a fixed compute budget. 

- **Ablations:** Compare against simple baselines: single large model, static model for each task type, or naive heuristics (e.g. choose model by prompt keywords). Also try **full-ensemble** or **chain-of-choice** approaches.  

- **Metrics:** In addition to quality/regret, report latency and cost (since routing trades cost vs quality). Calibration (probability vs outcome) and adherence to format (e.g. valid JSON if needed) are also relevant. If multiple models can contribute (ensembles), measure any blending errors.  

- **Mixed-agent flow:** Evaluate multi-turn flows with the coordinator as well. For example, provide a complex query that requires multiple steps; measure whether the orchestration yields a correct composite answer.  

Overall, the literature suggests focusing on **regret-minimization** as the core metric, supplemented by typical NLP benchmarks and system-level throughput.  

# Verifier Hierarchy 

Different output types need different verification. A sensible hierarchy is:  

- **Code:** Automatically compile/run generated code on test inputs. Use code coverage or unit tests to verify correctness. Tools like `pytest` or custom test suites can label success/failure. Static analysis (e.g. linting) may catch errors too.  
- **Math:** Use symbolic computation (Sympy) to check equations or numerical solvers for numerical answers. If multiple solutions, cross-validate.  
- **Retrieval (RAG):** Check factual claims against the source documents (via information retrieval) or known knowledge graphs. For example, verify named entities or dates via Wikipedia.  
- **Tools:** If the agent used tools (APIs, calculators), log the tool outputs and confirm they satisfy constraints (e.g. if it did a web search, check the link’s content).  
- **Multimodal:** For image inputs/outputs, use vision models as verifiers (CLIP for caption matching, OCR for text in images, etc.). Metrics like FID/BLEU are less interpretable here, so manual inspection or task-specific checks are often needed.  

The hierarchy means simpler “verifiers” (like unit tests) run first, and more costly checks (like human review) only if needed. There is no off-the-shelf source for this; it’s largely application-specific. The key is to label outputs as “correct/incorrect/unknown” so that routing or coordinator policies can be trained on quality feedback.

# Dataset Licensing & Decontamination 

**Licensing:**  Only use data with permissive licenses. For training any components, prefer open datasets (CC-BY, Apache 2.0, CC0). Avoid copyrighted text (non-BY) unless you have permission. For retrieval or RAG, use licensed corpora (Wikipedia, CCWeb, CC-News, etc.). Check licenses of benchmark datasets (e.g. use the vetted versions of common NLP benchmarks to avoid IP issues). 

**Decontamination:** Prevent benchmark leakage by filtering. Before training any models or adapters, remove any examples that overlap with evaluation prompts. If evaluation sets contain known copyrighted or proprietary content, ensure it wasn’t in your training corpora. Use tools like n-gram exact matching or paraphrase detection to flag contamination. Many teams remove common NLP benchmark questions from pretraining data.  

**Reproducibility:** Document data sources and versions. Use train/validation/test splits in a standard way. When releasing results, specify random seeds and computation environment. Ideally, containerize the serving code. 

There is no single citation for these best practices, but the community has emphasized them (e.g., see GPT-4o release notes on licensing, or BigScience on data transparency). The summary is: only use allowed data, exclude overlaps with eval, and clearly document everything for reproducibility.

# Adapter Capacity Ceilings 

Not all tasks can be solved by a LoRA adapter on a shared base. In practice:  
- **Where LoRA suffices:** If the base model already has the general capability and you just need to tweak it (e.g. domain shift, writing style, small factual update), LoRA works well. For moderately complex tasks (e.g. Q&A on a specific dataset, nerual style fine-tuning), low-rank adapters can approximate the needed changes.  
- **Where full experts are needed:** If the task is substantially different (e.g. a new modality like vision, or completely new knowledge domain, or complex reasoning tools), a full model or a larger adaptation may be required. There is no sharp cutoff, but empirically one finds diminishing returns on increasing LoRA rank beyond a point. If your adapter needs rank >64 or similar to achieve acceptable accuracy, it may be simpler to use a dedicated model or fine-tune the base fully. 

No academic source specifies exact limits, but engineers often test by increasing adapter rank until performance plateaus. For example, an 8B model plus a rank-32 adapter might work for medical Q&A, but a highly specialized legal reasoning task might still need its own 7B model.

# Vision Expert Strategy 

If vision processing is required and your text base model cannot handle images, a common approach is *on-demand multimodal routing*: when a vision-related query arrives, route it to a separate vision-enabled model (expert) before or in conjunction with the text model. For instance:  

- **Image-to-text step:** Use a vision model (e.g. Vision Transformer or CLIP) to convert the image into a text description or embeddings. Then feed those text embeddings into the text LLM. This keeps the text base untouched.  
- **Fusion model:** If available, call a pretrained vision-language model (like BLIP or CLIP-guided LLM) as a special expert. This could be an HTTP API or a local model. For multimodal interaction, route to it first to get a preliminary answer or context.  
- **Late fusion:** Another pattern is to process image and text in parallel: one expert handles vision, another handles text, then the coordinator merges their outputs.  

The best path depends on requirements. If vision truly stays outside the shared text base, then building an API layer is simplest: e.g. “if `query.image != null`, call `vision-model`, take its output string, then continue with text-model.” The actual expert might be a CLIP (for simple recognition) or a generative vision-language model.  

No single answer or source exists, but experience suggests using specialized, pre-trained VLMs for images (and keeping them as separate experts) is the practical solution. They typically return either textual descriptions or structured features that the text models can then consume.

# Long-Context Runtime Behavior 

On 16 GB GPUs, extremely long sessions stress the KV cache. Empirically, the first thing to break is the **KV cache capacity**. If context and batch are too large, you will hit GPU OOM. Without fancy on-the-fly compression, most frameworks will allocate KV buffers for max length (unless using vLLM’s PagedAttention). In that case, the default behavior is either an OOM error or silent performance collapse. 

Beyond raw memory, fragmentation can occur: holding many distinct context windows fragments the cache (though modern allocators manage it fairly well). Throughput will drop as more tokens accumulate (since cache grows each step). Models may also degrade in quality if the context window exceeds the model’s effective range (e.g. Llama’s RoPE issues beyond 32K). 

In practice, on 16 GB hardware one typically hits memory limits before correctness issues. If you try to maintain a long conversation (>8K) with an 8B model, you’ll likely exceed 16 GB unless using 4-bit quantization or paging. The safe strategy is to implement **context truncation or summarization**: when hitting the limit, summarize the oldest tokens (or flush them entirely). Anthropic’s guidance is to “reset context with a fresh agent” if coherence is lost. Logging the drop in throughput or seeing CUDA OOM errors would be expected failure modes under overload.  

# Related Systems and Projects 

Several adjacent projects address pieces of this pipeline:

- **Agent runtimes:** LangChain, BabyAGI, AutoGPT, Reflexion, etc. These are high-level frameworks for building multi-step LLM applications with tool use and memory. They embody many orchestration patterns (though often in Python scripts rather than training small routers).  
- **Routing research:** In addition to CSCR and R3, there are causal inference approaches to routing from observational data, and systems like AutoRouter (Google AI) that learn cheapest-accurate models. This fits into the bandit/regret literature.  
- **Memory-managed inference:** vLLM (Stanford) and Transformer Engine (NVIDIA) implement PagedAttention and KV quantization for large contexts. LoRAX and Punica focus on multi-LoRA serving. FlashAttention3 (NVIDIA) handles longer contexts via custom kernels.  
- **Model registries:** Hugging Face Hub, MLflow Model Registry, and ONNX model zoo allow versioning and loading models/adapters. Serving stacks like BentoML provide pluggable backends.  
- **Similar orchestration frameworks:** As noted, *LangGraph* (by Intent) and *CrewAI* provide graph-based routing and state management. Anthropics’ research also discusses agentic patterns (see their “building agentic AI” guide). The AdaptOrch benchmark and AgentOrchestra paper explore different topologies. Intent’s Context Engine (VS Code extension) is an example of large-scale code/task orchestration.  

In summary, while no single off-the-shelf product covers all needs, the pieces exist: orchestration guides, routing libraries, LoRA-serving systems, and training methods for caches and policies. Integrating these into a unified architecture is an active area of engineering and research. 

**Sources:** We drew on recent literature and engineering reports for each topic. For multi-agent patterns and routing primitives, see Galstian (2026). For multi-LoRA serving and adapter behaviors, see the Spheron blog and vLLM documentation. The base-model benchmarks come from AscentCore (April 2026). Router and RL techniques are from recent ArXiv papers. Learned caching approaches are from KVP (ArXiv 2602.10238) and LPC (OpenReview 2025). Additional context (e.g. VRAM formulas) is from Spheron’s engineering posts.