"""
Research taxonomy: Problem -> Hypotheses -> Search queries.

Every search query in this project MUST be traceable to a problem and a
hypothesis (principle #3/#5). Queries are tagged with `kind`:
  support   : looks for evidence that the hypothesis works
  refute    : looks for failure cases / negative results / ablations (principle #6)
  adjacent  : cross-field / under-explored principle (principle #8)

The harvester (research/harvest.py) iterates this structure. Adding a new
hypothesis = appending to this file; the DB dedups so re-running is cheap.
"""

TAXONOMY = {
    # ------------------------------------------------------------------ P1
    "P1_param_efficiency": {
        "problem": "Maximize capability per stored parameter at a fixed ~9B budget.",
        "hypotheses": {
            "H1.1_recurrent_depth_sharing": {
                "claim": "Weight-shared recurrent blocks iterated K times give the capability of a deeper model at fixed params.",
                "support": ["universal transformer recurrent depth", "looped transformer reasoning",
                            "recurrent depth latent reasoning test-time compute", "parameter sharing transformer layers",
                            "relaxed recursive transformers layer-wise LoRA", "ALBERT cross-layer parameter sharing",
                            "mixture of recursions adaptive token depth"],
                "refute": ["parameter sharing transformer performance degradation", "looped transformer length generalization failure",
                           "weight tying layers scaling law limitations"],
            },
            "H1.2_low_rank_tensor": {
                "claim": "Low-rank / tensor factorized weights preserve quality at lower param count.",
                "support": ["low-rank pretraining language model", "tensor train decomposition transformer",
                            "GaLore gradient low-rank projection", "monarch matrices structured", "kronecker factorized layers language model"],
                "refute": ["low-rank pretraining fails to match full rank", "structured matrices language model quality gap"],
            },
            "H1.3_sparse_moe": {
                "claim": "Sparse MoE gives more knowledge capacity per FLOP; with total param cap, fine-grained/shared experts matter.",
                "support": ["fine-grained mixture of experts scaling law", "shared experts DeepSeekMoE",
                            "mixture of experts parameter efficient upcycling", "expert merging compression MoE",
                            "mixture of a million experts product key"],
                "refute": ["mixture of experts dense comparison same total parameters", "MoE training instability router collapse",
                           "MoE reasoning underperforms dense"],
            },
            "H1.4_memory_layers": {
                "claim": "Product-key / memory layers store facts cheaply (lookups, not FLOPs).",
                "support": ["product key memory layers scale", "memory layers at scale", "large memory layers language model knowledge",
                            "ultra sparse memory network"],
                "refute": ["product key memory underutilization", "memory layer training difficulty"],
            },
        },
    },
    # ------------------------------------------------------------------ P2
    "P2_conditional_compute": {
        "problem": "Spend FLOPs where tokens need them (dynamic depth/width/routing).",
        "hypotheses": {
            "H2.1_mixture_of_depths": {
                "claim": "Per-token routing around blocks reduces FLOPs without loss.",
                "support": ["mixture of depths dynamically allocating compute", "early exit language model",
                            "adaptive computation time transformer", "token skipping layer skip LLM", "CALM confident adaptive language modeling"],
                "refute": ["mixture of depths autoregressive causality problem", "early exit degrades quality calibration",
                           "dynamic depth hardware inefficiency batching"],
            },
            "H2.2_learned_budget": {
                "claim": "A learned halting / budget controller allocates sequence-level compute.",
                "support": ["learned halting ponder net", "adaptive compute budget reasoning LLM", "difficulty-aware compute allocation reasoning"],
                "refute": ["ponder cost instability halting collapse", "overthinking reasoning models"],
            },
            "H2.3_activation_sparsity": {
                "claim": "Activation sparsity (ReLU^2, top-k) yields cheap inference.",
                "support": ["activation sparsity LLM relu", "deja vu contextual sparsity", "powerinfer sparse activation", "squared relu primer"],
                "refute": ["activation sparsity quality loss swiglu", "contextual sparsity predictor overhead"],
            },
        },
    },
    # ------------------------------------------------------------------ P3
    "P3_sequence_mixing": {
        "problem": "Long context at sub-quadratic cost without losing recall.",
        "hypotheses": {
            "H3.1_hybrid_ssm_attention": {
                "claim": "Mostly SSM/linear layers + few attention layers match transformers at lower cost.",
                "support": ["hybrid mamba attention language model", "jamba hybrid", "samba hybrid state space", "zamba shared attention",
                            "griffin gated linear recurrence local attention", "hymba hybrid heads", "mamba-2 state space duality"],
                "refute": ["state space models recall limitation", "repeat after me transformers better than state space copying",
                           "mamba in-context learning weakness", "linear attention associative recall gap"],
            },
            "H3.2_gated_delta_linear": {
                "claim": "Delta-rule / gated linear attention closes recall gap.",
                "support": ["gated delta networks", "DeltaNet parallelizing linear transformers delta rule", "gated linear attention hardware efficient",
                            "RWKV-7 goose", "RWKV eagle finch", "retentive network", "HGRN2 hierarchically gated", "xLSTM", "longhorn state space online learning"],
                "refute": ["linear attention fails long context retrieval", "linear RNN state size capacity limit"],
            },
            "H3.3_sparse_attention": {
                "claim": "Learned sparse attention keeps quality at long context.",
                "support": ["native sparse attention", "mixture of block attention", "sliding window attention global tokens", "multi-head latent attention",
                            "differential transformer", "landmark attention", "quest query aware sparsity kv"],
                "refute": ["sparse attention long context quality degradation", "sliding window attention fails needle"],
            },
            "H3.4_test_time_training_layers": {
                "claim": "Hidden state as a model trained at test time (TTT/Titans) improves long-context memory.",
                "support": ["learning to learn at test time RNN expressive hidden states", "titans learning to memorize at test time",
                            "test-time training long context", "fast weight programmers linear transformers", "atlas memory test time", "lattice memory"],
                "refute": ["test-time training layer throughput overhead", "fast weights instability"],
            },
        },
    },
    # ------------------------------------------------------------------ P4
    "P4_inference_time_scaling": {
        "problem": "Convert extra inference compute into capability with good gain/FLOP.",
        "hypotheses": {
            "H4.1_latent_reasoning": {
                "claim": "Reasoning in continuous latent space is cheaper than token CoT.",
                "support": ["coconut chain of continuous thought", "latent reasoning language model", "pause tokens", "implicit chain of thought",
                            "hierarchical reasoning model", "tiny recursive model reasoning"],
                "refute": ["continuous thought fails scaling", "pause tokens no improvement", "latent reasoning interpretability failure"],
            },
            "H4.2_verifier_search": {
                "claim": "Small model + verifier + search beats larger model at equal FLOPs.",
                "support": ["scaling test-time compute optimally", "process reward model step verification", "best-of-n verifier small model",
                            "monte carlo tree search LLM reasoning", "self-consistency", "generative verifier"],
                "refute": ["reward model overoptimization best of n", "self-consistency diminishing returns", "process reward model failure"],
            },
            "H4.3_speculative_decoding": {
                "claim": "Self-speculation (shared weights / early exit) gives speedups w/o separate draft.",
                "support": ["speculative decoding", "medusa multiple decoding heads", "EAGLE speculative sampling feature",
                            "self speculative decoding layer skip", "multi-token prediction"],
                "refute": ["speculative decoding batch throughput limitation", "multi-token prediction small model hurts"],
            },
        },
    },
    # ------------------------------------------------------------------ P5
    "P5_data_efficiency": {
        "problem": "Maximize capability per training token (information density).",
        "hypotheses": {
            "H5.1_quality_filtering": {
                "claim": "Model-based quality filtering + dedup gives large token-efficiency gains.",
                "support": ["fineweb-edu classifier filtering", "DCLM datacomp language model", "data selection influence functions LLM",
                            "semantic deduplication", "perplexity based data pruning"],
                "refute": ["data filtering reduces diversity", "quality filter bias benchmark contamination"],
            },
            "H5.2_synthetic_data": {
                "claim": "Synthetic textbook/reasoning data improves small models' reasoning.",
                "support": ["textbooks are all you need phi", "synthetic data pretraining rephrasing web", "tinystories small language model",
                            "cosmopedia", "synthetic continued pretraining"],
                "refute": ["model collapse recursive synthetic data", "synthetic data diversity limits"],
            },
            "H5.3_distillation": {
                "claim": "Logit/reasoning distillation from strong teachers is the dominant small-model lever.",
                "support": ["knowledge distillation pretraining language model", "minitron pruning distillation", "reasoning distillation small model",
                            "on-policy distillation", "gemma distillation pretraining"],
                "refute": ["distillation scaling laws capacity gap", "small models struggle learn from strong reasoners"],
            },
            "H5.4_curriculum_mixture": {
                "claim": "Curriculum and dynamic mixture improve sample efficiency.",
                "support": ["data mixing laws", "doremi domain reweighting", "curriculum learning language model pretraining", "annealing high quality data"],
                "refute": ["curriculum learning no benefit language model"],
            },
        },
    },
    # ------------------------------------------------------------------ P6
    "P6_training_efficiency": {
        "problem": "Train with minimal compute & memory (incl. 1GB-RAM research loop).",
        "hypotheses": {
            "H6.1_optimizers": {
                "claim": "Muon/Shampoo/SOAP-class optimizers cut tokens-to-target.",
                "support": ["muon optimizer scalable", "SOAP shampoo adam", "sophia second order optimizer", "schedule free optimizer", "lion optimizer"],
                "refute": ["optimizer benchmark fair comparison language model", "new optimizers speedup diminishes scale"],
            },
            "H6.2_memory_saving": {
                "claim": "8-bit/low-rank optimizer state, recomputation, reversible layers enable tiny-RAM training.",
                "support": ["8-bit optimizers block-wise quantization", "reversible transformer memory", "activation checkpointing",
                            "zero offload", "adafactor", "lomo full parameter fine-tuning low memory", "apollo sgd-like memory"],
                "refute": ["low precision optimizer state instability"],
            },
            "H6.3_quantized_training": {
                "claim": "Ternary/low-bit native training (BitNet) gives near-FP quality at a fraction of memory.",
                "support": ["bitnet 1.58 bit", "fp8 training LLM", "quantization aware training LLM", "scaling laws for precision"],
                "refute": ["bitnet scaling gap small models", "low bit training overtrained models degradation"],
            },
            "H6.4_backprop_alternatives": {
                "claim": "Forward-only / local learning might cut memory.",
                "support": ["forward gradient", "zeroth order optimization LLM MeZO", "local learning rules transformer", "forward-forward algorithm"],
                "refute": ["forward gradient variance scaling", "zeroth order pretraining infeasible"],
            },
            "H6.5_schedules_scaling": {
                "claim": "WSD schedules and maximal-update parametrization allow cheap hparam transfer.",
                "support": ["warmup stable decay schedule", "muP hyperparameter transfer", "compute optimal scaling chinchilla", "overtraining small models scaling"],
                "refute": ["muP transfer failure", "scaling law extrapolation error"],
            },
        },
    },
    # ------------------------------------------------------------------ P7
    "P7_memory_retrieval_tools": {
        "problem": "Offload knowledge and computation out of parameters.",
        "hypotheses": {
            "H7.1_retrieval": {
                "claim": "Retrieval-augmented pretraining lets small models match larger on knowledge.",
                "support": ["RETRO retrieval enhanced transformer", "retrieval augmented language model pretraining", "kNN-LM", "memorizing transformers"],
                "refute": ["retrieval augmentation hurts reasoning distraction", "RETRO benefits leakage"],
            },
            "H7.2_tools": {
                "claim": "Tool use (code exec, calculators) substitutes parametric capacity.",
                "support": ["toolformer", "program aided language models", "tool integrated reasoning math", "code interpreter reasoning small model"],
                "refute": ["tool use small models failure"],
            },
            "H7.3_continual": {
                "claim": "Continual learning without forgetting keeps the 9B model current.",
                "support": ["continual pretraining LLM forgetting", "replay continual learning language model", "model merging continual"],
                "refute": ["catastrophic forgetting continual pretraining"],
            },
        },
    },
    # ------------------------------------------------------------------ P8
    "P8_post_training": {
        "problem": "Post-training that maximally elicits reasoning in small models.",
        "hypotheses": {
            "H8.1_rl_verifiable": {
                "claim": "RL with verifiable rewards (GRPO) improves small-model reasoning.",
                "support": ["group relative policy optimization", "reinforcement learning verifiable rewards reasoning", "DPO direct preference optimization",
                            "self-play fine-tuning", "STaR self-taught reasoner"],
                "refute": ["does reinforcement learning really incentivize reasoning beyond base model", "spurious rewards RLVR", "RL small models limited"],
            },
        },
    },
    # ------------------------------------------------------------------ P9
    "P9_inference_compression": {
        "problem": "Low memory / latency deployment.",
        "hypotheses": {
            "H9.1_quant_kv": {
                "claim": "4-bit weights + compressed KV preserve quality.",
                "support": ["GPTQ", "AWQ activation aware quantization", "KV cache quantization", "KV cache compression eviction", "QuIP incoherence", "AQLM additive quantization"],
                "refute": ["quantization hurts reasoning long chain", "KV eviction failure"],
            },
        },
    },
    # ------------------------------------------------------------------ P10 cross-field
    "P10_cross_field": {
        "problem": "Principles successful elsewhere but under-used in LLMs.",
        "hypotheses": {
            "H10.1_cross": {
                "claim": "Ideas from neuroscience/numerics/compilers transfer.",
                "support": ["predictive coding neural network language", "hopfield networks is all you need", "deep equilibrium models",
                            "neural ODE language model", "complementary learning systems hippocampus model", "hypernetwork generate weights transformer",
                            "multigrid neural network training", "energy based language model", "diffusion language model", "byte latent transformer patches"],
                "refute": ["deep equilibrium model instability", "diffusion language model gap autoregressive"],
            },
        },
    },
}

# GitHub-specific queries (topic keywords). Star stratification is applied by harvester.
GITHUB_QUERIES = [
    "mixture of depths", "mixture of experts", "mamba", "state space model", "rwkv", "linear attention", "gated delta",
    "deltanet", "sparse attention", "flash attention", "test time training", "titans memory", "looped transformer",
    "universal transformer", "recurrent depth", "latent reasoning", "coconut continuous thought", "speculative decoding",
    "medusa", "eagle speculative", "knowledge distillation llm", "bitnet", "quantization llm", "gptq", "awq",
    "kv cache compression", "low rank training", "galore", "lora", "muon optimizer", "8bit optimizer", "schedule free",
    "sophia optimizer", "tinystories", "small language model", "nanogpt", "synthetic data generation llm",
    "data deduplication", "quality filter pretraining", "grpo", "dpo", "process reward model", "mcts llm",
    "self consistency", "retrieval augmented", "knn-lm", "memorizing transformer", "product key memory",
    "early exit transformer", "adaptive computation", "parameter sharing transformer", "hybrid attention ssm",
    "xlstm", "hyena", "retnet", "griffin hawk", "long context", "rope extension", "multi token prediction",
    "zeroth order optimization", "forward gradient", "reversible transformer", "activation checkpointing",
    "hierarchical reasoning model", "tiny recursive model", "deep equilibrium", "hopfield", "byte latent transformer",
    "diffusion language model", "mup", "model merging", "continual pretraining", "tool use llm", "program of thought",
]
STAR_BUCKETS = ["stars:0..1", "stars:2..10", "stars:11..100", "stars:101..1000", "stars:>1000"]

# ---------------------------------------------------------------------------------------------
# Session-1 Step-H expansion: hypotheses generated from L2 reading (see research/notes/evidence_log.md)
TAXONOMY["P11_s1_derived"] = {
    "problem": "Questions raised by session-1 reading: how to make looping pay off at iso-PARAMETER (not iso-FLOP).",
    "hypotheses": {
        "H11.1_looped_moe_routing_divergence": {
            "claim": "Looped MoE recovers per-loop expressivity via routing divergence; dense loops have an FFN bottleneck.",
            "support": ["looped mixture of experts routing divergence", "sparse layers looped language models",
                        "recursive mixture of experts shared layers", "MoEUT mixture of experts universal transformer",
                        "sparse universal transformer"],
            "refute": ["looped MoE expert collapse across iterations", "universal transformer MoE instability"],
        },
        "H11.2_loop_capacity_exponent": {
            "claim": "Recurrence-equivalence exponent phi (~0.46) can be raised by hyperconnections / injection / per-loop adapters.",
            "support": ["iso-depth scaling laws looped language models", "hyper-connections residual streams",
                        "input injection recurrent depth transformer", "per-iteration LoRA recursive transformer"],
            "refute": ["truncated backpropagation through depth looped transformer degrades"],
        },
        "H11.3_hybrid_role_dissociation": {
            "claim": "Attention = addressable store, recurrence = compressed prior; so attention layers should NOT be looped (KV cost) while recurrent layers can be.",
            "support": ["attention recalls recurrence controls hybrid", "hybrid model which layers need attention", "KV cache sharing across layers"],
            "refute": ["hybrid linear attention needle in haystack failure"],
        },
        "H11.4_self_spec_loops": {
            "claim": "Shallow loops of the same model are good drafts (self-speculation) — more so with shallow-deep self-distillation.",
            "support": ["self speculative decoding looped transformer", "early exit self distillation draft", "layerskip early exit speculative"],
            "refute": ["early exit draft low acceptance rate"],
        },
        "H11.5_layer_loop_vs_block_loop": {
            "claim": "Per-layer loop (Loopie) vs block loop (Huginn/Ouro) trade-offs differ in KV cost and stability.",
            "support": ["layer-wise looping transformer local refinement", "Loopie looped MoE", "Ouro looped language model entropy regularized exit"],
            "refute": ["looped transformer training instability residual scaling"],
        },
        "H11.6_kv_sharing_across_loops": {
            "claim": "Reusing/sharing KV across loop iterations removes the KV-memory multiplier of looping.",
            "support": ["cross-layer KV cache sharing", "KV cache reuse recurrent depth", "you only cache once YOCO"],
            "refute": ["cross layer kv sharing quality degradation"],
        },
    },
}
GITHUB_QUERIES += ["looped moe", "recurrent depth transformer", "huginn", "ouro loop", "mixture of recursions",
                   "relaxed recursive", "hyper connections", "yoco", "kv sharing", "layerskip", "self speculative",
                   "moeut", "sparse universal transformer", "gated attention", "kimi delta attention", "native sparse attention"]

# Session-2 Step-H expansion (from E4g reversal, R20/R21, R17, DPT): each hypothesis has support + refute queries.
TAXONOMY["P13_s2_derived"] = {
    "problem": "Make looping pay at matched compute and make memory/distillation gains robust.",
    "hypotheses": {
        "H13.1_anytime_tax": {"claim": "Training with variable depth costs quality at fixed depth; curricula reduce it.",
            "support": ["stochastic depth looped transformer anytime inference", "nested depth training anytime prediction", "early exit training cost final layer quality"],
            "refute": ["random depth training improves looped transformer generalization"]},
        "H13.2_token_adaptive_depth": {"claim": "Lookahead-supervised per-token depth beats uniform depth at matched compute.",
            "support": ["token adaptive recursion depth router", "learned halting supervised by loss improvement", "think harder per token adaptive iterations"],
            "refute": ["adaptive computation time no gain language modeling", "per token early exit batching inefficiency GPU"]},
        "H13.3_memory_loop_complement": {"claim": "Lookup memory and looping are complements (memory frees depth, loop adds depth).",
            "support": ["n-gram embedding memory reasoning depth effective layers", "memory layer looped transformer", "hash embedding language model scaling"],
            "refute": ["n-gram memory gains vanish with tokenizer vocabulary scaling", "large vocabulary replaces n-gram memory"]},
        "H13.4_distilled_pretraining_small": {"claim": "Logit distillation during pretraining is the largest token-efficiency lever for a 9B model.",
            "support": ["distilled pretraining induction heads", "top-k logit distillation pretraining", "teacher student capacity gap language model pretraining"],
            "refute": ["distillation hurts in-context learning", "knowledge distillation pretraining no benefit at scale"]},
        "H13.5_stable_loop_dynamics": {"claim": "Contractive injection (spectral radius<1) enables depth extrapolation.",
            "support": ["contractive recurrent depth extrapolation", "fixed point looped transformer convergence", "deep equilibrium language model"],
            "refute": ["looped transformer extrapolation fails beyond trained iterations"]},
        "H13.6_tokenizer_vs_memory": {"claim": "Byte-level vs BPE changes the value of n-gram memory.",
            "support": ["byte level language model n-gram cache", "tokenizer vocabulary size scaling law"],
            "refute": ["over-tokenized transformer vocabulary scaling"]},
    },
}
GITHUB_QUERIES += ["parcae looped", "adaptive looped transformer", "engram ngram memory", "hash embedding language model",
                   "distillation pretraining llm", "top-k logit distillation", "megadoc synthetic", "rephrase pretraining data",
                   "fixed point transformer", "deep equilibrium language model", "anytime inference early exit"]

# ---------------------------------------------------------------------------------------------
# Session-2 Step-H expansion: hypotheses generated from s2 experiments (E5, E6, E8, F007, F008) and reads R33–R36
TAXONOMY["P14_s2_derived"] = {
    "problem": "Explain s2 anomalies: memory vs in-context retrieval (E6), KD x memory additivity (E5b), seed variance (F008), loops vs arithmetic (E8).",
    "hypotheses": {
        "H14.1_memory_delays_induction": {
            "claim": "Token-indexed memory (n-gram tables) supplies bigram statistics early and delays/weakens induction-head formation, hurting in-context retrieval.",
            "support": ["induction head formation phase transition", "bigram statistics delay induction heads", "n-gram memorization versus in-context learning transformer",
                        "transient in-context learning emergence", "statistical induction heads n-gram"],
            "refute": ["n-gram embeddings improve in-context learning", "hash n-gram embedding in-context retrieval benchmark"],
        },
        "H14.2_kd_memory_complementarity": {
            "claim": "Soft-label KD and token-indexed memory address different deficits (target noise vs local pattern capacity) and therefore add.",
            "support": ["knowledge distillation small model data efficiency pretraining", "soft labels sample efficiency language model",
                        "distillation embedding tables memory"],
            "refute": ["distillation scaling law supervised outperforms", "knowledge distillation hurts in-context learning"],
        },
        "H14.3_seed_variance_small_lm": {
            "claim": "Small-LM results vary mostly with data order; variance-aware evaluation is needed before architectural ranking.",
            "support": ["random seed variance language model pretraining", "data ordering effect language model training",
                        "multiple seeds benchmark variance small transformers"],
            "refute": ["seed variance negligible language model scaling"],
        },
        "H14.4_position_bottleneck_arithmetic": {
            "claim": "Arithmetic failures of small LMs are positional; with digit-position embeddings, loops add length generalisation.",
            "support": ["abacus embeddings arithmetic transformer", "index hints addition length generalization", "looped transformer length generalization arithmetic",
                        "position coupling arithmetic transformer"],
            "refute": ["position embeddings arithmetic generalization failure"],
        },
        "H14.5_soft_weight_sharing": {
            "claim": "Soft tying (cosine/L2 penalties between layers) gives the loop inductive bias without loop FLOPs.",
            "support": ["looping inspired regularization layers cosine", "soft parameter sharing transformer layers", "layer similarity regularization depth"],
            "refute": ["weight tying regularization hurts language modeling"],
        },
        "H14.6_anytime_exit_training": {
            "claim": "Training intermediate exits (deep supervision through the coda) makes looped models anytime without the uniform-R tax.",
            "support": ["deep supervision early exit language model", "anytime neural network training intermediate classifiers", "layer dropout early exit loss"],
            "refute": ["early exit training degrades final layer quality"],
        },
    },
}
GITHUB_QUERIES += ["induction heads", "abacus embeddings", "index hints arithmetic", "early exit llm", "deep supervision transformer",
                   "soft weight sharing", "seed variance llm", "born again distillation", "logit distillation pretraining"]

# Session-2 Step-H (B1 knowledge bottleneck at 9B): corpus check found thin coverage of small-model + retrieval/tool scaling.
TAXONOMY["P15_knowledge_channel"] = {
    "problem": "Closed-book knowledge is the hardest axis at 9B (stored bits); which external channel (retrieval, tools, editable memory) closes it at least cost?",
    "hypotheses": {
        "H15.1_retrieval_vs_params_scaling": {
            "claim": "Retrieval-augmented small models match much larger closed-book models on knowledge tasks at far lower cost.",
            "support": ["retrieval augmented language model scaling datastore size", "small model retrieval matches larger model knowledge",
                        "RETRO scaling retrieval pretraining", "datastore scaling laws retrieval"],
            "refute": ["retrieval augmented generation fails reasoning small models", "retrieval noise hurts small language models",
                       "closed-book versus open-book gap does not close"],
        },
        "H15.2_parametric_vs_editable_memory": {
            "claim": "Editable token-indexed memory (Engram overlays) is a cheaper continual-knowledge channel than fine-tuning or RAG.",
            "support": ["knowledge editing memory layer large scale", "continual knowledge update without forgetting memory module"],
            "refute": ["knowledge editing ripple effects failure", "model editing does not generalize paraphrase"],
        },
        "H15.3_tool_use_small_models": {
            "claim": "Tool-use (search, code exec) post-training gives small models disproportionate gains on knowledge and math.",
            "support": ["tool integrated reasoning small language model", "agentic search small model reinforcement learning", "toolformer small model"],
            "refute": ["small language models tool use unreliable", "tool use claims small models evaluation failure"],
        },
    },
}
GITHUB_QUERIES += ["retrieval pretraining retro", "datastore scaling", "knowledge editing", "tool integrated reasoning", "agentic search rl"]

# ---- Step H (session 2, after F013/F014/E8i): problems derived from own anomalies ----
TAXONOMY["P16_s2_engram_dynamics"] = {
    "problem": "A token-indexed memory is update-starved (table lr x20 still improving), changes ICL emergence timing, and interferes with algorithmic spans. What is known about each?",
    "hypotheses": {
        "H16.1_sparse_row_lr": {
            "claim": "Sparse embedding rows need per-row lr far above dense params; optimum scales with update frequency.",
            "support": ["sparse embedding learning rate scaling frequency", "adagrad sparse features embedding rows convergence", "row-wise adaptive learning rate embedding table",
                        "hash embedding training dynamics learning rate", "muP embedding learning rate multiplier"],
            "refute": ["large embedding learning rate overfitting rare features", "embedding learning rate instability language model"],
        },
        "H16.2_emergence_timing_levers": {
            "claim": "In-context-learning emergence time is controllable by architecture/optimiser choices, not only data.",
            "support": ["induction head emergence time accelerate", "in-context learning phase transition timing architecture", "abrupt learning plateau shortening optimizer"],
            "refute": ["in-context learning emergence independent of architecture", "induction head formation data distribution only"],
        },
        "H16.3_domain_gated_memory": {
            "claim": "Lookup memories must be gated off on algorithmic spans (numbers, code) to avoid shortcut interference.",
            "support": ["n-gram shortcut arithmetic language model interference", "memorization interferes with algorithmic generalization", "token-type gating memory module"],
            "refute": ["memory layers improve arithmetic", "n-gram embeddings help code models"],
        },
    },
}
GITHUB_QUERIES += ["sparse embedding optimizer", "engram memory", "hash embedding language model", "induction head emergence"]
