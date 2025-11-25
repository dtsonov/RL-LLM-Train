# RL Task: Experiment Selection Under ML Constraints

This test defines a realistic **Reinforcement Learning (RL) task for LLMs**, modeling how an **ML engineer** selects which experiment to deploy based on performance, generalization, fairness, and varieties of operational constraints.

The task is designed to be run using **Claude Haiku 4.5**, following the assignment requirement of a **10–40% pass rate**.

## Task Description
ML Experiment Selection Under Multi-Objective Constraints.
This task trains a capability that LLMs are weak at but in the same time requires:
   - Integrate multiple metrics and resource constraints.
   - Respect fairness thresholds.
   - Follow tie breaking rules carefully when metrics are close.
   - Multistep reasoning (constraints + optimization)
   - Non trivial trade-off resolution
   - Multiple failure modes are good for RL training
   - Tunable difficulty via MAX_TOKENS

The model receives a small table of experiment results (`exp_a` → `exp_e`) with structured metrics:
- `train_F1`, `val_F1`, `val_loss`
- `majority_F1`, `minority_F1`
- `p95_latency_ms`, `gpu_mem_GB`
- `params_M` (model size in millions)
- Free-form `notes`

It must choose the **single best experiment** to deploy.

### Hard Constraints
val_F1 >= 0.85
fairness_gap (majority_F1 - minority_F1) <= 0.10
p95_latency_ms <= 45
gpu_mem_GB <= 7.5

The model is given a structured experiment table (exp_a → exp_e) with the following fields per experiment:

train_F1, val_F1, val_loss
majority_F1, minority_F1
p95_latency_ms, gpu_mem_GB
params_M (model size in millions of parameters)
Free-form notes

The model must select a single best experiment to deploy, based on a combination of hard constraints and soft tie-breaking rules.

"exp_a"
{"exp": "exp_a"}
{"choice": "exp_a"}

I choose exp_a
exp_a
"exp_d because it's best"

Example run configuration with 20% pass rate:
   MAX_TOKENS = 2250
   concurrent = False

Installation and run:
- pip install anthropic
- python main.py

Running 10 test iterations...
-  Run 1: SUCCESS - Got exp_a
-  Run 2: FAILURE - Got exp_c
...

Results:
--------
   MAX_TOKENS = 2000
   Passed: 1/10
   Failed: 9/10
   Pass Rate: 10.0%

   MAX_TOKENS = 2250
   Passed: 2/10
   Failed: 8/10
   Pass Rate: 20.0%

   MAX_TOKENS = 2250 - 2350
   Passed: 3/10
   Failed: 7/10
   Pass Rate: 30.0%

   MAX_TOKENS = 2500
   Passed: 4/10
   Failed: 6/10
   Pass Rate: 40.0%





