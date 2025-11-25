# tasks/exp_selection.py
#
# - The model sees a table of experiment runs with many metrics and constraints.
# - It must pick exactly one experiment id that best satisfies all constraints.
# - Grading is all or nothing on the chosen id.

from typing import Any
from anthropic.types import ToolUnionParam
import json


def get_tools() -> list[ToolUnionParam]:
    """
    Tools:
    - python_expression: for scratch reasoning (you can compute derived metrics, gaps, etc.).
    - submit_answer: for submitting the final chosen experiment id.
    """
    return [
        {
            "name": "python_expression",
            "description": (
                "Use this tool for scratch work. You can compute gaps, tradeoffs, or "
                "filter candidates using Python code. The grader only looks at what you "
                "submit via submit_answer."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python code executed with exec().",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the id of the chosen experiment.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "description": (
                            "Final answer: the string id of the chosen experiment, e.g. 'exp_a', "
                            "or a JSON object/string that contains that id under a simple key "
                            "like 'exp' or 'choice'."
                        )
                    }
                },
                "required": ["answer"],
            },
        },
    ]


def get_prompt() -> str:
    """
    ML experiment/log reasoning task (harder version).
    """
    return """
You are acting as an ML engineer selecting the best experiment run to deploy.

You are given a set of training runs for the same model family (classifier on an
imbalanced dataset). Each experiment changed some hyperparameters and produced
metrics on a held-out validation set.

Your job is to choose **exactly one** experiment id that is the best candidate
for deployment, based on all the constraints below.


High-level goal
---------------

We care about **balanced, robust generalization**, not just raw accuracy.

In particular:

1) Primary metric: validation F1_macro (higher is better).
2) We care about the **train–val gap** in F1_macro: large gaps indicate overfitting.
3) We must respect **latency and memory limits**:
   - Latency (p95) must be **<= 45 ms**.
   - GPU memory usage must be **<= 7.5 GB**.
4) We care about **fairness on the minority class**:
   - Minority-class F1 should not be dramatically worse than majority-class F1.
   - A rough rule: (majority_F1 - minority_F1) should be **<= 0.10**.
5) We prefer **small train–val gaps**:
   - As a soft rule, we strongly prefer gaps **<= 0.07**.
6) All else equal, we prefer **smaller, simpler models**, but NOT at the cost
   of noticeably worse F1 or fairness.


The candidates
--------------

Each experiment row:

- id: experiment id
- model: short model name
- params_M: number of parameters in millions
- train_F1: macro F1 on train
- val_F1: macro F1 on validation
- majority_F1: F1 on the majority class
- minority_F1: F1 on the minority class
- val_loss: final validation loss
- p95_latency_ms: approximate p95 inference latency in milliseconds
- gpu_mem_GB: peak GPU memory usage in GB
- notes: free-form observations from the training run

Table:

  id     model       params_M  train_F1  val_F1  majority_F1  minority_F1  val_loss  p95_latency_ms  gpu_mem_GB  notes
  -----  ----------  --------  --------  ------  -----------  -----------  --------  --------------  ----------  -------------------------------------------------------------
  exp_a  base_s      16.0      0.93      0.87    0.90         0.84         0.39      36              5.2         very stable, no divergence; smooth curves; mild underfit
  exp_b  base_m      42.0      0.97      0.89    0.94         0.82         0.36      44              7.4         slight overfit, some late spikes; near latency/mem limits
  exp_c  base_m_t    44.0      0.98      0.90    0.95         0.83         0.35      42              7.2         stronger val, but minority F1 lags; early overfit signs
  exp_d  base_s_do   18.0      0.94      0.88    0.91         0.85         0.38      40              6.1         dropout + LR decay; very stable; small gap
  exp_e  base_s_aug  17.0      0.95      0.86    0.93         0.80         0.40      32              5.5         heavy augmentation; majority up, minority somewhat worse


Derived info (you may compute via python_expression)
----------------------------------------------------

Useful derived quantities:

- train_val_gap = train_F1 - val_F1
- fairness_gap = majority_F1 - minority_F1

You must reason about these for each experiment.


Hard constraints (eligibility)
------------------------------

A run is **eligible** only if ALL of:

1) val_F1 >= 0.85.
2) p95_latency_ms <= 45 ms.
3) gpu_mem_GB <= 7.5 GB.
4) fairness_gap = (majority_F1 - minority_F1) <= 0.10.

Any run that violates any of these is **ineligible** and cannot be chosen.


Soft preferences among eligible runs
------------------------------------

Among the eligible runs, you must choose a single best candidate using:

1) Prefer higher val_F1, but:
   - If val_F1 differs by **<= 0.01**, treat them as roughly tied and look
     more carefully at gaps, fairness, and size.

2) Prefer smaller train_val_gap:
   - Gaps <= 0.07 are considered clearly better than gaps > 0.07,
     especially when val_F1 is within 0.01.

3) Prefer smaller fairness_gap:
   - If two runs have similar val_F1 and train_val_gap, choose the one with
     strictly smaller fairness_gap.

4) Prefer smaller models (params_M) when metrics are otherwise very similar:
   - If val_F1 differs by <= 0.01 AND both train_val_gap and fairness_gap
     are within 0.01 of each other, choose the smaller model.

5) Prefer stability in notes over runs that mention spikes / early overfit.


Your job
--------

1) Determine which experiments are eligible under the hard constraints.
2) Among the eligible runs, apply the soft preferences above to select the
   **single best** experiment id.
3) Use python_expression for any scratch work you need.
4) Finally, via submit_answer, return ONLY the chosen experiment id.


Final answer format
-------------------

Your submit_answer call must set "answer" to one of:

- The plain string:
      answer = "exp_x"

- Or a dict like:
      answer = {"exp": "exp_x"}
   or:
      answer = {"choice": "exp_x"}

- Or a JSON-encoded string of one of the above.

Constraints:

- The id must be exactly one of:
      "exp_a", "exp_b", "exp_c", "exp_d", "exp_e"
- Do NOT include explanations or extra keys in the final answer.
- Grading is strict: only the correct id is accepted.
    """.strip()


# ---- Grading helpers ----

_VALID_EXPS = {"exp_a", "exp_b", "exp_c", "exp_d", "exp_e"}


def _parse_answer(answer: Any) -> str | None:
    """
    Accept:
    - "exp_d"
    - {"exp": "exp_d"} or {"choice": "exp_d"}
    - JSON string of either of the above.
    """
    obj = answer

    # If it's a string, could be plain id or JSON
    if isinstance(answer, str):
        s = answer.strip()
        # Try JSON first
        try:
            parsed = json.loads(s)
            obj = parsed
        except Exception:
            # Not JSON → maybe plain id
            return s if s in _VALID_EXPS else None

    # Dict with a single relevant key
    if isinstance(obj, dict):
        for key in ("exp", "choice"):
            if key in obj:
                val = obj[key]
                if isinstance(val, str) and val in _VALID_EXPS:
                    return val
        return None

    return None


def grade(answer: Any) -> bool:
    """
    Grade the answer:

    - Parse it into a single experiment id.
    - Must equal the intended best choice.
    """
    chosen = _parse_answer(answer)
    if chosen is None:
        return False

    # Let's reason out the intended best choice.

    # First compute gaps mentally (documented here for clarity):
    #
    # exp_a:
    #   train_F1 = 0.93, val_F1 = 0.87  -> train_val_gap = 0.06
    #   fairness_gap = 0.90 - 0.84 = 0.06
    #   latency = 36 (OK), mem = 5.2 (OK), val_F1 >= 0.85 (OK)
    #
    # exp_b:
    #   train_F1 = 0.97, val_F1 = 0.89  -> gap = 0.08
    #   fairness_gap = 0.94 - 0.82 = 0.12  -> violates fairness ( > 0.10 ) → ineligible.
    #
    # exp_c:
    #   train_F1 = 0.98, val_F1 = 0.90  -> gap = 0.08
    #   fairness_gap = 0.95 - 0.83 = 0.12  -> violates fairness → ineligible.
    #
    # exp_d:
    #   train_F1 = 0.94, val_F1 = 0.88  -> gap = 0.06
    #   fairness_gap = 0.91 - 0.85 = 0.06
    #   latency = 40, mem = 6.1, val_F1 >= 0.85  -> eligible.
    #
    # exp_e:
    #   train_F1 = 0.95, val_F1 = 0.86  -> gap = 0.09
    #   fairness_gap = 0.93 - 0.80 = 0.13  -> violates fairness → ineligible.
    #
    # So the only eligible runs under the HARD constraints are:
    #   exp_a and exp_d.
    #
    # Compare exp_a vs exp_d:
    #   val_F1:
    #       exp_a = 0.87
    #       exp_d = 0.88
    #   → difference = 0.01 (treated as "rough tie").
    #
    #   train_val_gap:
    #       exp_a gap = 0.06
    #       exp_d gap = 0.06
    #   → identical.
    #
    #   fairness_gap:
    #       exp_a fairness_gap = 0.06
    #       exp_d fairness_gap = 0.06
    #   → identical.
    #
    #   params_M:
    #       exp_a = 16.0M
    #       exp_d = 18.0M
    #   → exp_a is slightly smaller.
    #
    #   notes:
    #       exp_a: "very stable, no divergence; smooth curves; mild underfit"
    #       exp_d: "dropout + LR decay; very stable; small gap"
    #   → both are stable; under our rules, tie-breaker becomes model size.
    #
    # Under the soft preferences:
    #   - val_F1 difference 0.01 → treat as roughly tied.
    #   - train_val_gap + fairness_gap are identical.
    #   - prefer smaller model → exp_a is the better choice.
    #
    expected = "exp_a"
    return chosen == expected
