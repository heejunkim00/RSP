"""Custom reward function for verl v0.5.0 — bridges simpleRL's `simplelr_*`
data_sources to hf_math_verify while falling back to v0.5.0 defaults for
everything else.

verl v0.5.0 loads this via `custom_reward_function.path` + `name=compute_score`
(see verl/trainer/ppo/reward.py get_custom_reward_fn). The loaded callable
is called by reward_manager/naive.py with:
    (data_source, solution_str, ground_truth, extra_info=None, **kwargs)
so we match that signature.

simpleRL's main_ppo.py used this mapping:
    data_source in {'openai/gsm8k'}         -> gsm8k
    data_source in {'lighteval/MATH', ...}  -> math
    'kk' in data_source                     -> kk
    'simplelr' in data_source               -> hf_math_verify  <-- the one we need
    'deepseek_r1' in data_source            -> deepseek_r1
We only port the `simplelr` branch here; everything else routes to v0.5.0's
default_compute_score, which already covers gsm8k/MATH and more.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Expose hf_math_verify.py and qwen_math_eval_toolkit/ as top-level imports.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import hf_math_verify  # noqa: E402  (loads qwen_math_eval_toolkit.parser transitively)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if isinstance(data_source, str) and "simplelr" in data_source:
        return hf_math_verify.compute_score(solution_str, ground_truth)

    # Everything else: fall back to verl v0.5.0's built-in table.
    from verl.utils.reward_score import default_compute_score

    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
