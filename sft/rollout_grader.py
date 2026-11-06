"""Binary correctness of a rollout: the last \\boxed{} answer vs. the ground truth."""
from mathruler.grader import extract_boxed_content, grade_answer


def is_correct(response: str, ground_truth: str) -> bool:
    return grade_answer(extract_boxed_content(response), ground_truth)
