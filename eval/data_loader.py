"""Load a benchmark from eval/data/<name>/<split>.jsonl (see scripts/download_data.py)."""
import os

from utils import load_jsonl


def load_data(data_name, split, data_dir="eval/data"):
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"{data_file} not found. Run `python scripts/download_data.py` first."
        )
    examples = list(load_jsonl(data_file))
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]
    return sorted(examples, key=lambda x: x["idx"])
