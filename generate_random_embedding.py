"""Draw one RSP and save it, for reuse across runs via `eval.py --embedding_path`.

    python generate_random_embedding.py --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
        --num_tokens 20 --seed 15 --save_dir embeddings/qwen25_math_1.5b_inst_L20_seed15

Writes <save_dir>/prefix_embeddings.pt containing {"weight": Tensor[L, d]}.
By default eval.py redraws the RSP every batch; a saved tensor fixes it instead.
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from model_utils import embedding_table_stats, sample_rsp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--num_tokens", default=20, type=int, help="RSP length L")
    parser.add_argument("--seed", default=15, type=int)
    parser.add_argument("--save_dir", required=True, type=str)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    stats = embedding_table_stats(model)
    embeds = sample_rsp(args.num_tokens, stats, args.seed)
    print(f"mu_E={stats['mean']:.3e}, sigma_E={stats['std']:.3e}, d={stats['hidden_dim']}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": embeds}, save_dir / "prefix_embeddings.pt")
    print(f"Saved {tuple(embeds.shape)} to {save_dir / 'prefix_embeddings.pt'}")


if __name__ == "__main__":
    main()
