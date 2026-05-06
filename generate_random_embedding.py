"""Generate random embeddings matching a model's embedding weight statistics."""
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig


def generate_random_embedding(model_path, num_tokens, seed=15, save_dir=None):
    """Generate random embeddings from N(embed_mean, embed_std).

    Args:
        model_path: HuggingFace model path
        num_tokens: Number of random embedding tokens to generate
        seed: Random seed for reproducibility
        save_dir: Directory to save the embedding. If None, returns the tensor.

    Returns:
        random_embeds tensor if save_dir is None
    """
    torch.manual_seed(seed)

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16
    )

    embed_weight = model.get_input_embeddings().weight.data
    embed_mean = embed_weight.mean()
    embed_std = embed_weight.std()
    hidden_dim = embed_weight.shape[1]

    print(f"Model: {model_path}")
    print(f"Embedding stats: mean={embed_mean:.6f}, std={embed_std:.6f}, hidden_dim={hidden_dim}")

    random_embeds = torch.randn(num_tokens, hidden_dim) * embed_std + embed_mean

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "prefix_embeddings.pt"
        torch.save({"weight": random_embeds}, save_path)
        print(f"Saved {num_tokens} random embeddings to {save_path}")

    del model
    torch.cuda.empty_cache()
    return random_embeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_tokens", type=int, default=20)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=15)
    args = parser.parse_args()

    generate_random_embedding(args.model_path, args.num_tokens, args.seed, args.save_dir)


if __name__ == "__main__":
    main()
