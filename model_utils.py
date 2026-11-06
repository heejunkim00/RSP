"""Model loading, RSP injection (prefix / suffix), and batched generation."""
import torch
import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_str, tokenizer):
        super().__init__()
        self.current_context = []
        self.tokenizer = tokenizer
        self.keywords_str = keywords_str

    def __call__(self, input_ids, scores, **kwargs):
        if len(self.current_context) == 0:
            self.current_context = [[] for _ in range(input_ids.shape[0])]
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            self.current_context[i].append(input_ids[i][-1].item())
            current_context = self.tokenizer.decode(self.current_context[i])
            sequences_should_be_stopped.append(
                any(w in current_context for w in self.keywords_str)
            )
        return all(sequences_should_be_stopped)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name_or_path, padding_side="left", attn_implementation=None):
    """Load a causal LM in bf16 and its tokenizer (left padding for batched generation)."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True, padding_side=padding_side, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_name_or_path)
    model_kwargs = {"config": config, "torch_dtype": torch.bfloat16}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# RSP sampling (Eq. 1 in the paper)
# ---------------------------------------------------------------------------
def embedding_table_stats(model):
    """Entrywise mean / std of the pretrained token-embedding table W_E."""
    embed_weight = model.get_input_embeddings().weight.data.cpu()
    return {
        "mean": embed_weight.mean(),
        "std": embed_weight.std(),
        "hidden_dim": embed_weight.shape[1],
    }


def sample_rsp(num_tokens, stats, seed):
    """Draw an RSP H in R^{L x d}: h_j ~ N(mu_E * 1, sigma_E^2 * I), deterministic in `seed`."""
    torch.manual_seed(seed)
    return torch.randn(num_tokens, stats["hidden_dim"]) * stats["std"] + stats["mean"]


# ---------------------------------------------------------------------------
# Suffix position: split the formatted prompt at the injection point
# ---------------------------------------------------------------------------
def split_prompt_for_suffix(prompt_str, model_name, suffix_position):
    """Split a chat-formatted prompt into (before, after) around the suffix insertion point.

    suffix_position:
        before_im_end   : ... question | <|im_end|>\\n<|im_start|>assistant\\n   (paper default)
        between_turns   : ... <|im_end|>\\n | <|im_start|>assistant\\n
        after_assistant : ... <|im_start|>assistant\\n |
    """
    name = model_name.lower()

    if "qwen" in name:
        if suffix_position == "before_im_end":
            marker = "<|im_end|>\n<|im_start|>assistant"
            parts = prompt_str.rsplit(marker, 1)
            if len(parts) == 2:
                return parts[0], marker + parts[1]
        elif suffix_position == "between_turns":
            marker = "<|im_end|>\n<|im_start|>assistant"
            parts = prompt_str.rsplit(marker, 1)
            if len(parts) == 2:
                return parts[0] + "<|im_end|>\n", "<|im_start|>assistant" + parts[1]
        elif suffix_position == "after_assistant":
            marker = "<|im_start|>assistant\n"
            parts = prompt_str.rsplit(marker, 1)
            if len(parts) == 2:
                return parts[0] + marker, parts[1]

    elif "llama" in name:
        if suffix_position == "before_im_end":
            marker = "<|eot_id|><|start_header_id|>assistant"
            parts = prompt_str.rsplit(marker, 1)
            if len(parts) == 2:
                return parts[0], marker + parts[1]
        elif suffix_position == "between_turns":
            marker = "<|eot_id|><|start_header_id|>assistant"
            parts = prompt_str.rsplit(marker, 1)
            if len(parts) == 2:
                return parts[0] + "<|eot_id|>", "<|start_header_id|>assistant" + parts[1]
        elif suffix_position == "after_assistant":
            marker = "<|end_header_id|>\n\n"
            idx = prompt_str.rfind(marker)
            if idx != -1:
                split_pos = idx + len(marker)
                return prompt_str[:split_pos], prompt_str[split_pos:]

    raise ValueError(f"Failed to split prompt: model={model_name}, position={suffix_position}")


# ---------------------------------------------------------------------------
# Embedding injection
# ---------------------------------------------------------------------------
def inject_embeddings(model, input_ids_before, attention_mask_before,
                      random_embeds, embedding_mode, num_tokens,
                      input_ids_after=None, attention_mask_after=None):
    """Build inputs_embeds with the RSP inserted.

    prefix : [H ; X]
    suffix : [X_before ; H ; X_after]   (X_after is empty for raw-text prompts)
    none   : X

    Returns (inputs_embeds, attention_mask).
    """
    device = model.device
    batch_size = input_ids_before.shape[0]

    with torch.no_grad():
        embeds_before = model.get_input_embeddings()(input_ids_before)

    if embedding_mode == "none":
        return embeds_before, attention_mask_before

    rand = random_embeds.to(device=device, dtype=embeds_before.dtype)
    rand = rand.unsqueeze(0).expand(batch_size, -1, -1)
    rand_mask = torch.ones(batch_size, num_tokens, dtype=torch.long, device=device)

    if embedding_mode == "prefix":
        input_embeds = torch.cat([rand, embeds_before], dim=1)
        new_mask = torch.cat([rand_mask, attention_mask_before], dim=1)
        return input_embeds, new_mask

    if embedding_mode == "suffix":
        if input_ids_after is not None and input_ids_after.shape[1] > 0:
            with torch.no_grad():
                embeds_after = model.get_input_embeddings()(input_ids_after)
            input_embeds = torch.cat([embeds_before, rand, embeds_after], dim=1)
            new_mask = torch.cat([attention_mask_before, rand_mask, attention_mask_after], dim=1)
        else:
            input_embeds = torch.cat([embeds_before, rand], dim=1)
            new_mask = torch.cat([attention_mask_before, rand_mask], dim=1)
        return input_embeds, new_mask

    raise ValueError(f"Unknown embedding_mode: {embedding_mode}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_completions(
    model, tokenizer, prompts, random_embeds, embedding_mode, num_tokens,
    batch_size=1, stop_words=None, disable_tqdm=False, prompts_after=None,
    **generation_kwargs
):
    """Generate completions with optional RSP injection.

    prompts       : list of prompt strings (for suffix mode, the "before" parts)
    prompts_after : list of "after" strings for suffix mode with chat markers, else None
    random_embeds : (num_tokens, hidden_dim) tensor, or None for the baseline
    """
    if stop_words is None:
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"]

    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        tokenized = tokenizer(
            batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=False
        )
        batch_input_ids = tokenized.input_ids.to(model.device)
        attention_mask = tokenized.attention_mask.to(model.device)

        common_kwargs = {
            "stopping_criteria": StoppingCriteriaList([KeywordsStoppingCriteria(stop_words, tokenizer)]),
            **generation_kwargs,
        }

        if embedding_mode == "none" or random_embeds is None:
            outputs = model.generate(
                input_ids=batch_input_ids, attention_mask=attention_mask, **common_kwargs
            )
            # input_ids path returns prompt + completion; strip the prompt
            outputs = outputs[:, batch_input_ids.shape[1]:]

        elif embedding_mode == "suffix" and prompts_after is not None:
            tok_after = tokenizer(
                prompts_after[i : i + batch_size], padding="longest",
                return_tensors="pt", add_special_tokens=False
            )
            input_embeds, new_mask = inject_embeddings(
                model, batch_input_ids, attention_mask, random_embeds, embedding_mode, num_tokens,
                input_ids_after=tok_after.input_ids.to(model.device),
                attention_mask_after=tok_after.attention_mask.to(model.device),
            )
            outputs = model.generate(inputs_embeds=input_embeds, attention_mask=new_mask, **common_kwargs)

        else:
            input_embeds, new_mask = inject_embeddings(
                model, batch_input_ids, attention_mask, random_embeds, embedding_mode, num_tokens
            )
            outputs = model.generate(inputs_embeds=input_embeds, attention_mask=new_mask, **common_kwargs)

        batch_generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for idx, prediction in enumerate(batch_generations):
            for stop_word in stop_words:
                batch_generations[idx] = batch_generations[idx].split(stop_word)[0]
        generations.extend(batch_generations)

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    if not disable_tqdm:
        progress.close()
    return generations
