"""Model loading and generation with 3 embedding injection modes."""
import torch
import tqdm
from pathlib import Path
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
        self.sample_stop_step = {}  # {sample_idx: num_generated_tokens at stop}

    def __call__(self, input_ids, scores, **kwargs):
        if len(self.current_context) == 0:
            self.current_context = [[] for _ in range(input_ids.shape[0])]
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            self.current_context[i].append(input_ids[i][-1].item())
            current_context = self.tokenizer.decode(self.current_context[i])
            should_be_stopped = any(w in current_context for w in self.keywords_str)
            sequences_should_be_stopped.append(should_be_stopped)
            if should_be_stopped and i not in self.sample_stop_step:
                self.sample_stop_step[i] = len(self.current_context[i])
        return all(sequences_should_be_stopped)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name_or_path, padding_side="left", attn_implementation=None):
    """Load model and tokenizer.

    Args:
        model_name_or_path: HuggingFace model id or local path
        padding_side: 'left' or 'right'
        attn_implementation: Optional attention implementation override
            (e.g., 'flash_attention_2', 'sdpa', 'eager').
            Default: None (use transformers default).
    """
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
    model_kwargs = {
        "config": config,
        "torch_dtype": torch.bfloat16,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Special tokens for Middle mode
# ---------------------------------------------------------------------------
QWEN_SPECIAL_TOKENS = [
    "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>",
    "<|quad_start|>", "<|quad_end|>",
    "<|vision_start|>", "<|vision_end|>",
    "<|vision_pad|>", "<|image_pad|>",
    "<|video_pad|>",
    "<|fim_prefix|>", "<|fim_middle|>",
    "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>",
]


def get_special_tokens_str(model_name, num_tokens):
    """Return concatenated special token string for middle mode.

    If num_tokens exceeds the available pool, tokens are cycled from the start.
    The actual token identity doesn't matter because their embeddings are replaced
    with random vectors anyway.
    """
    name = model_name.lower()
    if "qwen" in name:
        pool = QWEN_SPECIAL_TOKENS
        tokens = [pool[i % len(pool)] for i in range(num_tokens)]
        return "".join(tokens)
    elif "llama" in name:
        return "".join(f"<|reserved_special_token_{i}|>" for i in range(num_tokens))
    else:
        raise ValueError(f"Unsupported model for middle mode: {model_name}")


def find_special_token_positions(input_ids, tokenizer, model_name, num_tokens):
    """Find positions of special tokens in input_ids for middle mode."""
    name = model_name.lower()
    if "qwen" in name:
        tokens_list = [QWEN_SPECIAL_TOKENS[i % len(QWEN_SPECIAL_TOKENS)] for i in range(num_tokens)]
    elif "llama" in name:
        tokens_list = [f"<|reserved_special_token_{i}|>" for i in range(num_tokens)]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    special_ids = set()
    for t in tokens_list:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1:
            special_ids.add(ids[0])

    token_list = input_ids.tolist()
    positions = [i for i, tid in enumerate(token_list) if tid in special_ids]
    start = positions[-num_tokens]
    end = start + num_tokens
    return start, end


# ---------------------------------------------------------------------------
# Prompt construction for Middle mode (description text + special tokens)
# ---------------------------------------------------------------------------
def build_middle_prompt(question, model_name, num_tokens, prompt_type="mathstral"):
    """Build prompt with description text and special tokens for middle mode.

    Args:
        question: The question text
        model_name: Model name to determine which special tokens to use
        num_tokens: Number of special tokens
        prompt_type: Prompt type to determine ChatML or raw text

    Returns:
        Full prompt string with special tokens embedded
    """
    special_str = get_special_tokens_str(model_name, num_tokens)

    description = (
        f'There are {num_tokens} special tokens that contain compressed latent reasoning information '
        f'that might be useful for your reasoning.\n'
        f'If these tokens are useful for your case, you can use them as reference. If these tokens are not useful '
        f'for your case, you can ignore them and focus back to solving the problem.\n\n'
        f'Here are the {num_tokens} special tokens: {special_str}'
    )

    name = model_name.lower()

    if prompt_type in ["qwen25-math-cot", "qwen-boxed", "ltpo-step"]:
        # ChatML format
        user_content = f'{question}\n\n{description}'
        if "qwen" in name:
            return (
                f'<|im_start|>system\n'
                f'Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n'
                f'<|im_start|>user\n'
                f'{user_content}<|im_end|>\n'
                f'<|im_start|>assistant\n'
            )
        elif "llama" in name:
            return (
                f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
                f'Please reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>'
                f'<|start_header_id|>user<|end_header_id|>\n\n'
                f'{user_content}<|eot_id|>'
                f'<|start_header_id|>assistant<|end_header_id|>\n\n'
            )
    else:
        # Raw text format (e.g., mathstral)
        return (
            f'{question}\n\n{description}\n\n'
            f'Please reason step by step, and put your final answer within \\boxed{{}}.'
        )


# ---------------------------------------------------------------------------
# Suffix position: find split point for embedding concat
# ---------------------------------------------------------------------------
def split_prompt_for_suffix(prompt_str, model_name, suffix_position):
    """Split prompt string at the suffix insertion point.

    Args:
        prompt_str: The formatted prompt string
        model_name: Model name
        suffix_position: 'before_im_end', 'between_turns', 'after_assistant'

    Returns:
        (before, after) tuple of strings
    """
    name = model_name.lower()

    if "qwen" in name:
        if suffix_position == "before_im_end":
            # Split at: ...question | <|im_end|>\n<|im_start|>assistant\n
            marker = "<|im_end|>\n<|im_start|>assistant"
            parts = prompt_str.rsplit(marker, 1)
            if len(parts) == 2:
                return parts[0], marker + parts[1]
        elif suffix_position == "between_turns":
            # Split at: ...<|im_end|>\n | <|im_start|>assistant\n
            marker = "<|im_end|>\n<|im_start|>assistant"
            parts = prompt_str.rsplit(marker, 1)
            if len(parts) == 2:
                return parts[0] + "<|im_end|>\n", "<|im_start|>assistant" + parts[1]
        elif suffix_position == "after_assistant":
            # Split at: ...<|im_start|>assistant\n |
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
def inject_embeddings(model, tokenizer, input_ids_before, attention_mask_before,
                      random_embeds, embedding_mode, model_name, num_tokens,
                      input_ids_after=None, attention_mask_after=None):
    """Inject random embeddings according to the specified mode.

    Args:
        model: The language model
        tokenizer: Tokenizer
        input_ids_before: (batch, seq_len) - tokens before the injection point
        attention_mask_before: (batch, seq_len)
        random_embeds: (num_tokens, hidden_dim) random embedding tensor
        embedding_mode: 'none', 'prefix', 'middle', 'suffix'
        model_name: Model name
        num_tokens: Number of embedding tokens
        input_ids_after: (batch, seq_len) - tokens after injection point (suffix mode only)
        attention_mask_after: (batch, seq_len) - (suffix mode only)

    Returns:
        inputs_embeds, attention_mask (both modified)
    """
    device = model.device
    batch_size = input_ids_before.shape[0]

    with torch.no_grad():
        embeds_before = model.get_input_embeddings()(input_ids_before)

    if embedding_mode == "none":
        return embeds_before, attention_mask_before

    random_embeds_device = random_embeds.to(device=device, dtype=embeds_before.dtype)

    if embedding_mode == "prefix":
        # Concat random embeds before prompt embeds
        prefix = random_embeds_device.unsqueeze(0).expand(batch_size, -1, -1)
        input_embeds = torch.cat([prefix, embeds_before], dim=1)
        prefix_mask = torch.ones(batch_size, num_tokens, dtype=torch.long, device=device)
        new_mask = torch.cat([prefix_mask, attention_mask_before], dim=1)
        return input_embeds, new_mask

    elif embedding_mode == "prefix_after_bos":
        # Concat random embeds after BOS token: [BOS] + [RANDOM] + [rest of prompt]
        prefix = random_embeds_device.unsqueeze(0).expand(batch_size, -1, -1)
        bos_embeds = embeds_before[:, :1, :]          # first token (BOS)
        rest_embeds = embeds_before[:, 1:, :]          # rest of prompt
        input_embeds = torch.cat([bos_embeds, prefix, rest_embeds], dim=1)
        prefix_mask = torch.ones(batch_size, num_tokens, dtype=torch.long, device=device)
        bos_mask = attention_mask_before[:, :1]
        rest_mask = attention_mask_before[:, 1:]
        new_mask = torch.cat([bos_mask, prefix_mask, rest_mask], dim=1)
        return input_embeds, new_mask

    elif embedding_mode == "middle":
        # Replace special token positions with random embeds
        input_embeds = embeds_before.clone()
        for b in range(batch_size):
            start, end = find_special_token_positions(
                input_ids_before[b], tokenizer, model_name, num_tokens
            )
            input_embeds[b, start:end] = random_embeds_device
        return input_embeds, attention_mask_before

    elif embedding_mode == "suffix":
        # Concat: [before_embeds] + [random_embeds] + [after_embeds]
        rand_expanded = random_embeds_device.unsqueeze(0).expand(batch_size, -1, -1)
        rand_mask = torch.ones(batch_size, num_tokens, dtype=torch.long, device=device)

        if input_ids_after is not None and input_ids_after.shape[1] > 0:
            with torch.no_grad():
                embeds_after = model.get_input_embeddings()(input_ids_after)
            input_embeds = torch.cat([embeds_before, rand_expanded, embeds_after], dim=1)
            new_mask = torch.cat([attention_mask_before, rand_mask, attention_mask_after], dim=1)
        else:
            input_embeds = torch.cat([embeds_before, rand_expanded], dim=1)
            new_mask = torch.cat([attention_mask_before, rand_mask], dim=1)
        return input_embeds, new_mask

    else:
        raise ValueError(f"Unknown embedding_mode: {embedding_mode}")


# ---------------------------------------------------------------------------
# Entropy metrics computation
# ---------------------------------------------------------------------------
def compute_entropy_metrics(scores):
    """Compute per-step entropy, varentropy, and top-1 probability from generation scores.

    Args:
        scores: tuple of (batch_size, vocab_size) tensors, one per generation step

    Returns:
        dict with keys 'entropy', 'varentropy', 'top1_prob',
        each mapping to a list of lists (batch_size x num_steps) of floats
    """
    if not scores:
        return {"entropy": [], "varentropy": [], "top1_prob": []}

    batch_size = scores[0].shape[0]
    entropy = [[] for _ in range(batch_size)]
    varentropy = [[] for _ in range(batch_size)]
    top1_prob = [[] for _ in range(batch_size)]

    for score in scores:
        probs = torch.softmax(score.float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)

        ent = -torch.sum(probs * log_probs, dim=-1)  # (batch_size,)
        varent = torch.sum(probs * (log_probs + ent.unsqueeze(-1)) ** 2, dim=-1)
        top1 = probs.max(dim=-1).values

        for b in range(batch_size):
            entropy[b].append(ent[b].item())
            varentropy[b].append(varent[b].item())
            top1_prob[b].append(top1[b].item())

    return {"entropy": entropy, "varentropy": varentropy, "top1_prob": top1_prob}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_completions(
    model, tokenizer, prompts, random_embeds, embedding_mode, model_name,
    num_tokens, suffix_position="after_assistant", batch_size=1, stop_words=None,
    disable_tqdm=False, prompts_after=None, record_entropy=False, **generation_kwargs
):
    """Generate completions with optional random embedding injection.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompt strings (for suffix mode, these are the "before" parts)
        random_embeds: (num_tokens, hidden_dim) or None if mode is 'none'
        embedding_mode: 'none', 'prefix', 'middle', 'suffix'
        model_name: Model name
        num_tokens: Number of embedding tokens
        suffix_position: For suffix mode position
        batch_size: Batch size for generation
        stop_words: List of stop word strings
        prompts_after: List of "after" strings for suffix mode (same length as prompts)
        record_entropy: If True, compute and return per-step entropy metrics
        **generation_kwargs: Additional generation kwargs

    Returns:
        If record_entropy is False: List of generated strings
        If record_entropy is True: (List of generated strings, List of entropy dicts)
            Each entropy dict has keys: 'entropy', 'varentropy', 'top1_prob' (lists of float)
    """
    if stop_words is None:
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"]

    generations = []
    all_entropy = [] if record_entropy else None
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        tokenized = tokenizer(
            batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=False
        )
        batch_input_ids = tokenized.input_ids.to(model.device)
        attention_mask = tokenized.attention_mask.to(model.device)

        # Common generation kwargs
        stop_criteria = KeywordsStoppingCriteria(stop_words, tokenizer)
        common_kwargs = {
            "stopping_criteria": StoppingCriteriaList([stop_criteria]),
            **generation_kwargs,
        }
        if record_entropy:
            common_kwargs["output_scores"] = True
            common_kwargs["return_dict_in_generate"] = True

        if embedding_mode == "none" or random_embeds is None:
            # Baseline: use input_ids directly
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                **common_kwargs,
            )

        elif embedding_mode == "suffix" and prompts_after is not None:
            # Suffix: tokenize before and after separately, concat with random in between
            batch_after = prompts_after[i : i + batch_size]
            tok_after = tokenizer(
                batch_after, padding="longest", return_tensors="pt", add_special_tokens=False
            )
            ids_after = tok_after.input_ids.to(model.device)
            mask_after = tok_after.attention_mask.to(model.device)

            input_embeds, new_mask = inject_embeddings(
                model, tokenizer, batch_input_ids, attention_mask,
                random_embeds, embedding_mode, model_name, num_tokens,
                input_ids_after=ids_after, attention_mask_after=mask_after
            )
            outputs = model.generate(
                inputs_embeds=input_embeds,
                attention_mask=new_mask,
                **common_kwargs,
            )

        else:
            # Prefix or middle: single tokenization
            input_embeds, new_mask = inject_embeddings(
                model, tokenizer, batch_input_ids, attention_mask,
                random_embeds, embedding_mode, model_name, num_tokens
            )
            outputs = model.generate(
                inputs_embeds=input_embeds,
                attention_mask=new_mask,
                **common_kwargs,
            )

        # Extract sequences and entropy metrics
        if record_entropy:
            batch_entropy = compute_entropy_metrics(outputs.scores)
            sequences = outputs.sequences
            del outputs
            torch.cuda.empty_cache()
        else:
            batch_entropy = None
            sequences = outputs

        # For baseline (input_ids), strip input tokens from sequences before decode
        if embedding_mode == "none" or random_embeds is None:
            input_len = batch_input_ids.shape[1]
            sequences = sequences[:, input_len:]

        batch_generations = tokenizer.batch_decode(sequences, skip_special_tokens=True)

        # Remove stop sequences from output
        for idx, prediction in enumerate(batch_generations):
            for stop_word in stop_words:
                batch_generations[idx] = batch_generations[idx].split(stop_word)[0]

        generations.extend(batch_generations)

        # Collect entropy data per sample, trimmed to per-sample stop step
        if record_entropy and batch_entropy:
            for b in range(len(batch_prompts)):
                stop_step = stop_criteria.sample_stop_step.get(
                    b, len(batch_entropy["entropy"][b])
                )
                if b in stop_criteria.sample_stop_step:
                    stop_step -= 1  # exclude stop word token
                all_entropy.append({
                    "entropy": batch_entropy["entropy"][b][:stop_step],
                    "varentropy": batch_entropy["varentropy"][b][:stop_step],
                    "top1_prob": batch_entropy["top1_prob"][b][:stop_step],
                })

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    if not disable_tqdm:
        progress.close()

    if record_entropy:
        return generations, all_entropy
    return generations
