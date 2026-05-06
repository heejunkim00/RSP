# RSP

## Part 1 — RSP Eval

All reported numbers in this part were obtained on a single NVIDIA A6000 GPU.

### Setup

```bash
conda create -n rsp python=3.10 -y
conda activate rsp
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.3 --no-build-isolation
```

### Data

```bash
python scripts/download_data.py
```

Downloads MATH-500, GSM8K, and AIME 2024 to `eval/data/{dataset}/test.jsonl`.

### Run evaluation

`eval.py` runs a single (model, dataset, injection mode, num_embedding_tokens)
cell. Sweep over the cells listed below to reproduce the token-ablation table.

#### Selectable options

| Flag | Choices |
|---|---|
| `--model_name_or_path` | `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-Math-1.5B-Instruct`, `Qwen/Qwen2.5-Math-7B-Instruct`, `Qwen/Qwen2.5-Math-1.5B`, `Qwen/Qwen2.5-Math-7B` |
| `--data_names` | `math500`, `gsm8k`, `aime24` |
| `--prompt_type` | `qwen25-math-cot`, `mathstral` (use `--apply_chat_template` with `mathstral`) |
| `--embedding_mode` | `prefix`, `middle`, `suffix` |
| `--num_embedding_tokens` | `10`, `15`, `20` (ignored when `--embedding_mode none`) |

#### Example

Qwen2.5-Math-1.5B + MATH-500 + prefix + 20 tokens:

```bash
python eval.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B \
    --data_dir eval/data --data_names math500 --split test \
    --prompt_type qwen25-math-cot \
    --batch_size 32 --max_tokens_per_call 3072 \
    --temperature 0 --seed 0 \
    --embedding_mode prefix --num_embedding_tokens 20 \
    --output_dir output/prefix_qwen1.5b_tok20_math500
```

---

## Part 2 — DAPO with RSP

DAPO and DAPO+RSP training of Qwen2.5-Math-7B on the SimpleRL math dataset
(level 3–5), evaluated on GSM8K, MATH-500, College Math, Minerva Math, and
AIME 2024. We extend
[simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason), which
implements GRPO, with a DAPO implementation.

This part requires a **separate conda environment** (`rsp-dapo`) — different
CUDA / torch / transformers versions from Part 1 — and was run on H100 / B200
class GPUs. All commands below are relative to `dapo/`.

### Setup

```bash
conda create -n rsp-dapo python=3.10 -y
conda activate rsp-dapo

pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
grep -vE "^(flash-attn|verl|latex2sympy2)" dapo/requirements.txt | pip install -r /dev/stdin
pip install verl==0.5.0 latex2sympy2==1.9.1 --no-deps

# flash-attn must be built against a CUDA toolkit matching torch's CUDA 12.8.
# Skip the next two lines if `nvcc --version` already reports 12.x; otherwise
# install a matching toolkit into the env first:
conda install -c nvidia/label/cuda-12.8.0 cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
pip install flash-attn==2.8.3 --no-build-isolation

bash dapo/verl_patches/apply_patches.sh
```

### Data

```bash
cd dapo
python data/prepare_data.py            # train/test parquet
python eval/download_eval_data.py      # 5 benchmark jsonls
```

### Train

```bash
cd dapo
GPUS=0,1,2,3 bash train/train_dapo.sh        # baseline
GPUS=0,1,2,3 bash train/train_dapo_rsp.sh    # +RSP
```

| Setting | DAPO | DAPO + RSP |
| --- | --- | --- |
| Max prompt / response | 2048 / 2048 | 2028 / 2048 |
| RSP (suffix, 20 tokens) | — | enabled, in log-prob |
| Train batch / mini-batch | 1024 / 256 | 1024 / 256 |
| Rollout n / temperature / top-k | 8 / 1.0 / 50 | 8 / 1.0 / 50 |
| Learning rate | 5e-7 | 5e-7 |
| Clip low / high / c | 0.2 / 0.28 / 10.0 | 0.2 / 0.28 / 10.0 |
| KL loss / coef | off / 0.0 | off / 0.0 |
| Loss aggregation | token-mean | token-mean |
| Save / test freq, epochs | 10 / 10, 13 | 10 / 10, 13 |

Pretrained DAPO and DAPO+RSP checkpoints (steps 10–100) are available at
[rsp-anonymous-2026/RandomSoftPrompt](https://huggingface.co/rsp-anonymous-2026/RandomSoftPrompt)
for evaluation without retraining.

### Evaluate

```bash
cd dapo
bash eval/eval_dapo.sh --run_name <run_name> --init_model Qwen/Qwen2.5-Math-7B
```

| Benchmark | Decoding | n | Metric |
| --- | --- | --- | --- |
| GSM8K, MATH-500, College Math, Minerva Math | greedy (t=0) | 1 | mean@1 |
| AIME 2024 | t=1.0, top_p=0.95 | 32 | Avg@32 |
