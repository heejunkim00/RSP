#!/bin/bash
# Reproduce the accuracy tables (Table 2, Appendix Table 10).
#
# For every (model config, benchmark): one greedy baseline run, then
# prefix / suffix RSP runs for each seed in SEEDS.  L per (model, benchmark)
# follows Appendix Table "Number of RSP tokens".
#
# Usage (from the repo root):
#   bash scripts/main_tables.sh                       # all 7 configs, seeds 15..19
#   CONFIGS="qwen25_math_1.5b_inst" bash scripts/main_tables.sh
#   SEEDS="15" MODES="suffix" DATASETS="math500" bash scripts/main_tables.sh
#   python scripts/aggregate.py output/main_tables     # mean +- std over seeds
#
# Env overrides: CONFIGS, SEEDS, MODES, DATASETS, OUT_DIR, NUM_TEST_SAMPLE (smoke test),
#                MODEL_OVERRIDE (use a local model path for every selected config)
set -e
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-15 16 17 18 19}"
MODES="${MODES:-prefix suffix}"
DATASETS="${DATASETS:-math500 gsm8k aime24}"
OUT_DIR="${OUT_DIR:-output/main_tables}"
EXTRA=""
[ -n "${NUM_TEST_SAMPLE}" ] && EXTRA="--num_test_sample ${NUM_TEST_SAMPLE}"

# config_name | HF model | batch | prompt_type | chat_template(0/1) | L_math500 | L_gsm8k | L_aime24
ALL_CONFIGS=(
  "llama31_8b_inst            | meta-llama/Llama-3.1-8B-Instruct | 16 | mathstral       | 1 | 15 | 20 | 15"
  "qwen25_math_7b_inst        | Qwen/Qwen2.5-Math-7B-Instruct    | 16 | qwen25-math-cot | 0 | 20 | 20 | 20"
  "qwen25_math_1.5b_inst      | Qwen/Qwen2.5-Math-1.5B-Instruct  | 32 | qwen25-math-cot | 0 | 20 | 10 | 20"
  "qwen25_math_7b_base_chatml | Qwen/Qwen2.5-Math-7B             | 16 | qwen25-math-cot | 0 | 20 | 20 | 20"
  "qwen25_math_1.5b_base_chatml | Qwen/Qwen2.5-Math-1.5B         | 32 | qwen25-math-cot | 0 | 20 | 20 | 10"
  "qwen25_math_7b_base_raw    | Qwen/Qwen2.5-Math-7B             | 16 | mathstral       | 0 | 10 | 10 | 20"
  "qwen25_math_1.5b_base_raw  | Qwen/Qwen2.5-Math-1.5B           | 32 | mathstral       | 0 | 10 | 10 | 20"
)
CONFIGS="${CONFIGS:-llama31_8b_inst qwen25_math_7b_inst qwen25_math_1.5b_inst qwen25_math_7b_base_chatml qwen25_math_1.5b_base_chatml qwen25_math_7b_base_raw qwen25_math_1.5b_base_raw}"

max_tokens() { [ "$1" == "aime24" ] && echo 4096 || echo 3072; }

run_cell() {  # model batch prompt_type chat_flag dataset mode L seed out_dir
  local MODEL=$1 BATCH=$2 PT=$3 CT=$4 DS=$5 MODE=$6 L=$7 SEED=$8 OUT=$9
  local FLAGS=""
  [ "$CT" == "1" ] && FLAGS="$FLAGS --apply_chat_template"
  [ "$MODE" != "none" ] && FLAGS="$FLAGS --num_embedding_tokens $L --embedding_seed $SEED --suffix_position before_im_end"
  if [ -f "$OUT/$DS/"*_metrics.json ] 2>/dev/null; then echo "skip (done): $OUT/$DS"; return; fi
  echo "===== $OUT / $DS ====="
  python eval.py \
    --model_name_or_path "$MODEL" --data_dir eval/data --data_names "$DS" --split test \
    --prompt_type "$PT" --embedding_mode "$MODE" $FLAGS \
    --batch_size "$BATCH" --max_tokens_per_call "$(max_tokens "$DS")" --temperature 0 --seed 0 \
    --output_dir "$OUT" $EXTRA
}

for line in "${ALL_CONFIGS[@]}"; do
  IFS='|' read -r NAME MODEL BATCH PT CT L_MATH L_GSM L_AIME <<< "$(echo "$line" | tr -d ' ')"
  [[ " $CONFIGS " == *" $NAME "* ]] || continue
  [ -n "${MODEL_OVERRIDE}" ] && MODEL="${MODEL_OVERRIDE}"
  for DS in $DATASETS; do
    case $DS in math500) L=$L_MATH;; gsm8k) L=$L_GSM;; aime24) L=$L_AIME;; esac
    run_cell "$MODEL" "$BATCH" "$PT" "$CT" "$DS" none 0 0 "$OUT_DIR/$NAME/baseline"
    for MODE in $MODES; do
      for SEED in $SEEDS; do
        run_cell "$MODEL" "$BATCH" "$PT" "$CT" "$DS" "$MODE" "$L" "$SEED" "$OUT_DIR/$NAME/${MODE}_L${L}_seed${SEED}"
      done
    done
  done
done
echo "Done. Aggregate with: python scripts/aggregate.py $OUT_DIR"
