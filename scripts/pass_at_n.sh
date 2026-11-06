#!/bin/bash
# Reproduce the Pass@N experiment (Figure 4) on Qwen2.5-Math-1.5B-Instruct.
#
# Four conditions, 16 samples per problem:
#   (i)   baseline        : tau=1.0, no RSP                          (one run, n_sampling=16)
#   (ii)  rsp_single_seed : tau=1.0, one RSP shared by all samples    (one run, n_sampling=16)
#   (iii) rsp_indep_greedy: tau=0,   a fresh RSP per sample           (16 runs, embedding_seed 0..15)
#   (iv)  rsp_indep_temp  : tau=1.0, a fresh RSP per sample           (16 runs, embedding_seed 0..15)
#
# Usage (from the repo root):
#   bash scripts/pass_at_n.sh                  # math500 and aime24
#   DATASETS="aime24" bash scripts/pass_at_n.sh
#   python scripts/pass_at_n.py output/pass_at_n/math500 --plot pass_at_n_math500.pdf
#
# Env overrides: MODEL, DATASETS, OUT_DIR, N (samples per problem), NUM_TEST_SAMPLE (smoke test)
set -e
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}"   # override with a local path if needed
DATASETS="${DATASETS:-math500 aime24}"
OUT_DIR="${OUT_DIR:-output/pass_at_n}"
N="${N:-16}"
EXTRA=""
[ -n "${NUM_TEST_SAMPLE}" ] && EXTRA="--num_test_sample ${NUM_TEST_SAMPLE}"

COMMON="--model_name_or_path $MODEL --data_dir eval/data --split test --prompt_type qwen25-math-cot --batch_size 32 --seed 0 $EXTRA"
RSP="--embedding_mode suffix --suffix_position before_im_end"

for DS in $DATASETS; do
  case $DS in
    math500) L=10; MAXTOK=3072;;
    aime24)  L=20; MAXTOK=4096;;
    *) echo "unknown dataset $DS"; exit 1;;
  esac
  O="$OUT_DIR/$DS"

  echo "===== (i) baseline, tau=1.0, n=$N ====="
  python eval.py $COMMON --data_names $DS --max_tokens_per_call $MAXTOK \
    --temperature 1.0 --top_p 0.95 --n_sampling $N --embedding_mode none \
    --output_dir "$O/baseline"

  echo "===== (ii) RSP single seed, tau=1.0, n=$N ====="
  python eval.py $COMMON --data_names $DS --max_tokens_per_call $MAXTOK \
    --temperature 1.0 --top_p 0.95 --n_sampling $N $RSP --num_embedding_tokens $L --embedding_seed 15 \
    --output_dir "$O/rsp_single_seed"

  for ESEED in $(seq 0 $((N - 1))); do
    echo "===== (iii) RSP indep. seed $ESEED, greedy ====="
    python eval.py $COMMON --data_names $DS --max_tokens_per_call $MAXTOK \
      --temperature 0 --n_sampling 1 $RSP --num_embedding_tokens $L --embedding_seed $ESEED \
      --output_dir "$O/rsp_indep_greedy/eseed$ESEED"

    echo "===== (iv) RSP indep. seed $ESEED, tau=1.0 ====="
    python eval.py $COMMON --data_names $DS --max_tokens_per_call $MAXTOK \
      --temperature 1.0 --top_p 0.95 --n_sampling 1 $RSP --num_embedding_tokens $L --embedding_seed $ESEED \
      --output_dir "$O/rsp_indep_temp/eseed$ESEED"
  done

  python scripts/pass_at_n.py "$O" --plot "$O/pass_at_n_$DS.pdf"
done
