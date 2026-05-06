#!/bin/bash
# Apply RSP / GPPO patches on top of an installed verl 0.5.0.
#
# Usage:
#   conda activate rsp-dapo
#   bash verl_patches/apply_patches.sh
#
# What it does:
#   - Finds the verl install location via `python -c "import verl; ..."`.
#   - Copies each patched file in this directory to the corresponding
#     path inside the installed verl package, backing up the original
#     to <file>.orig (only on the first run).
#
# Idempotent: re-running after a verl reinstall reapplies the patches.
# Safe: keeps a .orig backup of each replaced file.

set -e

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Locate the installed verl package.
VERL_DIR=$(python -c "import os, verl; print(os.path.dirname(verl.__file__))" 2>/dev/null || true)
if [ -z "$VERL_DIR" ] || [ ! -d "$VERL_DIR" ]; then
    echo "[ERR] verl package not found. Install with: pip install verl==0.5.0" >&2
    exit 1
fi
echo "[INFO] verl install location: $VERL_DIR"

# 2. Files to patch (relative to PATCH_DIR and to VERL_DIR — they must match).
FILES=(
    "workers/fsdp_workers.py"
    "workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    "trainer/ppo/ray_trainer.py"
    "trainer/ppo/core_algos.py"
)

# 3. Apply each patch.
for rel in "${FILES[@]}"; do
    src="$PATCH_DIR/$rel"
    dst="$VERL_DIR/$rel"

    if [ ! -f "$src" ]; then
        echo "[ERR] missing patch source: $src" >&2
        exit 1
    fi
    if [ ! -f "$dst" ]; then
        echo "[ERR] missing target inside verl install: $dst" >&2
        echo "      Is verl version 0.5.0? Other versions may have different layout." >&2
        exit 1
    fi

    # Back up original if not already backed up.
    if [ ! -f "$dst.orig" ]; then
        cp "$dst" "$dst.orig"
        echo "[INFO] backed up: $dst -> $dst.orig"
    fi

    cp "$src" "$dst"
    echo "[OK]   patched: $rel"
done

echo ""
echo "[DONE] $((${#FILES[@]})) patches applied."
echo "       To revert: for each *.orig file under $VERL_DIR, rename back."
