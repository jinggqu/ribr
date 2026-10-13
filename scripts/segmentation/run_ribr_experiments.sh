#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
cd "${SCRIPT_DIR}/../.."

export_model_runtime_env

DEVICE="${DEVICE:-$(default_primary_device)}"
VARIANT="${VARIANT:-full}"
case "$VARIANT" in
    conv-only|gsbr-only|bri-only|bri-gsbr|bri-rrc|full) ;;
    *) echo "Unsupported VARIANT=$VARIANT" >&2; exit 1 ;;
esac
EXP_PREFIX="ribr-${VARIANT}"
validate_experiment_device "$DEVICE"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
DATA_ROOT="${DATA_ROOT:-${ULTRASOUND_DATA_ROOT:-${HOME}/datasets/ultrasound}}"
DATA_VIEW="${DATA_VIEW:-data_views/ribr-v1.csv}"
SPLIT_ROOT="${SPLIT_ROOT:-data_splits/ribr-v1}"
DERIVATIVE_MANIFEST="${DERIVATIVE_MANIFEST:-derivatives/resize256-v1/manifest.csv}"
SEEDS=(${SEEDS:-1 2 3})
TRAIN_DATASETS=(${TRAIN_DATASETS:-LN-INT BUSI DDTI TN3K Prostate})
IMG_SIZE="${IMG_SIZE:-256}"
AMORTIZED_EPOCHS="${AMORTIZED_EPOCHS:-120}"
AMORTIZED_LR="${AMORTIZED_LR:-1.5e-4}"
AMORTIZED_VAL_FREQ="${AMORTIZED_VAL_FREQ:-2}"
PATIENCE="${PATIENCE:-20}"
RUN_LOG="${RUN_LOG:-logs/ribr_experiments.log}"

DATA_PROTOCOL_ARGS=(
    --data-view "$DATA_VIEW"
    --split-root "$SPLIT_ROOT"
    --derivative-manifest "$DERIVATIVE_MANIFEST"
)
RIBR_SURFACE_ARGS=(
    --variant "$VARIANT"
    --checkpoint-score balanced_seg
    --boundary-loss-alpha 0.6
    --boundary-factor 5.0
    --band-sdf-alpha 0.18
    --speckle-smooth-alpha 0.03
    --residual-l2-alpha 0.06
    --gate-tau 0.65
    --gate-gamma 12.0
    --risk-route-scale 0.25
    --boundary-residual-scale 1.0
)
if [[ "${ALLOW_NONPAPER_SPLITS:-0}" == 1 ]]; then
    RIBR_SURFACE_ARGS+=(--allow-nonpaper-splits)
fi

prepare_run_log "$RUN_LOG"
exec > >(tee -a "$RUN_LOG") 2>&1

run_train_and_internal_test() {
    local seed="$1"
    local dataset="$2"
    uv run python -m src.methods.ribr.train_seg \
        --exp "${EXP_PREFIX}-${seed}" \
        --dataset "$dataset" \
        --data-root "$DATA_ROOT" \
        "${DATA_PROTOCOL_ARGS[@]}" \
        --img-size "$IMG_SIZE" \
        --seed "$seed" \
        --device "$DEVICE" \
        --output-dir "$OUTPUT_DIR" \
        --weak-augs \
        --patience "$PATIENCE" \
        --amortized-epochs "$AMORTIZED_EPOCHS" \
        --amortized-lr "$AMORTIZED_LR" \
        --amortized-val-freq "$AMORTIZED_VAL_FREQ" \
        "${RIBR_SURFACE_ARGS[@]}"
}

run_external_test() {
    local seed="$1"
    local train_dataset="$2"
    local test_dataset="$3"
    local checkpoint="${OUTPUT_DIR}/${EXP_PREFIX}-${seed}/${train_dataset}/${IMG_SIZE}/train/best_model.pt"
    uv run python -m src.methods.ribr.train_seg \
        --exp "${EXP_PREFIX}-${seed}" \
        --dataset "$test_dataset" \
        --data-root "$DATA_ROOT" \
        "${DATA_PROTOCOL_ARGS[@]}" \
        --img-size "$IMG_SIZE" \
        --seed "$seed" \
        --device "$DEVICE" \
        --output-dir "$OUTPUT_DIR" \
        --checkpoint "$checkpoint" \
        --test \
        "${RIBR_SURFACE_ARGS[@]}"
}

for dataset in "${TRAIN_DATASETS[@]}"; do
    EXTERNALS=($(resolve_external_datasets_for_task segmentation "$dataset"))
    for seed in "${SEEDS[@]}"; do
        echo "=== RIBR seed=${seed} train=${dataset} device=${DEVICE} ==="
        if has_completed_results "$EXP_PREFIX" "$seed" "$dataset" "$IMG_SIZE" test; then
            announce_skip_completed "ribr-${seed} ${dataset} internal test"
        elif has_training_complete_marker "$EXP_PREFIX" "$seed" "$dataset" "$IMG_SIZE"; then
            announce_resume_eval "ribr-${seed} ${dataset} internal test"
            uv run python -m src.methods.ribr.train_seg \
                --exp "${EXP_PREFIX}-${seed}" \
                --dataset "$dataset" \
                --data-root "$DATA_ROOT" \
                "${DATA_PROTOCOL_ARGS[@]}" \
                --img-size "$IMG_SIZE" \
                --seed "$seed" \
                --device "$DEVICE" \
                --output-dir "$OUTPUT_DIR" \
                --test \
                "${RIBR_SURFACE_ARGS[@]}"
        else
            run_train_and_internal_test "$seed" "$dataset"
        fi
        for external_dataset in "${EXTERNALS[@]}"; do
            echo "=== RIBR seed=${seed} external=${external_dataset} device=${DEVICE} ==="
            if has_completed_results "$EXP_PREFIX" "$seed" "$external_dataset" "$IMG_SIZE" test; then
                announce_skip_completed "ribr-${seed} ${dataset} -> ${external_dataset} external test"
                continue
            fi
            if ! internal_run_finished "$EXP_PREFIX" "$seed" "$dataset" "$IMG_SIZE"; then
                echo "Skip external eval until internal run finishes: ribr-${seed} ${dataset} -> ${external_dataset}"
                continue
            fi
            run_external_test "$seed" "$dataset" "$external_dataset"
        done
    done
done

print_completion_banner
