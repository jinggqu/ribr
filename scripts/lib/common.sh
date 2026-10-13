#!/usr/bin/env bash

export_model_runtime_env() {
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    export HF_HUB_DISABLE_TELEMETRY=1
    export HF_HUB_OFFLINE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

default_primary_device() {
    echo "${PRIMARY_DEVICE:-cuda:0}"
}

validate_experiment_device() {
    local device="$1"
    if [[ "$device" == "cpu" || "$device" == "mps" || "$device" =~ ^cuda(:[0-9]+)?$ ]]; then
        return 0
    fi
    echo "Unsupported DEVICE=${device}; use cpu, mps, cuda, or cuda:<index>." >&2
    return 1
}

prepare_run_log() {
    local run_log="$1"
    mkdir -p logs
    mkdir -p "$(dirname "$run_log")"
}

result_dir_for_slot() {
    local exp_prefix="$1"
    local seed="$2"
    local dataset="$3"
    local img_size="$4"
    local phase="${5:-test}"
    local output_root="${OUTPUT_DIR:-runs}"
    echo "${output_root}/${exp_prefix}-${seed}/${dataset}/${img_size}/${phase}"
}

has_completed_results() {
    local exp_prefix="$1"
    local seed="$2"
    local dataset="$3"
    local img_size="$4"
    local phase="${5:-test}"
    local result_root
    result_root="$(result_dir_for_slot "$exp_prefix" "$seed" "$dataset" "$img_size" "$phase")"
    if [[ ! -d "$result_root" ]]; then
        return 1
    fi
    [[ -n "$(find "$result_root" -mindepth 2 -maxdepth 2 -type f -name 'results.csv' -print -quit)" ]]
}

announce_skip_completed() {
    local label="$1"
    echo "Skip completed: ${label}"
}

announce_resume_eval() {
    local label="$1"
    echo "Resume eval only: ${label}"
}

training_complete_marker_path() {
    local exp_prefix="$1"
    local seed="$2"
    local dataset="$3"
    local img_size="$4"
    local output_root="${OUTPUT_DIR:-runs}"
    echo "${output_root}/${exp_prefix}-${seed}/${dataset}/${img_size}/train/train.complete"
}

has_training_complete_marker() {
    local exp_prefix="$1"
    local seed="$2"
    local dataset="$3"
    local img_size="$4"
    [[ -f "$(training_complete_marker_path "$exp_prefix" "$seed" "$dataset" "$img_size")" ]]
}

internal_run_finished() {
    local exp_prefix="$1"
    local seed="$2"
    local dataset="$3"
    local img_size="$4"
    if has_completed_results "$exp_prefix" "$seed" "$dataset" "$img_size" test; then
        return 0
    fi
    has_training_complete_marker "$exp_prefix" "$seed" "$dataset" "$img_size"
}

resolve_external_datasets_for_task() {
    local task="$1"
    shift
    local -a datasets=("$@")
    if [[ -n "${EXTERNAL_DATASETS+x}" ]]; then
        echo "$EXTERNAL_DATASETS"
        return
    fi
    if [[ -n "${EXTERNAL_DATASET+x}" ]]; then
        echo "$EXTERNAL_DATASET"
        return
    fi
    if [[ "${#datasets[@]}" -eq 1 ]]; then
        case "$task" in
        segmentation)
            default_segmentation_external_datasets_for "${datasets[0]}"
            ;;
        *)
            echo "Unsupported task for external dataset resolution: $task" >&2
            return 1
            ;;
        esac
        return
    fi
    echo ""
}

default_segmentation_external_datasets_for() {
    local dataset="$1"
    case "$dataset" in
    LN-INT)
        echo "LymphUS-Center-1 LymphUS-Center-2"
        ;;
    BUSI)
        echo "Breast-Lesions-USG BUS-UCLM"
        ;;
    *)
        echo ""
        ;;
    esac
}

print_completion_banner() {
    echo ""
    echo "================================================"
    echo "Complete!"
    echo "================================================"
}
