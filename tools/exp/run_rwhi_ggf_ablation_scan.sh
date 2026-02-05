#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PHASE="all"
DRY_RUN=0
CONTINUE_ON_ERROR=0

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE=2
MASTER_PORT_START=29601
MAX_ITERS=""
WORKERS_PER_GPU=""
EVAL_INTERVAL=""
BATCH_SIZE=""
LOG_ROOT=""

BASELINE_CONFIG="configs/racformer_r50_nuimg_704x256_f8.py"
RWHI_GGF_CONFIG="configs/racformer_with_rwhi_ggf.py"

# 训练排程（按你的实验要求）
BASELINE_TOTAL_EPOCHS=2
BASELINE_BATCH_SIZE=4
FOLLOWUP_TOTAL_EPOCHS=20
FOLLOWUP_BATCH_SIZE=2

NUM_RWHI_LIST=(300 450 600)
DIFFUSION_GAMMA_LIST=(0.1 0.2 0.3)
ST_TAU_LIST=(0.03 0.05 0.08)
GGA_BIAS_SCALE_LIST=(0.3 0.5 1.0)
GGA_BIAS_MIN_LIST=(-20 -40 -80)

BASE_NUM_RWHI=450
BASE_DIFFUSION_GAMMA=0.3
BASE_ST_TAU=0.05

declare -a COMMON_OVERRIDES=()
declare -a DONE_EXPERIMENTS=()
declare -a FAILED_EXPERIMENTS=()
NEXT_PORT="${MASTER_PORT_START}"
PORT_RESULT=""
MANIFEST_FILE=""

usage() {
  cat <<'EOF'
Usage:
  bash tools/exp/run_rwhi_ggf_ablation_scan.sh [options]

Options:
  --phase <ablation|scan|all>     Which stage to run (default: all)
  --gpus <ids>                    CUDA_VISIBLE_DEVICES value (default: 0,1)
  --nproc-per-node <n>            torchrun processes per node (default: 2)
  --master-port-start <port>      Starting master port, auto +1 per run (default: 29601)
  --max-iters <n>                 Pass --max_iters to train.py
  --batch-size <n>                Add override: batch_size=n
  --workers-per-gpu <n>           Add override: data.workers_per_gpu=n
  --eval-interval <n>             Add override: eval_config.interval=n
  --log-root <dir>                Output log directory (default: outputs/ablation_scan/<timestamp>)
  --dry-run                       Print commands only, do not execute
  --continue-on-error             Continue to next experiment when one fails
  -h, --help                      Show this help

Examples:
  # Full ablation + scan on 2 GPUs
  bash tools/exp/run_rwhi_ggf_ablation_scan.sh --phase all --gpus 0,1

  # Smoke only (max_iters=1)
  bash tools/exp/run_rwhi_ggf_ablation_scan.sh \
    --phase ablation \
    --max-iters 1 \
    --batch-size 2 \
    --workers-per-gpu 0 \
    --eval-interval 0
EOF
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

take_port() {
  PORT_RESULT="${NEXT_PORT}"
  NEXT_PORT=$((NEXT_PORT + 1))
}

init_common_overrides() {
  COMMON_OVERRIDES=()
  if [[ -n "${BATCH_SIZE}" ]]; then
    COMMON_OVERRIDES+=("batch_size=${BATCH_SIZE}")
  fi
  if [[ -n "${WORKERS_PER_GPU}" ]]; then
    COMMON_OVERRIDES+=("data.workers_per_gpu=${WORKERS_PER_GPU}")
  fi
  if [[ -n "${EVAL_INTERVAL}" ]]; then
    COMMON_OVERRIDES+=("eval_config.interval=${EVAL_INTERVAL}")
  fi
}

write_manifest_header_if_needed() {
  if (( DRY_RUN == 1 )); then
    return
  fi
  mkdir -p "${LOG_ROOT}"
  MANIFEST_FILE="${LOG_ROOT}/manifest.tsv"
  {
    echo -e "phase\texperiment\tstatus\tport\tconfig\tlog_file"
  } > "${MANIFEST_FILE}"
}

append_manifest() {
  local phase="$1"
  local experiment="$2"
  local status="$3"
  local port="$4"
  local config="$5"
  local log_file="$6"
  if (( DRY_RUN == 1 )); then
    return
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${phase}" "${experiment}" "${status}" "${port}" "${config}" "${log_file}" >> "${MANIFEST_FILE}"
}

run_experiment() {
  local phase="$1"
  local experiment="$2"
  local config="$3"
  shift 3
  local -a exp_overrides=("$@")

  take_port
  local port="${PORT_RESULT}"

  local -a cmd=(
    torchrun
    --master_port "${port}"
    --nproc_per_node "${NPROC_PER_NODE}"
    train.py
    --config "${config}"
  )

  if [[ -n "${MAX_ITERS}" ]]; then
    cmd+=(--max_iters "${MAX_ITERS}")
  fi

  local -a merged_overrides=("${COMMON_OVERRIDES[@]}" "${exp_overrides[@]}")
  if (( ${#merged_overrides[@]} > 0 )); then
    cmd+=(--override "${merged_overrides[@]}")
  fi

  local log_file="${LOG_ROOT}/${phase}_${experiment}.log"
  echo "[$(date '+%F %T')] phase=${phase} exp=${experiment} port=${port}"
  echo -n "CMD: "
  print_cmd "${cmd[@]}"

  if (( DRY_RUN == 1 )); then
    DONE_EXPERIMENTS+=("${phase}/${experiment}")
    return
  fi

  set +e
  {
    echo "phase=${phase}"
    echo "experiment=${experiment}"
    echo "port=${port}"
    echo -n "cmd="
    print_cmd "${cmd[@]}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${cmd[@]}"
  } 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e

  if (( rc == 0 )); then
    DONE_EXPERIMENTS+=("${phase}/${experiment}")
    append_manifest "${phase}" "${experiment}" "ok" "${port}" "${config}" "${log_file}"
    return
  fi

  FAILED_EXPERIMENTS+=("${phase}/${experiment}(rc=${rc})")
  append_manifest "${phase}" "${experiment}" "failed(rc=${rc})" "${port}" "${config}" "${log_file}"
  echo "[ERROR] ${phase}/${experiment} failed, rc=${rc}"
  if (( CONTINUE_ON_ERROR == 0 )); then
    exit "${rc}"
  fi
}

build_rwhi_only_overrides() {
  local -n dst="$1"
  dst=(
    "model.pts_bbox_head.use_rwhi=True"
    "model.pts_bbox_head.rwhi_cfg.enabled=True"
    "model.pts_bbox_head.use_alpha=True"
    "model.pts_bbox_head.rwhi_cfg.use_alpha=True"
    "model.pts_bbox_head.loss_alpha_anchor_weight=0.2"
    "model.pts_bbox_head.rwhi_cfg.num_rwhi=${BASE_NUM_RWHI}"
    "model.pts_bbox_head.rwhi_cfg.diffusion_gamma=${BASE_DIFFUSION_GAMMA}"
    "model.pts_bbox_head.rwhi_cfg.st_tau=${BASE_ST_TAU}"
    "model.pts_bbox_head.ggf_cfg.enabled=False"
    "model.pts_bbox_head.ggf_cfg.use_native_rgf=False"
    "model.pts_bbox_head.ggf_cfg.use_gga=False"
    "model.pts_bbox_head.ggf_cfg.use_mgc=False"
    "model.pts_bbox_head.transformer.ggf_cfg.enabled=False"
    "model.pts_bbox_head.transformer.ggf_cfg.use_native_rgf=False"
    "model.pts_bbox_head.transformer.ggf_cfg.use_gga=False"
    "model.pts_bbox_head.transformer.ggf_cfg.use_mgc=False"
  )
}

build_rwhi_ggf_overrides() {
  local -n dst="$1"
  local use_gga="$2"
  local use_mgc="$3"

  dst=(
    "model.pts_bbox_head.use_rwhi=True"
    "model.pts_bbox_head.rwhi_cfg.enabled=True"
    "model.pts_bbox_head.use_alpha=True"
    "model.pts_bbox_head.rwhi_cfg.use_alpha=True"
    "model.pts_bbox_head.loss_alpha_anchor_weight=0.2"
    "model.pts_bbox_head.rwhi_cfg.num_rwhi=${BASE_NUM_RWHI}"
    "model.pts_bbox_head.rwhi_cfg.diffusion_gamma=${BASE_DIFFUSION_GAMMA}"
    "model.pts_bbox_head.rwhi_cfg.st_tau=${BASE_ST_TAU}"
    "model.pts_bbox_head.ggf_cfg.enabled=True"
    "model.pts_bbox_head.ggf_cfg.use_native_rgf=True"
    "model.pts_bbox_head.ggf_cfg.use_gga=${use_gga}"
    "model.pts_bbox_head.ggf_cfg.use_mgc=${use_mgc}"
    "model.pts_bbox_head.transformer.ggf_cfg.enabled=True"
    "model.pts_bbox_head.transformer.ggf_cfg.use_native_rgf=True"
    "model.pts_bbox_head.transformer.ggf_cfg.use_gga=${use_gga}"
    "model.pts_bbox_head.transformer.ggf_cfg.use_mgc=${use_mgc}"
  )
}

run_ablation_phase() {
  echo "========== Run 4-group ablation =========="
  run_experiment "ablation" "baseline_racformer" "${BASELINE_CONFIG}" \
    "total_epochs=${BASELINE_TOTAL_EPOCHS}" \
    "batch_size=${BASELINE_BATCH_SIZE}"

  local -a overrides=(
    "total_epochs=${FOLLOWUP_TOTAL_EPOCHS}"
    "batch_size=${FOLLOWUP_BATCH_SIZE}"
    "model.pts_bbox_head.use_rwhi=False"
    "model.pts_bbox_head.rwhi_cfg.enabled=False"
    "model.pts_bbox_head.ggf_cfg.enabled=False"
    "model.pts_bbox_head.ggf_cfg.use_native_rgf=False"
    "model.pts_bbox_head.ggf_cfg.use_gga=False"
    "model.pts_bbox_head.ggf_cfg.use_mgc=False"
    "model.pts_bbox_head.transformer.ggf_cfg.enabled=False"
    "model.pts_bbox_head.transformer.ggf_cfg.use_native_rgf=False"
    "model.pts_bbox_head.transformer.ggf_cfg.use_gga=False"
    "model.pts_bbox_head.transformer.ggf_cfg.use_mgc=False"
  )
  run_experiment "ablation" "pseudo_baseline_new_path" "${RWHI_GGF_CONFIG}" "${overrides[@]}"

  build_rwhi_only_overrides overrides
  overrides+=(
    "total_epochs=${FOLLOWUP_TOTAL_EPOCHS}"
    "batch_size=${FOLLOWUP_BATCH_SIZE}"
  )
  run_experiment "ablation" "rwhi_only" "${RWHI_GGF_CONFIG}" "${overrides[@]}"

  build_rwhi_ggf_overrides overrides "True" "False"
  overrides+=(
    "total_epochs=${FOLLOWUP_TOTAL_EPOCHS}"
    "batch_size=${FOLLOWUP_BATCH_SIZE}"
  )
  run_experiment "ablation" "rwhi_ggf_rgf_gga" "${RWHI_GGF_CONFIG}" "${overrides[@]}"
}

run_scan_phase() {
  echo "========== Run parameter scan =========="
  local -a followup_schedule=(
    "total_epochs=${FOLLOWUP_TOTAL_EPOCHS}"
    "batch_size=${FOLLOWUP_BATCH_SIZE}"
  )
  local -a overrides=()
  local num_rwhi
  local gamma
  local tau
  local bias_scale
  local bias_min
  local bias_min_tag

  for num_rwhi in "${NUM_RWHI_LIST[@]}"; do
    build_rwhi_only_overrides overrides
    overrides+=("${followup_schedule[@]}")
    overrides+=("model.pts_bbox_head.rwhi_cfg.num_rwhi=${num_rwhi}")
    run_experiment "scan_rwhi" "num_rwhi_${num_rwhi}" "${RWHI_GGF_CONFIG}" "${overrides[@]}"
  done

  for gamma in "${DIFFUSION_GAMMA_LIST[@]}"; do
    build_rwhi_only_overrides overrides
    overrides+=("${followup_schedule[@]}")
    overrides+=("model.pts_bbox_head.rwhi_cfg.diffusion_gamma=${gamma}")
    run_experiment "scan_rwhi" "diffusion_gamma_${gamma//./p}" "${RWHI_GGF_CONFIG}" "${overrides[@]}"
  done

  for tau in "${ST_TAU_LIST[@]}"; do
    build_rwhi_only_overrides overrides
    overrides+=("${followup_schedule[@]}")
    overrides+=("model.pts_bbox_head.rwhi_cfg.st_tau=${tau}")
    run_experiment "scan_rwhi" "st_tau_${tau//./p}" "${RWHI_GGF_CONFIG}" "${overrides[@]}"
  done

  build_rwhi_ggf_overrides overrides "False" "False"
  overrides+=("${followup_schedule[@]}")
  run_experiment "scan_ggf" "rgf_only" "${RWHI_GGF_CONFIG}" "${overrides[@]}"

  build_rwhi_ggf_overrides overrides "True" "False"
  overrides+=("${followup_schedule[@]}")
  run_experiment "scan_ggf" "rgf_plus_gga" "${RWHI_GGF_CONFIG}" "${overrides[@]}"

  build_rwhi_ggf_overrides overrides "True" "True"
  overrides+=("${followup_schedule[@]}")
  run_experiment "scan_ggf" "rgf_plus_gga_plus_mgc" "${RWHI_GGF_CONFIG}" "${overrides[@]}"

  for bias_scale in "${GGA_BIAS_SCALE_LIST[@]}"; do
    for bias_min in "${GGA_BIAS_MIN_LIST[@]}"; do
      bias_min_tag="${bias_min//-/neg}"
      build_rwhi_ggf_overrides overrides "True" "False"
      overrides+=("${followup_schedule[@]}")
      overrides+=(
        "model.pts_bbox_head.ggf_cfg.gga_cfg.bias_scale=${bias_scale}"
        "model.pts_bbox_head.transformer.ggf_cfg.gga_cfg.bias_scale=${bias_scale}"
        "model.pts_bbox_head.ggf_cfg.gga_cfg.bias_min=${bias_min}"
        "model.pts_bbox_head.transformer.ggf_cfg.gga_cfg.bias_min=${bias_min}"
      )
      run_experiment "scan_gga" "bias_scale_${bias_scale//./p}_bias_min_${bias_min_tag}" "${RWHI_GGF_CONFIG}" "${overrides[@]}"
    done
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      PHASE="$2"
      shift 2
      ;;
    --gpus)
      CUDA_VISIBLE_DEVICES_VALUE="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --master-port-start)
      MASTER_PORT_START="$2"
      NEXT_PORT="${MASTER_PORT_START}"
      shift 2
      ;;
    --max-iters)
      MAX_ITERS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --workers-per-gpu)
      WORKERS_PER_GPU="$2"
      shift 2
      ;;
    --eval-interval)
      EVAL_INTERVAL="$2"
      shift 2
      ;;
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${PHASE}" in
  ablation|scan|all)
    ;;
  *)
    echo "Invalid --phase: ${PHASE}" >&2
    usage
    exit 1
    ;;
esac

if [[ -z "${LOG_ROOT}" ]]; then
  LOG_ROOT="outputs/ablation_scan/$(date '+%Y%m%d_%H%M%S')"
fi

init_common_overrides
write_manifest_header_if_needed

echo "Root dir: ${ROOT_DIR}"
echo "Phase: ${PHASE}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES_VALUE}"
echo "nproc_per_node: ${NPROC_PER_NODE}"
echo "Master port start: ${MASTER_PORT_START}"
echo "Dry run: ${DRY_RUN}"
if [[ -n "${MAX_ITERS}" ]]; then
  echo "Max iters: ${MAX_ITERS}"
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  echo "Override batch_size: ${BATCH_SIZE}"
fi
if [[ -n "${WORKERS_PER_GPU}" ]]; then
  echo "Override workers_per_gpu: ${WORKERS_PER_GPU}"
fi
if [[ -n "${EVAL_INTERVAL}" ]]; then
  echo "Override eval interval: ${EVAL_INTERVAL}"
fi
if (( DRY_RUN == 0 )); then
  echo "Log root: ${LOG_ROOT}"
  echo "Manifest: ${MANIFEST_FILE}"
fi

if [[ "${PHASE}" == "ablation" || "${PHASE}" == "all" ]]; then
  run_ablation_phase
fi

if [[ "${PHASE}" == "scan" || "${PHASE}" == "all" ]]; then
  run_scan_phase
fi

echo "========== Done =========="
echo "Completed: ${#DONE_EXPERIMENTS[@]}"
echo "Failed: ${#FAILED_EXPERIMENTS[@]}"
if (( ${#DONE_EXPERIMENTS[@]} > 0 )); then
  printf '  - %s\n' "${DONE_EXPERIMENTS[@]}"
fi
if (( ${#FAILED_EXPERIMENTS[@]} > 0 )); then
  printf '  - %s\n' "${FAILED_EXPERIMENTS[@]}"
  exit 1
fi
