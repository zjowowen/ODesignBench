#!/usr/bin/env bash
# Run AlphaFold3 via a bundled Apptainer/Singularity runtime when Docker is unavailable.
# Arguments are compatible with refold/run_af3.sh:
#   $1 exp_name
#   $2 input_json
#   $3 output_dir
#   $4 gpus (single gpu id or comma list; first id will be used)
#   $5 run_data_pipeline (True/False)
#   $6 cache_dir
#   $7 num_diffusion_samples (optional, default=1)

set -euo pipefail

exp_name="${1:-af3_job}"
input_json="${2:?input_json is required}"
output_path="${3:?output_dir is required}"
gpus="${4:-0}"
run_data_pipeline="${5:-False}"
cache_dir="${6:-}"
num_diffusion_samples="${7:-1}"

AF3_SIF_IMAGE="${AF3_SIF_IMAGE:-/mnt/shared-storage-user/ailab-lgl/aurodesign/design_pipeline/af3_ppi.sif}"
AF3_WORKDIR="${AF3_WORKDIR:-/mnt/shared-storage-user/ailab-lgl/aurodesign/design_pipeline}"
AF3_CODE_SUBDIR="${AF3_CODE_SUBDIR:-alphafold3_full}"
AF3_BASE="${AF3_BASE:-/path/to/af3}"
AF3_PUBLIC_DB="${AF3_PUBLIC_DB:-/mnt/shared-storage-user/beam/share/af3/af3_database}"
AF3_MODEL_DIR="${AF3_MODEL_DIR:-}"
AF3_APPTAINER_BIN="${AF3_APPTAINER_BIN:-/mnt/shared-storage-user/ailab-lgl/aurodesign/design_pipeline/apptainer/bin/apptainer}"
AF3_ASSETS="${AF3_ASSETS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../assets}"
AF3_APPTAINER_UNSQUASH="${AF3_APPTAINER_UNSQUASH:-0}"
AF3_APPTAINER_USERNS="${AF3_APPTAINER_USERNS:-1}"
AF3_APPTAINER_NO_PRIVS="${AF3_APPTAINER_NO_PRIVS:-1}"
AF3_APPTAINER_WRITABLE_TMPFS="${AF3_APPTAINER_WRITABLE_TMPFS:-1}"

if [[ -z "$AF3_MODEL_DIR" ]]; then
  if [[ -d "$AF3_BASE/models" ]]; then
    AF3_MODEL_DIR="$AF3_BASE/models"
  elif [[ -d "$AF3_BASE/model" ]]; then
    AF3_MODEL_DIR="$AF3_BASE/model"
  fi
fi

if [[ ! -x "$AF3_APPTAINER_BIN" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    AF3_APPTAINER_BIN="$(command -v apptainer)"
  elif command -v singularity >/dev/null 2>&1; then
    AF3_APPTAINER_BIN="$(command -v singularity)"
  else
    echo "ERROR: apptainer runtime not found." >&2
    echo "  Tried AF3_APPTAINER_BIN=$AF3_APPTAINER_BIN" >&2
    exit 1
  fi
fi

if [[ ! -f "$AF3_SIF_IMAGE" ]]; then
  echo "ERROR: AF3_SIF_IMAGE not found: $AF3_SIF_IMAGE" >&2
  exit 1
fi
if [[ ! -d "$AF3_WORKDIR/$AF3_CODE_SUBDIR" ]]; then
  echo "ERROR: AF3 code dir not found: $AF3_WORKDIR/$AF3_CODE_SUBDIR" >&2
  exit 1
fi
if [[ ! -d "$AF3_PUBLIC_DB" ]]; then
  echo "ERROR: AF3_PUBLIC_DB not found: $AF3_PUBLIC_DB" >&2
  exit 1
fi
if [[ ! -d "$AF3_MODEL_DIR" ]]; then
  echo "ERROR: AF3 model dir not found: $AF3_MODEL_DIR" >&2
  exit 1
fi
if [[ ! -d "$AF3_ASSETS" ]]; then
  echo "ERROR: AF3_ASSETS not found: $AF3_ASSETS" >&2
  exit 1
fi

GPU0="${gpus%%,*}"
export CUDA_VISIBLE_DEVICES="$GPU0"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU0"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$GPU0"

mkdir -p "$output_path"
AF3_LOG_DIR="$(dirname "$output_path")/af3_log"
mkdir -p "$AF3_LOG_DIR"
AF3_WRAPPER_LOG="${AF3_WRAPPER_LOG:-$output_path/af3_wrapper_gpu${GPU0}.log}"
AF3_APPTAINER_TMPDIR="${AF3_APPTAINER_TMPDIR:-$AF3_LOG_DIR/apptainer_tmp_gpu${GPU0}_$$}"
AF3_APPTAINER_CACHEDIR="${AF3_APPTAINER_CACHEDIR:-$AF3_LOG_DIR/apptainer_cache_gpu${GPU0}_$$}"
mkdir -p "$AF3_APPTAINER_TMPDIR" "$AF3_APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="$AF3_APPTAINER_TMPDIR"
export APPTAINER_CACHEDIR="$AF3_APPTAINER_CACHEDIR"
export SINGULARITY_TMPDIR="$AF3_APPTAINER_TMPDIR"
export SINGULARITY_CACHEDIR="$AF3_APPTAINER_CACHEDIR"

exec > >(tee -a "$AF3_WRAPPER_LOG") 2>&1

BIND_ARGS=(
  -B "$AF3_WORKDIR/$AF3_CODE_SUBDIR:/root/af3_code"
  -B "$AF3_MODEL_DIR:/root/models"
  -B "$AF3_PUBLIC_DB:/root/public_databases"
  -B "$AF3_ASSETS:/assets"
  -B "$input_json:$input_json"
  -B "$output_path:$output_path"
  -B "$output_path:/root/af_output"
)

if [[ -d "$AF3_WORKDIR/$AF3_CODE_SUBDIR/site_packages/share" ]]; then
  BIND_ARGS+=(-B "$AF3_WORKDIR/$AF3_CODE_SUBDIR/site_packages/share:/usr/lib/python3.11/site-packages/share")
fi

GPU_LIB_BINDS=(
  "/usr/local/cuda/lib64:/usr/local/cuda/lib64"
  "/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/lib64"
  "/usr/local/cuda:/usr/local/cuda"
  "/usr/local/nvidia/lib64:/usr/local/nvidia/lib64"
  "/usr/local/cuda-12.8/compat/lib.real:/usr/local/cuda-12.8/compat/lib.real"
)
for bind_spec in "${GPU_LIB_BINDS[@]}"; do
  host_path="${bind_spec%%:*}"
  if [[ -e "$host_path" ]]; then
    BIND_ARGS+=(-B "$bind_spec")
  fi
done

if [[ -n "${HOME:-}" && -d "${HOME:-}" ]]; then
  BIND_ARGS+=(-B "${HOME}:${HOME}")
fi

if [[ -n "$cache_dir" && "$cache_dir" != "None" && "$cache_dir" != "" ]]; then
  mkdir -p "$cache_dir"
  BIND_ARGS+=(-B "$cache_dir:$cache_dir")
fi

PY_SCRIPT="/root/af3_code/multi_run_alphafold_dugang.py"
if [[ ! -f "$AF3_WORKDIR/$AF3_CODE_SUBDIR/multi_run_alphafold_dugang.py" ]]; then
  echo "ERROR: missing multi_run_alphafold_dugang.py under $AF3_WORKDIR/$AF3_CODE_SUBDIR" >&2
  exit 1
fi

echo "[af3_ppi_sif] exp_name=$exp_name gpu=$GPU0 input_json=$input_json output=$output_path"
echo "[af3_ppi_sif] apptainer_tmp=$AF3_APPTAINER_TMPDIR apptainer_cache=$AF3_APPTAINER_CACHEDIR unsquash=$AF3_APPTAINER_UNSQUASH userns=$AF3_APPTAINER_USERNS no_privs=$AF3_APPTAINER_NO_PRIVS writable_tmpfs=$AF3_APPTAINER_WRITABLE_TMPFS"

APPTAINER_CMD=("$AF3_APPTAINER_BIN" exec --nv)
if [[ "$AF3_APPTAINER_WRITABLE_TMPFS" == "1" || "$AF3_APPTAINER_WRITABLE_TMPFS" == "true" || "$AF3_APPTAINER_WRITABLE_TMPFS" == "TRUE" ]]; then
  APPTAINER_CMD+=(--writable-tmpfs)
fi
if [[ "$AF3_APPTAINER_UNSQUASH" == "1" || "$AF3_APPTAINER_UNSQUASH" == "true" || "$AF3_APPTAINER_UNSQUASH" == "TRUE" ]]; then
  APPTAINER_CMD+=(--unsquash)
fi
if [[ "$AF3_APPTAINER_USERNS" == "1" || "$AF3_APPTAINER_USERNS" == "true" || "$AF3_APPTAINER_USERNS" == "TRUE" ]]; then
  APPTAINER_CMD+=(--userns)
fi
if [[ "$AF3_APPTAINER_NO_PRIVS" == "1" || "$AF3_APPTAINER_NO_PRIVS" == "true" || "$AF3_APPTAINER_NO_PRIVS" == "TRUE" ]]; then
  APPTAINER_CMD+=(--no-privs)
fi
APPTAINER_CMD+=(
  --env "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/cuda-12.8/compat/lib.real"
  --env "CUDA_HOME=/usr/local/cuda"
  --env "CUDA_ROOT=/usr/local/cuda"
  --env "NVIDIA_VISIBLE_DEVICES=all"
)

run_once() {
  "${APPTAINER_CMD[@]}" \
    "${BIND_ARGS[@]}" \
    "$AF3_SIF_IMAGE" \
    /bin/bash -lc "python '$PY_SCRIPT' \
      --json_path '$input_json' \
      --model_dir /root/models \
      --output_dir '$output_path' \
      --run_data_pipeline='$run_data_pipeline' \
      --gpu_device 0 \
      --num_diffusion_samples '$num_diffusion_samples'"
}

max_retries=3
attempt=1
while true; do
  echo "[af3_ppi_sif] attempt ${attempt}/${max_retries}"
  if run_once; then
    echo "[af3_ppi_sif] success on attempt ${attempt}"
    break
  fi
  rc=$?
  echo "[af3_ppi_sif] attempt ${attempt} failed with exit code ${rc}"
  if [[ "$attempt" -ge "$max_retries" ]]; then
    exit "$rc"
  fi
  if [[ "$rc" -eq 255 ]]; then
    attempt=$((attempt + 1))
    AF3_APPTAINER_TMPDIR="${AF3_LOG_DIR}/apptainer_tmp_gpu${GPU0}_$$_r${attempt}"
    AF3_APPTAINER_CACHEDIR="${AF3_LOG_DIR}/apptainer_cache_gpu${GPU0}_$$_r${attempt}"
    mkdir -p "$AF3_APPTAINER_TMPDIR" "$AF3_APPTAINER_CACHEDIR"
    export APPTAINER_TMPDIR="$AF3_APPTAINER_TMPDIR"
    export APPTAINER_CACHEDIR="$AF3_APPTAINER_CACHEDIR"
    export SINGULARITY_TMPDIR="$AF3_APPTAINER_TMPDIR"
    export SINGULARITY_CACHEDIR="$AF3_APPTAINER_CACHEDIR"
    sleep 3
    continue
  fi
  exit "$rc"
done
