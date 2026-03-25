#!/bin/bash
# ODesignBench copy of run_af3_a800.sh with /root/af_output bind mount fix.
# AF3 resources are from the original path.
# This script runs AlphaFold3 using the AF3 Docker image (not apptainer/sif).

# Parse arguments from refold_api.py
# $1: exp_name
# $2: input_json
# $3: output_dir
# $4: gpus (comma-separated, e.g., "0,1,2,3")
# $5: run_data_pipeline (True/False)
# $6: cache_dir
# $7: num_diffusion_samples (optional, default: 1)

exp_name="$1"
input_json="$2"
output_path="$3"
gpus="$4"
run_data_pipeline="$5"
cache_dir="$6"
num_diffusion_samples="${7:-1}"  # Default to 1 if not provided

# AF3 installation directories (override via env vars)
AF3_BASE="${AF3_BASE:-/path/to/af3}"
AF3_PUBLIC_DB="${AF3_PUBLIC_DB:-/path/to/public_databases}"

# AF3 Docker image name (override via env var if your image tag differs)
AF3_DOCKER_IMAGE="${AF3_DOCKER_IMAGE:-alphafold3}"

# Whether to patch AF3 to allow MSA injection in alphafoldserver dialect.
# Set to 'false' to disable the patch (e.g., if using a pre-patched image).
AF3_DIALECT_PATCH="${AF3_DIALECT_PATCH:-true}"

AF3_ASSETS="${AF3_ASSETS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../assets}"

# Create output directory and log directory (for debugging)
mkdir -p "$output_path"
AF3_LOG_DIR="$(dirname "$output_path")/af3_log"
mkdir -p "$AF3_LOG_DIR"

# Convert comma-separated gpus to space-separated for launch.py
gpus_space_separated=${gpus//,/ }

# Build docker volume mounts
# - /root/af_output: some containers use this default output path
# - /app/alphafold/log: capture AF3 logs for debugging (see af3_log/ after run)
BIND_MOUNTS=(
    -v "$AF3_BASE/models:/root/models"
    -v "$AF3_PUBLIC_DB:/root/public_databases"
    -v "$output_path:$output_path"
    -v "$output_path:/root/af_output"
    -v "$AF3_LOG_DIR:/app/alphafold/log"
    -v "$input_json:$input_json"
    # Mount assets so MSA paths are accessible inside the container.
    -v "$AF3_ASSETS:/assets"
)

if [[ "$AF3_BASE" == "/path/to/af3" || ! -d "$AF3_BASE/models" ]]; then
    echo "ERROR: Set AF3_BASE to your AlphaFold3 installation directory (must contain models/ for docker volume mounts)." >&2
    exit 1
fi

if [[ "$AF3_PUBLIC_DB" == "/path/to/public_databases" || ! -d "$AF3_PUBLIC_DB" ]]; then
    echo "ERROR: Set AF3_PUBLIC_DB to your AlphaFold3 public_databases directory." >&2
    exit 1
fi

# Optional bind for local home paths referenced by input/output files.
if [[ -n "${HOME:-}" && -d "${HOME:-}" ]]; then
    BIND_MOUNTS+=(-v "${HOME}:${HOME}")
fi

# Add cache_dir bind mount if provided
if [[ -n "$cache_dir" && "$cache_dir" != "None" && "$cache_dir" != "" ]]; then
    mkdir -p "$cache_dir"
    BIND_MOUNTS+=(-v "$cache_dir:$cache_dir")
fi

# Limit JAX GPU memory preallocation to reduce OOM risk.
# Use the non-deprecated env var to avoid conflicts inside AF3/JAX runtime.
# If users still export XLA_PYTHON_CLIENT_MEM_FRACTION, treat it as fallback.
if [[ -z "${XLA_CLIENT_MEM_FRACTION:-}" && -n "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" ]]; then
    export XLA_CLIENT_MEM_FRACTION="$XLA_PYTHON_CLIENT_MEM_FRACTION"
fi
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.6}"
unset XLA_PYTHON_CLIENT_MEM_FRACTION

# ---------------------------------------------------------------------------
# Dialect patch: allow MSA fields in alphafoldserver dialect.
# The native AF3 Server API rejects MSA fields in proteinChain. This patch
# extends ProteinChain.from_alphafoldserver_dict to accept unpairedMsa,
# unpairedMsaPath, pairedMsa, pairedMsaPath, and templates fields, enabling
# pre-computed MSA injection via JSON for tasks like PBP target MSA.
#
# Strategy: run gen_af3_patch.py inside a throwaway AF3 container (where the
# source file exists at /app/alphafold/...). The script writes the patched
# file to /tmp on the host (via -v /tmp:/tmp). The main AF3 container then
# mounts this patched file over the original at startup.
# ---------------------------------------------------------------------------
PATCHED_FILE="/tmp/alphafold3_folding_input_patched.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$AF3_DIALECT_PATCH" == "true" ]]; then
    if [[ ! -f "$PATCHED_FILE" ]]; then
        echo "Generating AF3 dialect patch (one-time, cached at $PATCHED_FILE)..."
        docker run --rm \
            -v /tmp:/tmp \
            -v "$SCRIPT_DIR:/opt/odesign_patch" \
            "$AF3_DOCKER_IMAGE" \
            python3 /opt/odesign_patch/gen_af3_patch.py
        patch_status=$?
        if [[ $patch_status -ne 0 ]]; then
            echo "ERROR: Failed to generate AF3 dialect patch (exit $patch_status)" >&2
            exit 1
        fi
        echo "Patch generated successfully."
    fi
fi

# Build the full docker run command
# If patched, mount the patched file over the original
if [[ "$AF3_DIALECT_PATCH" == "true" && -f "$PATCHED_FILE" ]]; then
    BIND_MOUNTS+=(-v "$PATCHED_FILE:/app/alphafold/src/alphafold3/common/folding_input.py")
fi

# Execute AlphaFold3 using Docker
# Limit GPUs visible to container: if $gpus is set (e.g. "0,1"), pass it via NVIDIA_VISIBLE_DEVICES.
# Otherwise default to all GPUs. Also pass via --gpus to ensure the Docker runtime enforces it.
if [[ -n "$gpus" && "$gpus" != "all" ]]; then
    GPU_SPEC="device=$gpus"
    NV_VISIBLE="$gpus"
else
    GPU_SPEC="all"
    NV_VISIBLE="all"
fi

docker run --rm \
    --gpus "$GPU_SPEC" \
    --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES="$NV_VISIBLE" \
    -e XLA_CLIENT_MEM_FRACTION="$XLA_CLIENT_MEM_FRACTION" \
    -e PROMPT_COMMAND= \
    "${BIND_MOUNTS[@]}" \
    "$AF3_DOCKER_IMAGE" \
    python /app/alphafold/run_alphafold.py \
    --json_path "$input_json" \
    --output_dir "$output_path" \
    --run_data_pipeline="$run_data_pipeline" \
    --gpu_device 0 \
    --num_diffusion_samples "$num_diffusion_samples"
