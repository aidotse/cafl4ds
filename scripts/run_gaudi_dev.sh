#!/bin/bash

# --- Usage Function ---
print_usage() {
    echo "Usage: $0 [OPTIONS] <image_name> <device_id> [command...]"
    echo ""
    echo "Options:"
    echo "  -m, --mount <path>   Optional read-only bind mount for data/models (e.g., /mnt/stl10 or /host:/container)"
    echo "  -r, --root           Run as root (disables default user mapping)"
    echo "  -h, --help           Show this help message and exit"
    echo ""
    echo "Example: $0 -m /mnt/stl10 --root my_image all bash"
}


# --- Mandatory Positional Arguments ---
IMAGE_NAME="$1"
DEVICE_ID="$2"

# Remove the image name and device ID from the arguments list so we can pass the rest as the command ("$@")
shift 2

# --- Default Variables ---
DATA_MOUNT=""
RUN_AS_ROOT=""


# --- Parse Script Options ---
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mount)
            DATA_MOUNT="$2"
            shift 2
            ;;
        -r|--root)
            RUN_AS_ROOT=1
            shift 1
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1"
            print_usage
            exit 1
            ;;
        *)
            # Break out of the loop when we hit the first non-flag argument (the image name)
            break
            ;;
    esac
done


if [ -z "$IMAGE_NAME" ] || [ -z "$DEVICE_ID" ]; then
    echo "Error: You must provide both an image name and a device ID."
    print_usage
    exit 1
fi

# Safeguard: Check if DEVICE_ID is a number or the exact string "all"
if ! [[ "$DEVICE_ID" =~ ^[0-9]+$ ]] && [[ "$DEVICE_ID" != "all" ]]; then
    echo "Error: Invalid device ID ('$DEVICE_ID')."
    print_usage
    exit 1
fi

# Dynamic hardware isolation
if [ "$DEVICE_ID" == "all" ]; then
    # Expose the whole accel directory and infiniband for RoCE
    ISOLATION_FLAGS="--device /dev/accel:/dev/accel --device /dev/infiniband:/dev/infiniband"
else
    # Map the specific accel node to the container
    # Physical /dev/accel/accel0 -> HPU 0
    ISOLATION_FLAGS="--device /dev/accel/accel$DEVICE_ID:/dev/accel/accel$DEVICE_ID"
fi

# Optional read-only bind mount for data/models that live OUTSIDE the repo (the repo itself is
# always mounted at /workspace). This launcher hardcodes no dataset path — opt in with DATA_MOUNT:
#   DATA_MOUNT=/mnt/stl10                     -> mounts /mnt/stl10 at the same path, read-only
#   DATA_MOUNT=/host/models:/workspace/models -> explicit `host:container[:opts]` spec, used as-is
# Unset (default) -> no extra mount. Example:
#   DATA_MOUNT=/mnt/stl10 ./scripts/run_gaudi_dev.sh <image> 0 \
#       python scripts/run_loop.py device=hpu data_root=/mnt/stl10
if [ -n "${DATA_MOUNT:-}" ]; then
    case "$DATA_MOUNT" in
        *:*) DATA_FLAGS="-v $DATA_MOUNT" ;;                 # explicit host:container[:opts], verbatim
        *)   DATA_FLAGS="-v $DATA_MOUNT:$DATA_MOUNT:ro" ;;  # bare host path -> same path, read-only
    esac
else
    DATA_FLAGS=""
fi

# Run as the host user (not root) so files the container writes into the mounted repo are owned
# by you, not root. The image has no passwd entry for your uid and Habana's backend autoload calls
# getpwuid(), so bind /etc/passwd + /etc/group read-only to resolve the name, and point HOME at a
# writable path. Device nodes are world-accessible (/dev/accel/*, /dev/infiniband/uverbs* are 0666),
# so single-card HPU access works fine unprivileged. Escape hatch: set RUN_AS_ROOT=1 to keep the old
# root behaviour (e.g. should a privileged multi-card run ever need it).
if [ -n "${RUN_AS_ROOT:-}" ]; then
    USER_FLAGS=""
else
    USER_FLAGS="--user $(id -u):$(id -g) -e HOME=/tmp -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro"
fi

# Remove the image name from the arguments list so we can pass the rest
shift 2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Allocate a TTY only when stdin is one. Lets the script drive an interactive
# shell normally, while still working headlessly (CI, piped commands) where
# `docker run -t` would fail with "cannot attach stdin to a TTY-enabled container".
if [ -t 0 ]; then
    TTY_FLAGS="-it"
else
    TTY_FLAGS="-i"
fi

# --- KERNEL & STABILITY FLAGS ---
# --ipc=host:            Prevents PyTorch DataLoader crashes (Shared Mem).
# --ulimit memlock=-1:   Allows unlimited memory pinning for DMA transfers.
# --ulimit stack=64MB:   Prevents C++ stack overflows in the driver.
# --cap-add=IPC_LOCK:    Required for memory pinning capabilities.

# --- HARDWARE ACCESS FLAGS ---
# --net=host:            Required for Gaudi 2 internal RoCE interconnects.
# --privileged:          (Optional) The "Sledgehammer". Use if you get
#                        "Permission Denied" on /dev/infiniband or cannot reset cards.

# --- DEBUGGING & CONFIGURATION ---
# HABANA_VISIBLE_DEVICES: Controls which cards are seen. Use 'all' for full node,
#                         or '0' to isolate a single card for testing.
# PT_HPU...LOGGING:       Set to '1' to see warnings if operations fall back to CPU
#                         (slaughters performance). Keep '0' for clean logs.
# PT_HPU_RECIPE_CACHE_CONFIG: Ensure the Gaudi compiler caches do not collide by injecting a unique path for each container.

# --- ENVIRONMENT ---
# -v ~/.cache/huggingface:/workspace/cache : should match the HF_HOME env variable in the Dockerfile,
#   so that we only download models when absolutely necessary.
# DDP_SHARED_ID:            a common ID for all ranks in DDP, used to coordinate logging, etc. Can't use e.g.
#                           os.getppid or such because of Docker deterministic PID namespaces. Can't use
#                           TORCHELASTIC_RUN_ID without manually also setting --rdzv-id. Easiest just to pass directly.
#

# Other flags tested:
#--user root \
#--device /dev/habanalabs:/dev/habanalabs \
#--device /dev/hl*:/dev/hl* \
#--device /dev/infiniband:/dev/infiniband \
# -v /sys/class/infiniband:/sys/class/infiniband:ro \
docker run $TTY_FLAGS --rm \
  --runtime=habana \
  $USER_FLAGS \
  $ISOLATION_FLAGS \
  --cap-add=sys_nice \
  --cap-add=IPC_LOCK \
  --ipc=host \
  --net=host \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864 \
  -e PT_HPU_RECIPE_CACHE_CONFIG=/tmp/recipe_cache_$DEVICE_ID,False,1024 \
  -e HABANA_VISIBLE_DEVICES=$DEVICE_ID \
  -e PT_HPU_ENABLE_CPU_FALLBACK_LOGGING=1 \
  -e DDP_SHARED_ID=$RANDOM \
  $DATA_FLAGS \
  -v "$PROJECT_ROOT":/workspace \
  -v ~/.cache/huggingface:/workspace/cache \
  "$IMAGE_NAME" \
  "$@"
