#!/bin/bash
# =============================================================================
# Runara AMD MI300X Benchmark - Main Orchestration Script
# =============================================================================
# Launches MI300X instance, runs Llama-3.3-70B FP8 benchmark, collects results,
# and auto-destroys instance to prevent cost leaks.
#
# Features:
# - Uses saved snapshot if available (skips vLLM install)
# - Uses persistent volume for model weights (skips download)
# - Creates snapshot after first successful setup
#
# Usage: ./runara-benchmark.sh [--poc] [--setup-only] [--no-destroy]
#   --poc          Run only first benchmark (128→2048) for quick validation
#   --setup-only   Setup environment and create snapshot, don't run benchmark
#   --no-destroy   Keep droplet running after benchmark (for debugging)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DROPLET_NAME="runara-$(date +%s)"
SIZE="gpu-mi300x1-192gb"
REGION="tor1"
BASE_IMAGE="gpu-amd-base"  # Ubuntu 24.04 + ROCm 6.4.0
SNAPSHOT_NAME="runara-mi300x-vllm-ready"
VOLUME_NAME="runara-models"
VOLUME_SIZE="500"  # GB - enough for multiple large models
MODEL="meta-llama/Llama-3.3-70B-Instruct"
QUANTIZATION="fp8"
MAX_RUNTIME_SECONDS=7200  # 2 hours absolute max
RESULTS_DIR="./results/$(date +%Y%m%d-%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCTL="/snap/bin/doctl"

# Parse arguments
POC_MODE=false
SETUP_ONLY=false
NO_DESTROY=false

for arg in "$@"; do
    case $arg in
        --poc) POC_MODE=true ;;
        --setup-only) SETUP_ONLY=true ;;
        --no-destroy) NO_DESTROY=true ;;
    esac
done

echo "============================================================"
echo "🚀 Runara AMD MI300X Benchmark"
echo "============================================================"
echo "Model: $MODEL"
echo "Precision: $QUANTIZATION"
echo "GPU: 1x MI300X (192GB)"
echo "Region: $REGION"
if $POC_MODE; then
    echo "Mode: POC (single benchmark: 128→2048)"
elif $SETUP_ONLY; then
    echo "Mode: SETUP ONLY (create snapshot, no benchmark)"
else
    echo "Mode: Full (9 benchmark scenarios)"
fi
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# Prerequisite Checks
# -----------------------------------------------------------------------------
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    if ! command -v $DOCTL &> /dev/null; then
        echo "❌ doctl not found. Install with: snap install doctl"
        exit 1
    fi
    
    if ! $DOCTL account get &> /dev/null; then
        echo "❌ doctl not authenticated. Run: doctl auth init"
        exit 1
    fi
    
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "❌ HF_TOKEN not set. Export your Hugging Face token."
        exit 1
    fi
    
    echo "✅ Prerequisites OK"
}

# -----------------------------------------------------------------------------
# Cleanup Function (CRITICAL - prevents cost leaks)
# -----------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    
    if [[ "$NO_DESTROY" == "true" ]]; then
        echo "   --no-destroy flag set, keeping droplet running"
        echo "   ⚠️  Remember to destroy manually: $DOCTL compute droplet delete $DROPLET_ID --force"
        return
    fi
    
    if [[ -n "${DROPLET_ID:-}" ]]; then
        echo "   Destroying droplet $DROPLET_ID ($DROPLET_NAME)..."
        $DOCTL compute droplet delete "$DROPLET_ID" --force 2>/dev/null || true
        echo "   ✅ Droplet destroyed"
    fi
    
    # Kill watchdog if running
    if [[ -n "${WATCHDOG_PID:-}" ]]; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
    fi
    
    # Note: Volume is NOT destroyed - it persists for next run
    echo "   📦 Volume '$VOLUME_NAME' preserved for next run"
    echo "🧹 Cleanup complete"
}

# ALWAYS cleanup on exit, error, or interrupt
trap cleanup EXIT INT TERM ERR

# -----------------------------------------------------------------------------
# Timeout Watchdog (CRITICAL - prevents runaway costs)
# -----------------------------------------------------------------------------
start_watchdog() {
    (
        sleep $MAX_RUNTIME_SECONDS
        echo ""
        echo "⚠️  MAX RUNTIME ($MAX_RUNTIME_SECONDS sec) EXCEEDED!"
        echo "⚠️  Force destroying all Runara instances..."
        $DOCTL compute droplet list --tag-name runara --format ID --no-header 2>/dev/null | \
            xargs -I {} $DOCTL compute droplet delete {} --force 2>/dev/null || true
        exit 1
    ) &
    WATCHDOG_PID=$!
    echo "⏱️  Watchdog started (max ${MAX_RUNTIME_SECONDS}s / $(($MAX_RUNTIME_SECONDS/3600))h)"
}

# -----------------------------------------------------------------------------
# Check/Create Persistent Volume
# -----------------------------------------------------------------------------
setup_volume() {
    echo ""
    echo "📦 Checking persistent volume..."
    
    VOLUME_ID=$($DOCTL compute volume list --format ID,Name --no-header | grep "$VOLUME_NAME" | awk '{print $1}' || true)
    
    if [[ -n "$VOLUME_ID" ]]; then
        echo "   ✅ Found existing volume: $VOLUME_NAME ($VOLUME_ID)"
    else
        echo "   Creating new volume: $VOLUME_NAME (${VOLUME_SIZE}GB)..."
        VOLUME_ID=$($DOCTL compute volume create "$VOLUME_NAME" \
            --region "$REGION" \
            --size "${VOLUME_SIZE}GiB" \
            --desc "Runara model storage - persistent between runs" \
            --format ID --no-header)
        echo "   ✅ Created volume: $VOLUME_ID"
    fi
    
    export VOLUME_ID
}

# -----------------------------------------------------------------------------
# Check for Existing Snapshot
# -----------------------------------------------------------------------------
check_snapshot() {
    echo ""
    echo "📸 Checking for saved snapshot..."
    
    SNAPSHOT_ID=$($DOCTL compute snapshot list --format ID,Name --no-header | grep "$SNAPSHOT_NAME" | awk '{print $1}' || true)
    
    if [[ -n "$SNAPSHOT_ID" ]]; then
        echo "   ✅ Found snapshot: $SNAPSHOT_NAME ($SNAPSHOT_ID)"
        echo "   Will use snapshot (vLLM pre-installed, faster startup)"
        USE_SNAPSHOT=true
        IMAGE_TO_USE="$SNAPSHOT_ID"
    else
        echo "   No snapshot found. Will use base image and create snapshot after setup."
        USE_SNAPSHOT=false
        IMAGE_TO_USE="$BASE_IMAGE"
    fi
    
    export USE_SNAPSHOT IMAGE_TO_USE SNAPSHOT_ID
}

# -----------------------------------------------------------------------------
# Create Snapshot (after first successful setup)
# -----------------------------------------------------------------------------
create_snapshot() {
    echo ""
    echo "📸 Creating snapshot for future runs..."
    
    # Check if snapshot already exists
    EXISTING=$($DOCTL compute snapshot list --format Name --no-header | grep "$SNAPSHOT_NAME" || true)
    if [[ -n "$EXISTING" ]]; then
        echo "   Snapshot already exists, skipping"
        return
    fi
    
    echo "   Creating snapshot from droplet $DROPLET_ID..."
    echo "   (This may take 5-10 minutes)"
    
    $DOCTL compute droplet-action snapshot "$DROPLET_ID" \
        --snapshot-name "$SNAPSHOT_NAME" \
        --wait
    
    echo "   ✅ Snapshot created: $SNAPSHOT_NAME"
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
main() {
    check_prerequisites
    mkdir -p "$RESULTS_DIR"
    
    # Start watchdog
    start_watchdog
    
    # Setup persistent volume
    setup_volume
    
    # Check for snapshot
    check_snapshot
    
    # -------------------------------------------------------------------------
    # Step 1: Create GPU Droplet
    # -------------------------------------------------------------------------
    echo ""
    echo "📦 Creating MI300X GPU droplet..."
    echo "   Name: $DROPLET_NAME"
    echo "   Size: $SIZE"
    echo "   Image: $IMAGE_TO_USE"
    echo "   Region: $REGION"
    echo "   Volume: $VOLUME_NAME"
    echo ""
    
    # Get SSH key ID
    SSH_KEY_ID=$($DOCTL compute ssh-key list --format ID --no-header | head -1)
    if [[ -z "$SSH_KEY_ID" ]]; then
        echo "❌ No SSH keys found in DigitalOcean. Add one first."
        exit 1
    fi
    echo "   SSH Key ID: $SSH_KEY_ID"
    
    # Create droplet with volume attached
    DROPLET_INFO=$($DOCTL compute droplet create "$DROPLET_NAME" \
        --size "$SIZE" \
        --image "$IMAGE_TO_USE" \
        --region "$REGION" \
        --ssh-keys "$SSH_KEY_ID" \
        --volumes "$VOLUME_ID" \
        --tag-name runara \
        --tag-name auto-destroy \
        --wait \
        --format ID,PublicIPv4 \
        --no-header)
    
    DROPLET_ID=$(echo "$DROPLET_INFO" | awk '{print $1}')
    DROPLET_IP=$(echo "$DROPLET_INFO" | awk '{print $2}')
    
    echo ""
    echo "✅ Droplet created!"
    echo "   ID: $DROPLET_ID"
    echo "   IP: $DROPLET_IP"
    
    # Save droplet info for emergency cleanup
    echo "$DROPLET_ID" > "$RESULTS_DIR/droplet_id.txt"
    echo "$DROPLET_IP" > "$RESULTS_DIR/droplet_ip.txt"
    
    # -------------------------------------------------------------------------
    # Step 2: Wait for SSH
    # -------------------------------------------------------------------------
    echo ""
    echo "⏳ Waiting for SSH access..."
    
    for i in {1..60}; do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
            root@"$DROPLET_IP" "echo 'SSH OK'" 2>/dev/null; then
            echo "✅ SSH connected!"
            break
        fi
        echo "   Attempt $i/60 - waiting..."
        sleep 10
    done
    
    # -------------------------------------------------------------------------
    # Step 3: Mount Volume & Setup
    # -------------------------------------------------------------------------
    echo ""
    echo "💾 Mounting persistent volume..."
    
    ssh -o StrictHostKeyChecking=no root@"$DROPLET_IP" << 'MOUNT_SCRIPT'
# Mount the volume
VOLUME_DEV=$(ls /dev/disk/by-id/ | grep scsi-0DO_Volume | head -1)
if [[ -n "$VOLUME_DEV" ]]; then
    mkdir -p /mnt/models
    
    # Check if already formatted
    if ! blkid /dev/disk/by-id/$VOLUME_DEV; then
        echo "Formatting volume..."
        mkfs.ext4 /dev/disk/by-id/$VOLUME_DEV
    fi
    
    mount /dev/disk/by-id/$VOLUME_DEV /mnt/models
    echo "Volume mounted at /mnt/models"
    
    # Set HuggingFace cache to volume
    mkdir -p /mnt/models/huggingface
    export HF_HOME=/mnt/models/huggingface
    echo "export HF_HOME=/mnt/models/huggingface" >> ~/.bashrc
else
    echo "Warning: Volume not found"
fi
MOUNT_SCRIPT
    
    # -------------------------------------------------------------------------
    # Step 4: Setup Environment (if not using snapshot)
    # -------------------------------------------------------------------------
    if [[ "$USE_SNAPSHOT" == "false" ]]; then
        echo ""
        echo "🔧 First-time setup: Installing vLLM..."
        
        ssh -o StrictHostKeyChecking=no root@"$DROPLET_IP" << 'SETUP_SCRIPT'
export HF_HOME=/mnt/models/huggingface
pip install --upgrade pip
pip install vllm requests aiohttp huggingface_hub
echo "vLLM installed"
SETUP_SCRIPT
        
        # Create snapshot for future runs
        create_snapshot
    else
        echo ""
        echo "⚡ Using snapshot - vLLM already installed"
    fi
    
    # If setup-only mode, exit here
    if $SETUP_ONLY; then
        echo ""
        echo "✅ SETUP COMPLETE"
        echo "   Snapshot created: $SNAPSHOT_NAME"
        echo "   Volume ready: $VOLUME_NAME"
        echo ""
        echo "Next run will be much faster!"
        return
    fi
    
    # -------------------------------------------------------------------------
    # Step 5: Run Benchmark
    # -------------------------------------------------------------------------
    echo ""
    echo "🏃 Running benchmark on MI300X..."
    echo ""
    
    # Copy benchmark script to remote
    scp -o StrictHostKeyChecking=no \
        "$SCRIPT_DIR/remote-benchmark.sh" \
        root@"$DROPLET_IP":/root/benchmark.sh
    
    # Run benchmark
    ssh -o StrictHostKeyChecking=no root@"$DROPLET_IP" \
        "export HF_TOKEN='$HF_TOKEN' && export POC_MODE=$POC_MODE && export HF_HOME=/mnt/models/huggingface && bash /root/benchmark.sh" \
        2>&1 | tee "$RESULTS_DIR/benchmark.log"
    
    # -------------------------------------------------------------------------
    # Step 6: Download Results
    # -------------------------------------------------------------------------
    echo ""
    echo "📥 Downloading results..."
    
    scp -o StrictHostKeyChecking=no -r \
        root@"$DROPLET_IP":/root/results/* \
        "$RESULTS_DIR/" 2>/dev/null || true
    
    echo ""
    echo "============================================================"
    echo "✅ BENCHMARK COMPLETE"
    echo "============================================================"
    echo "Results saved to: $RESULTS_DIR"
    echo ""
    ls -la "$RESULTS_DIR"
    echo ""
    echo "💰 Droplet will be destroyed now (volume preserved)"
    # cleanup() will be called by trap
}

# Run main
main "$@"
