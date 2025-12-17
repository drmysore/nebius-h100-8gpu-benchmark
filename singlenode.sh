#!/bin/bash
# singlenode.sh - Run 8-GPU single-node training with a single command
# Usage: source singlenode.sh [exercise1|exercise2] [optimized]
#
# Examples:
#   source singlenode.sh                    # Run Exercise 1 training (default config)
#   source singlenode.sh exercise1          # Run Exercise 1 training
#   source singlenode.sh exercise1 optimized # Run Exercise 1 with optimized config
#   source singlenode.sh exercise2          # Run Exercise 2 benchmarks
#   source singlenode.sh exercise2 health   # Run health check only
#   source singlenode.sh exercise2 single   # Run single GPU benchmarks
#   source singlenode.sh exercise2 distributed # Run distributed benchmarks

# =============================================================================
# Configuration
# =============================================================================
NPROC_PER_NODE=8
REPO_PATH="/home/supreethlab/repos/nebius-h100-8gpu-benchmark"

# Parse arguments
EXERCISE="${1:-exercise1}"
CONFIG_TYPE="${2:-default}"

# =============================================================================
# Environment Setup
# =============================================================================
echo "=============================================="
echo "8-GPU Single-Node Training Launcher"
echo "=============================================="
echo "Node: $(hostname)"
echo "IP: $(hostname -I | awk '{print $1}')"
echo "GPUs: $NPROC_PER_NODE"
echo "Exercise: $EXERCISE"
echo "Config: $CONFIG_TYPE"
echo "=============================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-finetune

# Set optimizations
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "Environment configured"

# =============================================================================
# Exercise 1: LLM Training
# =============================================================================
if [ "$EXERCISE" == "exercise1" ]; then
    echo ""
    echo "Starting Exercise 1: LLM Fine-Tuning (8 GPUs)"
    echo "=============================================="

    # Select config file
    if [ "$CONFIG_TYPE" == "optimized" ]; then
        CONFIG_FILE="configs/training_config_optimized.yaml"
        echo "Using optimized configuration"
    else
        CONFIG_FILE="configs/training_config.yaml"
        echo "Using default configuration"
    fi

    WORK_DIR="$REPO_PATH/exercise1"

    # Create output directories
    mkdir -p /home/supreethlab/training/checkpoints
    mkdir -p /home/supreethlab/training/logs

    echo ""
    echo "Starting training on $NPROC_PER_NODE GPUs..."
    echo ""

    cd $WORK_DIR
    torchrun --nproc_per_node=$NPROC_PER_NODE \
        scripts/train_function_calling.py \
        --config $CONFIG_FILE

    echo ""
    echo "=============================================="
    echo "Exercise 1 completed"
    echo "Checkpoints: /home/supreethlab/training/checkpoints/"
    echo "=============================================="

# =============================================================================
# Exercise 2: GPU Benchmarks
# =============================================================================
elif [ "$EXERCISE" == "exercise2" ]; then
    echo ""
    echo "Starting Exercise 2: GPU Benchmarks"
    echo "=============================================="

    WORK_DIR="$REPO_PATH/exercise2"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    cd $WORK_DIR

    # Select benchmark mode
    case $CONFIG_TYPE in
        health)
            echo "Running health check..."
            python scripts/benchmark.py --mode health
            ;;
        single)
            echo "Running single GPU benchmarks..."
            OUTPUT_FILE="results/benchmark_single_$TIMESTAMP.json"
            python scripts/benchmark.py --mode single --output $OUTPUT_FILE
            echo "Results saved to: $OUTPUT_FILE"
            ;;
        distributed)
            echo "Running distributed benchmarks (8 GPUs)..."
            OUTPUT_FILE="results/benchmark_distributed_$TIMESTAMP.json"
            torchrun --nproc_per_node=$NPROC_PER_NODE \
                scripts/benchmark.py --mode distributed --output $OUTPUT_FILE
            echo "Results saved to: $OUTPUT_FILE"
            ;;
        full|default|*)
            echo "Running full benchmark suite..."

            echo ""
            echo "Step 1/3: Health check"
            python scripts/benchmark.py --mode health --output results/benchmark_health_$TIMESTAMP.json

            echo ""
            echo "Step 2/3: Single GPU benchmarks"
            python scripts/benchmark.py --mode single --output results/benchmark_single_$TIMESTAMP.json

            echo ""
            echo "Step 3/3: Distributed benchmarks (8 GPUs)"
            torchrun --nproc_per_node=$NPROC_PER_NODE \
                scripts/benchmark.py --mode distributed --output results/benchmark_distributed_$TIMESTAMP.json
            ;;
    esac

    echo ""
    echo "=============================================="
    echo "Exercise 2 completed"
    echo "Results: $WORK_DIR/results/"
    echo "=============================================="

else
    echo "Unknown exercise: $EXERCISE"
    echo ""
    echo "Usage: source singlenode.sh [exercise1|exercise2] [config]"
    echo ""
    echo "Exercise 1 (LLM Training):"
    echo "  source singlenode.sh exercise1           # Default config"
    echo "  source singlenode.sh exercise1 optimized # Optimized config"
    echo ""
    echo "Exercise 2 (Benchmarks):"
    echo "  source singlenode.sh exercise2           # Full benchmark suite"
    echo "  source singlenode.sh exercise2 health    # Health check only"
    echo "  source singlenode.sh exercise2 single    # Single GPU benchmarks"
    echo "  source singlenode.sh exercise2 distributed # Multi-GPU benchmarks"
fi

echo ""
echo "Done."
