# Nebius H100 GPU Cluster Demo Guide (Single Node)

This document provides a complete walkthrough for demonstrating GPU cluster validation and LLM fine-tuning on a single Nebius AI Cloud node with 8x H100 GPUs.

## Table of Contents

1. [Overview](#overview)
2. [Hardware Configuration](#hardware-configuration)
3. [Pre-Demo Checklist](#pre-demo-checklist)
4. [Demo Part 1: Hardware Validation](#demo-part-1-hardware-validation)
5. [Demo Part 2: GPU Benchmarks (Exercise 2)](#demo-part-2-gpu-benchmarks-exercise-2)
6. [Demo Part 3: LLM Fine-Tuning (Exercise 1)](#demo-part-3-llm-fine-tuning-exercise-1)
7. [Demo Part 4: Performance Analysis](#demo-part-4-performance-analysis)
8. [Results Summary](#results-summary)
9. [Troubleshooting](#troubleshooting)
10. [Appendix](#appendix)

---

## Overview

This demonstration showcases two primary capabilities on a single 8-GPU node:

1. **Exercise 1 - LLM Fine-Tuning**: Training a 7B parameter language model for function calling using distributed training across 8 GPUs.

2. **Exercise 2 - GPU Cluster Acceptance Testing**: Validating node performance through comprehensive benchmarks including compute, memory bandwidth, and inter-GPU communication.

### Repository Structure

```
nebius-h100-8gpu-benchmark/
├── exercise1/                    # LLM Fine-Tuning
│   ├── scripts/
│   │   ├── train_function_calling.py
│   │   └── train_job.sbatch
│   └── configs/
│       ├── training_config.yaml
│       └── training_config_optimized.yaml
├── exercise2/                    # GPU Benchmarks
│   ├── scripts/
│   │   └── benchmark.py
│   └── results/
├── singlenode.sh                 # Single-node launcher
├── hardware_info.sh              # Hardware discovery
├── OPTIMIZATION.md               # Performance tuning guide
└── DEMO.md                       # This file
```

---

## Hardware Configuration

### Node Specifications

| Component | Specification |
|-----------|---------------|
| CPU | Intel Xeon Platinum 8468 (128 cores, 2 sockets) |
| RAM | 1.5 TB DDR5 |
| GPU | 8x NVIDIA H100 80GB HBM3 |
| GPU Memory | 640 GB HBM3 total |
| Interconnect | NVLink 4.0 (all GPUs fully connected) |
| Driver | 570.195.03 |
| CUDA | 12.1 |
| PyTorch | 2.3.0+cu121 |
| NCCL | 2.20.5 |

### GPU Topology

All 8 GPUs are connected via NVLink 4.0 with 18 links each (NV18), providing 900 GB/s bidirectional bandwidth between any pair of GPUs.

```
GPU Layout:
├── NUMA Node 0: GPU 0-3 (CPU cores 0-63)
└── NUMA Node 1: GPU 4-7 (CPU cores 64-127)
```

---

## Pre-Demo Checklist

### 1. Verify Conda Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-finetune
python -c "import torch; print(f'PyTorch: {torch.__version__}, GPUs: {torch.cuda.device_count()}')"
```

### 2. Verify GPU Availability

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

Expected: 8 GPUs with 81559 MiB each.

### 3. Check Disk Space

```bash
df -h /home/supreethlab/training/
```

Ensure sufficient space for checkpoints (approximately 2 GB per checkpoint).

---

## Demo Part 1: Hardware Validation

### Show Hardware Configuration

```bash
cd /home/supreethlab/repos/nebius-h100-8gpu-benchmark
./hardware_info.sh summary
```

Expected output:
```
==============================================
NODE SUMMARY
==============================================
Hostname: computeinstance-e00e9ncccc9zh9nxw2
IP Address: 10.2.0.129

Hardware:
  - CPU: Intel(R) Xeon(R) Platinum 8468
  - Cores: 128
  - RAM: 1.5Ti
  - GPUs: 8x NVIDIA H100 80GB HBM3
  - GPU Memory: 81559 MiB per GPU
```

### Show GPU Topology

```bash
./hardware_info.sh gpu
```

Key points to highlight:
- All GPUs show NV18 connections (18 NVLink links)
- Two NUMA domains for optimal CPU-GPU affinity
- Full mesh connectivity between all GPUs

### Show Full GPU Status

```bash
nvidia-smi
```

Verify:
- All 8 GPUs visible
- Temperature within normal range (30-50C idle)
- Memory available (~80 GB each)

---

## Demo Part 2: GPU Benchmarks (Exercise 2)

### Run Full Benchmark Suite

```bash
cd /home/supreethlab/repos/nebius-h100-8gpu-benchmark
source singlenode.sh exercise2
```

This runs three benchmark phases:
1. Health check (GPU detection)
2. Single GPU benchmarks (MatMul, Memory bandwidth)
3. Distributed benchmarks (NCCL AllReduce, Training throughput)

### Run Individual Benchmarks

```bash
# Health check only
source singlenode.sh exercise2 health

# Single GPU benchmarks only
source singlenode.sh exercise2 single

# Distributed benchmarks only (8 GPUs)
source singlenode.sh exercise2 distributed
```

### Benchmark Results Explained

#### Health Check

| Metric | Expected | Description |
|--------|----------|-------------|
| GPU Count | 8 | All GPUs detected |
| Memory | 79.19 GB each | Available HBM3 memory |
| Status | PASS | No hardware errors |

#### Single GPU Performance

| Test | Expected | Description |
|------|----------|-------------|
| MatMul (BF16) | >700 TFLOPS | H100 theoretical peak: 989 TFLOPS |
| Memory Bandwidth | >2.5 TB/s | H100 spec: 3.35 TB/s |

#### Distributed Performance

| Test | Expected | Description |
|------|----------|-------------|
| NCCL AllReduce | >400 GB/s | NVLink aggregate bandwidth |
| Training Throughput | >300,000 tok/s | 8-GPU parallel throughput |

### View Results

```bash
cat exercise2/results/benchmark_distributed.json | python -m json.tool
```

---

## Demo Part 3: LLM Fine-Tuning (Exercise 1)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2-7B-Instruct |
| Method | LoRA (Low-Rank Adaptation) |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| Trainable Parameters | 161M (2.1% of 7.7B) |
| Precision | BF16 |
| Batch Size | 2 per GPU |
| Gradient Accumulation | 4 steps |
| Effective Batch Size | 64 (8 GPUs) |
| Max Sequence Length | 4096 |
| Training Steps | 100 (demo) |
| Dataset | Glaive Function Calling |

### Run Training (Default Configuration)

```bash
cd /home/supreethlab/repos/nebius-h100-8gpu-benchmark
source singlenode.sh exercise1
```

### Run Training (Optimized Configuration)

```bash
source singlenode.sh exercise1 optimized
```

Optimized configuration uses:
- Batch size 4 (vs 2)
- 8 DataLoader workers (vs 4)
- Prefetch factor 4 (vs 2)

### Training Progress

Watch for these milestones in the output:

```
Step 1:   loss=1.79, lr=6.67e-06
Step 10:  loss=1.32, lr=1.97e-05
Step 20:  loss=0.69, lr=1.85e-05
Step 50:  loss=0.44, lr=1.05e-05  [Checkpoint saved, Eval: 0.43]
Step 100: loss=0.39, lr=1.31e-07  [Final checkpoint, Eval: 0.40]
```

### Training Success Criteria

| Criterion | Expected | Verification |
|-----------|----------|--------------|
| Completion | Exit code 0 | No errors in output |
| Final Loss | < 0.5 | Check last training loss |
| Eval Loss | < 0.5 | Check evaluation loss |
| Checkpoint | Saved | `ls training/checkpoints/final/` |

### View Training Output

```bash
# List all checkpoints
ls -la /home/supreethlab/training/checkpoints/

# Verify final model files
ls -la /home/supreethlab/training/checkpoints/final/
```

---

## Demo Part 4: Performance Analysis

### Monitor GPU During Training

```bash
# Real-time GPU utilization
watch -n 1 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'

# Detailed metrics (power, temperature, memory)
nvidia-smi dmon -s pucvmet -d 1 -c 10
```

### Key Metrics to Observe

| Metric | Expected During Training |
|--------|--------------------------|
| GPU Utilization | 80-100% |
| Memory Used | 50-70 GB per GPU |
| Temperature | 40-70C |
| Power Draw | 300-600W per GPU |

### Performance Summary

| Metric | Result |
|--------|--------|
| MatMul Performance | 730.97 TFLOPS |
| Memory Bandwidth | 3.02 TB/s |
| NCCL AllReduce | 442.28 GB/s |
| Training Throughput | 331,842 tokens/s |
| Per-GPU Throughput | 41,480 tokens/s |
| Final Training Loss | 0.3895 |
| Final Eval Loss | 0.396 |

---

## Results Summary

### Exercise 2: Benchmark Results

| Test | Threshold | Result | Status |
|------|-----------|--------|--------|
| GPU Health | 8/8 GPUs | 8/8 | PASS |
| MatMul (BF16) | >100 TFLOPS | 730.97 TFLOPS | PASS |
| Memory Bandwidth | >1.0 TB/s | 3.02 TB/s | PASS |
| NCCL AllReduce | >400 GB/s | 442.28 GB/s | PASS |
| Training Throughput | >4000 tok/s/GPU | 41,480 tok/s/GPU | PASS |

### Exercise 1: Training Results

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen2-7B-Instruct |
| Parameters | 7.7B total, 161M trainable |
| Training Steps | 100 |
| Final Training Loss | 0.3895 |
| Final Eval Loss | 0.396 |
| Training Time | ~6 minutes |
| Checkpoint Size | ~650 MB |

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Use default config (smaller batch size): `source singlenode.sh exercise1`
2. Reduce batch size in config: `per_device_train_batch_size: 1`
3. Verify no other processes using GPU: `nvidia-smi`

#### Slow Training

**Possible causes:**
1. DataLoader bottleneck - increase `dataloader_num_workers`
2. Disk I/O - ensure SSD storage for checkpoints
3. Other processes - check `htop` and `nvidia-smi`

#### Gradient Checkpointing Warning

```
UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly
```

**Solution:** Ignore - this is a deprecation warning that does not affect training.

### Verification Commands

```bash
# Check GPU health
nvidia-smi -q | grep -E "Product Name|Total|Free|Temp"

# Check running processes
ps aux | grep python

# Check disk space
df -h /home/supreethlab/

# Kill stuck processes
pkill -f torchrun
```

---

## Appendix

### A. File Locations

| Item | Path |
|------|------|
| Repository | `/home/supreethlab/repos/nebius-h100-8gpu-benchmark/` |
| Training Checkpoints | `/home/supreethlab/training/checkpoints/` |
| Training Logs | `/home/supreethlab/training/logs/` |
| Benchmark Results | `exercise2/results/` |

### B. Configuration Files

| File | Purpose |
|------|---------|
| `training_config.yaml` | Default training settings |
| `training_config_optimized.yaml` | Higher batch size, more workers |
| `benchmark_config.yaml` | Benchmark thresholds |

### C. Scripts

| Script | Usage |
|--------|-------|
| `singlenode.sh` | `source singlenode.sh [exercise] [config]` |
| `hardware_info.sh` | `./hardware_info.sh [all|gpu|summary]` |

### D. Quick Commands

```bash
# Run default training
source singlenode.sh

# Run optimized training
source singlenode.sh exercise1 optimized

# Run full benchmarks
source singlenode.sh exercise2

# Hardware summary
./hardware_info.sh summary

# GPU topology
./hardware_info.sh gpu

# Monitor GPUs
watch -n 1 nvidia-smi
```

---

## Demo Timeline (Suggested)

| Time | Section | Duration |
|------|---------|----------|
| 0:00 | Overview | 1 min |
| 1:00 | Hardware validation | 2 min |
| 3:00 | GPU benchmarks | 2 min |
| 5:00 | LLM training demo | 4 min |
| 9:00 | Results summary | 2 min |
| 11:00 | Q&A | 4 min |

**Total Duration: 12-15 minutes**

---

## Author

Supreeth Mysore

## Date

2025-12-17

## Version

1.0
