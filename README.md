# Nebius H100 8-GPU Benchmark

GPU cluster testing and LLM fine-tuning on a single-node 8x NVIDIA H100 80GB configuration on Nebius AI Cloud.

## Cluster Configuration

| Component | Details |
|-----------|---------|
| Node | poc-h100-node0 |
| GPUs | 8x NVIDIA H100 80GB HBM3 |
| GPU Memory | 79.19 GB per GPU (633.5 GB total) |
| CUDA | 12.1 |
| PyTorch | 2.3.0+cu121 |
| NCCL | 2.20.5 |

## Repository Structure

```
.
├── exercise1/                    # LLM Fine-Tuning for Function Calling
│   ├── scripts/
│   │   ├── train_function_calling.py
│   │   ├── preprocess_data.py
│   │   ├── evaluate.py
│   │   └── train_job.sbatch
│   ├── configs/
│   │   ├── training_config.yaml
│   │   └── ds_config_zero3.json
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       └── terraform.tfvars.example
│
├── exercise2/                    # GPU Cluster Acceptance Testing
│   ├── scripts/
│   │   └── benchmark.py
│   ├── configs/
│   │   └── benchmark_config.yaml
│   ├── results/
│   │   ├── benchmark_health.json
│   │   ├── benchmark_single.json
│   │   └── benchmark_distributed.json
│   ├── tests/
│   │   └── test_benchmark.py
│   ├── k8s/
│   │   └── benchmark-job.yaml
│   ├── .github/workflows/
│   │   └── ci.yml
│   └── Dockerfile
│
└── README.md
```

## Exercise 1: LLM Fine-Tuning for Function Calling

Fine-tuning Qwen2-7B on the Glaive function calling dataset using LoRA.

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2-7B-Instruct |
| Parameters | 7.7B total, 161M trainable (LoRA) |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| Precision | BF16 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 2 per GPU |
| Gradient Accumulation | 4 steps |
| Effective Batch Size | 64 (8 GPUs) |
| Learning Rate | 2e-5 |
| Scheduler | Cosine |
| Max Sequence Length | 4096 |

### Running Training

```bash
cd exercise1

# Single GPU
python scripts/train_function_calling.py --config configs/training_config.yaml

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 scripts/train_function_calling.py \
  --config configs/training_config.yaml
```

## Exercise 2: GPU Cluster Acceptance Testing

Benchmark results for validating GPU cluster performance.

### Benchmark Results Summary

| Test | Result | Expected | Status |
|------|--------|----------|--------|
| GPU Health | All 8 GPUs healthy | - | PASS |
| MatMul (BF16) | 728.8 TFLOPS | >100 TFLOPS | PASS |
| Memory Bandwidth | 3.02 TB/s | >3.0 TB/s | PASS |
| NCCL AllReduce | 442.21 GB/s | >400 GB/s | PASS |
| Training Throughput | 331,742 tokens/s | >2000/GPU | PASS |

### Matrix Multiplication Benchmark (BF16)

| Matrix Size | TFLOPS | Time/Op |
|-------------|--------|---------|
| 4096x4096 | 728.8 | 0.189 ms |
| 8192x8192 | 673.6 | 1.632 ms |
| 16384x16384 | 669.8 | 13.133 ms |

### NCCL AllReduce Bandwidth (8 GPUs via NVLink)

| Message Size | Bandwidth | Latency |
|--------------|-----------|---------|
| 1 MB | 41.95 GB/s | 0.044 ms |
| 10 MB | 203.94 GB/s | 0.090 ms |
| 100 MB | 387.05 GB/s | 0.474 ms |
| 1 GB | 442.21 GB/s | 4.150 ms |

### Running Benchmarks

```bash
cd exercise2

# Health check
python scripts/benchmark.py --mode health

# Single GPU benchmark
python scripts/benchmark.py --mode single

# Multi-GPU benchmark (8 GPUs)
torchrun --nproc_per_node=8 scripts/benchmark.py --mode distributed
```

## Infrastructure

| Setting | Value |
|---------|-------|
| Cloud Provider | Nebius AI Cloud |
| Platform | gpu-h100-sxm |
| Preset | 8gpu-128vcpu-1600gb |
| Interconnect | NVLink |

## Benchmark Date

2025-12-15
