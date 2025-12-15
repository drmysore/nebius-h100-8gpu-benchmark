# Nebius H100 8-GPU Benchmark Results

This repository contains GPU cluster acceptance testing results for a single-node configuration with 8x NVIDIA H100 80GB GPUs on Nebius AI Cloud.

## Cluster Configuration

| Component | Details |
|-----------|---------|
| Node | poc-h100-node0 |
| GPUs | 8x NVIDIA H100 80GB HBM3 |
| GPU Memory | 79.19 GB per GPU (633.5 GB total) |
| CUDA | 12.1 |
| PyTorch | 2.3.0+cu121 |
| NCCL | 2.20.5 |

## Benchmark Results Summary

| Test | Result | Expected | Status |
|------|--------|----------|--------|
| GPU Health | All 8 GPUs healthy | - | PASS |
| MatMul (BF16) | 728.8 TFLOPS | >100 TFLOPS | PASS |
| Memory Bandwidth | 3.02 TB/s | >3.0 TB/s | PASS |
| NCCL AllReduce | 442.21 GB/s | >400 GB/s | PASS |
| Training Throughput | 331,742 tokens/s | >2000/GPU | PASS |

## Detailed Results

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

### Training Benchmark

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 style transformer (12 layers, 768 hidden) |
| Batch Size | 8 per GPU |
| Sequence Length | 512 |
| Total Throughput | 331,742.7 tokens/second |
| Per-GPU Throughput | 41,467.8 tokens/second |

## Running the Benchmark

Single GPU test:

```bash
python scripts/benchmark.py --mode single
```

Multi-GPU test (8 GPUs):

```bash
torchrun --nproc_per_node=8 scripts/benchmark.py --mode distributed
```

## Repository Contents

| File | Description |
|------|-------------|
| scripts/benchmark.py | Main benchmark script |
| configs/benchmark_config.yaml | Benchmark configuration |
| results/ | JSON benchmark results |

## Infrastructure

| Setting | Value |
|---------|-------|
| Platform | gpu-h100-sxm |
| Preset | 8gpu-128vcpu-1600gb |
| Interconnect | NVLink |

## Benchmark Date

2025-12-15
