# Nebius H100 GPU Cluster - LLM Fine-Tuning and Acceptance Testing

This repository contains solutions for two exercises demonstrating GPU cluster deployment, validation, and ML workload execution on Nebius AI Cloud.

- **Exercise 1**: Multi-node LLM fine-tuning for function calling using Slurm/Soperator
- **Exercise 2**: GPU cluster acceptance testing with automated benchmarks and CI/CD

## Cluster Configuration

| Component | Specification |
|-----------|---------------|
| Platform | Nebius AI Cloud |
| GPUs | 8x NVIDIA H100 80GB HBM3 |
| GPU Memory | 79.19 GB per GPU (633.5 GB total) |
| Interconnect | NVLink (intra-node) |
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
│   │   └── train_job.sbatch      # Slurm job script
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

---

## Exercise 1: LLM Fine-Tuning for Function Calling

Fine-tuning Qwen2-7B on the Glaive function calling dataset using LoRA and DeepSpeed ZeRO-3.

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

### Running Training via Slurm (Recommended)

The primary way to run training is via Slurm/Soperator:

```bash
cd exercise1

# Submit the training job
sbatch scripts/train_job.sbatch

# Monitor job status
squeue -u $USER
watch -n 5 squeue -u $USER

# View training logs
tail -f /mnt/shared/logs/train_<JOB_ID>.log

# Check GPU utilization on nodes
srun --jobid=<JOB_ID> nvidia-smi -l 5
```

### Alternative: Manual torchrun (Debug Mode)

For debugging or single-node testing:

```bash
cd exercise1

# Single GPU
python scripts/train_function_calling.py --config configs/training_config.yaml

# Multi-GPU (8 GPUs on single node)
torchrun --nproc_per_node=8 scripts/train_function_calling.py \
    --config configs/training_config.yaml
```

### Success Criteria

Training is considered successful when:

| Criterion | Expected | Location |
|-----------|----------|----------|
| Job completes | Exit code 0 | `squeue` shows COMPLETED |
| Final checkpoint saved | `final/` directory exists | `/mnt/shared/checkpoints/final/` |
| Training loss converges | < 0.5 | Training logs |
| No OOM errors | None | Error logs |
| Throughput | > 2000 tokens/s/GPU | Training logs |

Output artifacts:
- **Checkpoints**: `/mnt/shared/checkpoints/`
- **Logs**: `/mnt/shared/logs/train_<JOB_ID>.log`
- **TensorBoard**: `/mnt/shared/logs/tensorboard/`
- **Summary**: `/mnt/shared/logs/train_<JOB_ID>_summary.txt`

---

## Exercise 2: GPU Cluster Acceptance Testing

Comprehensive benchmark suite for validating GPU cluster performance before production workloads.

### Acceptance Test Results Summary

| Test | Metric | Threshold | Observed | Status |
|------|--------|-----------|----------|--------|
| GPU Health | All GPUs visible | 8/8 | 8/8 | PASS |
| MatMul (BF16) | TFLOPS | > 100 | 728.8 | PASS |
| Memory Bandwidth | TB/s | > 1.0 | 3.02 | PASS |
| NCCL AllReduce | GB/s | > 400 | 442.21 | PASS |
| Training Throughput | tokens/s | > 4000/GPU | 41,468/GPU | PASS |

**Overall Status: PASS**

### Detailed Benchmark Results

#### GPU Health Check
All 8 GPUs detected and operational:
- Model: NVIDIA H100 80GB HBM3
- Memory: 79.19 GB per GPU
- CUDA Compute Capability: 9.0

#### Matrix Multiplication Benchmark (BF16)

| Matrix Size | TFLOPS | Time/Op |
|-------------|--------|---------|
| 4096x4096 | 728.8 | 0.189 ms |
| 8192x8192 | 673.6 | 1.632 ms |
| 16384x16384 | 669.8 | 13.133 ms |

#### Memory Bandwidth

| Metric | Value |
|--------|-------|
| Bandwidth | 3.02 TB/s |
| Test Size | 1 GB |

#### NCCL AllReduce Bandwidth (8 GPUs via NVLink)

| Message Size | Bandwidth | Latency |
|--------------|-----------|---------|
| 1 MB | 41.95 GB/s | 0.044 ms |
| 10 MB | 203.94 GB/s | 0.090 ms |
| 100 MB | 387.05 GB/s | 0.474 ms |
| 1 GB | 442.21 GB/s | 4.150 ms |

#### Training Throughput

| Metric | Value |
|--------|-------|
| Total throughput | 331,742 tokens/s |
| Per-GPU throughput | 41,468 tokens/s |
| World size | 8 GPUs |

### Running Benchmarks

#### Using Docker (Recommended)

```bash
# Pull the container image
docker pull ghcr.io/<your-org>/nebius-h100-8gpu-benchmark:latest

# Health check only
docker run --gpus all ghcr.io/<your-org>/nebius-h100-8gpu-benchmark:latest \
    --mode health

# Single GPU benchmarks (matmul + memory bandwidth)
docker run --gpus all ghcr.io/<your-org>/nebius-h100-8gpu-benchmark:latest \
    --mode single --output /results/benchmark_single.json

# Multi-GPU distributed benchmarks (NCCL + training)
docker run --gpus all ghcr.io/<your-org>/nebius-h100-8gpu-benchmark:latest \
    torchrun --nproc_per_node=8 /app/scripts/benchmark.py \
    --mode distributed --output /results/benchmark_distributed.json

# Full test suite
docker run --gpus all ghcr.io/<your-org>/nebius-h100-8gpu-benchmark:latest \
    --mode full --output /results/benchmark_full.json
```

#### Running Directly

```bash
cd exercise2

# Health check
python scripts/benchmark.py --mode health

# Single GPU benchmark
python scripts/benchmark.py --mode single --output results/benchmark_single.json

# Multi-GPU benchmark (8 GPUs)
torchrun --nproc_per_node=8 scripts/benchmark.py \
    --mode distributed --output results/benchmark_distributed.json
```

### Building the Container Locally

```bash
cd exercise2

# Build the image
docker build -t gpu-cluster-test:local .

# Run tests
docker run --rm gpu-cluster-test:local pytest /app/tests -v

# Run benchmarks (requires GPU)
docker run --gpus all gpu-cluster-test:local --mode health
```

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) automates:

1. **Lint and Test**: Runs flake8 linting and pytest unit tests (CPU-only)
2. **Build**: Builds the Docker image using buildx
3. **Push**: Pushes to GitHub Container Registry (ghcr.io) on main branch
4. **Test Container**: Validates the built image runs correctly
5. **Release**: Creates GitHub releases with container tags on version tags

**Triggering the pipeline:**
- Push to `main` or `develop` branches
- Pull requests to `main`
- Git tags matching `v*` (e.g., `v1.0.0`)

**Container image naming:**
```
ghcr.io/<owner>/<repo>:latest      # Latest from main
ghcr.io/<owner>/<repo>:sha-<hash>  # Specific commit
ghcr.io/<owner>/<repo>:v1.0.0      # Release version
```

### Running on Kubernetes

Deploy the benchmark job to a Kubernetes cluster with GPU nodes:

```bash
cd exercise2/k8s

# Single-node benchmark (8 GPUs)
kubectl apply -f benchmark-job.yaml

# Watch job progress
kubectl get jobs -w

# View logs
kubectl logs -f job/gpu-benchmark

# Get results
kubectl cp gpu-benchmark-<pod>:/results/benchmark_results.json ./results.json

# Clean up
kubectl delete -f benchmark-job.yaml
```

**Resource requirements:**
- 8x NVIDIA GPUs (nvidia.com/gpu: 8)
- 1400 Gi memory limit
- 64 CPU cores

---

## Demo Plan (5-7 minutes)

### 1. Repository Overview (1 min)
- Show repository structure
- Explain exercise1 (LLM fine-tuning) vs exercise2 (acceptance testing)

### 2. Exercise 2: Acceptance Testing Results (2 min)
```bash
# Show benchmark results
cat exercise2/results/benchmark_distributed.json | python -m json.tool

# Highlight key metrics
# - GPU health: 8/8 PASS
# - MatMul: 728.8 TFLOPS
# - NCCL: 442.21 GB/s
# - Training: 331,742 tokens/s
```

### 3. Exercise 1: Training Configuration (2 min)
```bash
# Show Slurm job script
cat exercise1/scripts/train_job.sbatch

# Show training config
cat exercise1/configs/training_config.yaml

# Explain DeepSpeed ZeRO-3 config
cat exercise1/configs/ds_config_zero3.json
```

### 4. Failure Modes and Recovery (1-2 min)
- **NCCL timeout**: Check InfiniBand connectivity, NCCL environment variables
- **OOM errors**: Reduce batch size, enable gradient checkpointing
- **Node failure**: Slurm automatically reschedules; resume from checkpoint
- **Storage issues**: Verify shared filesystem mount, check disk space

### 5. Q&A

---

## Troubleshooting

### NCCL Communication Failures
```bash
# Check InfiniBand status
ibstat

# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Verify GPU-to-GPU connectivity
nvidia-smi topo -m
```

### Out of Memory (OOM)
- Reduce `batch_size` in training config
- Enable `gradient_checkpointing: true`
- Increase DeepSpeed ZeRO stage (currently ZeRO-3)

### Job Failures
```bash
# Check Slurm job status
scontrol show job <JOB_ID>

# View error logs
cat /mnt/shared/logs/train_<JOB_ID>.err
```

---

## Benchmark Date

2025-12-15

## Author

Supreeth Mysore
