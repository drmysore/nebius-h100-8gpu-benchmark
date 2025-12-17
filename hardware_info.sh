#!/bin/bash
# hardware_info.sh - Discover single-node hardware configuration
# Usage: ./hardware_info.sh [all|cpu|gpu|memory|ib|summary]

MODE="${1:-all}"

print_header() {
    echo ""
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

# CPU info
cpu_info() {
    print_header "CPU INFORMATION"
    lscpu | grep -E "Model name|Socket|Core|Thread|CPU\(s\):|NUMA" | head -8
}

# Memory info
memory_info() {
    print_header "MEMORY"
    free -h
}

# GPU info
gpu_info() {
    print_header "GPU DEVICES"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

    print_header "GPU TOPOLOGY (NVLink)"
    nvidia-smi topo -m

    print_header "GPU STATUS"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw --format=csv
}

# InfiniBand info
ib_info() {
    print_header "INFINIBAND ADAPTERS"
    if command -v ibstat &> /dev/null; then
        ibstat 2>/dev/null | grep -E "CA |Port |State|Physical|Rate" | head -20
    else
        echo "ibstat not available"
    fi

    print_header "NETWORK DEVICES"
    lspci | grep -i -E "mellanox|infiniband|network" | head -10
}

# Storage info
storage_info() {
    print_header "STORAGE"
    df -h | grep -E "Filesystem|/dev/" | head -5
}

# Quick summary
summary_info() {
    print_header "NODE SUMMARY"
    echo "Hostname: $(hostname)"
    echo "IP Address: $(hostname -I | awk '{print $1}')"
    echo ""
    echo "Hardware:"
    echo "  - CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
    echo "  - Cores: $(nproc)"
    echo "  - RAM: $(free -h | grep Mem | awk '{print $2}')"
    echo "  - GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "  - GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1) per GPU"
    echo ""
    echo "Software:"
    echo "  - Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "  - CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | cut -d',' -f1 || echo 'N/A')"
}

# All info
all_info() {
    summary_info
    cpu_info
    memory_info
    gpu_info
    ib_info
    storage_info
}

# Main
case $MODE in
    all)
        all_info
        ;;
    cpu)
        cpu_info
        ;;
    gpu)
        gpu_info
        ;;
    memory)
        memory_info
        ;;
    ib)
        ib_info
        ;;
    storage)
        storage_info
        ;;
    summary)
        summary_info
        ;;
    *)
        echo "Usage: ./hardware_info.sh [all|cpu|gpu|memory|ib|storage|summary]"
        echo ""
        echo "Options:"
        echo "  all     - Full hardware info (default)"
        echo "  cpu     - CPU information"
        echo "  gpu     - GPU topology and status"
        echo "  memory  - RAM information"
        echo "  ib      - InfiniBand/network info"
        echo "  storage - Disk information"
        echo "  summary - Quick summary"
        ;;
esac
