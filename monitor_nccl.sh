#!/bin/bash
# monitor_nccl.sh - Monitor NCCL/NVLink traffic during training (single node)

echo -e "\033[1;36m╔════════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[1;36m║\033[0m        \033[1;33mNCCL / NVLink Monitor (Single Node)\033[0m                       \033[1;36m║\033[0m"
echo -e "\033[1;36m╚════════════════════════════════════════════════════════════════╝\033[0m"
echo ""

MODE="${1:-all}"

show_nvlink() {
    echo -e "\033[1;32m=== NVLink Status ===\033[0m"
    nvidia-smi nvlink -s 2>/dev/null | head -30 || echo "NVLink stats not available"
    echo ""
}

show_gpu_p2p() {
    echo -e "\033[1;32m=== GPU P2P Bandwidth Matrix ===\033[0m"
    nvidia-smi topo -p2p r 2>/dev/null || nvidia-smi topo -m | head -15
    echo ""
}

show_topology() {
    echo -e "\033[1;32m=== GPU Topology ===\033[0m"
    nvidia-smi topo -m
    echo ""
}

show_nccl_env() {
    echo -e "\033[1;32m=== NCCL Environment ===\033[0m"
    echo "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-not set}"
    echo "NCCL_DEBUG=${NCCL_DEBUG:-not set}"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-not set}"
    echo ""
}

live_monitor() {
    echo -e "\033[1;33mStarting live monitor (Ctrl+C to stop)...\033[0m"
    echo ""
    watch -n 2 '
        echo "=== GPU Utilization ==="
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv
        echo ""
        echo "=== NVLink Traffic ==="
        nvidia-smi nvlink -g 0 2>/dev/null | head -10 || echo "N/A"
    '
}

case $MODE in
    nvlink)
        show_nvlink
        ;;
    p2p)
        show_gpu_p2p
        ;;
    topo)
        show_topology
        ;;
    env)
        show_nccl_env
        ;;
    live)
        live_monitor
        ;;
    all|*)
        show_nccl_env
        show_nvlink
        show_topology
        echo -e "\033[1;90mTip: Use './monitor_nccl.sh live' for continuous monitoring\033[0m"
        ;;
esac
