#!/bin/bash
#
# FLTrust No-Attack Baseline Experiment
# 50 clients, 10 servers, 40 rounds × 5 epochs
# CIFAR-10, ResNet18, No attack (mc=0, mp=0)
#

echo "=========================================="
echo "FLTrust No-Attack Baseline Experiment"
echo "=========================================="
echo "Configuration:"
echo "  - Defense: FLTrust (FIXED version)"
echo "  - Servers: 10"
echo "  - Clients: 50"
echo "  - Rounds: 40 × 5 epochs = 200 total epochs"
echo "  - Attack: None (mc=0, mp=0)"
echo "  - Dataset: CIFAR-10 (non-IID, Dirichlet α=1)"
echo "  - Model: ResNet18"
echo "  - GPU: cuda:1"
echo "=========================================="
echo

echo "Starting experiment at: $(date)"
echo "Expected runtime: ~2-3 hours"
echo

# Run experiment
python main.py

exit_code=$?

echo
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✓ Experiment completed successfully!"
    echo "Finished at: $(date)"
    echo
    echo "Results saved to:"
    ls -lh new_ms_res/ | tail -3
else
    echo "✗ Experiment failed with exit code: $exit_code"
fi
echo "=========================================="

exit $exit_code
