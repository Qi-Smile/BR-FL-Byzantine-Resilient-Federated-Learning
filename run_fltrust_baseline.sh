#!/bin/bash
#
# FLTrust Experiment Runner
# Configuration: 50 clients, 10 servers, 40 rounds × 5 epochs, No attack
# Dataset: CIFAR-10 with ResNet18
#

set -e  # Exit on error

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
echo "=========================================="
echo

# Step 1: Prepare data distribution
echo "[1/2] Preparing data distribution..."
echo "This may take a few minutes for 50 clients..."
python create_data.py

if [ $? -eq 0 ]; then
    echo "✓ Data preparation completed"
else
    echo "✗ Data preparation failed!"
    exit 1
fi

echo

# Step 2: Run federated learning experiment
echo "[2/2] Starting federated learning experiment..."
echo "Expected runtime: ~2-3 hours"
echo "Results will be saved to: new_ms_res/cifar10_ResNet18_FLtrust_*"
echo

python main.py

if [ $? -eq 0 ]; then
    echo
    echo "=========================================="
    echo "✓ Experiment completed successfully!"
    echo "=========================================="
    echo "Check results in: new_ms_res/"
    ls -lh new_ms_res/ | tail -5
else
    echo
    echo "✗ Experiment failed!"
    exit 1
fi
