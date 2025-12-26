#!/bin/bash
#
# FLTrust No-Attack Baseline Experiment (CPU Mode)
# 50 clients, 10 servers, 40 rounds × 5 epochs
# CIFAR-10, ResNet18, No attack (mc=0, mp=0)
#
# NOTE: Running on CPU due to CUDA segmentation fault issue
# GPU crashes at training start despite all troubleshooting attempts
#

LOG_FILE="fltrust_cpu_experiment.log"

echo "==========================================" | tee $LOG_FILE
echo "FLTrust No-Attack Baseline Experiment" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
echo "Configuration:" | tee -a $LOG_FILE
echo "  - Defense: FLTrust (FIXED version)" | tee -a $LOG_FILE
echo "  - Servers: 10" | tee -a $LOG_FILE
echo "  - Clients: 50" | tee -a $LOG_FILE
echo "  - Rounds: 40 × 5 epochs = 200 total epochs" | tee -a $LOG_FILE
echo "  - Attack: None (mc=0, mp=0)" | tee -a $LOG_FILE
echo "  - Dataset: CIFAR-10 (non-IID, Dirichlet α=1)" | tee -a $LOG_FILE
echo "  - Model: ResNet18" | tee -a $LOG_FILE
echo "  - Device: CPU (due to CUDA crash)" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
echo | tee -a $LOG_FILE

echo "Starting experiment at: $(date)" | tee -a $LOG_FILE
echo "Expected runtime: ~10-12 hours (CPU is slower than GPU)" | tee -a $LOG_FILE
echo | tee -a $LOG_FILE

# Run experiment
python main.py 2>&1 | tee -a $LOG_FILE

exit_code=$?

echo | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
if [ $exit_code -eq 0 ]; then
    echo "✓ Experiment completed successfully!" | tee -a $LOG_FILE
    echo "Finished at: $(date)" | tee -a $LOG_FILE
    echo | tee -a $LOG_FILE
    echo "Results saved to:" | tee -a $LOG_FILE
    ls -lh new_ms_res/ | tail -3 | tee -a $LOG_FILE
else
    echo "✗ Experiment failed with exit code: $exit_code" | tee -a $LOG_FILE
fi
echo "==========================================" | tee -a $LOG_FILE

exit $exit_code
