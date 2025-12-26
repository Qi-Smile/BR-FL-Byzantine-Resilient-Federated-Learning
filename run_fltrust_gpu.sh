#!/bin/bash
#
# FLTrust GPU Experiment (PyTorch 2.5.1)
# 50 clients, 10 servers, 40 rounds × 5 epochs
# CIFAR-10, ResNet18, No attack (mc=0, mp=0)
# Using NEW PyTorch 2.5.1 environment (fixes CUDA crash)
#

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_new

LOG_FILE="fltrust_gpu_experiment.log"

echo "==========================================" | tee $LOG_FILE
echo "FLTrust GPU Experiment (PyTorch 2.5.1)" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
echo "Configuration:" | tee -a $LOG_FILE
echo "  - Defense: FLTrust (FIXED version)" | tee -a $LOG_FILE
echo "  - Servers: 10" | tee -a $LOG_FILE
echo "  - Clients: 50" | tee -a $LOG_FILE
echo "  - Rounds: 40 × 5 epochs = 200 total epochs" | tee -a $LOG_FILE
echo "  - Attack: None (mc=0, mp=0)" | tee -a $LOG_FILE
echo "  - Dataset: CIFAR-10 (non-IID, Dirichlet α=1)" | tee -a $LOG_FILE
echo "  - Model: ResNet18" | tee -a $LOG_FILE
echo "  - Device: GPU cuda:1" | tee -a $LOG_FILE
echo "  - PyTorch: 2.5.1+cu121 (NEW!)" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE
echo | tee -a $LOG_FILE

python --version | tee -a $LOG_FILE
python -c "import torch; print(f'PyTorch: {torch.__version__}')" | tee -a $LOG_FILE
python -c "import torch; print(f'CUDA: {torch.version.cuda}')" | tee -a $LOG_FILE
echo | tee -a $LOG_FILE

echo "Starting experiment at: $(date)" | tee -a $LOG_FILE
echo "Expected runtime: ~2-3 hours (GPU)" | tee -a $LOG_FILE
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
