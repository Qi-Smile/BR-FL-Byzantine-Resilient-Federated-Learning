#\!/bin/bash
echo "=================================="
echo "FLTrust Experiment Status Monitor"
echo "=================================="
echo
echo "[GPU Experiment - PyTorch 2.5.1]"
screen -ls | grep fltrust_gpu
echo "Speed: ~28 it/s | ETA: 2-3 hours"
echo "Log: tail -f fltrust_gpu_experiment.log"
echo "Recent progress:"
tail -5 fltrust_gpu_experiment.log | grep -E "(Epoch|Accuracy|Round)" | tail -3
echo
echo "[CPU Experiment - Backup]"
screen -ls | grep fltrust_cpu
echo "Speed: ~1.6 it/s | ETA: 10-12 hours"
echo "Log: tail -f fltrust_cpu_experiment.log"
echo "Recent progress:"
tail -5 fltrust_cpu_experiment.log | grep -E "(Epoch|Accuracy|Round)" | tail -3
echo
echo "=================================="
echo "Commands:"
echo "  GPU: screen -r fltrust_gpu"
echo "  CPU: screen -r fltrust_cpu"
echo "  Stop CPU: screen -S fltrust_cpu -X quit"
echo "=================================="

