# Byzantine-Resilient FL Experiments

## New Defense Methods Implemented

This implementation adds 2 new robust aggregation methods:

1. **CoMed (Coordinate-wise Median)**
   - Computes element-wise median across client updates
   - Reference: Yin et al., ICML 2018

2. **MultiKrum**
   - Extension of Krum that selects top-m updates and averages them
   - Parameter: m = n - f - 2 for maximum robustness
   - Reference: Blanchard et al., NeurIPS 2017

3. **GeoMed (Geometric Median)** [Already implemented]
   - Computes point minimizing sum of L2 distances
   - Uses Weiszfeld's algorithm
   - Reference: Pillutla et al., IEEE TSP 2022

## Experiment Setup

- **Dataset**: CIFAR-10
- **Model**: ResNet-18
- **Clients**: 50 (C = 50)
- **Servers**: 10 (P = 10)
- **Byzantine Clients**: 5 (10%)
- **Byzantine Servers**: 1 (10%)
- **Data Heterogeneity**: Dirichlet(α = 1.0)
- **Local Epochs**: 5
- **Communication Rounds**: 20

## Attack Matrix

**Server Attacks (4 types)**:
- Noise
- Random
- SignFlip
- Backward

**Client Attacks (5 types)**:
- Noise
- Random
- SignFlip
- Backward
- LabelFlip

**Total**: 4 × 5 = 20 attack combinations per defense method

## Running Experiments

### Option 1: Run All Experiments Sequentially

```bash
# Make scripts executable
chmod +x run_experiments.py analyze_results.py

# Run all experiments (CoMed, MultiKrum, GeoMed) × 20 attack combinations
python run_experiments.py --mode sequential

# Run only one defense method
python run_experiments.py --mode sequential --defense CoMed
```

### Option 2: Run with Screen Sessions

```bash
# Start experiments in screen sessions (easier to monitor)
python run_experiments.py --mode screen

# Monitor progress
screen -ls
screen -r exp_CoMed_Noise_Noise

# Detach from screen: Ctrl+A, D
```

### Option 3: List All Experiments

```bash
# Show list of all experiments to be run
python run_experiments.py --mode list
```

### Option 4: Run Single Experiment Manually

```bash
# Edit config/cifar10_resnet18.yaml:
# - Set defense: 'CoMed' (or 'MultiKrum', 'GeoMed')
# - Set server_attacks: 'Noise'
# - Set client_attacks: 'LabelFlip'
# - Set server_attack_ratio: 0.1
# - Set client_attack_ratio: 0.1

# Run with pytorch_new environment (avoids cuDNN issues)
source activate pytorch_new
python main.py
```

## Analyzing Results

### Generate Summary Report

```bash
# Analyze all results and generate CSV summary
python analyze_results.py

# This creates: experiment_summary.csv
```

### Generate Pivot Tables

```bash
# Create pivot tables (Server Attack × Client Attack) for each defense
python analyze_results.py --pivot

# This creates:
# - pivot_CoMed_mean.csv
# - pivot_CoMed_std.csv
# - pivot_MultiKrum_mean.csv
# - pivot_MultiKrum_std.csv
# - pivot_GeoMed_mean.csv
# - pivot_GeoMed_std.csv
```

### Result Metrics

For each experiment, the analysis reports:
- **Mean Accuracy (%)**: Average accuracy across all benign clients
- **Std Accuracy (%)**: Standard deviation across benign clients
- **Min/Max Accuracy (%)**: Range of accuracies

## Output Structure

```
new_ms_res/
├── cifar10_ResNet18_CoMed_Noise_Noise_1_0.1_0.1/
│   ├── test_acc.xlsx
│   └── test_loss.xlsx
├── cifar10_ResNet18_CoMed_Noise_Random_1_0.1_0.1/
│   ├── test_acc.xlsx
│   └── test_loss.xlsx
...
```

## Expected Runtime

- Single experiment: ~2-3 hours (20 rounds, CIFAR-10, GPU)
- Total for one defense method: ~40-60 hours (20 experiments)
- Total for all 3 methods: ~120-180 hours

## Implementation Details

### CoMed
- Location: `Defense/CoMed.py`
- Server-side: Applies coordinate-wise median
- Client-side: Averages server updates

### MultiKrum
- Location: `Defense/MultiKrum.py`
- Server-side: Selects top-m updates (m = n - f - 2)
- Client-side: Averages server updates
- Requires: n ≥ 2f + 3

### Integration
- main.py: Lines 18-19 (imports), Lines 187-189 (client aggregation), Lines 340-358 (server aggregation)

## Troubleshooting

### CUDA/cuDNN Issues
- Use `pytorch_new` conda environment
- PyTorch 2.5.1 + CUDA 12.1
- Avoids cuDNN version conflicts

### Out of Memory
- Reduce batch size in config: `batch_size: 64` (from 128)
- Use CPU mode: `GPU: False`

### Check Experiment Progress
```bash
# Monitor screen sessions
screen -ls

# Check latest results
ls -lhtr new_ms_res/
tail new_ms_res/*/test_acc.xlsx
```
