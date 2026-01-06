#!/usr/bin/env python3
"""
Safe Operating Region (SOR) Experiments Runner

This script runs a grid of 25 experiments to validate the SOR theory:
- Fixed defense parameters: alpha=0.35, beta=0.35, gamma=1.0
- Varying Byzantine ratios: 5 mc values × 5 mp values = 25 experiments
- mc (client Byzantine ratio): [0.05, 0.10, 0.15, 0.20, 0.25]
- mp (server Byzantine ratio): [0.0, 0.1, 0.2, 0.3, 0.4]

Usage:
    python run_sor_experiments.py              # Run all experiments
    python run_sor_experiments.py --monitor    # Monitor running experiments
    python run_sor_experiments.py --dry-run    # Show experiment queue without running
"""

import os
import subprocess
import time
import yaml
import argparse
import shutil
from datetime import datetime

# ============ EXPERIMENT CONFIGURATION ============

# Byzantine ratio grid
MC_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]  # Client Byzantine ratio
MP_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]       # Server Byzantine ratio

# Fixed defense parameters (SOR theory validation)
FIXED_ALPHA = 0.35
FIXED_BETA = 0.35
FIXED_GAMMA = 1.0

# Experiment settings
FIXED_ATTACK = 'Noise'           # Both server and client attacks
DIRICHLET_RATE = 1               # Data distribution parameter
GLOBAL_ROUNDS = 20               # Global training rounds
LOCAL_EPOCHS = 5                 # Local epochs per round
NUM_WORKERS = 2                  # DataLoader workers

# Infrastructure
CONFIG_TEMPLATE = 'config/cifar10_resnet18.yaml'
CONDA_ENV = 'pytorch_new'
MAX_CONCURRENT = 4               # 2 GPUs, run 4 experiments in parallel (2 per GPU)
GPUS = [0, 1]                    # Available GPU IDs
OUTPUT_DIR = 'sor_res/'          # Results directory
CHECK_INTERVAL = 60              # Seconds between status checks

# ============ HELPER FUNCTIONS ============

def load_yaml(file_path):
    """Load YAML configuration file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(config, file_path):
    """Save configuration to YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config_for_sor(mc, mp, gpu_id):
    """
    Create experiment-specific config for SOR experiment.

    Args:
        mc: Client Byzantine ratio
        mp: Server Byzantine ratio
        gpu_id: GPU ID to use

    Returns:
        config: Updated configuration dictionary
    """
    config = load_yaml(CONFIG_TEMPLATE)

    # Set defense parameters (CRITICAL: enables manual mode in main.py)
    if 'defense_paras' not in config:
        config['defense_paras'] = {}

    config['defense_paras']['alpha'] = FIXED_ALPHA
    config['defense_paras']['beta'] = FIXED_BETA
    config['defense_paras']['gamma'] = FIXED_GAMMA

    # Set defense method
    config['general_paras']['defense'] = 'Ours'
    config['general_paras']['server_attacks'] = FIXED_ATTACK
    config['general_paras']['client_attacks'] = FIXED_ATTACK
    config['general_paras']['output_dir'] = OUTPUT_DIR

    # Set Byzantine ratios
    config['fed_paras']['server_attack_ratio'] = mp
    config['fed_paras']['client_attack_ratio'] = mc
    config['fed_paras']['dirichlet_rate'] = DIRICHLET_RATE
    config['fed_paras']['round'] = GLOBAL_ROUNDS

    # Training parameters
    config['train_paras']['epoch'] = LOCAL_EPOCHS
    config['train_paras']['GPU'] = True
    config['train_paras']['cuda_number'] = gpu_id

    # Dataset parameters
    config['dataset_paras']['num_workers'] = NUM_WORKERS
    config['dataset_paras']['batch_size'] = 64  # Reduce from 128 to avoid GPU OOM

    return config


def get_result_dir(mc, mp):
    """
    Get expected result directory name for experiment.

    Must match the naming convention from utils/help.py:
    cifar10_ResNet18_Ours_a0.35_b0.35_g1.0_Noise_Noise_1_{mp}_{mc}
    """
    return f"{OUTPUT_DIR}cifar10_ResNet18_Ours_a{FIXED_ALPHA}_b{FIXED_BETA}_g{FIXED_GAMMA}_Noise_Noise_{DIRICHLET_RATE}_{mp}_{mc}"


def check_experiment_completed(mc, mp):
    """Check if experiment has completed successfully."""
    result_dir = get_result_dir(mc, mp)
    result_file = os.path.join(result_dir, 'test_acc.xlsx')
    return os.path.exists(result_file)


def get_screen_name(mc, mp, gpu_id):
    """Generate screen session name for experiment."""
    return f"sor_exp_mc{mc}_mp{mp}_gpu{gpu_id}"


def get_config_path(mc, mp, gpu_id):
    """Generate temporary config file path for experiment."""
    return f"config/sor_exp_mc{mc}_mp{mp}_gpu{gpu_id}.yaml"


def check_screen_exists(screen_name):
    """Check if a screen session exists."""
    result = subprocess.run(
        ['screen', '-ls'],
        capture_output=True,
        text=True
    )
    return screen_name in result.stdout


def start_experiment(mc, mp, gpu_id):
    """
    Start a single SOR experiment in a detached screen session.

    Args:
        mc: Client Byzantine ratio
        mp: Server Byzantine ratio
        gpu_id: GPU ID to use

    Returns:
        screen_name: Name of the screen session
    """
    screen_name = get_screen_name(mc, mp, gpu_id)
    config_path = get_config_path(mc, mp, gpu_id)

    # Create experiment-specific config
    config = update_config_for_sor(mc, mp, gpu_id)
    save_yaml(config, config_path)

    # Build screen command
    cmd = f"""screen -dmS {screen_name} bash -c '
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && \
conda activate {CONDA_ENV} && \
echo "Starting SOR experiment: mc={mc}, mp={mp}, GPU={gpu_id}" && \
echo "Config: {config_path}" && \
echo "Results: {get_result_dir(mc, mp)}" && \
python main.py --config {config_path} && \
echo "Experiment completed successfully" && \
rm -f {config_path} || echo "Experiment failed or interrupted"'
"""

    subprocess.run(cmd, shell=True, executable='/bin/bash', check=True)
    print(f"  ✓ Started: mc={mc}, mp={mp} on GPU {gpu_id} (screen: {screen_name})")
    return screen_name


def generate_experiment_queue():
    """
    Generate queue of experiments to run.

    Returns:
        queue: List of experiment dictionaries
        stats: Statistics dictionary
    """
    queue = []
    completed = []
    gpu_idx = 0

    for mc in MC_VALUES:
        for mp in MP_VALUES:
            if check_experiment_completed(mc, mp):
                completed.append({'mc': mc, 'mp': mp})
                continue

            # Assign GPU in round-robin fashion
            gpu_id = GPUS[gpu_idx % len(GPUS)]
            gpu_idx += 1

            queue.append({
                'mc': mc,
                'mp': mp,
                'gpu_id': gpu_id,
                'name': f"mc{mc}_mp{mp}"
            })

    stats = {
        'total': len(MC_VALUES) * len(MP_VALUES),
        'queued': len(queue),
        'completed': len(completed),
        'completion_rate': len(completed) / (len(MC_VALUES) * len(MP_VALUES)) * 100
    }

    return queue, stats, completed


def get_running_experiments():
    """Get list of currently running SOR experiment screen sessions."""
    result = subprocess.run(
        ['screen', '-ls'],
        capture_output=True,
        text=True
    )

    running = []
    for line in result.stdout.split('\n'):
        if 'sor_exp_mc' in line:
            # Extract screen name
            parts = line.strip().split('\t')
            if parts:
                screen_full = parts[0]
                # Format: "12345.sor_exp_mc0.05_mp0.0_gpu0"
                screen_name = screen_full.split('.', 1)[1] if '.' in screen_full else screen_full
                running.append(screen_name)

    return running


def monitor_experiments():
    """Display current status of SOR experiments."""
    print("\n" + "="*70)
    print("SOR EXPERIMENTS - STATUS MONITOR")
    print("="*70)

    queue, stats, completed = generate_experiment_queue()
    running = get_running_experiments()

    print(f"\n📊 Overall Progress:")
    print(f"  Total experiments: {stats['total']}")
    print(f"  Completed: {stats['completed']} ({stats['completion_rate']:.1f}%)")
    print(f"  Running: {len(running)}")
    print(f"  Queued: {stats['queued']}")

    if running:
        print(f"\n🔄 Running Experiments ({len(running)}):")
        for screen_name in running:
            print(f"  • {screen_name}")

    if completed:
        print(f"\n✓ Completed Experiments ({len(completed)}):")
        for exp in completed[:10]:  # Show first 10
            print(f"  • mc={exp['mc']}, mp={exp['mp']}")
        if len(completed) > 10:
            print(f"  ... and {len(completed) - 10} more")

    if queue:
        print(f"\n⏳ Queued Experiments ({len(queue)}):")
        for exp in queue[:10]:  # Show first 10
            print(f"  • mc={exp['mc']}, mp={exp['mp']} → GPU {exp['gpu_id']}")
        if len(queue) > 10:
            print(f"  ... and {len(queue) - 10} more")

    print("\n" + "="*70)
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def cleanup_orphaned_configs():
    """Remove orphaned config files from failed experiments."""
    config_pattern = "config/sor_exp_mc*_mp*_gpu*.yaml"
    orphaned = []

    import glob
    for config_file in glob.glob(config_pattern):
        # Extract screen name from config file
        basename = os.path.basename(config_file).replace('.yaml', '')
        # Check if corresponding screen exists
        if not check_screen_exists(basename):
            orphaned.append(config_file)
            try:
                os.remove(config_file)
                print(f"  Removed orphaned config: {config_file}")
            except Exception as e:
                print(f"  Failed to remove {config_file}: {e}")

    return orphaned


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description='Run SOR experiments')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor running experiments')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show experiment queue without running')
    parser.add_argument('--cleanup', action='store_true',
                       help='Cleanup orphaned config files')
    parser.add_argument('--interval', type=int, default=CHECK_INTERVAL,
                       help=f'Check interval in seconds (default: {CHECK_INTERVAL})')

    args = parser.parse_args()

    # Monitor mode
    if args.monitor:
        while True:
            monitor_experiments()
            time.sleep(args.interval)

    # Cleanup mode
    if args.cleanup:
        print("Cleaning up orphaned config files...")
        orphaned = cleanup_orphaned_configs()
        print(f"Removed {len(orphaned)} orphaned config files")
        return

    # Generate experiment queue
    print("\n" + "="*70)
    print("SOR EXPERIMENTS - QUEUE GENERATION")
    print("="*70)
    print(f"\nExperiment Configuration:")
    print(f"  Defense: Ours (alpha={FIXED_ALPHA}, beta={FIXED_BETA}, gamma={FIXED_GAMMA})")
    print(f"  Attack: {FIXED_ATTACK} (both server and client)")
    print(f"  mc values: {MC_VALUES}")
    print(f"  mp values: {MP_VALUES}")
    print(f"  Training: {GLOBAL_ROUNDS} rounds × {LOCAL_EPOCHS} epochs")
    print(f"  GPUs: {GPUS}")
    print(f"  Max concurrent: {MAX_CONCURRENT}")

    queue, stats, completed = generate_experiment_queue()

    print(f"\n📊 Queue Statistics:")
    print(f"  Total experiments: {stats['total']}")
    print(f"  Already completed: {stats['completed']} ({stats['completion_rate']:.1f}%)")
    print(f"  To be queued: {stats['queued']}")

    if stats['queued'] == 0:
        print("\n✓ All experiments already completed!")
        return

    # Dry run mode
    if args.dry_run:
        print(f"\n⏳ Experiment Queue ({len(queue)} experiments):")
        for i, exp in enumerate(queue, 1):
            result_dir = get_result_dir(exp['mc'], exp['mp'])
            print(f"  {i:2d}. mc={exp['mc']}, mp={exp['mp']} → GPU {exp['gpu_id']}")
            print(f"      Output: {result_dir}")
        print("\nDry run complete. Use without --dry-run to start experiments.")
        return

    # Create output directory
    print(f"\n✓ Starting {stats['queued']} experiments...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run experiments
    print("\n" + "="*70)
    print("STARTING EXPERIMENTS")
    print("="*70 + "\n")

    running_screens = []
    queue_idx = 0

    try:
        while queue_idx < len(queue) or running_screens:
            # Start new experiments up to MAX_CONCURRENT
            while len(running_screens) < MAX_CONCURRENT and queue_idx < len(queue):
                exp = queue[queue_idx]
                screen_name = start_experiment(exp['mc'], exp['mp'], exp['gpu_id'])
                running_screens.append({
                    'screen': screen_name,
                    'mc': exp['mc'],
                    'mp': exp['mp'],
                    'gpu': exp['gpu_id'],
                    'start_time': time.time()
                })
                queue_idx += 1
                time.sleep(2)  # Brief pause between launches

            # Check for completed experiments
            time.sleep(args.interval)

            still_running = []
            for exp_info in running_screens:
                if check_screen_exists(exp_info['screen']):
                    still_running.append(exp_info)
                else:
                    # Experiment finished
                    elapsed = time.time() - exp_info['start_time']
                    if check_experiment_completed(exp_info['mc'], exp_info['mp']):
                        print(f"  ✓ Completed: mc={exp_info['mc']}, mp={exp_info['mp']} "
                              f"(GPU {exp_info['gpu']}) in {elapsed/60:.1f} min")
                    else:
                        print(f"  ✗ Failed: mc={exp_info['mc']}, mp={exp_info['mp']} "
                              f"(GPU {exp_info['gpu']}) - no result file found")

                    # Cleanup config file
                    config_file = get_config_path(exp_info['mc'], exp_info['mp'], exp_info['gpu'])
                    if os.path.exists(config_file):
                        os.remove(config_file)

            running_screens = still_running

            # Status update
            if running_screens or queue_idx < len(queue):
                _, current_stats, _ = generate_experiment_queue()
                print(f"\n📊 Progress: {current_stats['completed']}/{current_stats['total']} completed, "
                      f"{len(running_screens)} running, {len(queue) - queue_idx} queued")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        print(f"Running experiments: {len(running_screens)}")
        print("These will continue in the background. Use 'screen -ls' to see them.")
        return

    # Final summary
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)

    _, final_stats, _ = generate_experiment_queue()
    print(f"\n✓ Final Status:")
    print(f"  Total experiments: {final_stats['total']}")
    print(f"  Completed: {final_stats['completed']} ({final_stats['completion_rate']:.1f}%)")
    print(f"  Results directory: {OUTPUT_DIR}")

    if final_stats['queued'] > 0:
        print(f"\n⚠️  Warning: {final_stats['queued']} experiments did not complete successfully")
        print("  Run the script again to retry failed experiments.")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
