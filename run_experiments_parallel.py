#!/usr/bin/env python3
"""
Parallel experiment runner with continuous monitoring and GPU load balancing.

Features:
- Concurrent execution: max 12 experiments at a time
- Continuous monitoring: checks every 5 minutes for completed experiments
- Automatic refill: starts new experiments as soon as slots become available
- GPU balancing: alternates between cuda0 and cuda1
- Screen sessions: one per experiment for easy monitoring
- Uses pytorch_new environment
"""

import os
import subprocess
import yaml
import time
import argparse
from pathlib import Path
from datetime import datetime


# Experiment configuration
DEFENSE_METHODS = ['CoMed', 'GeoMed']
SERVER_ATTACKS = ['Noise', 'Random', 'SignFlip', 'Backward']
CLIENT_ATTACKS = ['Noise', 'Random', 'SignFlip', 'Backward', 'LabelFlip']

CONFIG_TEMPLATE = 'config/cifar10_resnet18.yaml'
CONDA_ENV = 'pytorch_new'
MAX_CONCURRENT = 4  # Maximum concurrent experiments (2 per GPU)
GPUS = [0, 1]  # Two 4090 GPUs


def update_config(defense, server_attack, client_attack, gpu_id, config_path):
    """
    Update configuration file for an experiment.

    Args:
        defense: Defense method name
        server_attack: Server attack type
        client_attack: Client attack type
        gpu_id: GPU device ID (0 or 1)
        config_path: Path to config file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update defense and attack settings
    config['general_paras']['defense'] = defense
    config['general_paras']['server_attacks'] = server_attack
    config['general_paras']['client_attacks'] = client_attack

    # Set attack ratios
    config['fed_paras']['server_attack_ratio'] = 0.1  # 10% Byzantine servers
    config['fed_paras']['client_attack_ratio'] = 0.1  # 10% Byzantine clients

    # Set number of rounds
    config['fed_paras']['round'] = 20

    # GPU configuration
    config['train_paras']['GPU'] = True
    config['train_paras']['cuda_number'] = gpu_id

    # Ensure results go to ms_res
    config['general_paras']['output_dir'] = 'ms_res/'

    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_running_experiments():
    """
    Get list of currently running experiment screen sessions.

    Returns:
        List of screen session names
    """
    try:
        result = subprocess.run(
            ['screen', '-ls'],
            capture_output=True,
            text=True
        )

        # Parse screen -ls output
        running = []
        for line in result.stdout.split('\n'):
            if 'exp_' in line and 'Detached' in line:
                # Extract screen name
                parts = line.strip().split('.')
                if len(parts) >= 2:
                    screen_name = parts[1].split()[0]
                    running.append(screen_name)

        return running
    except Exception as e:
        print(f"⚠ Error checking running experiments: {e}")
        return []


def check_experiment_completed(defense, server_attack, client_attack):
    """
    Check if experiment has completed by verifying signal file or result files exist.

    Args:
        defense: Defense method name
        server_attack: Server attack type
        client_attack: Client attack type

    Returns:
        Tuple (completed: bool, gpu_id: int or None)
    """
    # Check for signal file first
    signal_file = f"signals/cifar10_ResNet18_{defense}_{server_attack}_{client_attack}_1_0.1_0.1.done"

    if os.path.exists(signal_file):
        # Read GPU info from signal file
        gpu_id = None
        try:
            with open(signal_file, 'r') as f:
                for line in f:
                    if line.startswith('gpu_id='):
                        gpu_id = int(line.strip().split('=')[1])
                        break
            # Remove signal file after reading
            os.remove(signal_file)
        except:
            pass
        return True, gpu_id

    # Fallback: check if result file exists
    result_dir = f"ms_res/cifar10_ResNet18_{defense}_{server_attack}_{client_attack}_1_0.1_0.1"
    result_file = os.path.join(result_dir, 'test_acc.xlsx')

    if os.path.exists(result_file):
        return True, None

    return False, None


def start_experiment(defense, server_attack, client_attack, gpu_id):
    """
    Start a single experiment in a screen session.

    Args:
        defense: Defense method name
        server_attack: Server attack type
        client_attack: Client attack type
        gpu_id: GPU device ID

    Returns:
        Screen session name if successful, None otherwise
    """
    screen_name = f"exp_{defense}_{server_attack}_{client_attack}_gpu{gpu_id}"

    # Create experiment-specific config file to avoid race conditions
    exp_config_path = f"config/exp_{defense}_{server_attack}_{client_attack}_gpu{gpu_id}.yaml"
    update_config(defense, server_attack, client_attack, gpu_id, CONFIG_TEMPLATE)

    # Copy the updated config to experiment-specific file
    import shutil
    shutil.copy(CONFIG_TEMPLATE, exp_config_path)

    # Create screen command with experiment-specific config
    cmd = f"screen -dmS {screen_name} bash -c 'eval \"$(conda shell.bash hook)\" && conda activate {CONDA_ENV} && python main.py --config {exp_config_path}; rm -f {exp_config_path}; exec bash'"

    try:
        subprocess.run(cmd, shell=True, executable='/bin/bash', check=True)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ✓ Started: {screen_name}")
        return screen_name
    except Exception as e:
        print(f"❌ Failed to start {screen_name}: {e}")
        return None


def generate_experiment_queue():
    """
    Generate queue of all experiments to run, skipping already completed ones.

    Returns:
        List of experiment dicts
    """
    queue = []
    skipped = 0
    gpu_idx = 0

    for defense in DEFENSE_METHODS:
        for server_attack in SERVER_ATTACKS:
            for client_attack in CLIENT_ATTACKS:
                # Check if experiment already completed
                result_dir = f"ms_res/cifar10_ResNet18_{defense}_{server_attack}_{client_attack}_1_0.1_0.1"
                result_file = os.path.join(result_dir, 'test_acc.xlsx')

                if os.path.exists(result_file):
                    skipped += 1
                    continue  # Skip completed experiments

                # Alternate GPUs
                gpu_id = GPUS[gpu_idx % len(GPUS)]
                gpu_idx += 1

                queue.append({
                    'defense': defense,
                    'server_attack': server_attack,
                    'client_attack': client_attack,
                    'gpu_id': gpu_id,
                    'name': f"{defense}_{server_attack}_{client_attack}_gpu{gpu_id}"
                })

    if skipped > 0:
        print(f"⏭ Skipped {skipped} already completed experiments")

    return queue


def run_experiments_parallel(check_interval=5, dry_run=False):
    """
    Run experiments in parallel with continuous monitoring.

    Args:
        check_interval: Minutes to wait between checks (default: 5)
        dry_run: If True, only print what would be done
    """
    queue = generate_experiment_queue()
    total_experiments = len(queue)
    started_experiments = {}  # screen_name -> exp_info
    failed = []

    print("="*80)
    print("Parallel Experiment Runner (Continuous Monitoring)")
    print("="*80)
    print(f"Total experiments: {total_experiments}")
    print(f"Max concurrent: {MAX_CONCURRENT}")
    print(f"GPUs: {GPUS}")
    print(f"Check interval: {check_interval} minutes")
    print(f"Conda environment: {CONDA_ENV}")
    print("="*80)
    print()

    if dry_run:
        print("DRY RUN MODE - No experiments will be started\n")
        for i, exp in enumerate(queue, 1):
            print(f"{i:2d}. {exp['name']}")
        print(f"\nGPU distribution:")
        gpu_counts = {}
        for exp in queue:
            gpu_counts[exp['gpu_id']] = gpu_counts.get(exp['gpu_id'], 0) + 1
        for gpu_id, count in sorted(gpu_counts.items()):
            print(f"   GPU {gpu_id}: {count} experiments")
        return

    queue_idx = 0
    start_time = datetime.now()

    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        while queue_idx < total_experiments or len(started_experiments) > 0:
            # Get currently running experiments
            running = get_running_experiments()
            running_set = set(running)

            # Check for completed experiments and track freed GPUs
            completed_now = []
            failed_now = []
            freed_gpus = []  # Track which GPUs became available

            for screen_name in list(started_experiments.keys()):
                exp = started_experiments[screen_name]

                # First check signal file (experiment completed successfully)
                completed, signal_gpu_id = check_experiment_completed(exp['defense'], exp['server_attack'], exp['client_attack'])

                if completed:
                    # Experiment completed - get GPU from signal or exp record
                    gpu_id = signal_gpu_id if signal_gpu_id is not None else exp['gpu_id']
                    freed_gpus.append(gpu_id)
                    completed_now.append((screen_name, gpu_id))
                    del started_experiments[screen_name]
                    # Kill the screen session since experiment is done
                    subprocess.run(['screen', '-X', '-S', screen_name, 'quit'],
                                   capture_output=True, timeout=5)
                elif screen_name not in running_set:
                    # Screen died but no results - experiment failed
                    gpu_id = exp['gpu_id']
                    freed_gpus.append(gpu_id)
                    failed_now.append((screen_name, gpu_id))
                    failed.append(exp['name'])
                    del started_experiments[screen_name]

            if completed_now:
                timestamp = datetime.now().strftime("%H:%M:%S")
                for screen_name, gpu_id in completed_now:
                    print(f"[{timestamp}] ✓ Completed: {screen_name} (GPU {gpu_id} freed)")

            if failed_now:
                timestamp = datetime.now().strftime("%H:%M:%S")
                for screen_name, gpu_id in failed_now:
                    print(f"[{timestamp}] ❌ Failed: {screen_name} (GPU {gpu_id} freed, no results)")

            # Start new experiments on freed GPUs first, then balance
            while queue_idx < total_experiments and len(started_experiments) < MAX_CONCURRENT:
                # Determine which GPU to use
                if freed_gpus:
                    # Use a freed GPU first
                    target_gpu = freed_gpus.pop(0)
                else:
                    # Count experiments per GPU and use least loaded
                    gpu_counts = {gpu: 0 for gpu in GPUS}
                    for exp in started_experiments.values():
                        gpu_counts[exp['gpu_id']] += 1
                    target_gpu = min(gpu_counts, key=gpu_counts.get)

                # Find next experiment in queue that matches this GPU, or reassign GPU
                exp = queue[queue_idx]
                queue_idx += 1

                # Override the pre-assigned GPU with the target GPU (freed or least-loaded)
                exp['gpu_id'] = target_gpu
                exp['name'] = f"{exp['defense']}_{exp['server_attack']}_{exp['client_attack']}_gpu{target_gpu}"

                screen_name = start_experiment(
                    exp['defense'],
                    exp['server_attack'],
                    exp['client_attack'],
                    exp['gpu_id']
                )

                if screen_name:
                    started_experiments[screen_name] = exp
                    print(f"   Progress: {queue_idx}/{total_experiments} queued, {len(started_experiments)} running (GPU {target_gpu})")
                    time.sleep(2)  # Brief delay between starts
                else:
                    failed.append(exp['name'])

            # If all experiments queued and none running, we're done
            if queue_idx >= total_experiments and len(started_experiments) == 0:
                break

            # Status update
            if len(started_experiments) > 0:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] Status: {len(started_experiments)} running, "
                      f"{queue_idx}/{total_experiments} queued, "
                      f"{total_experiments - queue_idx - len(started_experiments)} pending")

                # Show GPU distribution
                gpu_running = {0: 0, 1: 0}
                for screen_name in started_experiments:
                    if 'gpu0' in screen_name:
                        gpu_running[0] += 1
                    elif 'gpu1' in screen_name:
                        gpu_running[1] += 1
                print(f"   GPU 0: {gpu_running[0]} | GPU 1: {gpu_running[1]}")

                print(f"\n💡 Monitor: screen -ls | Attach: screen -r exp_<name> | Detach: Ctrl+A, D")
                print(f"⏰ Next check in {check_interval} minutes...\n")

                # Wait before next check
                time.sleep(check_interval * 60)

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        print("Current state:")
        print(f"  Queued: {queue_idx}/{total_experiments}")
        print(f"  Running: {len(started_experiments)} experiments")
        if started_experiments:
            print("\nStill running:")
            for screen_name in started_experiments:
                print(f"  - {screen_name}")
        return

    # Final summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*80}")
    print("All Experiments Completed!")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successfully started: {total_experiments - len(failed)}")
    print(f"Failed to start: {len(failed)}")
    print(f"Total time: {elapsed}")
    print()
    print("💡 Check results:")
    print("   ls -lht ms_res/               # View result directories")
    print("   python generate_hybrid_attack_table.py  # Generate result table")
    print(f"{'='*80}\n")


def monitor_experiments():
    """
    Display current status of running experiments.
    """
    running = get_running_experiments()

    print("="*80)
    print(f"Running Experiments ({len(running)} active)")
    print("="*80)

    if not running:
        print("No experiments currently running in screen sessions.")
    else:
        for i, screen_name in enumerate(running, 1):
            print(f"{i:2d}. {screen_name}")

        # Count by GPU
        gpu_counts = {0: 0, 1: 0}
        for screen_name in running:
            if 'gpu0' in screen_name:
                gpu_counts[0] += 1
            elif 'gpu1' in screen_name:
                gpu_counts[1] += 1

        print(f"\nGPU distribution:")
        print(f"  GPU 0: {gpu_counts[0]} experiments")
        print(f"  GPU 1: {gpu_counts[1]} experiments")

    print("="*80)
    print("\n💡 Commands:")
    print("   screen -r <name>  # Attach to experiment")
    print("   Ctrl+A, D         # Detach from screen")
    print("   screen -X -S <name> quit  # Kill experiment")
    print()


def cleanup_screens():
    """
    Clean up all experiment screen sessions.
    """
    running = get_running_experiments()

    if not running:
        print("No experiment screen sessions to clean up.")
        return

    print(f"Found {len(running)} experiment screen sessions.")
    confirm = input("Are you sure you want to kill all of them? (yes/no): ")

    if confirm.lower() != 'yes':
        print("Cancelled.")
        return

    for screen_name in running:
        try:
            subprocess.run(['screen', '-X', '-S', screen_name, 'quit'], check=True)
            print(f"✓ Killed: {screen_name}")
        except Exception as e:
            print(f"❌ Failed to kill {screen_name}: {e}")

    print(f"\n✓ Cleanup complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Byzantine-Resilient FL experiments in parallel with continuous monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (12 concurrent, 5 min checks)
  python run_experiments_parallel.py

  # Dry run to see experiment queue
  python run_experiments_parallel.py --dry-run

  # Custom check interval (10 minutes)
  python run_experiments_parallel.py --interval 10

  # Monitor running experiments
  python run_experiments_parallel.py --monitor

  # Clean up all screens
  python run_experiments_parallel.py --cleanup
        """
    )

    parser.add_argument('--interval', type=float, default=5,
                        help='Minutes to wait between completion checks (default: 5)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show experiment queue without starting')
    parser.add_argument('--monitor', action='store_true',
                        help='Monitor currently running experiments')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up all experiment screen sessions')

    args = parser.parse_args()

    if args.monitor:
        monitor_experiments()
    elif args.cleanup:
        cleanup_screens()
    else:
        run_experiments_parallel(
            check_interval=args.interval,
            dry_run=args.dry_run
        )
