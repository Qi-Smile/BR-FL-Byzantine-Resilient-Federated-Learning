#!/usr/bin/env python3
"""
Experiment script for running Byzantine-Resilient Federated Learning
with various defense methods and attack combinations.

This script runs experiments for 3 defense methods (CoMed, MultiKrum, GeoMed)
across 20 attack combinations (4 server attacks × 5 client attacks).
"""

import os
import subprocess
import yaml
import time
from pathlib import Path

# Define experiment parameters
DEFENSE_METHODS = ['CoMed', 'MultiKrum', 'GeoMed']
SERVER_ATTACKS = ['Noise', 'Random', 'SignFlip', 'Backward']
CLIENT_ATTACKS = ['Noise', 'Random', 'SignFlip', 'Backward', 'LabelFlip']

CONFIG_TEMPLATE = 'config/cifar10_resnet18.yaml'
CONDA_ENV = 'pytorch_new'  # Use pytorch_new environment to avoid cuDNN issues

def update_config(defense, server_attack, client_attack, config_path):
    """
    Update the configuration file with the specified defense and attack methods.

    Args:
        defense: Defense method name
        server_attack: Server attack type
        client_attack: Client attack type
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

    # Set number of rounds to 20 (as per user requirement)
    config['fed_paras']['round'] = 20

    # Ensure GPU is enabled
    config['train_paras']['GPU'] = True
    config['train_paras']['cuda_number'] = 1

    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✓ Updated config: {defense} | Server: {server_attack} | Client: {client_attack}")


def run_experiment(defense, server_attack, client_attack, screen_name=None):
    """
    Run a single experiment with the specified defense and attack combination.

    Args:
        defense: Defense method name
        server_attack: Server attack type
        client_attack: Client attack type
        screen_name: Optional screen session name

    Returns:
        Subprocess object or None
    """
    # Update configuration
    update_config(defense, server_attack, client_attack, CONFIG_TEMPLATE)

    exp_name = f"{defense}_{server_attack}_{client_attack}"

    if screen_name:
        # Run in screen session
        cmd = f"screen -dmS {screen_name} bash -c 'source activate {CONDA_ENV} && python main.py; exec bash'"
        print(f"🚀 Starting experiment in screen: {screen_name}")
        print(f"   Monitor with: screen -r {screen_name}")
    else:
        # Run directly
        cmd = f"source activate {CONDA_ENV} && python main.py"
        print(f"🚀 Running experiment: {exp_name}")

    try:
        process = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
        return process
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return None


def run_all_experiments_sequential():
    """
    Run all experiments sequentially (one after another).
    """
    total_experiments = len(DEFENSE_METHODS) * len(SERVER_ATTACKS) * len(CLIENT_ATTACKS)
    print(f"\n{'='*70}")
    print(f"Starting Sequential Experiment Suite")
    print(f"{'='*70}")
    print(f"Total experiments: {total_experiments}")
    print(f"Defense methods: {DEFENSE_METHODS}")
    print(f"Server attacks: {SERVER_ATTACKS}")
    print(f"Client attacks: {CLIENT_ATTACKS}")
    print(f"{'='*70}\n")

    completed = 0
    failed = 0

    for defense in DEFENSE_METHODS:
        for server_attack in SERVER_ATTACKS:
            for client_attack in CLIENT_ATTACKS:
                completed += 1
                exp_name = f"{defense}_{server_attack}_{client_attack}"

                print(f"\n[{completed}/{total_experiments}] Running: {exp_name}")
                print(f"{'─'*70}")

                # Run experiment
                process = run_experiment(defense, server_attack, client_attack)

                if process:
                    # Wait for completion
                    process.wait()

                    if process.returncode == 0:
                        print(f"✓ Completed successfully")
                    else:
                        print(f"⚠ Finished with return code: {process.returncode}")
                        failed += 1
                else:
                    print(f"❌ Failed to start")
                    failed += 1

                print(f"{'─'*70}")

    print(f"\n{'='*70}")
    print(f"Experiment Suite Completed")
    print(f"{'='*70}")
    print(f"Total: {total_experiments} | Completed: {completed} | Failed: {failed}")
    print(f"{'='*70}\n")


def run_experiments_in_screen():
    """
    Run experiments using screen sessions (allows monitoring individual experiments).
    Note: Only one experiment runs at a time to avoid resource conflicts.
    """
    total_experiments = len(DEFENSE_METHODS) * len(SERVER_ATTACKS) * len(CLIENT_ATTACKS)
    print(f"\n{'='*70}")
    print(f"Starting Experiments in Screen Sessions")
    print(f"{'='*70}")
    print(f"Total experiments: {total_experiments}")
    print(f"Use 'screen -ls' to list sessions")
    print(f"Use 'screen -r <name>' to attach to a session")
    print(f"{'='*70}\n")

    exp_num = 0
    for defense in DEFENSE_METHODS:
        for server_attack in SERVER_ATTACKS:
            for client_attack in CLIENT_ATTACKS:
                exp_num += 1
                screen_name = f"exp_{defense}_{server_attack}_{client_attack}"

                print(f"[{exp_num}/{total_experiments}] {screen_name}")
                run_experiment(defense, server_attack, client_attack, screen_name)

                # Brief delay to allow screen session to start
                time.sleep(2)

    print(f"\n✓ All {total_experiments} experiments queued in screen sessions")
    print(f"⚠ Note: Run one at a time to avoid resource conflicts")
    print(f"Monitor progress with: screen -ls")


def generate_experiment_list():
    """
    Generate a list of all experiments to be run.
    """
    experiments = []
    for defense in DEFENSE_METHODS:
        for server_attack in SERVER_ATTACKS:
            for client_attack in CLIENT_ATTACKS:
                experiments.append({
                    'defense': defense,
                    'server_attack': server_attack,
                    'client_attack': client_attack,
                    'name': f"{defense}_{server_attack}_{client_attack}"
                })
    return experiments


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Byzantine-Resilient FL experiments')
    parser.add_argument('--mode', choices=['sequential', 'screen', 'list'],
                        default='sequential',
                        help='Execution mode: sequential (run one by one), screen (use screen sessions), list (show experiment list)')
    parser.add_argument('--defense', choices=DEFENSE_METHODS + ['all'],
                        default='all',
                        help='Defense method to run (default: all)')

    args = parser.parse_args()

    if args.mode == 'list':
        experiments = generate_experiment_list()
        print(f"\nTotal experiments: {len(experiments)}\n")
        for i, exp in enumerate(experiments, 1):
            print(f"{i:2d}. {exp['name']}")
        print()

    elif args.mode == 'sequential':
        if args.defense != 'all':
            # Filter defense methods
            original_methods = DEFENSE_METHODS.copy()
            DEFENSE_METHODS.clear()
            DEFENSE_METHODS.append(args.defense)

        run_all_experiments_sequential()

        if args.defense != 'all':
            DEFENSE_METHODS.clear()
            DEFENSE_METHODS.extend(original_methods)

    elif args.mode == 'screen':
        if args.defense != 'all':
            original_methods = DEFENSE_METHODS.copy()
            DEFENSE_METHODS.clear()
            DEFENSE_METHODS.append(args.defense)

        run_experiments_in_screen()

        if args.defense != 'all':
            DEFENSE_METHODS.clear()
            DEFENSE_METHODS.extend(original_methods)
