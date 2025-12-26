#!/usr/bin/env python
"""
Defense Configuration Experiments for BR-FL

This script implements three experiments to analyze the defense configuration:
- Experiment A: Clipping Rate vs Attack Intensity (heatmap data)
- Experiment B: Impact of gamma on robustness and communication
- Experiment C: Fundamental Threshold Validation

Usage:
    python defense_config_experiments.py --experiment A
    python defense_config_experiments.py --experiment B
    python defense_config_experiments.py --experiment C
"""

import torch
import argparse
import numpy as np
import pandas as pd
import os
import copy
import math
from datetime import datetime
from torch.utils.data import DataLoader

# Import utilities
from utils.help import load_config, get_device
from utils.utility import get_dataset, load_model, get_attacker_index, get_round_index
from utils.utility import test_model, normal_train_model

# Import attacks
from Attack.Noise import NoiseAttacks
from Attack.Random import RandomAttacks
from Attack.SignFlip import SignFlipAttacks
from Attack.Backward import BackwardAttacks

# Import defenses
from Defense.Avg import AvgDefense
from Defense.TreamMean import TreamMeanDefense


def setup_experiment(config_path='./config/cifar10_resnet18.yaml'):
    """Load configuration and setup basic experiment parameters"""
    config = load_config(config_path)
    device = get_device(config['train_paras']['cuda_number'], config['train_paras']['GPU'])
    return config, device


def run_federated_training(config, device, alpha, beta, gamma, mc, mp, attack_type='Random',
                          max_rounds=40, verbose=True):
    """
    Run a single federated learning experiment with specified parameters.

    Args:
        config: Configuration dict
        device: Training device
        alpha: Server-side clipping rate
        beta: Client-side clipping rate
        gamma: Broadcast fraction
        mc: Client attack ratio
        mp: Server attack ratio
        attack_type: Type of attack ('Random', 'Noise', 'SignFlip', 'Backward')
        max_rounds: Number of global rounds
        verbose: Print progress

    Returns:
        dict: Results including accuracy curves and final accuracy
    """

    # Update config with experiment parameters
    config['fed_paras']['client_attack_ratio'] = mc
    config['fed_paras']['server_attack_ratio'] = mp
    config['general_paras']['server_attacks'] = attack_type
    config['general_paras']['client_attacks'] = attack_type
    config['fed_paras']['round'] = max_rounds

    # Calculate trimmed rates based on alpha and beta
    # Following the logic in main.py
    trimmed_client_rate = alpha
    trimmed_server_rate = beta

    # Load dataset
    dataset_name = config['dataset_paras']['name']
    dirichlet_rate = config['fed_paras']['dirichlet_rate']
    train_dataset_list, test_dataset, _ = get_dataset(dataset_name, dirichlet_rate)

    server_number = config['fed_paras']['server_number']
    client_number = config['fed_paras']['client_number']

    # Create data loaders
    dataset_train_loader_list = [
        DataLoader(train_dataset_list[client_id],
                  batch_size=config['dataset_paras']['batch_size'],
                  num_workers=config['dataset_paras']['num_workers'],
                  pin_memory=True, shuffle=True)
        for client_id in range(client_number)
    ]

    dataset_test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config['dataset_paras']['batch_size'],
        num_workers=config['dataset_paras']['num_workers'],
        pin_memory=True, shuffle=False
    )

    # Initialize models
    model_name = config['model_paras']['name']
    num_classes = config['dataset_paras']['num_classes']

    base_model = load_model(model_name, num_classes)
    server_model_list = [copy.deepcopy(base_model) for _ in range(server_number)]
    client_model_list = [copy.deepcopy(base_model) for _ in range(client_number)]

    # Get attacker indices
    server_attacker_index, client_attacker_index = get_attacker_index(
        server_number, client_number, mp, mc, 'order'
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment Configuration:")
        print(f"  alpha={alpha}, beta={beta}, gamma={gamma}")
        print(f"  mc={mc}, mp={mp}, attack={attack_type}")
        print(f"  Servers: {server_number}, Clients: {client_number}")
        print(f"  Server attackers: {len(server_attacker_index)}/{server_number}")
        print(f"  Client attackers: {len(client_attacker_index)}/{client_number}")
        print(f"{'='*60}\n")

    # Initialize tracking
    accuracy_history = []
    loss_history = []

    # Initialize update lists
    server_send_updates = [copy.deepcopy(server_model_list[idx].state_dict())
                          for idx in range(server_number)]
    client_send_updates = [copy.deepcopy(client_model_list[idx].state_dict())
                          for idx in range(client_number)]

    # Training loop
    for each_round in range(max_rounds):

        # Store old models for certain attacks
        if attack_type in ['SignFlip', 'Backward']:
            old_server_model_list = copy.deepcopy(server_model_list)
            old_client_model_list = copy.deepcopy(client_model_list)

        # 1. Servers send global model to clients
        select_server_id, server_to_clients, update_counts = get_round_index(
            server_number, client_number
        )

        global_state_dict = AvgDefense(server_send_updates,
                                       index=list(range(server_number)),
                                       device=device)

        # 2. Clients aggregate server updates (using trimmed mean with beta)
        defense_state_dict = TreamMeanDefense(server_send_updates,
                                             index=list(range(server_number)),
                                             trimmed_rate=trimmed_server_rate)

        # 3. Local training
        for client_id in range(client_number):
            # Load aggregated model
            if client_id in client_attacker_index:
                client_model_list[client_id].load_state_dict(global_state_dict)
            else:
                client_model_list[client_id].load_state_dict(defense_state_dict)

            # Train
            for ep in range(config['train_paras']['epoch']):
                client_model_list[client_id], _ = normal_train_model(
                    model=client_model_list[client_id],
                    train_loader=dataset_train_loader_list[client_id],
                    device=device,
                    epoch=ep + each_round * config['train_paras']['epoch'],
                    criterion_name=config['train_paras']['criterion_name'],
                    optimizer_name=config['train_paras']['optimizer_name'],
                    lr=config['train_paras']['lr']
                )

        # 4. Clients upload updates (with attacks if malicious)
        for client_id in range(client_number):
            if client_id in client_attacker_index and config['attack_paras']['client_is_attack']:
                # Apply attack
                if attack_type == 'Noise':
                    client_send_updates[client_id] = NoiseAttacks(
                        client_model_list[client_id],
                        config['attack_paras']['noise_mean'],
                        config['attack_paras']['noise_std'],
                        device
                    )
                elif attack_type == 'Random':
                    client_send_updates[client_id] = RandomAttacks(
                        client_model_list[client_id],
                        config['attack_paras']['random_lower'],
                        config['attack_paras']['random_upper'],
                        device
                    )
                elif attack_type == 'SignFlip':
                    client_send_updates[client_id] = SignFlipAttacks(
                        client_model_list[client_id],
                        old_client_model_list[client_id],
                        config['attack_paras']['SignFlipScaleFactor']
                    )
                elif attack_type == 'Backward':
                    client_send_updates[client_id] = BackwardAttacks(
                        client_model_list[client_id],
                        old_client_model_list[client_id]
                    )
            else:
                # Benign client
                client_send_updates[client_id] = copy.deepcopy(
                    client_model_list[client_id].state_dict()
                )

        # 5. Servers aggregate client updates (using trimmed mean with alpha)
        for server_id in range(server_number):
            if server_id in server_attacker_index:
                # Malicious server: just average
                res_model_state_dict = AvgDefense(
                    client_send_updates,
                    server_to_clients[server_id],
                    device
                )
                if res_model_state_dict is not None:
                    server_model_list[server_id].load_state_dict(res_model_state_dict)

                # Apply server attack
                if config['attack_paras']['server_is_attack']:
                    if attack_type == 'Noise':
                        server_send_updates[server_id] = NoiseAttacks(
                            server_model_list[server_id],
                            config['attack_paras']['noise_mean'],
                            config['attack_paras']['noise_std'],
                            device
                        )
                    elif attack_type == 'Random':
                        server_send_updates[server_id] = RandomAttacks(
                            server_model_list[server_id],
                            config['attack_paras']['random_lower'],
                            config['attack_paras']['random_upper'],
                            device
                        )
                    elif attack_type == 'SignFlip':
                        server_send_updates[server_id] = SignFlipAttacks(
                            server_model_list[server_id],
                            old_server_model_list[server_id],
                            config['attack_paras']['SignFlipScaleFactor']
                        )
                    elif attack_type == 'Backward':
                        server_send_updates[server_id] = BackwardAttacks(
                            server_model_list[server_id],
                            old_server_model_list[server_id]
                        )
                else:
                    server_send_updates[server_id] = copy.deepcopy(
                        server_model_list[server_id].state_dict()
                    )
            else:
                # Benign server: use trimmed mean defense
                res_model_state_dict = TreamMeanDefense(
                    client_send_updates,
                    server_to_clients[server_id],
                    trimmed_rate=trimmed_client_rate
                )
                if res_model_state_dict is not None:
                    server_model_list[server_id].load_state_dict(res_model_state_dict)

                server_send_updates[server_id] = copy.deepcopy(
                    server_model_list[server_id].state_dict()
                )

        # 6. Evaluate
        # Test with a benign client model
        benign_clients = [i for i in range(client_number) if i not in client_attacker_index]
        if benign_clients:
            test_client_id = benign_clients[0]
            test_acc, test_loss = test_model(
                model=client_model_list[test_client_id],
                test_loader=dataset_test_loader,
                device=device,
                criterion_name=config['train_paras']['criterion_name']
            )

            accuracy_history.append(test_acc)
            loss_history.append(test_loss)

            if verbose and (each_round % 5 == 0 or each_round == max_rounds - 1):
                print(f"Round {each_round+1}/{max_rounds}: "
                      f"Acc={test_acc:.2f}%, Loss={test_loss:.4f}")

    # Calculate communication cost (simplified: number of parameters transmitted)
    # gamma affects how many servers each client communicates with
    total_params = sum(p.numel() for p in base_model.parameters())
    comm_cost_per_round = total_params * client_number * gamma  # Client receives from gamma*P servers
    total_comm_cost = comm_cost_per_round * max_rounds

    return {
        'accuracy_history': accuracy_history,
        'loss_history': loss_history,
        'final_accuracy': accuracy_history[-1] if accuracy_history else 0.0,
        'communication_cost': total_comm_cost,
        'params': {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'mc': mc,
            'mp': mp,
            'attack': attack_type
        }
    }


def experiment_a_clipping_vs_attack(output_dir='./defense_config_results/exp_a'):
    """
    Experiment A: Clipping Rate vs Attack Intensity

    Tests different (alpha, beta) configurations against various attack intensities.
    Outputs a 3x6 accuracy matrix for heatmap visualization.
    """
    print("\n" + "="*60)
    print("EXPERIMENT A: Clipping Rate vs Attack Intensity")
    print("="*60)

    # Load base config
    config, device = setup_experiment('./config/cifar10_resnet18.yaml')

    # Fixed parameters
    gamma = 1.0
    attack_type = 'Random'
    max_rounds = 40

    # Clipping rate configurations
    configs = [
        {'alpha': 0.25, 'beta': 0.25},
        {'alpha': 0.35, 'beta': 0.35},
        {'alpha': 0.45, 'beta': 0.45},
    ]

    # Attack intensity levels
    attack_levels = [
        {'mc': 0.05, 'mp': 0.05},
        {'mc': 0.10, 'mp': 0.10},
        {'mc': 0.15, 'mp': 0.10},
        {'mc': 0.10, 'mp': 0.15},
        {'mc': 0.15, 'mp': 0.15},
        {'mc': 0.20, 'mp': 0.20},
    ]

    # Result matrix: [configs x attack_levels]
    results_matrix = np.zeros((len(configs), len(attack_levels)))
    all_results = []

    # Run experiments
    for i, clip_config in enumerate(configs):
        for j, attack_level in enumerate(attack_levels):
            print(f"\n[{i*len(attack_levels)+j+1}/{len(configs)*len(attack_levels)}] "
                  f"Testing alpha={clip_config['alpha']}, beta={clip_config['beta']} "
                  f"vs mc={attack_level['mc']}, mp={attack_level['mp']}")

            result = run_federated_training(
                config=copy.deepcopy(config),
                device=device,
                alpha=clip_config['alpha'],
                beta=clip_config['beta'],
                gamma=gamma,
                mc=attack_level['mc'],
                mp=attack_level['mp'],
                attack_type=attack_type,
                max_rounds=max_rounds,
                verbose=False
            )

            results_matrix[i, j] = result['final_accuracy']
            all_results.append(result)

            print(f"  → Final Accuracy: {result['final_accuracy']:.2f}%")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save matrix as numpy array
    np.save(os.path.join(output_dir, f'accuracy_matrix_{timestamp}.npy'), results_matrix)

    # Save as CSV for easy viewing
    df = pd.DataFrame(
        results_matrix,
        index=[f"α={c['alpha']}, β={c['beta']}" for c in configs],
        columns=[f"mc={a['mc']}, mp={a['mp']}" for a in attack_levels]
    )
    df.to_csv(os.path.join(output_dir, f'accuracy_matrix_{timestamp}.csv'))

    # Save detailed results
    detailed_df = pd.DataFrame([
        {
            'alpha': r['params']['alpha'],
            'beta': r['params']['beta'],
            'mc': r['params']['mc'],
            'mp': r['params']['mp'],
            'final_accuracy': r['final_accuracy']
        }
        for r in all_results
    ])
    detailed_df.to_csv(os.path.join(output_dir, f'detailed_results_{timestamp}.csv'), index=False)

    print(f"\n{'='*60}")
    print("EXPERIMENT A COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    print("\nAccuracy Matrix (%):")
    print(df)

    return results_matrix, all_results


def experiment_b_gamma_impact(output_dir='./defense_config_results/exp_b'):
    """
    Experiment B: Impact of gamma on robustness and communication

    Tests different gamma values with fixed attack intensity.
    Outputs accuracy curves and communication costs.
    """
    print("\n" + "="*60)
    print("EXPERIMENT B: Impact of Gamma")
    print("="*60)

    # Load base config
    config, device = setup_experiment('./config/cifar10_resnet18.yaml')

    # Fixed parameters
    mc, mp = 0.1, 0.1
    alpha, beta = 0.35, 0.40
    attack_type = 'Random'
    max_rounds = 40

    # Gamma values to test
    gamma_values = [0.3, 0.5, 0.7, 1.0]

    all_results = []

    # Run experiments
    for gamma in gamma_values:
        print(f"\nTesting gamma={gamma}")

        result = run_federated_training(
            config=copy.deepcopy(config),
            device=device,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            mc=mc,
            mp=mp,
            attack_type=attack_type,
            max_rounds=max_rounds,
            verbose=True
        )

        all_results.append(result)
        print(f"  → Final Accuracy: {result['final_accuracy']:.2f}%")
        print(f"  → Communication Cost: {result['communication_cost']:.2e}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save accuracy curves
    max_len = max(len(r['accuracy_history']) for r in all_results)
    accuracy_curves = np.zeros((len(gamma_values), max_len))

    for i, result in enumerate(all_results):
        accuracy_curves[i, :len(result['accuracy_history'])] = result['accuracy_history']

    np.save(os.path.join(output_dir, f'accuracy_curves_{timestamp}.npy'), accuracy_curves)

    # Save summary
    summary_df = pd.DataFrame([
        {
            'gamma': r['params']['gamma'],
            'final_accuracy': r['final_accuracy'],
            'communication_cost': r['communication_cost'],
            'comm_cost_relative': r['communication_cost'] / all_results[-1]['communication_cost']
        }
        for r in all_results
    ])
    summary_df.to_csv(os.path.join(output_dir, f'gamma_impact_{timestamp}.csv'), index=False)

    print(f"\n{'='*60}")
    print("EXPERIMENT B COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    print("\nGamma Impact Summary:")
    print(summary_df)

    return all_results


def experiment_c_threshold_validation(output_dir='./defense_config_results/exp_c'):
    """
    Experiment C: Fundamental Threshold Validation

    Tests various (mc, mp) combinations to verify theoretical threshold.
    Uses most conservative configuration.
    """
    print("\n" + "="*60)
    print("EXPERIMENT C: Threshold Validation")
    print("="*60)

    # Load base config
    config, device = setup_experiment('./config/cifar10_resnet18.yaml')

    # Most conservative configuration
    alpha, beta, gamma = 0.45, 0.45, 1.0
    attack_type = 'Random'
    max_rounds = 40

    # Test points with expected outcomes
    test_points = [
        (0.10, 0.10, 'success'),   # Should succeed
        (0.15, 0.10, 'success'),   # Should succeed
        (0.10, 0.20, 'success'),   # Should succeed
        (0.15, 0.15, 'success'),   # Should succeed
        (0.20, 0.15, 'failure'),   # Should fail
        (0.15, 0.20, 'failure'),   # Should fail
        (0.20, 0.20, 'failure'),   # Should fail
    ]

    all_results = []

    # Run experiments
    for mc, mp, expected in test_points:
        print(f"\nTesting mc={mc}, mp={mp} (expected: {expected})")

        result = run_federated_training(
            config=copy.deepcopy(config),
            device=device,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            mc=mc,
            mp=mp,
            attack_type=attack_type,
            max_rounds=max_rounds,
            verbose=False
        )

        result['expected'] = expected
        all_results.append(result)

        # Determine if actually succeeded (threshold: 70% accuracy)
        actual = 'success' if result['final_accuracy'] > 70.0 else 'failure'
        match = '✓' if actual == expected else '✗'

        print(f"  → Final Accuracy: {result['final_accuracy']:.2f}%")
        print(f"  → Expected: {expected}, Actual: {actual} {match}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save accuracy curves for all test points
    max_len = max(len(r['accuracy_history']) for r in all_results)
    accuracy_curves = np.zeros((len(test_points), max_len))

    for i, result in enumerate(all_results):
        accuracy_curves[i, :len(result['accuracy_history'])] = result['accuracy_history']

    np.save(os.path.join(output_dir, f'threshold_curves_{timestamp}.npy'), accuracy_curves)

    # Save summary
    summary_df = pd.DataFrame([
        {
            'mc': r['params']['mc'],
            'mp': r['params']['mp'],
            'final_accuracy': r['final_accuracy'],
            'expected': r['expected'],
            'actual': 'success' if r['final_accuracy'] > 70.0 else 'failure',
            'match': r['expected'] == ('success' if r['final_accuracy'] > 70.0 else 'failure')
        }
        for r in all_results
    ])
    summary_df.to_csv(os.path.join(output_dir, f'threshold_validation_{timestamp}.csv'), index=False)

    print(f"\n{'='*60}")
    print("EXPERIMENT C COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    print("\nThreshold Validation Summary:")
    print(summary_df)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Defense Configuration Experiments for BR-FL'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['A', 'B', 'C', 'all'],
        required=True,
        help='Which experiment to run: A (clipping vs attack), B (gamma impact), C (threshold), or all'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./defense_config_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Defense Configuration Experiments")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.experiment == 'A' or args.experiment == 'all':
        experiment_a_clipping_vs_attack(os.path.join(args.output, 'exp_a'))

    if args.experiment == 'B' or args.experiment == 'all':
        experiment_b_gamma_impact(os.path.join(args.output, 'exp_b'))

    if args.experiment == 'C' or args.experiment == 'all':
        experiment_c_threshold_validation(os.path.join(args.output, 'exp_c'))

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
