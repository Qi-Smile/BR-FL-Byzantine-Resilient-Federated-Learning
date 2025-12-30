#!/usr/bin/env python3
"""
Analyze Byzantine-Resilient Federated Learning experiment results.

This script analyzes the results from all experiments and computes
statistics for benign clients.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def analyze_single_experiment(result_dir):
    """
    Analyze a single experiment's results.

    Args:
        result_dir: Path to experiment result directory

    Returns:
        Dictionary containing analysis results
    """
    acc_file = os.path.join(result_dir, 'test_acc.xlsx')

    if not os.path.exists(acc_file):
        return None

    try:
        # Read accuracy data
        df_acc = pd.read_excel(acc_file)

        # Identify benign clients (columns starting with "Benign")
        benign_columns = [col for col in df_acc.columns if col.startswith('Benign')]

        if len(benign_columns) == 0:
            print(f"Warning: No benign clients found in {result_dir}")
            return None

        # Get final epoch accuracies for benign clients
        final_acc = df_acc[benign_columns].iloc[-1]

        results = {
            'num_benign_clients': len(benign_columns),
            'final_acc_mean': final_acc.mean(),
            'final_acc_std': final_acc.std(),
            'final_acc_min': final_acc.min(),
            'final_acc_max': final_acc.max(),
            'benign_accuracies': final_acc.values
        }

        return results

    except Exception as e:
        print(f"Error analyzing {result_dir}: {e}")
        return None


def find_experiment_results(output_dir='new_ms_res'):
    """
    Find all experiment result directories.

    Args:
        output_dir: Base output directory

    Returns:
        List of result directory paths
    """
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return []

    result_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains result files
            if os.path.exists(os.path.join(item_path, 'test_acc.xlsx')):
                result_dirs.append(item_path)

    return sorted(result_dirs)


def parse_experiment_name(dirname):
    """
    Parse experiment name to extract defense and attack types.

    Expected format: cifar10_ResNet18_{defense}_{server_attack}_{client_attack}_...

    Args:
        dirname: Directory name

    Returns:
        Dictionary with parsed components or None
    """
    parts = dirname.split('_')

    if len(parts) < 5:
        return None

    try:
        return {
            'defense': parts[2],
            'server_attack': parts[3],
            'client_attack': parts[4],
        }
    except IndexError:
        return None


def generate_summary_report(output_dir='new_ms_res', save_path='experiment_summary.csv'):
    """
    Generate a summary report of all experiments.

    Args:
        output_dir: Base output directory
        save_path: Path to save summary CSV

    Returns:
        DataFrame containing summary results
    """
    result_dirs = find_experiment_results(output_dir)

    if len(result_dirs) == 0:
        print(f"No experiment results found in {output_dir}")
        return None

    print(f"Found {len(result_dirs)} experiment results")

    summary_data = []

    for result_dir in result_dirs:
        dirname = os.path.basename(result_dir)
        exp_info = parse_experiment_name(dirname)

        if exp_info is None:
            print(f"Warning: Could not parse experiment name: {dirname}")
            continue

        results = analyze_single_experiment(result_dir)

        if results is None:
            continue

        summary_data.append({
            'Defense': exp_info['defense'],
            'Server Attack': exp_info['server_attack'],
            'Client Attack': exp_info['client_attack'],
            'Benign Clients': results['num_benign_clients'],
            'Mean Accuracy (%)': results['final_acc_mean'],
            'Std Accuracy (%)': results['final_acc_std'],
            'Min Accuracy (%)': results['final_acc_min'],
            'Max Accuracy (%)': results['final_acc_max'],
        })

        print(f"✓ {exp_info['defense']:<12} | "
              f"{exp_info['server_attack']:<10} vs {exp_info['client_attack']:<10} | "
              f"Acc: {results['final_acc_mean']:.2f}% ± {results['final_acc_std']:.2f}%")

    if len(summary_data) == 0:
        print("No valid experiment results to summarize")
        return None

    # Create DataFrame
    df_summary = pd.DataFrame(summary_data)

    # Save to CSV
    df_summary.to_csv(save_path, index=False)
    print(f"\n✓ Summary saved to: {save_path}")

    return df_summary


def generate_pivot_tables(df_summary, output_prefix='pivot'):
    """
    Generate pivot tables for easier analysis.

    Args:
        df_summary: Summary DataFrame
        output_prefix: Prefix for output files
    """
    if df_summary is None or len(df_summary) == 0:
        return

    for defense in df_summary['Defense'].unique():
        df_defense = df_summary[df_summary['Defense'] == defense]

        # Create pivot table: Server Attack × Client Attack
        pivot_mean = df_defense.pivot_table(
            values='Mean Accuracy (%)',
            index='Server Attack',
            columns='Client Attack',
            aggfunc='mean'
        )

        pivot_std = df_defense.pivot_table(
            values='Std Accuracy (%)',
            index='Server Attack',
            columns='Client Attack',
            aggfunc='mean'
        )

        # Save pivot tables
        pivot_mean.to_csv(f"{output_prefix}_{defense}_mean.csv")
        pivot_std.to_csv(f"{output_prefix}_{defense}_std.csv")

        print(f"\n{defense} Defense - Mean Accuracy (%):")
        print(pivot_mean.round(2))
        print(f"\n{defense} Defense - Std Accuracy (%):")
        print(pivot_std.round(2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--output-dir', default='new_ms_res',
                        help='Output directory containing experiment results')
    parser.add_argument('--summary', default='experiment_summary.csv',
                        help='Path to save summary CSV')
    parser.add_argument('--pivot', action='store_true',
                        help='Generate pivot tables')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Analyzing Experiment Results")
    print(f"{'='*80}\n")

    # Generate summary report
    df_summary = generate_summary_report(args.output_dir, args.summary)

    # Generate pivot tables if requested
    if args.pivot and df_summary is not None:
        print(f"\n{'='*80}")
        print(f"Generating Pivot Tables")
        print(f"{'='*80}\n")
        generate_pivot_tables(df_summary)

    print(f"\n{'='*80}")
    print(f"Analysis Complete")
    print(f"{'='*80}\n")
