#!/usr/bin/env python3
"""
Quick functional test for new defense methods.
Tests CoMed and MultiKrum with dummy data.
"""

import torch
from Defense.CoMed import CoMedDefense
from Defense.MultiKrum import MultiKrumDefense
from Defense.GeoMed import GeoMedDefense

def create_dummy_updates(num_clients, param_shape=(10, 10), add_outlier=False):
    """Create dummy model updates for testing."""
    updates = []
    for i in range(num_clients):
        state_dict = {
            'layer1.weight': torch.randn(param_shape),
            'layer1.bias': torch.randn(param_shape[0]),
        }
        updates.append(state_dict)

    # Add outlier if requested
    if add_outlier:
        outlier = {
            'layer1.weight': torch.randn(param_shape) * 100,  # Large outlier
            'layer1.bias': torch.randn(param_shape[0]) * 100,
        }
        updates.append(outlier)

    return updates

def test_comed():
    """Test CoMed defense."""
    print("Testing CoMed...")

    # Create dummy data
    updates = create_dummy_updates(num_clients=10, add_outlier=True)
    device = torch.device('cpu')

    # Test defense
    result = CoMedDefense(updates_list=updates, index=list(range(len(updates))), device=device)

    assert result is not None, "CoMed returned None"
    assert 'layer1.weight' in result, "Missing layer1.weight"
    assert 'layer1.bias' in result, "Missing layer1.bias"
    assert result['layer1.weight'].shape == (10, 10), "Incorrect weight shape"
    assert result['layer1.bias'].shape == (10,), "Incorrect bias shape"

    print("✓ CoMed test passed")

def test_multikrum():
    """Test MultiKrum defense."""
    print("Testing MultiKrum...")

    # Create dummy data
    updates = create_dummy_updates(num_clients=10, add_outlier=True)
    device = torch.device('cpu')

    # Test with m=5 (select top 5)
    result = MultiKrumDefense(
        updates_list=updates,
        num_attackers=2,
        index=list(range(len(updates))),
        device=device,
        m=5
    )

    assert result is not None, "MultiKrum returned None"
    assert 'layer1.weight' in result, "Missing layer1.weight"
    assert 'layer1.bias' in result, "Missing layer1.bias"
    assert result['layer1.weight'].shape == (10, 10), "Incorrect weight shape"
    assert result['layer1.bias'].shape == (10,), "Incorrect bias shape"

    print("✓ MultiKrum test passed")

def test_geomed():
    """Test GeoMed defense."""
    print("Testing GeoMed...")

    # Create dummy data
    updates = create_dummy_updates(num_clients=10, add_outlier=True)
    device = torch.device('cpu')

    # Test defense
    result = GeoMedDefense(updates_list=updates, index=list(range(len(updates))), device=device)

    assert result is not None, "GeoMed returned None"
    assert 'layer1.weight' in result, "Missing layer1.weight"
    assert 'layer1.bias' in result, "Missing layer1.bias"
    assert result['layer1.weight'].shape == (10, 10), "Incorrect weight shape"
    assert result['layer1.bias'].shape == (10,), "Incorrect bias shape"

    print("✓ GeoMed test passed")

def test_robustness():
    """Test that defenses are robust to outliers."""
    print("\nTesting robustness to outliers...")

    device = torch.device('cpu')

    # Normal updates (mean around 0)
    normal_updates = create_dummy_updates(num_clients=8, add_outlier=False)

    # Malicious updates (mean around 100)
    malicious_updates = create_dummy_updates(num_clients=2, param_shape=(10, 10))
    for update in malicious_updates:
        for key in update:
            update[key] = update[key] * 100 + 100

    # Combine
    all_updates = normal_updates + malicious_updates

    # Test CoMed
    result_comed = CoMedDefense(all_updates, list(range(10)), device)
    median_value = result_comed['layer1.weight'].abs().mean().item()

    # Test MultiKrum
    result_mkrum = MultiKrumDefense(all_updates, num_attackers=2, index=list(range(10)), device=device, m=5)
    mkrum_value = result_mkrum['layer1.weight'].abs().mean().item()

    # Test GeoMed
    result_geomed = GeoMedDefense(all_updates, list(range(10)), device)
    geomed_value = result_geomed['layer1.weight'].abs().mean().item()

    print(f"  CoMed aggregated value: {median_value:.2f} (expected < 10)")
    print(f"  MultiKrum aggregated value: {mkrum_value:.2f} (expected < 10)")
    print(f"  GeoMed aggregated value: {geomed_value:.2f} (expected < 10)")

    # All should be close to 0, not 100 (outliers rejected)
    assert median_value < 10, f"CoMed not robust to outliers: {median_value}"
    assert mkrum_value < 10, f"MultiKrum not robust to outliers: {mkrum_value}"
    assert geomed_value < 10, f"GeoMed not robust to outliers: {geomed_value}"

    print("✓ Robustness test passed")

if __name__ == '__main__':
    print("="*70)
    print("Functional Tests for Defense Methods")
    print("="*70)

    try:
        test_comed()
        test_multikrum()
        test_geomed()
        test_robustness()

        print("\n" + "="*70)
        print("All tests passed! ✓")
        print("="*70)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
