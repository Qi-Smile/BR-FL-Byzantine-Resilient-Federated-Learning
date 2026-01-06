import torch
import copy
import numpy as np
from scipy.stats import norm


def compute_z_score(n_benign, n_malicious, trim_ratio):
    """
    Compute z-score for ALIE attack based on trimmed mean boundary.

    Formula from paper (Baruch et al., NeurIPS 2019):
    - n_total = n_benign + n_malicious
    - s = floor(n_total × trim_ratio)  # Number of updates trimmed
    - p = (n_total - s - n_malicious) / n_benign
    - z = Φ^(-1)(p)  # Inverse CDF of standard normal

    Args:
        n_benign: Number of benign clients in server
        n_malicious: Number of malicious clients in server
        trim_ratio: Trimmed mean clipping rate

    Returns:
        z-score value
    """
    if n_benign == 0:
        return 0.0  # Fallback

    n_total = n_benign + n_malicious
    s = int(np.floor(n_total * trim_ratio))

    # Position in sorted updates where malicious should be
    p = (n_total - s - n_malicious) / n_benign

    # Clip to valid probability range
    p = np.clip(p, 0.01, 0.99)

    # Compute z-score using inverse CDF
    z = norm.ppf(p)

    return z


def ALIEAttack(updates: torch.nn.Module, benign_updates: list[dict],
               n_malicious: int, trim_ratio: float, alie_direction: int,
               device) -> dict:
    """
    ALIE (A Little Is Enough) Attack - Baruch et al., NeurIPS 2019

    Generates malicious updates at trimmed mean boundary using statistics
    of benign clients' updates.

    Args:
        updates: Malicious client's trained model
        benign_updates: List of state_dicts from benign clients in same server
        n_malicious: Number of malicious clients in this server
        trim_ratio: Trimmed mean clipping rate (0 to 0.5)
        alie_direction: Attack direction (+1 or -1)
        device: PyTorch device

    Returns:
        Malicious update state_dict
    """

    if len(benign_updates) == 0:
        # Fallback: no benign clients in server, return original
        print("Warning: No benign clients in server for ALIE attack, using original update")
        return copy.deepcopy(updates.state_dict())

    n_benign = len(benign_updates)

    # Compute z-score dynamically
    z = compute_z_score(n_benign, n_malicious, trim_ratio)

    print(f"ALIE: n_benign={n_benign}, n_mal={n_malicious}, trim={trim_ratio:.2f}, z={z:.3f}")

    attack_update_state_dict = {}

    # Iterate over each parameter
    for k in benign_updates[0].keys():
        # Stack benign updates: [n_benign, param_shape]
        benign_params = torch.stack([
            benign_update[k].float().to(device)
            for benign_update in benign_updates
        ])

        # Compute element-wise statistics
        mu = torch.mean(benign_params, dim=0)
        sigma = torch.std(benign_params, dim=0) + 1e-8  # Add epsilon for stability

        # ALIE formula: μ + direction × z × σ
        attack_update_state_dict[k] = mu + alie_direction * z * sigma

    return attack_update_state_dict
