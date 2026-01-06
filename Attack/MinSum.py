import torch
import copy


def MinSumAttack(updates: torch.nn.Module, benign_updates: list[dict],
                 num_iterations: int, learning_rate: float, device) -> dict:
    """
    Min-Sum Attack - Variant that minimizes sum of distances

    Same as Min-Max but optimizes sum instead of max.
    Often more aggressive but potentially more detectable.

    Args:
        updates: Malicious client's trained model
        benign_updates: List of state_dicts from benign clients
        num_iterations: Optimization iterations
        learning_rate: Learning rate for gamma
        device: PyTorch device

    Returns:
        Optimized malicious update state_dict
    """

    if len(benign_updates) == 0:
        print("Warning: No benign clients for MinSum attack")
        return copy.deepcopy(updates.state_dict())

    # Compute benign mean
    benign_mean = {}
    for k in benign_updates[0].keys():
        benign_params = torch.stack([
            benign_update[k].float().to(device)
            for benign_update in benign_updates
        ])
        benign_mean[k] = torch.mean(benign_params, dim=0)

    # Attack direction: -μ / ||μ||
    mu_norm = torch.tensor(0.0, device=device)
    for k in benign_mean.keys():
        mu_norm += torch.sum(benign_mean[k] ** 2)
    mu_norm = torch.sqrt(mu_norm) + 1e-8

    attack_direction = {}
    for k in benign_mean.keys():
        attack_direction[k] = -benign_mean[k] / mu_norm

    # Initialize γ
    gamma = torch.tensor(1.0, requires_grad=True, device=device)
    optimizer = torch.optim.SGD([gamma], lr=learning_rate)

    # Optimize γ to minimize sum of distances
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        current_malicious = {}
        for k in benign_mean.keys():
            current_malicious[k] = benign_mean[k] + gamma * attack_direction[k]

        # Compute SUM of distances (not max)
        sum_distance = torch.tensor(0.0, device=device)
        for benign_update in benign_updates:
            distance = torch.tensor(0.0, device=device)
            for k in current_malicious.keys():
                benign_param = benign_update[k].float().to(device)
                diff = current_malicious[k] - benign_param
                distance += torch.sum(diff ** 2)

            sum_distance += distance  # Sum instead of max

        loss = sum_distance
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            gamma.clamp_(-10.0, 10.0)

    print(f"MinSum: Optimized gamma={gamma.item():.3f} after {num_iterations} iterations")

    # Generate final update
    final_update = {}
    with torch.no_grad():
        for k in benign_mean.keys():
            final_update[k] = benign_mean[k] + gamma.item() * attack_direction[k]

    return final_update
