import torch
import copy


def MinMaxAttack(updates: torch.nn.Module, benign_updates: list[dict],
                 num_iterations: int, learning_rate: float, device) -> dict:
    """
    Min-Max Attack - Shejwalkar & Houmansadr, NDSS 2021

    Optimizes scaling factor γ to minimize maximum L2 distance to benign updates.
    Attack direction: -μ / ||μ|| (negative normalized mean)

    Args:
        updates: Malicious client's trained model (not used in this version)
        benign_updates: List of state_dicts from benign clients in same server
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for gamma optimization
        device: PyTorch device

    Returns:
        Optimized malicious update state_dict
    """

    if len(benign_updates) == 0:
        print("Warning: No benign clients for MinMax attack")
        return copy.deepcopy(updates.state_dict())

    # Step 1: Compute benign mean μ
    benign_mean = {}
    for k in benign_updates[0].keys():
        benign_params = torch.stack([
            benign_update[k].float().to(device)
            for benign_update in benign_updates
        ])
        benign_mean[k] = torch.mean(benign_params, dim=0)

    # Step 2: Compute attack direction: -μ / ||μ||
    # First compute ||μ|| across all parameters
    mu_norm = torch.tensor(0.0, device=device)
    for k in benign_mean.keys():
        mu_norm += torch.sum(benign_mean[k] ** 2)
    mu_norm = torch.sqrt(mu_norm) + 1e-8

    # Normalized attack direction
    attack_direction = {}
    for k in benign_mean.keys():
        attack_direction[k] = -benign_mean[k] / mu_norm

    # Step 3: Initialize γ (scaling factor)
    gamma = torch.tensor(1.0, requires_grad=True, device=device)
    optimizer = torch.optim.SGD([gamma], lr=learning_rate)

    # Step 4: Optimize γ to minimize max distance
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Current malicious update: μ + γ × direction
        current_malicious = {}
        for k in benign_mean.keys():
            current_malicious[k] = benign_mean[k] + gamma * attack_direction[k]

        # Compute distances to all benign updates
        max_distance = torch.tensor(0.0, device=device)
        for benign_update in benign_updates:
            distance = torch.tensor(0.0, device=device)
            for k in current_malicious.keys():
                benign_param = benign_update[k].float().to(device)
                diff = current_malicious[k] - benign_param
                distance += torch.sum(diff ** 2)

            max_distance = torch.max(max_distance, distance)

        # Loss: minimize maximum distance
        loss = max_distance
        loss.backward()
        optimizer.step()

        # Clamp γ to prevent extreme values
        with torch.no_grad():
            gamma.clamp_(-10.0, 10.0)

    print(f"MinMax: Optimized gamma={gamma.item():.3f} after {num_iterations} iterations")

    # Step 5: Generate final malicious update
    final_update = {}
    with torch.no_grad():
        for k in benign_mean.keys():
            final_update[k] = benign_mean[k] + gamma.item() * attack_direction[k]

    return final_update
