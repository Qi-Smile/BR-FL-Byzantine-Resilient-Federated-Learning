import torch


def GeoMedDefense(updates_list: list[dict], index, device, max_iter=100, eps=1e-6) -> dict:
    """
    Geometric Median defense using Weiszfeld's algorithm.

    Computes the geometric median of model updates to robustly aggregate
    updates from multiple clients/servers in the presence of Byzantine attacks.
    The geometric median minimizes the sum of L2 distances to all points,
    making it more robust to outliers than the arithmetic mean.

    Args:
        updates_list (list[dict]): List of model updates (state_dicts) from clients/servers.
        index (list): Indices of the clients/servers to be aggregated.
        device (torch.device): The device on which to perform computation (CPU or GPU).
        max_iter (int): Maximum iterations for Weiszfeld algorithm. Default: 100.
        eps (float): Convergence threshold and minimum distance to prevent division by zero. Default: 1e-6.

    Returns:
        dict: Aggregated model parameters as a state dict, or None if index is empty.
    """

    # Return None if no clients/servers are provided
    if len(index) == 0:
        return None

    # Extract the state dictionaries of the selected clients/servers
    model_state_dict_list = [updates_list[i] for i in index]
    num_clients = len(model_state_dict_list)

    # Initialize the defense state dictionary
    defense_state_dict = {}

    # Iterate over each parameter in the model
    for param_name in model_state_dict_list[0].keys():
        # Flatten each client's parameter tensor and convert to float
        tensors_list = [torch.flatten(model_state_dict[param_name]).float().to(device)
                        for model_state_dict in model_state_dict_list]

        # Stack tensors into a matrix: [num_clients, num_params_in_layer]
        stacked_tensor = torch.stack(tensors_list)

        # Initialize median as the mean of all updates (good starting point)
        median = torch.mean(stacked_tensor, dim=0)  # Shape: [num_params_in_layer]

        # Weiszfeld algorithm for computing geometric median
        for iteration in range(max_iter):
            # Compute L2 distances from each update to current median
            # distances shape: [num_clients]
            distances = torch.norm(stacked_tensor - median.unsqueeze(0), p=2, dim=1)

            # Prevent division by zero by clamping to minimum eps
            distances = torch.clamp(distances, min=eps)

            # Compute weights as inverse distances
            weights = 1.0 / distances

            # Normalize weights to sum to 1
            weights = weights / torch.sum(weights)

            # Compute weighted sum to get new median
            # Broadcasting: weights [num_clients] * stacked_tensor [num_clients, num_params]
            new_median = torch.sum(stacked_tensor * weights.unsqueeze(1), dim=0)

            # Check convergence
            if torch.norm(new_median - median) < eps:
                break

            median = new_median

        # Reshape median back to original parameter shape
        defense_state_dict[param_name] = median.view(model_state_dict_list[0][param_name].shape)

    return defense_state_dict
