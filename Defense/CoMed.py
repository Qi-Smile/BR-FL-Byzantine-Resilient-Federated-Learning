import torch


def CoMedDefense(updates_list: list[dict], index, device) -> dict:
    """
    Coordinate-wise Median (CoMed) defense mechanism.

    Computes the coordinate-wise median across all model updates,
    which is robust to Byzantine attacks. For each parameter dimension,
    takes the median value across all clients.

    Reference:
    Yin et al., "Byzantine-robust distributed learning: Towards optimal
    statistical rates", ICML 2018

    Args:
        updates_list (list[dict]): List of model updates (state_dicts) from clients/servers.
        index (list): Indices of the clients/servers to be aggregated.
        device (torch.device): The device on which to perform computation (CPU or GPU).

    Returns:
        dict: Aggregated model parameters as a state dict, or None if index is empty.
    """

    # Return None if no clients/servers are provided
    if len(index) == 0:
        return None

    # Extract the state dictionaries of the selected clients/servers
    model_state_dict_list = [updates_list[i] for i in index]

    if not model_state_dict_list:
        raise ValueError("The updates list is empty!")

    defense_state_dict = {}

    # Iterate over each parameter in the model
    for param_name in model_state_dict_list[0].keys():
        # Flatten each client's parameter tensor
        tensors_list = [torch.flatten(model_state_dict[param_name]).float().to(device)
                        for model_state_dict in model_state_dict_list]

        # Stack tensors into a matrix: [num_clients, num_params_in_layer]
        stacked_tensor = torch.stack(tensors_list)

        # Compute coordinate-wise median along the client dimension (dim=0)
        # median returns (values, indices), we only need values
        median_tensor = torch.median(stacked_tensor, dim=0).values

        # Reshape median back to original parameter shape
        defense_state_dict[param_name] = median_tensor.view(
            model_state_dict_list[0][param_name].shape
        )

    return defense_state_dict
