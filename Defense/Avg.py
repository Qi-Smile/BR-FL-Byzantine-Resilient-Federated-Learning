import torch
import copy

def AvgDefense(updates_list: list[dict], index, device, weight_list=None) -> dict:
    """
    Computes the weighted average of the updates provided by multiple clients/servers.
    
    Args:
        updates_list (list): List of model updates (state_dicts) from multiple servers/clients.
        index (list): The indices of the clients/servers to be averaged.
        device (torch.device): The device on which to perform the averaging (CPU or GPU).
        weight_list (torch.Tensor): Optional list of weights for each update.
        
    Returns:
        defense_state_dict (dict): Averaged model parameters as a state dict.
    """
    
    if len(index) == 0:
        return None  # Return None if no clients/servers are provided
    
    # Extract the state dictionaries of the selected clients/servers
    model_state_dict_list = [updates_list[i] for i in index]

    length = len(model_state_dict_list)
    
    # Default to equal weights if no weight list is provided
    if weight_list is None:
        weight_list = torch.ones(length, dtype=torch.float).to(device) / length
    else:
        # Normalize the provided weights to ensure they sum to 1
        weight_list = torch.tensor(weight_list, dtype=torch.float).to(device)
        weight_list /= weight_list.sum()

    # Initialize the defense state dictionary
    defense_state_dict = {}

    # Iterate through the selected model state dictionaries and compute the weighted sum
    for id, model_state_dict in enumerate(model_state_dict_list):
        for name, param in model_state_dict.items():
            if name not in defense_state_dict:
                # Initialize the tensor for each parameter
                defense_state_dict[name] = torch.zeros_like(param.data, dtype=torch.float).to(device)
            # Add the weighted parameter update
            defense_state_dict[name] += weight_list[id] * param.data.to(device)
            
    return defense_state_dict
