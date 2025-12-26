import torch
import math


def TreamMeanDefense(updates_list: list[dict], index, trimmed_rate) -> dict:
    
    if len(index) == 0:
        return None

    trimmed_number = math.ceil(len(index) * trimmed_rate)

    model_state_dict_list = [updates_list[i] for i in index]
    num_clients = len(model_state_dict_list)
    sta_index = trimmed_number
    end_index = num_clients - trimmed_number

    if not model_state_dict_list:
        raise ValueError("The updates list is empty!")
    if sta_index >= end_index:
        raise ValueError("The number of attackers does not satisfy requirements")

    defense_state_dict = {}

    # Iterate over parameter names in the first client's model (as all models have the same structure)
    for k in model_state_dict_list[0].keys():
        # Flatten each parameter tensor from each client's model
        tensors_list = [torch.flatten(model_state_dict[k]) for model_state_dict in model_state_dict_list]
        
        # Stack the tensors into one tensor of shape [num_clients, num_params]
        stacked_tensor = torch.stack(tensors_list)
        
        # Sort the stacked tensor along the client dimension
        sort_tensor, _ = torch.sort(stacked_tensor, dim=0)

        # Exclude attacker updates by selecting the middle values
        avg_tensor = sort_tensor[sta_index:end_index]
        avg_tensor = torch.mean(avg_tensor.float(), dim=0)

        # Reshape the averaged tensor back to the original parameter shape
        defense_state_dict[k] = avg_tensor.view(model_state_dict_list[0][k].shape)

    return defense_state_dict
