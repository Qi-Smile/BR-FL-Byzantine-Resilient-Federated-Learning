import torch
import copy

def RandomAttacks(updates: torch.nn.Module, random_lower, random_upper, device) ->dict:
    # Deep copy the model updates to avoid modifying the original model


    attack_update_state_dict = copy.deepcopy(updates.state_dict())


    for k, v in attack_update_state_dict.items():
        # Ensure that the tensor is in float format before applying random values
        if v.dtype == torch.long:
            tensor = (random_lower + (random_upper - random_lower) * torch.rand_like(v, dtype=torch.float)).long().to(device)
            attack_update_state_dict[k] = tensor
        else:
            # print(attack_updates.state_dict()[k])
            tensor = (random_lower + (random_upper - random_lower) * torch.rand_like(v, dtype=torch.float)).long().to(device)
            # print(tensor)
            attack_update_state_dict[k] = tensor

    
    # attack_updates = copy.deepcopy(updates)
    # attack_updates.load_state_dict(attack_update_state_dict)
    # for key, val in attack_updates.state_dict().items():
    #     print(key, val)
    #     break
    
    return attack_update_state_dict