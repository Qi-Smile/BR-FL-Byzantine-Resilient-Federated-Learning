
import torch
import copy

def NoiseAttacks(updates: torch.nn.Module, noise_mean, noise_std, device)->dict:
    # Deep copy the model to avoid modifying the original one

    attack_update_state_dict = copy.deepcopy(updates.state_dict())
    
    # Loop through each parameter in the state_dict of the model
    for k, v in attack_update_state_dict.items():
        # Generate noise with the specified mean and std, matching the size of the parameters
        noise = torch.normal(mean=noise_mean, std=noise_std, size=v.size()).to(device)
        
        # Update the parameters with noise
        attack_update_state_dict[k] = v + noise
    
    return attack_update_state_dict