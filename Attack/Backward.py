import torch
import copy


def BackwardAttacks(updates, old_updates: torch.nn.Module, scale_factor=100) -> dict:
    # Initialize a dictionary to store the scaled parameters
    attack_update_state_dict = {}

    # Iterate over the old model's state dictionary items (name and params)
    for name, params in old_updates.state_dict().items():
        # Scale the parameter values and store them in the new state dictionary
        attack_update_state_dict[name] = scale_factor * params

    # Return the modified state dictionary
    return attack_update_state_dict