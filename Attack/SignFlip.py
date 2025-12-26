import torch
import copy



def SignFlipAttacks(updates:torch.nn.Module, old_updates:torch.nn.Module, scale_factor=1):
  
  attack_update_state_dict = copy.deepcopy(updates.state_dict())
  for k in attack_update_state_dict.keys():

      attack_update_state_dict[k] = (1+scale_factor)*old_updates.state_dict()[k] - scale_factor*updates.state_dict()[k]
  
  return attack_update_state_dict
