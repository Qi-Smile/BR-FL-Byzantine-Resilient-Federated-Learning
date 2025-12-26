import torch
from Defense.Avg import AvgDefense


def euclidean_distance(model1_state_dict: dict, model2_state_dict: dict) -> float:
    distance = 0.0
    for key in model1_state_dict.keys():
        param1 = model1_state_dict[key].float()
        param2 = model2_state_dict[key].float()
        distance += torch.norm(param1 - param2, p=2, dtype=torch.float).item()  # Euclidean distance (L2 norm)
    return distance

def compute_scores(distances, i, n, f):


    s = [distances[j][i] for j in range(i)] + [
            distances[i][j] for j in range(i + 1, n)
        ]

    # select n-f-2 smallest distances
    _s = sorted(s)[: n - f - 2]
        

    return sum(_s)


def KrumDefense(updates_list: list[torch.nn.Module], num_attackers, index,device,k=1)->torch.nn.Module:

    # for each client select n-f neighbor for compute scores and select k for avg
    # n < 2*f + 2
    # server_list 是模型列表

    if len(index) == 0:
        return None

    model_list_state_dict = [updates_list[i] for i in index]  
    num_clients = len(model_list_state_dict)
    distances = {}
    for i in range(num_clients-1):
        distances[i] = {}
        for j in range(i + 1, num_clients):
            distances[i][j] = euclidean_distance( model_list_state_dict[i],  model_list_state_dict[j])
    
    scores = [(i, compute_scores(distances, i, num_clients, num_attackers)) for i in range(num_clients)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    top_k_indices = list(map(lambda x: x[0], sorted_scores))[:k]

    defense_model = AvgDefense(model_list_state_dict, top_k_indices, device)
    

    return defense_model
