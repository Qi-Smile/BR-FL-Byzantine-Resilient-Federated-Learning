import torch, copy,wandb
from utils.utility import vectorize_model_state_dict, vectorize_model, load_model_grad
import torch.nn as nn
import torch.nn.functional as F





import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import wandb

def FLtrustDefense(global_model: torch.nn.Module, updates_list: list[dict], central_dataloader, server_index,
                   index, lr, local_ep, optimizer_name, criterion_name, device, round_num,
                    fltrust_global_lr=0.2) -> dict:
    """
    FLTrust defense mechanism implementation.
    
    Args:
        global_model: The server's global model (torch.nn.Module).
        updates_list: List of model updates from clients (state_dict).
        central_dataloader: Dataloader for trusted central dataset (root dataset).
        server_index: Index of the server.
        index: Indices of clients participating in the current round.
        lr: Learning rate for training the server model.
        local_ep: Number of epochs for server model training.
        optimizer_name: The name of the optimizer (e.g., 'Adam').
        criterion_name: The name of the criterion (e.g., 'CrossEntropy').
        device: Device (CPU/GPU) for computations.
        round_num: The current round of training.
    
    Returns:
        The updated global model's state_dict.
    """
    
    if len(index) == 0:
        return None

    # Deep copy of the model to track changes
    pre_model = copy.deepcopy(global_model)

    # Move the global model to the specified device (CPU/GPU)
    global_model.to(device)

    # Optimizer setup
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=fltrust_global_lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Criterion setup
    if criterion_name == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

    # Vectorize the pre-trained model weights
    pre_model_weight = vectorize_model(pre_model)

    # Train the global model on the central (trusted) dataset
    for ep in range(local_ep):
        global_model.train()
        running_loss = 0.0
        for images, labels in central_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping to prevent numerical explosion
            torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

    # Vectorize the current global model after training on the central dataset
    cur_model_weight = vectorize_model(global_model)

    # Compute the weight difference (global model update)
    global_weight_diff = (cur_model_weight - pre_model_weight)

    # Compute the weight differences for all clients
    group_weight_diff_list = [
        (vectorize_model_state_dict(updates_list[i]) - pre_model_weight) for i in index
    ]

    relu = nn.ReLU()
    client_number = len(group_weight_diff_list)

    # Trust scores initialization
    TS = []
    # Numerical stability: prevent division by very small norms
    eps = 1e-3  # Increased from 1e-6 to handle convergence better

    for idx in range(client_number):
        # Compute trust score using cosine similarity and apply ReLU to ensure non-negativity
        ts = relu(F.cosine_similarity(group_weight_diff_list[idx], global_weight_diff, dim=0)).item()
        TS.append(ts)

        # Normalize the client update based on the magnitude of the server update
        # Add eps to prevent division by very small norms (numerical stability)
        client_norm = torch.norm(group_weight_diff_list[idx])
        client_norm_safe = torch.clamp(client_norm, min=eps)
        group_weight_diff_list[idx] = (torch.norm(global_weight_diff) / client_norm_safe) * group_weight_diff_list[idx]

    # Stack the client weight differences into a tensor for weighted aggregation
    group_weight_diff_list = torch.stack(group_weight_diff_list)

    # Log the trust scores
    # for client_idx in range(client_number):
    #     wandb.log({f"Server:{server_index}_TS_client_idx": {client_idx: TS}, "Round": round_num})

    # Convert trust scores to a tensor and normalize them
    TS = torch.tensor(TS).to(device=group_weight_diff_list.device)
    weight_sum = torch.sum(TS, dim=0)
    if weight_sum > 0:
        weight_list = TS / weight_sum
    else:
        weight_list = TS  # Fallback if no trustable clients (should not happen)

    # Compute the weighted sum of client updates
    return_weight_diff = (weight_list.unsqueeze(1) * group_weight_diff_list).sum(dim=0)

    # Load the aggregated model update back into the global model
    load_model_grad(global_model, return_weight_diff, pre_model)

    return global_model.state_dict()


