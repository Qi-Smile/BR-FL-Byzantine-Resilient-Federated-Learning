import torch
from Defense.Avg import AvgDefense


def euclidean_distance(model1_state_dict: dict, model2_state_dict: dict) -> float:
    """
    Compute the Euclidean distance (L2 norm) between two model state dictionaries.
    """
    distance = 0.0
    for key in model1_state_dict.keys():
        param1 = model1_state_dict[key].float()
        param2 = model2_state_dict[key].float()
        distance += torch.norm(param1 - param2, p=2, dtype=torch.float).item()
    return distance


def compute_scores(distances, i, n, f):
    """
    Compute the score for the i-th update as the sum of distances
    to its n-f-2 nearest neighbors.

    Args:
        distances: Dictionary of pairwise distances
        i: Index of the update to score
        n: Total number of updates
        f: Number of Byzantine workers

    Returns:
        float: Score (sum of distances to n-f-2 nearest neighbors)
    """
    # Collect all distances from update i to other updates
    s = [distances[j][i] for j in range(i)] + [
        distances[i][j] for j in range(i + 1, n)
    ]

    # Select n-f-2 smallest distances
    _s = sorted(s)[: n - f - 2]

    return sum(_s)


def MultiKrumDefense(updates_list: list[dict], num_attackers, index, device, m=1) -> dict:
    """
    Multi-Krum defense mechanism.

    An extension of Krum that selects m updates with the lowest scores
    and averages them. This is more robust than Krum (m=1) while maintaining
    Byzantine resilience.

    Reference:
    Blanchard et al., "Machine learning with adversaries: Byzantine tolerant
    gradient descent", NeurIPS 2017

    Args:
        updates_list (list[dict]): List of model updates (state_dicts) from clients/servers.
        num_attackers (int): Number of Byzantine attackers (f).
        index (list): Indices of the clients/servers to be aggregated.
        device (torch.device): The device on which to perform computation.
        m (int): Number of top updates to select and average. Default: 1 (original Krum).
                 Typically set to m = n - f - 2.

    Returns:
        dict: Aggregated model parameters as a state dict, or None if index is empty.

    Note:
        Requires n >= 2*f + 3 for Byzantine resilience.
        If m = 1, this reduces to the original Krum algorithm.
        If m = n - f - 2, this maximizes robustness by including all trusted updates.
    """

    if len(index) == 0:
        return None

    model_list_state_dict = [updates_list[i] for i in index]
    num_clients = len(model_list_state_dict)

    # Verify Byzantine resilience condition
    if num_clients < 2 * num_attackers + 3:
        raise ValueError(
            f"Multi-Krum requires n >= 2*f + 3. Got n={num_clients}, f={num_attackers}"
        )

    # Adjust m if it exceeds available updates
    m = min(m, num_clients - num_attackers)

    # Compute pairwise distances between all updates
    distances = {}
    for i in range(num_clients - 1):
        distances[i] = {}
        for j in range(i + 1, num_clients):
            distances[i][j] = euclidean_distance(
                model_list_state_dict[i], model_list_state_dict[j]
            )

    # Compute scores for each update
    scores = [
        (i, compute_scores(distances, i, num_clients, num_attackers))
        for i in range(num_clients)
    ]

    # Sort by score (lower is better) and select top m updates
    sorted_scores = sorted(scores, key=lambda x: x[1])
    top_m_indices = [x[0] for x in sorted_scores[:m]]

    # Average the top m updates
    defense_model = AvgDefense(model_list_state_dict, top_m_indices, device)

    return defense_model
