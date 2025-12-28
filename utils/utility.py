
from torchvision.models import mobilenet_v2, resnet18

import pickle,random
from Net.resnet import ResNet18, ResNet34
from Net.MLP import MLP
import os
import torch
import matplotlib.pyplot as plt 
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def vectorize_model(model: torch.nn.Module):
    # Return a one-dimensional vector by flattening and concatenating all parameters
    return torch.cat([p.view(-1) for p in model.state_dict().values()])

def vectorize_model_state_dict(model_state_dict: dict):
    # Return a one-dimensional vector by flattening and concatenating all parameters
    return torch.cat([p.view(-1) for p in model_state_dict.values()])




def load_model_grad(model:torch.nn.Module, weight_diff, global_weight:torch.nn.Module):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(model.parameters()):
        p.data =  weight_diff[index_bias:index_bias+p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()



def denormalize(tensor, mean, std):
    tensor = tensor.clone()  # 复制张量
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def imshow(img):
    img = denormalize(img, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])  # 去归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')  # 关闭坐标轴
                # 保存为 PNG 格式
    plt.savefig('light_image.pdf', bbox_inches='tight', pad_inches=0)



def get_attacker_index(server_attacker_rate: float, client_attacker_rate: float, 
                       server_number: int, client_number: int, 
                       method: str = 'order'):
    """
    Generates attacker indices for servers and clients based on the given method.

    Args:
        server_attacker_rate (float): Proportion of servers to be attackers.
        client_attacker_rate (float): Proportion of clients to be attackers.
        server_number (int): Total number of servers.
        client_number (int): Total number of clients.
        method (str): Method to generate attacker indices ('order' or 'random').

    Returns:
        server_attacker_index (torch.Tensor): Indices of servers that are attackers.
        client_attacker_index (torch.Tensor): Indices of clients that are attackers.

    Raises:
        ValueError: If an unsupported method is provided or if the rates are invalid.
    """
    
    # Validate the input rates and numbers
    if not (0 <= server_attacker_rate <= 1):
        raise ValueError('server_attacker_rate must be between 0 and 1.')
    if not (0 <= client_attacker_rate <= 1):
        raise ValueError('client_attacker_rate must be between 0 and 1.')
    if server_number <= 0 or client_number <= 0:
        raise ValueError('server_number and client_number must be positive integers.')

    if method == 'order':
        # Generate indices in order
        server_attacker_index: torch.Tensor = torch.arange(0, int(max((server_attacker_rate * server_number), 0)))
        client_attacker_index: torch.Tensor = torch.arange(0, int(max((client_attacker_rate * client_number), 0)))

    elif method == 'random':
        # Generate random indices
        server_attacker_index: torch.Tensor = torch.randperm(server_number)[:int(server_attacker_rate * server_number)]
        client_attacker_index: torch.Tensor = torch.randperm(client_number)[:int(client_attacker_rate * client_number)]

    else:
        raise ValueError('Unsupported generating attacker method! Choose "order" or "random".')
    
    return server_attacker_index, client_attacker_index


def get_round_index(server_number, client_number,):

    # Generate a list of client indices
    clients = list(range(client_number))

    # Shuffle the clients to introduce randomness
    random.shuffle(clients)

    # Now evenly distribute the clients to the servers
    server_to_clients = defaultdict(list)
    client_to_server = {}

    # Distribute clients to servers in a round-robin or balanced way
    for idx, client in enumerate(clients):
        server_idx = idx % server_number  # Round-robin assignment after shuffling
        server_to_clients[server_idx].append(client)
        client_to_server[client] = server_idx  # Track which server the client is assigned to



    update_counts = torch.zeros(server_number, dtype=torch.int)

    return client_to_server, server_to_clients, update_counts


def LabelFlip_test_model(model:torch.nn.Module, test_loader, device, num_classes=10,
                criterion_name='CrossEntropy',):
   

    model.eval()
    model.to(device)

    test_loss = 0
    correct = 0
    total = 0
    if criterion_name == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader), desc=f"Testing"):
            data, target = data.to(device), target.to(device)
            new_target = (num_classes - target - 1)
            output = model(data)
            test_loss += criterion(output, new_target).item()  # sum up batch loss
            total += new_target.size(0)
            # Get the index of the max log-probability
            pred = output.argmax(dim=1)
            correct += pred.eq(new_target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100*correct / total
    
    return accuracy, test_loss



def LabelFlip_train_model(model:torch.nn.Module, train_loader, epoch, device, label_flip_rate=0.2,
                        num_classes=10, criterion_name='CrossEntropy', 
                       lr=0.001, optimizer_name='Adam'):


    import torch
import random
from tqdm import tqdm

def LabelFlip_train_model(model: torch.nn.Module, train_loader, epoch, device, label_flip_rate=0.2,
                          num_classes=10, criterion_name='CrossEntropy',
                          lr=0.001, optimizer_name='Adam'):
    
    model.train()
    model.to(device)

    # Initialize optimizer
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize criterion
    if criterion_name == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()

    running_loss = 0.0

    for data, target in tqdm(train_loader, total=len(train_loader), desc=f"Training epoch: {epoch}"):
        data, target = data.to(device), target.to(device)
        
        # Apply label flipping based on the label_flip_rate
        flip_mask = torch.rand(len(target)) < label_flip_rate  # Create a mask for which labels to flip
        new_target = target.clone()  # Clone original targets to avoid modification
        new_target[flip_mask] = (num_classes - target[flip_mask] - 1)  # Flip selected labels

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, new_target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate the average loss for the epoch
    average_loss = running_loss / len(train_loader)

    return model, average_loss












def test_model(model:torch.nn.Module, test_loader, device,
                criterion_name='CrossEntropy',):
   

    model.eval()
    model.to(device)

    test_loss = 0
    correct = 0
    total = 0
    if criterion_name == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader), desc=f"Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            total += target.size(0)
            # Get the index of the max log-probability
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()


    test_loss /= len(test_loader)
    accuracy = 100*correct / total
    
    return accuracy, test_loss


def normal_train_model(model:torch.nn.Module, train_loader,epoch, device,criterion_name='CrossEntropy', 
                       lr=0.001, optimizer_name='Adam'):


    model.train()
    model.to(device)
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if criterion_name == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

    running_loss = 0.0  

    for data, target in tqdm(train_loader, total=len(train_loader), desc=f"Training epoch: {epoch}"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    # Calculate the average loss for the epoch
    average_loss = running_loss / len(train_loader)


    return model, average_loss







def load_model(model_name, num_classes):
    """
    Load a model architecture based on the model name.

    Args:
        model_name (str): Name of the model to load ('Resnet18', 'Resnet34', 'MobileNetV2', 'MLP').
        num_classes (int): Number of output classes for the model.

    Returns:
        model: A PyTorch model instance corresponding to the specified architecture.

    Raises:
        ValueError: If the model name is not supported.
    """
    
    # if model_name == 'Resnet18':
    #     model = ResNet18(num_classes=num_classes)

    # elif model_name == 'Resnet34':
    #     model = ResNet34(num_classes=num_classes)
    if model_name == 'MobileNetV2':
        model = mobilenet_v2(num_classes=num_classes)
        model.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        return model

    elif model_name == 'ResNet18':
        # Use custom ResNet18 implementation (works on CUDA)
        # torchvision resnet18 causes segfault on this CUDA environment
        model = ResNet18(num_classes=num_classes)

    elif model_name == 'Resnet34':
        model = ResNet34(num_classes=num_classes)

    elif model_name == 'MLP':
        model = MLP(input_size=28*28, hidden_sizes=[128, 64], output_size=num_classes)

    else:
        raise ValueError(f"Unsupported model name: {model_name}. Choose from 'Resnet18', 'Resnet34', 'MobileNetV2', 'MLP'.")

    return model



def get_dataset(dataset_name,Dalpha):



    # read data from the saved file
    file_path = os.path.join('data','Distribution', dataset_name)
    tmp = 'Dalpha=' + str(Dalpha)
    file_path = os.path.join(file_path, tmp)

    os.makedirs(file_path, exist_ok=True)

    dataset_train_path = os.path.join(file_path, 'dataset_train_list.pkl')
    dataset_test_path = os.path.join(file_path, 'dataset_test.pkl')
    central_dataset_path = os.path.join(file_path, 'central_dataset.pkl')


    with open(dataset_train_path, 'rb') as f:
        dataset_train_list = pickle.load(f)

    with open(dataset_test_path, 'rb') as f:
        dataset_test = pickle.load(f)
    
    with open(central_dataset_path, 'rb') as f:
        central_dataset = pickle.load(f)


    return dataset_train_list, dataset_test, central_dataset






# def get_test_weight(config, adversial=False,malicious=False):

      # for different number test samples
#     dataset_name = config['dataset_paras']['name']
#     Dalpha = config['fed_paras']['dirichlet_rate']
#     # print(Dalpha)
#     client_number = config['fed_paras']['client_number']
#     client_attacker_rate = config['fed_paras']['client_attack_ratio']

#     client_attacker_num = int(max((client_attacker_rate * client_number), 0))


#     file_path = os.path.join(config['dataset_paras']['save_path'],'Distribution',dataset_name)
    
#     tmp = 'dalpha=' + str(Dalpha)
#     file_path = os.path.join(file_path, tmp)

#     os.makedirs(file_path, exist_ok=True)

#     dataset_test_path = os.path.join(file_path, 'dataset_test_list.pkl')

#     with open(dataset_test_path, 'rb') as f:
#         dataset_test_list = pickle.load(f)

#     test_sample_list = []

#     if malicious == False:
#         for idx in range(len(dataset_test_list)):
#             # 获取该数据集对应的人的标识符
#             if idx < client_attacker_num:
#                 pass
#             else:
#                 person_id = idx
#                 person_dataset = dataset_test_list[person_id]
#                 # 初始化一个字典，用于存储该人的每个类别的样本数量
#                 val = len(person_dataset)
#                 test_sample_list.append(val)
#     else:
#         for idx in range(len(dataset_test_list)):
#         # 获取该数据集对应的人的标识符
#             if idx >= client_attacker_num:
#                 pass
#             else:
#                 person_id = idx
#                 person_dataset = dataset_test_list[person_id]
#                 # 初始化一个字典，用于存储该人的每个类别的样本数量
#                 val = len(person_dataset)
#                 test_sample_list.append(val)


#     # print('the length of test_dataset is: ', len(test_sample_list))
#     # print('the test sample is:', test_sample_list)
#     total_num = sum(test_sample_list)
#     for idx in range(len(test_sample_list)):
#         test_sample_list[idx] = test_sample_list[idx] / total_num
#     # print('the total number of sample is:', total_num)
#     # print('the ratio of each client is:', test_sample_list)

#     return test_sample_list

