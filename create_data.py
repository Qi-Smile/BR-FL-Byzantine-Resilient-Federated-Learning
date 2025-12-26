
import torch
import copy
import pickle
import os
import numpy as np
from torch.utils.data import Subset, random_split
from utils.help import load_config
from utils.help import load_dataset
import random


def split_data(dataset, train_ratio, random_seed=None):
    # dataset = dataloader.dataset

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    if random_seed is not None:
        torch.manual_seed(random_seed)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    return train_dataset, test_dataset




def dirichlet_split_noniid(train_labels, num_class, alpha, n_clients, seed):

    # The Dirichlet distribution of argument alpha 
    # divides the set of sample indexes into subsets of n_clients

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_classes = num_class
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, N) Category label distribution matrix X, which records the proportion of each category to each client

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # (K, ...) Records the sample index set corresponding to K categories

    client_idcs = [[] for _ in range(n_clients)]
    # Record the sample index set corresponding to N clients
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split divides the sample of class k into N subsets according to proportion
        #  i indicates the i th client, and idcs indicates its corresponding sample index set idcs
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs


def get_dataset_labels(dataset):
    
    labels = []
    # for sample in dataset:
    #     print(sample[1])
    labels = [sample[1] for sample in dataset]

    return labels


def dirichlet_distribution(dataset, iid, dataset_name, num_class, clients, DIRICHLET_ALPHA, seed=0):

    total_samples = len(dataset)
    n_clients = clients
    np.random.seed(seed)
    torch.manual_seed(seed)

    if iid:
        samples_per_client = total_samples // n_clients
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        subsets = [indices[i*samples_per_client: (i+1)*samples_per_client] for i in range(0, n_clients)]
    else:
        train_labels = np.array(get_dataset_labels(dataset))
        
        subsets = dirichlet_split_noniid(train_labels, num_class=num_class,
                                         alpha=DIRICHLET_ALPHA, n_clients=n_clients,seed=seed,)

    local_dataset = []
    for subset_indices in subsets:
    
        subset = Subset(dataset, subset_indices)
        local_dataset.append(subset)

    return local_dataset



def sample_subset(trainset, percentage, seed=0):


    if seed is not None:
        random.seed(seed)

    train_size = int(len(trainset)*percentage)
    test_size = len(trainset) - train_size

    if not 0 <= percentage <= 1:
        raise ValueError("The percentage must be between 0 and 1!")

    subset,_ = random_split(trainset, [train_size, test_size])
    # print(len(subset))

    return subset

def generate_distribution(config):

    import torch.utils.data

    client_number = config['fed_paras']['client_number']
    Dalpha = config['fed_paras']['dirichlet_rate']
    dataset_name = config['dataset_paras']['name']

    trainset = load_dataset(dataset_name=dataset_name,data_dir=config['dataset_paras']['save_path'],
                            normalise=config['dataset_paras']['normilizer'], trained=True)
    testset = load_dataset(dataset_name=dataset_name,data_dir=config['dataset_paras']['save_path'],
                            normalise=config['dataset_paras']['normilizer'], trained=False)
    

    # num_clients = config['fed_paras']['client_number']

    num_group = config['fed_paras']['num_groups']


    

    expanded_data_list = [copy.deepcopy(trainset) for _ in range(num_group)]
    expanded_data = torch.utils.data.ConcatDataset(expanded_data_list)
    # 0.01 不太行
    central_dataset = sample_subset(trainset=expanded_data, percentage=0.1, seed=config['general_paras']['random_seed'])

    all_dataset_list = dirichlet_distribution(dataset=expanded_data, iid=config['fed_paras']['iid'], 
                                              dataset_name=config['dataset_paras']['name'], 
                                              num_class=config['dataset_paras']['num_classes'], clients=client_number,
                                               DIRICHLET_ALPHA=  Dalpha,
                                               seed=config['general_paras']['random_seed']) 


    
    Dalpha = config['fed_paras']['dirichlet_rate']

    dir_save = config['dataset_paras']['save_path']
    file_path = os.path.join(dir_save,'Distribution', dataset_name)
    file_path = os.path.join(file_path, 'Dalpha=' + str(Dalpha))


    os.makedirs(file_path, exist_ok=True)

    dataset_train_path = os.path.join(file_path, 'dataset_train_list.pkl')
    dataset_test_path = os.path.join(file_path, 'dataset_test.pkl')
    central_dataset_path = os.path.join(file_path, 'central_dataset.pkl')


    with open(dataset_train_path, 'wb') as f:
        pickle.dump(all_dataset_list, f)

    with open(dataset_test_path, 'wb') as f:
        pickle.dump(testset, f)

    with open(central_dataset_path, 'wb') as f:
        pickle.dump(central_dataset, f)




def generate_pfl_distribution(config):

    import torch.utils.data

    Dalpha = config['fed_paras']['dirichlet_rate']
    client_number = config['fed_paras']['client_number']
    split_rate = config['fed_paras']['split_ration']
    dataset_name = config['dataset_paras']['name']

    trainset = load_dataset(dataset_name=dataset_name,data_dir=config['dataset_paras']['save_path'],
                            normalise=config['dataset_paras']['normilizer'], trained=True)
    testset = load_dataset(dataset_name=dataset_name, data_dir=config['dataset_paras']['save_path'],
                            normalise=config['dataset_paras']['normilizer'], trained=False)


    num_group = config['fed_paras']['num_groups']
    
    central_dataset = sample_subset(trainset, 0.01, seed=config['general_paras']['random_seed'])


    # print(len(trainset))
    # print(len(testset))

    combined_data = torch.utils.data.ConcatDataset([trainset, testset])
    # print(len(combined_data))
    # print(combined_data[0])
    expanded_data_list = [copy.deepcopy(combined_data) for _ in range(num_group)]
    expanded_data = torch.utils.data.ConcatDataset(expanded_data_list)

    all_dataset_list = dirichlet_distribution(dataset=expanded_data, iid=config['fed_paras']['iid'], 
                                              dataset_name=config['dataset_paras']['Name'], 
                                              num_class=config['dataset_paras']['num_classes'], clients=client_number//num_group,
                                               DIRICHLET_ALPHA=  Dalpha,
                                               seed=config['general_paras']['random_seed'])                                                                                                                                                                                            

    # all_dataset_list = []
    # for idx in range(num_group):
    #     all_dataset_list.extend(dataset_list)

    dataset_train_list = []
    dataset_test_list = []
    for i in range(client_number):

        train_data, test_data = split_data(all_dataset_list[i], split_rate, config)
        
        # from collections import Counter
        # train_label_counts = Counter(item[1] for item in train_data)  # The number of samples for each category in the training set
        # test_label_counts = Counter(item[1] for item in test_data)  # The number of samples for each category in the testing set

        dataset_train_list.append(train_data)
        dataset_test_list.append(test_data)


    dir_save = config['dataset_paras']['save_path']
    file_path = os.path.join(dir_save,'Distribution', dataset_name)
    file_path = os.path.join(file_path, 'Dalpha=' + str(Dalpha))

    os.makedirs(file_path, exist_ok=True)
    

    dataset_train_path = os.path.join(file_path, 'dataset_train_list.pkl')
    dataset_test_path = os.path.join(file_path, 'dataset_test_list.pkl')
    central_dataset_path = os.path.join(file_path, 'central_dataset.pkl')

    with open(dataset_train_path, 'wb') as f:
        pickle.dump(dataset_train_list, f)

    with open(dataset_test_path, 'wb') as f:
        pickle.dump(dataset_test_list, f)

    with open(central_dataset_path, 'wb') as f:
        pickle.dump(central_dataset, f)
    
    



if __name__ == '__main__':

    config_path = './config/cifar10_resnet18.yaml'
    # config_path = './config/mnist_mlp.yaml'
    # print(os.getcwd())
    config_list = load_config(config_path)
    # generate_pfl_distribution(config=config_list)

    generate_distribution(config=config_list)




