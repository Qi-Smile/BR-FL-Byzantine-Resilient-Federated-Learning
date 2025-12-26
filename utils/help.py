
import yaml
import torchvision
import torch
import datetime


def load_config(file_name):
    with open(file_name, 'r') as file:
        config = yaml.safe_load(file)
    return config



def get_file_name(config):

    dataset_name = config['dataset_paras']['name']
    model_name = config['model_paras']['name']
    server_attacks = config['general_paras']['server_attacks']
    client_attacks = config['general_paras']['client_attacks']
    defense_method = config['general_paras']['defense']
    dirichlet_rate = config['fed_paras']['dirichlet_rate'] 
    server_attacker_rate = config['fed_paras']['server_attack_ratio']
    client_attacker_rate = config['fed_paras']['client_attack_ratio']
    # now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    
    runs_name = (dataset_name + '_' + model_name + '_' + server_attacks + '_'+ client_attacks + '_' 
                + defense_method + str(dirichlet_rate) + '_'+str(server_attacker_rate)+ '_' +str(client_attacker_rate))
    

    save_file = dataset_name + '_' + model_name +'_' + defense_method + '_' + server_attacks + '_'+client_attacks +  '_'+ str(dirichlet_rate)+'_'+ str(server_attacker_rate) + '_' + str(client_attacker_rate)

    return runs_name, save_file


def get_device(cuda_number, GPU:bool = True):

    device = torch.device(
            'cuda:{}'.format(cuda_number) if torch.cuda.is_available() and GPU == True and cuda_number != -1 else 'cpu')
     
    return device

def load_dataset(dataset_name, data_dir, normalise=True, trained=True):


    if dataset_name == 'mnist':
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
           
        ])
        if normalise == True:  
            # only one channel
            # data_transform.transforms.append(torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
            data_transform.transforms.append(torchvision.transforms.Normalize([0.1307], [0.3081]))

        dataset = torchvision.datasets.MNIST(root='data', train=trained, transform=data_transform, download=True)


    elif dataset_name == 'emnist':
        data_transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.ToTensor(),

        ])
        if normalise == True:  

            data_transform.transforms.append(torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]))
            dataset = torchvision.datasets.EMNIST(root=data_dir, split='balanced', train=trained, download=False, transform=data_transform)
           
    elif dataset_name == 'cifar10':
        data_transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            
        ])


        if normalise == True:
            if trained == True:
                
                data_transform.transforms.append(torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
            else:
                data_transform.transforms.append(torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))

        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=trained, transform=data_transform, download=True)
    

    elif dataset_name == 'cifar100':
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        if normalise == True:  
            data_transform.transforms.append(torchvision.transforms.Normalize(mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                                              std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]))

        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=trained, transform=data_transform, download=True)


    elif dataset_name == 'imagenet1k':

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),

           
        ])
        dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=data_transform)


    elif dataset_name == 'gtsrb':
        # the pictures from gtsrb are not the same
        mean_nums = [0.485, 0.456, 0.406]
        std_nums = [0.229, 0.224, 0.225]
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
        ])
        if trained == True:
            split = 'train'
        else:
            split = 'test'
        if normalise == True:  
            data_transform.transforms.append(torchvision.transforms.Normalize(mean_nums, std_nums))
        dataset = torchvision.datasets.GTSRB(root=data_dir, split=split, transform=data_transform, download=True)


    elif dataset_name == 'tiny_imagenet':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),

        ])
        if normalise == True:  
            
            transform.transforms.append(torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        # from CustomDataset import TinyImageNet
        # data_dir = './data/tiny-imagenet-200/'
        # train_dataset = TinyImageNet(data_dir, transform=transform, train=trained)
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    return dataset

