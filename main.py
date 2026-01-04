import torch
from utils.help import load_config, get_device, get_file_name
from utils.utility import get_dataset, load_model, get_attacker_index, get_round_index
from utils.utility import test_model,normal_train_model, LabelFlip_train_model, LabelFlip_test_model
import wandb
from wandb import AlertLevel
from torch.utils.data import DataLoader
import copy, math
import argparse
from Attack.Noise import NoiseAttacks
from Attack.Random import RandomAttacks
from Attack.SignFlip import SignFlipAttacks
from Attack.Backward import BackwardAttacks
from Defense.Avg import AvgDefense
from Defense.Krum import KrumDefense
from Defense.FLtrust import FLtrustDefense
from Defense.TreamMean import TreamMeanDefense
from Defense.GeoMed import GeoMedDefense
from Defense.CoMed import CoMedDefense
from Defense.MultiKrum import MultiKrumDefense
import os
import pandas as pd


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/cifar10_resnet18.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    project_name = 'TON'
    path = args.config
    # path = './config/mnist_mlp.yaml'
    config = load_config(path)
    
    device = get_device(config['train_paras']['cuda_number'], config['train_paras']['GPU'])
    dataset_name = config['dataset_paras']['name']
    model_name = config['model_paras']['name']

    server_attacks = config['general_paras']['server_attacks']
    client_attacks = config['general_paras']['client_attacks']
    server_number = config['fed_paras']['server_number']    
    client_number = config['fed_paras']['client_number']

    defense_method = config['general_paras']['defense']
    dirichlet_rate = config['fed_paras']['dirichlet_rate'] 
    server_attacker_rate = config['fed_paras']['server_attack_ratio']
    client_attacker_rate = config['fed_paras']['client_attack_ratio']
    global_round = config['fed_paras']['round']
    local_epoch = config['train_paras']['epoch']

    runs_name, file_name = get_file_name(config)


    wandb.init(project=project_name, name=runs_name, config=config)

    wandb.alert(
                title="{}".format(runs_name),
                text="{} Code starts running".format(runs_name),
                level=AlertLevel.WARN,
                wait_duration=1, )
    
    train_dataset_list, test_dataset, central_dataset = get_dataset(dataset_name, dirichlet_rate)
    dataset_train_loader_list = [
                DataLoader(train_dataset_list[client_id], batch_size=config['dataset_paras']['batch_size'],
                           num_workers=config['dataset_paras']['num_workers'], pin_memory=False,
                           shuffle=True) for
                client_id in range(client_number)]
    dataset_test_loader = DataLoader(dataset=test_dataset, batch_size=config['dataset_paras']['batch_size'],
                            num_workers=config['dataset_paras']['num_workers'], pin_memory=False,
                                         shuffle=False)
    if defense_method == 'FLtrust':
        central_dataset_loader = DataLoader(dataset=central_dataset, batch_size=config['dataset_paras']['batch_size'],
                            num_workers=config['dataset_paras']['num_workers'], pin_memory=False,
                                         shuffle=False)
    else:
        central_dataset_loader = None

    base_model = load_model(model_name, config['dataset_paras']['num_classes'])

    # obtain the attacker index
    server_attacker_index, client_attacker_index = get_attacker_index(server_attacker_rate=server_attacker_rate,
        client_attacker_rate=client_attacker_rate, server_number=server_number, client_number=client_number)

    # Create global optimizer for FLTrust (preserves state across rounds)
    # Following official FLTrust implementation: SGD with momentum
    # IMPORTANT: Optimizer must be created AFTER model is moved to device
    fltrust_global_optimizer = None

    column_names = []

    for i in range(client_number):
        if i in client_attacker_index:
            column_names.append('Malicious'+ 'Client{}'.format(i))
        else:
            column_names.append('Benign'+ 'Client{}'.format(i))
             

    base_model = base_model.to(device)

    # Create global optimizer for FLTrust AFTER model is on device
    if defense_method == 'FLtrust':
        fltrust_global_optimizer = torch.optim.SGD(base_model.parameters(), lr=0.01, momentum=0.9)

    # for name, param in base_model.state_dict().items():
    #     print(name,param.size())

    client_model_list = [copy.deepcopy(base_model) for _ in range(client_number)]
    server_model_list = [copy.deepcopy(base_model) for _ in range(server_number)]

    # # old_model_list = copy.deepcopy(model_list)
    # zeros_model = copy.deepcopy(base_model)
    # zeros_model = zeros_model.to(device)
    # for key in zeros_model.state_dict():
    #     torch.nn.init.zeros_(zeros_model.state_dict()[key])


    res_acc_list = []
    res_acc_list = [copy.deepcopy(res_acc_list) for i in range(client_number)]
    res_test_loss_list = []
    res_test_loss_list = [copy.deepcopy(res_test_loss_list) for i in range(client_number)]


    server_send_updates: list[dict] = [copy.deepcopy(server_model_list[idx].state_dict()) for idx in range(server_number)]
    client_send_updates: list[dict] = [copy.deepcopy(client_model_list[idx].state_dict()) for idx in range(client_number)]

    if defense_method == 'Ours' or defense_method == 'ablation1' or defense_method == 'ablation2':
        
        trimmed_client_rate = (config['fed_paras']['client_attack_ratio']) / (0.5-config['fed_paras']['server_attack_ratio'])
        # pre_attacker_client_num = math.ceil(trimmed_client_rate * (client_number/server_number))
        # pre_attacker_client_num = max(pre_attacker_client_num,1)
        # trimmed_client_rate = pre_attacker_client_num / (client_number/server_number)
        trimmed_client_rate = math.ceil(trimmed_client_rate*10)
        trimmed_client_rate = trimmed_client_rate / 10
        if trimmed_client_rate == 0:
            trimmed_server_rate = config['fed_paras']['server_attack_ratio']
        else:
            trimmed_server_rate = (config['fed_paras']['client_attack_ratio'] / trimmed_client_rate) + config['fed_paras']['server_attack_ratio']
        
        if trimmed_client_rate >= 0.5:
            raise ValueError("The client attack rate is too high!")
        
        if trimmed_server_rate >= 0.5:
            raise ValueError("The server attack rate is too high!")
        
        print(trimmed_client_rate)
        print(trimmed_server_rate)

        if trimmed_client_rate > 0.4:
            trimmed_client_rate = 0.4
        if trimmed_server_rate > 0.4:
            trimmed_server_rate = 0.4


    elif defense_method == 'FedMs':

        trimmed_server_rate = config['fed_paras']['server_attack_ratio']
        if trimmed_server_rate >= 0.5:
            raise ValueError("The server attack rate is too high!")
            
        # predict_client_attacker_num =  math.ceil(client_number * trimmed_client_rate)
        # predict_server_attacker_num = math.ceil(server_number * trimmed_server_rate)


    # client_id = 0
    # all_train_ep = 0
    # for ep in range(50):
    #     (client_model_list[client_id], tmp_train_loss) = normal_train_model(model=client_model_list[client_id],
    #                                                     train_loader=dataset_train_loader_list[client_id], device=device, epoch=all_train_ep,
    #                                                     criterion_name=config['train_paras']['criterion_name'],
    #                                                     optimizer_name=config['train_paras']['optimizer_name'], lr=config['train_paras']['lr'], 
    #                                                                                    )
    #     (test_acc, test_loss) = test_model(model=client_model_list[client_id], test_loader=dataset_test_loader, device=device,
    #                                                     criterion_name=config['train_paras']['criterion_name'])
    #     print('Client{}: Epoch{}: Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(client_id, all_train_ep, test_loss, test_acc))
    

    for each_round in range(global_round):

        if server_attacks == 'SignFlip' or server_attacks == 'Backward':
            old_server_model_list = copy.deepcopy(server_model_list)
        if client_attacks == 'SignFlip' or client_attacks == 'Backward':
            old_client_model_list = copy.deepcopy(client_model_list)

        # 1. the servers send their global model updates(state_dict) to the clients

        select_server_id, server_to_clients, update_counts = get_round_index(server_number, client_number)

        global_state_dict:dict = AvgDefense(server_send_updates, index=list(range(server_number)), device=device)

        # 2. the clients use different defense methods to obtain the updated models state dict
        if defense_method == 'Avg' or defense_method == 'FLtrust' or defense_method == 'Krum' or defense_method == 'GeoMed' or defense_method == 'CoMed' or defense_method == 'MultiKrum' or defense_method == 'ablation2':
            # Avg all the server models state dict
            defense_state_dict:dict = AvgDefense(server_send_updates, index=list(range(server_number)), device=device)
        
        elif defense_method == 'FedMs' or defense_method == 'Ours' or defense_method == 'ablation1':
            defense_state_dict:dict = TreamMeanDefense(server_send_updates, index=list(range(server_number)),
                                              trimmed_rate=trimmed_server_rate)
        
        # 3. conduct local training
        for client_id in range(client_number):

            if client_id in client_attacker_index:
                client_model_list[client_id].load_state_dict(global_state_dict)
                
            else:
                client_model_list[client_id].load_state_dict(defense_state_dict)

            for ep in range(local_epoch):
                all_train_ep = ep + each_round * local_epoch
                if client_attacks != 'LabelFlip':
                    (client_model_list[client_id], tmp_train_loss) = normal_train_model(model=client_model_list[client_id],
                                                        train_loader=dataset_train_loader_list[client_id], device=device, epoch=all_train_ep,
                                                        criterion_name=config['train_paras']['criterion_name'],
                                                        optimizer_name=config['train_paras']['optimizer_name'], lr=config['train_paras']['lr'], 
                                                                                       )
                    (test_acc, test_loss) = test_model(model=client_model_list[client_id], test_loader=dataset_test_loader, device=device,
                                                        criterion_name=config['train_paras']['criterion_name'])
                elif client_attacks == 'LabelFlip':
                    (client_model_list[client_id], tmp_train_loss) = LabelFlip_train_model(model=client_model_list[client_id], label_flip_rate=config['attack_paras']['LabelFlipRate'], 
                                                        train_loader=dataset_train_loader_list[client_id], device=device, epoch=all_train_ep,
                                                        criterion_name=config['train_paras']['criterion_name'], num_classes=config['dataset_paras']['num_classes'],
                                                        optimizer_name=config['train_paras']['optimizer_name'], lr=config['train_paras']['lr'], 
                                                                                       )
                    (test_acc, test_loss) = test_model(model=client_model_list[client_id], test_loader=dataset_test_loader, device=device,
                                                        criterion_name=config['train_paras']['criterion_name'])
                    (label_test_acc, label_test_loss) = LabelFlip_test_model(model=client_model_list[client_id], test_loader=dataset_test_loader, device=device,
                                                        criterion_name=config['train_paras']['criterion_name'], num_classes=config['dataset_paras']['num_classes'])
                
                if client_id in client_attacker_index:
                    wandb.log({'Testing Accuracy of malicious client{}'.format(client_id):test_acc,
                                        'Testing Loss of malicious client{}'.format(client_id): test_loss, 
                                        'Training loss of malicious client {}'.format(client_id): tmp_train_loss,
                                        'Epoch': all_train_ep})
                    if client_attacks == 'LabelFlip':
                        wandb.log({'LabelFlip Testing Accuracy of malicious client{}'.format(client_id):label_test_acc,
                                        'LabelFlip Testing Loss of malicious client{}'.format(client_id): label_test_loss,
                                        'Epoch': all_train_ep})
                else:
                    wandb.log({'Testing Accuracy of benign client{}'.format(client_id):test_acc,
                                        'Testing Loss of benign client{}'.format(client_id): test_loss, 
                                        'Training loss of benign client {}'.format(client_id): tmp_train_loss,
                                        'Epoch': all_train_ep})
                res_acc_list[client_id].append(test_acc)
                res_test_loss_list[client_id].append(test_loss)
                print('Client{}: Epoch{}: Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(client_id, all_train_ep, test_loss, test_acc))
        
        # 4. local clients upload their updates to the servers

            if client_id in client_attacker_index:
                
                if config['attack_paras']['client_is_attack'] == True:
                    # conduct client attacks
                    if client_attacks == 'Noise':
                        client_send_updates[client_id] = NoiseAttacks(client_model_list[client_id], config['attack_paras']['noise_mean'],
                                                                        config['attack_paras']['noise_std'], device)
                    elif client_attacks == 'Random':
                        client_send_updates[client_id] = RandomAttacks(client_model_list[client_id], config['attack_paras']['random_lower'],
                                                                        config['attack_paras']['random_upper'], device)
                    elif client_attacks == 'SignFlip':
                        client_send_updates[client_id] = SignFlipAttacks(client_model_list[client_id], old_client_model_list[client_id], config['attack_paras']['SignFlipScaleFactor'])
                    elif client_attacks == 'Backward':
                        client_send_updates[client_id] = BackwardAttacks(client_model_list[client_id], old_client_model_list[client_id],)
                    elif client_attacks == 'LabelFlip':
                        client_send_updates[client_id] = copy.deepcopy(client_model_list[client_id].state_dict())
                else:
                        client_send_updates[client_id] = copy.deepcopy(client_model_list[client_id].state_dict())
            else:
                # benign client
                client_send_updates[client_id] = copy.deepcopy(client_model_list[client_id].state_dict())

        # 5. the servers receive the updates from the clients and conduct defense or attacks


        if defense_method == 'FLtrust':
            # Use deepcopy to avoid modifying base_model directly
            global_avg_model = copy.deepcopy(base_model)
            global_avg_model.load_state_dict(global_state_dict)
            res_model_state_dict = FLtrustDefense(global_model=global_avg_model,
                                                updates_list=client_send_updates,central_dataloader=central_dataset_loader, server_index=server_number,
                                                index=range(0,client_number), lr=config['train_paras']['lr'], local_ep=local_epoch,
                                        criterion_name=config['train_paras']['criterion_name'], optimizer_name=config['train_paras']['optimizer_name'],
                                        device=device, round_num=each_round,
                                         fltrust_global_lr=config['train_paras']['fltrust_global_lr'],
                                         global_optimizer=fltrust_global_optimizer)
            
        for server_id in range(server_number):

            if server_id in server_attacker_index:
                # attackers first avg their group client models 
                res_model_state_dict = AvgDefense(client_send_updates,server_to_clients[server_id] , device)
                if res_model_state_dict != None:
                    server_model_list[server_id].load_state_dict(res_model_state_dict)

                # server_send_updates[server_id] = copy.deepcopy(server_model_list[server_id].state_dict())

                if config['attack_paras']['server_is_attack'] == True:
                    # and then conduct attacks
                    if server_attacks == 'Noise':       
                        server_send_updates[server_id] = NoiseAttacks(server_model_list[server_id], config['attack_paras']['noise_mean'],
                                                                    config['attack_paras']['noise_std'], device)
                    elif server_attacks== 'Random':           
                        server_send_updates[server_id] = RandomAttacks(server_model_list[server_id], config['attack_paras']['random_lower'],
                                                                    config['attack_paras']['random_upper'], device)
                    elif server_attacks == 'SignFlip':
                        server_send_updates[server_id] = SignFlipAttacks(server_model_list[server_id], old_server_model_list[server_id], config['attack_paras']['SignFlipScaleFactor']) 
                    
                    elif server_attacks == 'Backward':
                        server_send_updates[server_id] = BackwardAttacks(server_model_list[server_id], old_server_model_list[server_id],)
                else:
                    server_send_updates[server_id] = copy.deepcopy(server_model_list[server_id].state_dict())   
                 
            else:
                # benign server conduct defense
                if defense_method == 'Avg' or defense_method == 'FedMs' or defense_method == 'ablation1':
                    res_model_state_dict = AvgDefense(client_send_updates,server_to_clients[server_id] , device)
                    if res_model_state_dict != None:
                        server_model_list[server_id].load_state_dict(res_model_state_dict)
                    else:
                        pass
                elif defense_method == 'Krum':
                    res_model_state_dict = KrumDefense(updates_list=client_send_updates,
                                            num_attackers=(max(1,math.ceil(config['fed_paras']['client_attack_ratio']*client_number*(server_number/client_number)))),
                                            index=server_to_clients[server_id],
                                        device=device,k=config['defense_paras']['Krum_m'])
                    if res_model_state_dict != None:
                        server_model_list[server_id].load_state_dict(res_model_state_dict)

                elif defense_method == 'FLtrust':
                    # res_model_state_dict = FLtrustDefense(global_model=server_model_list[server_id],
                    #                             updates_list=client_send_updates,central_dataloader=central_dataset_loader, server_index=server_id,
                    #                             index=server_to_clients[server_id], lr=config['train_paras']['lr'], local_ep=local_epoch,
                    #                     criterion_name=config['train_paras']['criterion_name'], optimizer_name=config['train_paras']['optimizer_name'],
                    #                     device=device, round_num=each_round)
                    if res_model_state_dict != None:
                        server_model_list[server_id].load_state_dict(res_model_state_dict)

                elif defense_method == 'GeoMed':
                    res_model_state_dict = GeoMedDefense(updates_list=client_send_updates,
                                                         index=server_to_clients[server_id],
                                                         device=device)
                    if res_model_state_dict != None:
                        server_model_list[server_id].load_state_dict(res_model_state_dict)

                elif defense_method == 'CoMed':
                    res_model_state_dict = CoMedDefense(updates_list=client_send_updates,
                                                        index=server_to_clients[server_id],
                                                        device=device)
                    if res_model_state_dict != None:
                        server_model_list[server_id].load_state_dict(res_model_state_dict)

                elif defense_method == 'MultiKrum':
                    num_clients_per_server = len(server_to_clients[server_id])
                    num_attackers = max(1, math.ceil(config['fed_paras']['client_attack_ratio'] * num_clients_per_server))
                    # Set m = n - f - 2 for maximum robustness
                    m = max(1, num_clients_per_server - num_attackers - 2)
                    res_model_state_dict = MultiKrumDefense(updates_list=client_send_updates,
                                                            num_attackers=num_attackers,
                                                            index=server_to_clients[server_id],
                                                            device=device,
                                                            m=m)
                    if res_model_state_dict != None:
                        server_model_list[server_id].load_state_dict(res_model_state_dict)

                elif defense_method == 'Ours' or defense_method == 'ablation2':
                    res_model_state_dict = TreamMeanDefense(client_send_updates,server_to_clients[server_id],
                                        trimmed_rate=trimmed_client_rate)
                    if res_model_state_dict != None:
                        server_model_list[server_id].load_state_dict(res_model_state_dict)

                # update the sending
                server_send_updates[server_id] = copy.deepcopy(server_model_list[server_id].state_dict())

    # save the res name 
    save_path = os.path.join(config['general_paras']['output_dir'],file_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # modify it to pd
    res_acc_file_name = os.path.join(save_path, 'test_acc.xlsx')
    res_test_loss_file_name = os.path.join(save_path, 'test_loss.xlsx')
        
   
    # Pad shorter lists with None or another value to match lengths
    max_len = max(len(sublist) for sublist in res_acc_list)
    res_acc_list = [sublist + [None] * (max_len - len(sublist)) for sublist in res_acc_list]
    res_test_loss_list = [sublist + [None] * (max_len - len(sublist)) for sublist in res_test_loss_list]

    # Convert the list of sublists into a DataFrame and transpose it
    df_acc = pd.DataFrame(res_acc_list).transpose()
    df_test_loss = pd.DataFrame(res_test_loss_list).transpose()

    df_acc.columns = column_names
    df_test_loss.columns = column_names

    df_acc.to_excel(res_acc_file_name, index=False)
    df_test_loss.to_excel(res_test_loss_file_name, index=False)

    # Write completion signal file for the parallel runner
    signal_dir = 'signals'
    os.makedirs(signal_dir, exist_ok=True)
    signal_file = os.path.join(signal_dir, f"{file_name}.done")
    with open(signal_file, 'w') as f:
        f.write(f"gpu_id={config['train_paras']['cuda_number']}\n")
        f.write(f"defense={defense_method}\n")
        f.write(f"server_attack={server_attacks}\n")
        f.write(f"client_attack={client_attacks}\n")
        f.write(f"status=success\n")

    wandb.alert(
            title=runs_name,
            text="{} End of code run!".format(runs_name),
            level=AlertLevel.WARN,
            wait_duration=1, )
        
    wandb.finish()         
        


        



    









