import sys
import os

# Dynamically add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)  # Add project_root to sys.path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # 新增
from utils.help import load_config, get_file_name
from utils.utility import get_attacker_index

if __name__ == '__main__':
    # 配置文件路径
    path = './config/cifar10_resnet18.yaml'
    # path = './config/mnist_mlp.yaml'
    config = load_config(path)

    # 从配置文件加载参数
    dataset_name = config['dataset_paras']['name']
    model_name = config['model_paras']['name']

    server_attacks = config['general_paras']['server_attacks']
  
    server_number = config['fed_paras']['server_number']
    client_number = config['fed_paras']['client_number']

    defense_method = config['general_paras']['defense']
    dirichlet_rate = config['fed_paras']['dirichlet_rate']
    server_attacker_rate = config['fed_paras']['server_attack_ratio']
    client_attacker_rate = config['fed_paras']['client_attack_ratio']
    global_round = config['fed_paras']['round']
    local_epoch = config['train_paras']['epoch']

    client_attacks_list = ['Noise', 'Random', 'SignFlip', 'Backward', 'LabelFlip']
    defense_list = ['Avg', 'Ours']
    label_list = ['Avg', 'BR-FL']

    attack_name = config['general_paras']['server_attacks']

    # 获取攻击者索引
    server_attacker_index, client_attacker_index = get_attacker_index(
        server_attacker_rate=server_attacker_rate,
        client_attacker_rate=client_attacker_rate,
        server_number=server_number,
        client_number=client_number
    )

    all_epoch = global_round * local_epoch

    # server attack 对应各种client attack
    # 然后 固定client attack 对应各种防御方法
    # 再就是固定防御方法 对于每个client idx的属性
    all_acc_list = []
    all_loss_list = []

    for client_attack_name in client_attacks_list:
        draw_acc_list = []
        draw_loss_list = []
        config['general_paras']['client_attacks'] = client_attack_name

        for defense_name in defense_list:
            config['general_paras']['defense'] = defense_name
            

            runs_name, file_name = get_file_name(config)
            save_path = os.path.join(config['general_paras']['output_dir'], file_name)

            res_acc_file_name = os.path.join(save_path, 'test_acc.xlsx')
            res_test_loss_file_name = os.path.join(save_path, 'test_loss.xlsx')

            # 加载Excel文件为DataFrame
            df_acc = pd.read_excel(res_acc_file_name)
            df_test_loss = pd.read_excel(res_test_loss_file_name)

            client_acc_list = []
            client_loss_list = []

            for idx in range(client_number):
                if idx in client_attacker_index:
                    continue  # 跳过攻击者客户端
                else:
                    benign_column = 'BenignClient' + str(idx)
                    client_acc_list.append(df_acc[benign_column].values.tolist())
                    client_loss_list.append(df_test_loss[benign_column].values.tolist())

            # for idx in range(len(client_acc_list)):
            #     print(client_acc_list[idx][all_epoch-1])
                
            draw_acc_list.append(client_acc_list)
            draw_loss_list.append(client_loss_list)

        all_acc_list.append(draw_acc_list)
        all_loss_list.append(draw_loss_list)

    final_acc_list = []
    final_loss_list = []
    final_acc_var_list = []  # 新增：存储每轮的精度方差
    final_loss_var_list = []  # 新增：存储每轮的损失方差

    for client_attack_idx, client_attack_name in enumerate(client_attacks_list):
        # 遍历每对攻击 每对攻击下 是 各种防御方法
        tmp_acc_list = []
        tmp_loss_list = []
        tmp_acc_var_list = []  # 存储每种防御方法的精度方差
        tmp_loss_var_list = []  # 存储每种防御方法的损失方差

        for method_idx, defense_name in enumerate(defense_list):
            # 遍历每个防御方法
            this_acc_avg_list = []
            this_loss_avg_list = []
            this_acc_var_list = []  # 当前方法的每轮精度方差
            this_loss_var_list = []  # 当前方法的每轮损失方差
            # have problem here
            for epoch in range(all_epoch):
                # 收集所有客户端在该轮次的精度和损失
                acc_values = 0.0
                loss_values = 0.0
                # print(len(draw_acc_list[method_idx]))
                acc_values = [all_acc_list[client_attack_idx][method_idx][client_idx][epoch]
                              for client_idx in range(len(all_acc_list[client_attack_idx][method_idx]))]
                # loss_values = [all_loss_list[client_attack_idx][method_idx][client_idx][epoch]
                #                for client_idx in range(len(all_loss_list[client_attack_idx][method_idx]))]
                

                # # 计算平均精度和损失
                # if epoch == all_epoch - 1:
                #     print(len(acc_values))
                #     print(acc_values)

                avg_acc = np.mean(acc_values)
                # avg_loss = np.mean(loss_values)

                                # 计算标准差
                std_acc = np.std(acc_values, ddof=1)  # 样本标准差
                # std_loss = np.std(loss_values, ddof=1)

                # 将结果存储到列表中
                this_acc_avg_list.append(avg_acc)
                # this_loss_avg_list.append(avg_loss)
                this_acc_var_list.append(std_acc)
                # this_loss_var_list.append(std_loss)

            # 将每种防御方法的结果存储到临时列表中
            tmp_acc_list.append(this_acc_avg_list)
            # tmp_loss_list.append(this_loss_avg_list)
            tmp_acc_var_list.append(this_acc_var_list)
            # tmp_loss_var_list.append(this_loss_var_list)

        # 将所有防御方法的结果存储到最终结果列表中
        final_acc_list.append(tmp_acc_list)
        # final_loss_list.append(tmp_loss_list)
        final_acc_var_list.append(tmp_acc_var_list)
        # final_loss_var_list.append(tmp_loss_var_list)

    for client_attack_idx, client_attack_name in enumerate(client_attacks_list):
        print("Server Attack:{}, Client Attack:{}".format(server_attacks,client_attack_name))
        for defense_idx, defense_name in enumerate(defense_list):
            print("Defense Method:{}".format(defense_name))
            print("Final Accuracy:", final_acc_list[client_attack_idx][defense_idx][all_epoch-1])
            print("Final Accuracy Variance:", final_acc_var_list[client_attack_idx][defense_idx][all_epoch-1])
            # print("Final Loss:", final_loss_list[client_attack_idx][defense_idx][all_epoch-1])
            # print("Final Loss Variance:", final_loss_var_list[client_attack_idx][defense_idx][all_epoch-1])
            print("")
