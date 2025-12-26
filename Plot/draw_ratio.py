import sys
import os

# Dynamically add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)  # Add project_root to sys.path


import pandas as pd
import matplotlib.pyplot as plt
from utils.help import load_config,get_file_name
from utils.utility import  get_attacker_index

import numpy as np
import seaborn as sns


if __name__ == '__main__':

    path = './config/cifar10_resnet18.yaml'
    # path = './config/mnist_mlp.yaml'   
    config = load_config(path) 

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

   
    defense_list = ['Ours','Avg','Krum','FLtrust', 'FedMs',]
    label_list = ['BR-FL','FedAvg','Krum','FLTrust', 'Fed-MS',]

    attack_name = config['general_paras']['server_attacks']
    
   
    server_attacker_index, client_attacker_index = get_attacker_index(server_attacker_rate=server_attacker_rate,
        client_attacker_rate=client_attacker_rate, server_number=server_number, client_number=client_number)

    all_epoch = global_round * local_epoch

    benign_client_num = client_number - len(client_attacker_index)


    range_epoch = range(all_epoch)  # Range of epochs


    draw_acc_list = []
    draw_loss_list = []


    for defense_name in defense_list:
        config['general_paras']['defense'] = defense_name

        runs_name, file_name = get_file_name(config)
        save_path = os.path.join(config['general_paras']['output_dir'], file_name)

        res_acc_file_name = os.path.join(save_path, 'test_acc.xlsx')
        res_test_loss_file_name = os.path.join(save_path, 'test_loss.xlsx')

                # Load the Excel files into DataFrames
        df_acc = pd.read_excel(res_acc_file_name)
        df_test_loss = pd.read_excel(res_test_loss_file_name)
        # print(df_acc)
        client_acc_list = []
        client_loss_list = []
        for idx in range(client_number):
            
            if idx in client_attacker_index:
                continue
            else:
                benign_column = 'BenignClient' + str(idx)
                client_acc_list.append((df_acc[benign_column].values).tolist())
                client_loss_list.append((df_test_loss[benign_column].values).tolist())
                # lowess_smooth_acc = sm.nonparametric.lowess(client_acc_list, all_epoch, frac=0)
                # lowess_smooth_loss = sm.nonparametric.lowess(client_loss_list, all_epoch, frac=0)

        # print(client_acc_list)
        draw_acc_list.append(client_acc_list)
        draw_loss_list.append(client_loss_list)
    

        # Set font properties and styles
    plt.rcParams['font.family'] = 'Times New Roman'
    xticks = range(all_epoch)
    line_width = 4
    tick_fontsize = 30
    label_fontsize = 30
    legend_size = 25
    legend_font_props = {'size': legend_size, 'weight': 'bold'}

    # plt.subplots_adjust(left=0.13, right=0.99, top=0.98, bottom=0.13)

    # Create a new figure with specified size
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.13, right=0.99, top=0.98, bottom=0.13)
    

    # Plotting
    for idx in range(len(defense_list)):
        # Prepare data for plotting
        data = pd.DataFrame({
            'Epoch': np.tile(range_epoch, benign_client_num),
            'Accuracy': np.concatenate(draw_acc_list[idx])
        })
        # Use lineplot to plot
        sns.lineplot(
            x='Epoch',
            y='Accuracy',
            data=data,
            label=label_list[idx],
            errorbar='sd',
            estimator='mean',
            linewidth=line_width
        )

    # Set y-axis limits
    plt.ylim(0, 80)

    # Configure legend
    le = plt.legend(
        loc="lower right",
        fontsize=legend_size,
        frameon=False,
        bbox_to_anchor=(1.0, 0),
        prop=legend_font_props
    )

    # Set line widths in the legend
    for line in le.get_lines():
        line.set_linewidth(4)  # Adjust line width as needed

    # Set labels and their font sizes
    plt.xlabel("Epochs", fontsize=label_fontsize, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=label_fontsize, fontweight='bold')

    # Set tick parameters
    plt.xticks(fontsize=tick_fontsize, fontweight='bold')
    plt.yticks(fontsize=tick_fontsize, fontweight='bold')

    # Enable grid lines
    plt.grid(True)

    # Save the plot as a PDF

    save_name = dataset_name + '_' + attack_name + '_' +'ms:'+str(server_attacker_rate) +'_'+ 'mc:'+str(client_attacker_rate)+ '.pdf'

    save_path = os.path.join('Plot_res', 'Ratio')
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, save_name)
    plt.savefig(save_path)
    plt.show()
    plt.clf()





    
            




        




