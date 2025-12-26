import sys
import os

# Dynamically add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)  # Add project_root to sys.path

import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils.help import load_config, get_file_name
from utils.utility import get_attacker_index
import statsmodels.api as sm

if __name__ == '__main__':

    # Load the configuration file
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

    # Defense methods to be evaluated
    defense_list = ['Avg', 'Krum', 'FLtrust', 'FedMs', 'Ours']
    label_list = ['FedAvg','Krum','FLTrust', 'Fed-MS','BR-FL',]

    attack_name = config['general_paras']['server_attacks']

    # Get attacker indexes
    server_attacker_index, client_attacker_index = get_attacker_index(
        server_attacker_rate=server_attacker_rate,
        client_attacker_rate=client_attacker_rate, 
        server_number=server_number, 
        client_number=client_number
    )

    all_epoch = global_round * local_epoch  # Total number of epochs

    draw_acc_list = []
    draw_loss_list = []

    # Process each defense method
    for defense_name in defense_list:
        config['general_paras']['defense'] = defense_name

        runs_name, file_name = get_file_name(config)
        save_path = os.path.join(config['general_paras']['output_dir'], file_name)

        # Load accuracy and loss data from Excel files
        res_acc_file_name = os.path.join(save_path, 'test_acc.xlsx')
        # res_test_loss_file_name = os.path.join(save_path, 'test_loss.xlsx')

        df_acc = pd.read_excel(res_acc_file_name)
        # df_test_loss = pd.read_excel(res_test_loss_file_name)

        client_acc_list = []
        # client_loss_list = []

        # Collect data from benign clients only
        for idx in range(client_number):
            if idx in client_attacker_index:
                continue  # Skip attackers
            benign_column = f'BenignClient{idx}'
            client_acc_list.append(df_acc[benign_column].values.tolist())
            # client_loss_list.append(df_test_loss[benign_column].values.tolist())

        draw_acc_list.append(client_acc_list)
        # draw_loss_list.append(client_loss_list)

    final_acc_list = []
    final_loss_list = []

    # Calculate average accuracy and loss for each epoch
    for method_idx, defense_name in enumerate(defense_list):
        this_acc_list = []
        this_loss_list = []

        for epoch in range(all_epoch):
            acc_sum = sum(draw_acc_list[method_idx][client_idx][epoch] 
                          for client_idx in range(len(draw_acc_list[method_idx])))
            # # loss_sum = sum(draw_loss_list[method_idx][client_idx][epoch] 
            #                for client_idx in range(len(draw_loss_list[method_idx])))

            # Compute average accuracy and loss
            avg_acc = acc_sum / len(draw_acc_list[method_idx])
            # avg_loss = loss_sum / len(draw_loss_list[method_idx])

            this_acc_list.append(avg_acc)
            # this_loss_list.append(avg_loss)

        final_acc_list.append(this_acc_list)
        final_loss_list.append(this_loss_list)

    # Plot configuration
    xticks = [i for i in range(0, all_epoch, 25)]  # Display every 25 epochs
    line_width = 4
    plt.rcParams['font.family'] = 'Times New Roman'
    markers = [None] * len(defense_list)
    ms = [10] * len(defense_list)
    line_style = ['-'] * len(defense_list)
    tick_fontsize = 24
    label_fontsize = 24
    legend_size = 25
    legend_font_props = {'size': legend_size, 'weight': 'bold'}

    # Create a new figure for accuracy plot
    plt.figure(figsize=(10, 8))

    # Plot accuracy for each defense method
    for idx in range(len(defense_list)):
        plt.plot(range(all_epoch), final_acc_list[idx], 
                 marker=markers[idx], ms=ms[idx], 
                 linestyle=line_style[idx], 
                 label=label_list[idx], linewidth=line_width)

    # Set labels and ticks
    plt.xlabel('Epochs', fontsize=label_fontsize, fontweight='bold')
    plt.ylabel('Accuracy(%)', fontsize=label_fontsize, fontweight='bold')
    plt.xticks(xticks, fontsize=tick_fontsize, fontweight='bold')
    plt.yticks(fontsize=tick_fontsize, fontweight='bold')

    # Add legend and grid
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), prop=legend_font_props)
    plt.grid(True)

    # Adjust layout and set y-axis limit
    plt.ylim(0, 105)
    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.97)

    # Save the plot as a PDF
    save_name = f'{dataset_name}_{attack_name}_draw_acc.pdf'
    save_path = os.path.join('Plot_res', 'draw1')
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, save_name)
    plt.savefig(save_path)
    plt.show()
    plt.clf()
