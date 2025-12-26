
import sys
import os

# Dynamically add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)  # Add project_root to sys.path


import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，保证每次运行结果一致
np.random.seed(42)

def generate_dirichlet_data(alpha, num_clients=50, num_labels=10):
    """
    生成基于 Dirichlet 分布的样本数据。
    每个客户有 num_labels 个类别的分布。
    """
    data = np.random.dirichlet(alpha * np.ones(num_labels), size=num_clients)
    # print(f"客户 0 的分布: {data[0]}")
    # print(f"客户 1 的分布: {data[1]}")
    return data

def plot_single_distribution(data, alpha, pre, num_labels, res_dir='Plot_res'):
    """
    画出某个 alpha 对应的客户分布图，并保存为单独的文件。
    """

    line_width = 4
    plt.rcParams['font.family'] = 'Times New Roman'
    markers = [None,None,None,None,None,None,]
    ms = [10,10,10,10,10,10,10,10]
    # line_style = ['-','--',':','-.','dashed','-',]
    line_style = ['-','-','-','-','-','-',]
    tick_fontsize = 30
    label_fontsize = 30
    legend_size = 30
    legend_font_props = {'size': legend_size, 'weight': 'bold'}
    fig, ax = plt.subplots(figsize=(10, 8))

    # 只展示前 pre 个客户的分布
    for client_id in range(pre):
        for label_id in range(num_labels):
            size = data[client_id, label_id] * 3000  # 调整气泡大小
            ax.scatter(client_id, label_id, s=size, alpha=0.6, edgecolors='black')

    
    ax.set_xlabel('Client ID', fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel('Label', fontsize=label_fontsize, fontweight='bold')

    # 设置所有客户和标签的刻度
    ax.set_xticks(np.arange(pre))  # 前 pre 个客户
    ax.set_yticks(np.arange(num_labels))  # 所有标签

    # # 调整刻度标签的字体大小，避免重叠
    # ax.tick_params(axis='x', labelsize=tick_fontsize)
    # ax.tick_params(axis='y', labelsize=tick_fontsize)
    # 设置刻度标签的字体样式
    ax.set_xticklabels(np.arange(pre), fontsize=tick_fontsize, fontweight='bold')
    ax.set_yticklabels(np.arange(num_labels), fontsize=tick_fontsize, fontweight='bold')

    save_path = os.path.join(res_dir, 'draw_dis')
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'alpha={alpha}.pdf')
    plt.tight_layout()
    
    plt.savefig(save_path)  # 单独保存图像
    plt.show()

if __name__ == '__main__':
    # 设置 alpha 值和 pre 个客户展示
    alpha_values = [0.25, 1, 5, 1000]
    pre = 10  # 展示前 pre 个客户

    # 为每个 alpha 生成独立的分布图
    for alpha in alpha_values:
        data = generate_dirichlet_data(alpha, num_clients=50, num_labels=10)
        plot_single_distribution(data, alpha, pre, num_labels=10, res_dir='Plot_res')
