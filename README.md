# Byzantine-Resilient Federated Learning (BR-FL)

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

### Overview

This repository implements **Byzantine-Resilient Federated Learning (BR-FL)**, a hierarchical federated learning framework with dual-sided defense mechanisms. The project evaluates various defense strategies against Byzantine attacks in a multi-server, multi-client federated learning environment.

**Key Innovation**: BR-FL employs trimmed mean aggregation at both the server-side and client-side, providing robust defense against coordinated Byzantine attacks in hierarchical federated learning systems.

### Features

- 🛡️ **Multiple Defense Mechanisms**
  - FedAvg (baseline)
  - Krum
  - FLTrust
  - Trimmed Mean (FedMs)
  - **GeoMed** (Geometric Median) - NEW!
  - **BR-FL (Ours)** - Dual-sided defense with trimmed mean

- ⚔️ **Byzantine Attack Simulations**
  - Noise Attack
  - Random Attack
  - Sign-Flip Attack
  - Backward Attack
  - Label-Flip Attack

- 🏗️ **Hierarchical Architecture**
  - Multi-server, multi-client setup
  - Configurable attack ratios for both servers and clients
  - Support for non-IID data distribution (Dirichlet)

- 📊 **Comprehensive Evaluation**
  - Accuracy tracking for benign and malicious participants
  - Integration with Weights & Biases for experiment tracking
  - Defense configuration analysis experiments

- 🧪 **Datasets**
  - CIFAR-10 with ResNet18
  - MNIST with MLP

### Requirements

```
Python >= 3.8
PyTorch >= 2.1.0
CUDA >= 12.1 (for GPU support)
```

See [requirement.txt](requirement.txt) for full dependencies.

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Ton
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

3. **Prepare datasets** (first time only)
   ```bash
   python create_data.py
   ```

### Quick Start

#### 1. Basic Experiment

Edit configuration in `config/cifar10_resnet18.yaml`:

```yaml
general_paras:
  defense: 'Ours'  # Options: Avg, Krum, FLtrust, FedMs, GeoMed, Ours
  server_attacks: 'Random'
  client_attacks: 'Random'

fed_paras:
  server_number: 10
  client_number: 50
  server_attack_ratio: 0.1
  client_attack_ratio: 0.1
```

Run the experiment:
```bash
python main.py
```

#### 2. Test GeoMed Defense

```bash
# Modify config file
sed -i "s/defense: 'Ours'/defense: 'GeoMed'/" config/cifar10_resnet18.yaml

# Run experiment
python main.py
```

#### 3. Defense Configuration Experiments

Run comprehensive defense analysis:

```bash
# Experiment A: Clipping Rate vs Attack Intensity
python defense_config_experiments.py --experiment A

# Experiment B: Impact of Gamma
python defense_config_experiments.py --experiment B

# Experiment C: Threshold Validation
python defense_config_experiments.py --experiment C

# Run all experiments
python defense_config_experiments.py --experiment all
```

See [DEFENSE_CONFIG_EXPERIMENTS_README.md](DEFENSE_CONFIG_EXPERIMENTS_README.md) for details.

### Project Structure

```
Ton/
├── Attack/                    # Byzantine attack implementations
│   ├── Noise.py
│   ├── Random.py
│   ├── SignFlip.py
│   └── Backward.py
├── Defense/                   # Defense mechanisms
│   ├── Avg.py                # FedAvg baseline
│   ├── Krum.py               # Krum defense
│   ├── FLtrust.py            # FLTrust defense
│   ├── TreamMean.py          # Trimmed Mean
│   └── GeoMed.py             # Geometric Median (NEW)
├── Net/                       # Neural network models
│   ├── resnet.py             # ResNet18/34
│   └── MLP.py                # Multi-layer perceptron
├── config/                    # Configuration files
│   ├── cifar10_resnet18.yaml
│   └── mnist_mlp.yaml
├── utils/                     # Utility functions
│   ├── help.py               # Config & dataset loaders
│   └── utility.py            # Training & testing utilities
├── Plot/                      # Visualization scripts
├── main.py                    # Main experiment script
├── create_data.py             # Data preparation
├── defense_config_experiments.py  # Defense analysis (NEW)
└── test_geomed.py             # GeoMed unit tests (NEW)
```

### Configuration Parameters

#### Defense Parameters (`defense_paras`)
- `alpha`: Server-side clipping rate (default: 0.35)
- `beta`: Client-side clipping rate (default: 0.35)
- `gamma`: Broadcast fraction (default: 1.0)

#### Federated Learning Parameters (`fed_paras`)
- `round`: Number of global rounds
- `server_number`: Number of servers
- `client_number`: Number of clients
- `server_attack_ratio`: Ratio of malicious servers
- `client_attack_ratio`: Ratio of malicious clients
- `dirichlet_rate`: Data heterogeneity (1000=IID, lower=more non-IID)

#### Training Parameters (`train_paras`)
- `lr`: Learning rate
- `epoch`: Local training epochs
- `optimizer_name`: Optimizer (Adam/SGD)
- `cuda_number`: GPU device ID

### Results

Results are saved to:
- `new_ms_res/` (CIFAR-10) or `ms_res/` (MNIST)
- Excel files: `test_acc.xlsx`, `test_loss.xlsx`
- Weights & Biases dashboard (if enabled)

### Defense Configuration Experiments

Three experiments to analyze BR-FL theoretical properties:

**Experiment A**: Clipping Rate vs Attack Intensity
- Tests 3 configurations × 6 attack levels
- Output: 3×6 accuracy heatmap

**Experiment B**: Impact of Gamma
- Tests 4 gamma values (0.3, 0.5, 0.7, 1.0)
- Output: Accuracy curves + communication costs

**Experiment C**: Threshold Validation
- Tests 7 attack configurations
- Validates fundamental threshold: ⌊2mcP⌋ + mpP < P/2

### Citation

If you use this code in your research, please cite:

```bibtex
@article{brfl2024,
  title={Byzantine-Resilient Federated Learning with Dual-Sided Defense},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments

- Weights & Biases for experiment tracking
- PyTorch team for the deep learning framework

---

<a name="中文"></a>
## 中文

### 项目概述

本项目实现了**拜占庭鲁棒联邦学习 (BR-FL)**，这是一个具有双侧防御机制的分层联邦学习框架。项目评估了多种防御策略在多服务器、多客户端联邦学习环境中对抗拜占庭攻击的效果。

**核心创新**：BR-FL 在服务器端和客户端同时采用修剪均值聚合，为分层联邦学习系统中的协同拜占庭攻击提供鲁棒防御。

### 主要特性

- 🛡️ **多种防御机制**
  - FedAvg（基线）
  - Krum
  - FLTrust
  - Trimmed Mean (FedMs)
  - **GeoMed**（几何中值）- 新增！
  - **BR-FL (Ours)** - 双侧防御（修剪均值）

- ⚔️ **拜占庭攻击模拟**
  - 噪声攻击
  - 随机攻击
  - 符号翻转攻击
  - 反向攻击
  - 标签翻转攻击

- 🏗️ **分层架构**
  - 多服务器、多客户端设置
  - 服务器和客户端的攻击比例可配置
  - 支持非独立同分布数据（Dirichlet 分布）

- 📊 **全面评估**
  - 跟踪良性和恶意参与者的准确率
  - 集成 Weights & Biases 进行实验跟踪
  - 防御配置分析实验

- 🧪 **数据集**
  - CIFAR-10 配合 ResNet18
  - MNIST 配合 MLP

### 环境要求

```
Python >= 3.8
PyTorch >= 2.1.0
CUDA >= 12.1（GPU 支持）
```

完整依赖见 [requirement.txt](requirement.txt)。

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd Ton
   ```

2. **安装依赖**
   ```bash
   pip install -r requirement.txt
   ```

3. **准备数据集**（首次运行）
   ```bash
   python create_data.py
   ```

### 快速开始

#### 1. 基础实验

编辑配置文件 `config/cifar10_resnet18.yaml`：

```yaml
general_paras:
  defense: 'Ours'  # 选项: Avg, Krum, FLtrust, FedMs, GeoMed, Ours
  server_attacks: 'Random'
  client_attacks: 'Random'

fed_paras:
  server_number: 10
  client_number: 50
  server_attack_ratio: 0.1
  client_attack_ratio: 0.1
```

运行实验：
```bash
python main.py
```

#### 2. 测试 GeoMed 防御

```bash
# 修改配置文件
sed -i "s/defense: 'Ours'/defense: 'GeoMed'/" config/cifar10_resnet18.yaml

# 运行实验
python main.py
```

#### 3. 防御配置实验

运行综合防御分析：

```bash
# 实验A: 裁剪率 vs 攻击强度
python defense_config_experiments.py --experiment A

# 实验B: Gamma 的影响
python defense_config_experiments.py --experiment B

# 实验C: 阈值验证
python defense_config_experiments.py --experiment C

# 运行所有实验
python defense_config_experiments.py --experiment all
```

详见 [DEFENSE_CONFIG_EXPERIMENTS_README.md](DEFENSE_CONFIG_EXPERIMENTS_README.md)。

### 项目结构

```
Ton/
├── Attack/                    # 拜占庭攻击实现
│   ├── Noise.py              # 噪声攻击
│   ├── Random.py             # 随机攻击
│   ├── SignFlip.py           # 符号翻转攻击
│   └── Backward.py           # 反向攻击
├── Defense/                   # 防御机制
│   ├── Avg.py                # FedAvg 基线
│   ├── Krum.py               # Krum 防御
│   ├── FLtrust.py            # FLTrust 防御
│   ├── TreamMean.py          # 修剪均值
│   └── GeoMed.py             # 几何中值（新增）
├── Net/                       # 神经网络模型
│   ├── resnet.py             # ResNet18/34
│   └── MLP.py                # 多层感知机
├── config/                    # 配置文件
│   ├── cifar10_resnet18.yaml # CIFAR-10 配置
│   └── mnist_mlp.yaml        # MNIST 配置
├── utils/                     # 工具函数
│   ├── help.py               # 配置和数据集加载
│   └── utility.py            # 训练和测试工具
├── Plot/                      # 可视化脚本
├── main.py                    # 主实验脚本
├── create_data.py             # 数据准备
├── defense_config_experiments.py  # 防御分析（新增）
└── test_geomed.py             # GeoMed 单元测试（新增）
```

### 配置参数

#### 防御参数 (`defense_paras`)
- `alpha`: 服务器端裁剪率（默认：0.35）
- `beta`: 客户端裁剪率（默认：0.35）
- `gamma`: 广播比例（默认：1.0）

#### 联邦学习参数 (`fed_paras`)
- `round`: 全局轮数
- `server_number`: 服务器数量
- `client_number`: 客户端数量
- `server_attack_ratio`: 恶意服务器比例
- `client_attack_ratio`: 恶意客户端比例
- `dirichlet_rate`: 数据异构性（1000=IID，越低越非IID）

#### 训练参数 (`train_paras`)
- `lr`: 学习率
- `epoch`: 本地训练轮数
- `optimizer_name`: 优化器（Adam/SGD）
- `cuda_number`: GPU 设备编号

### 实验结果

结果保存至：
- `new_ms_res/`（CIFAR-10）或 `ms_res/`（MNIST）
- Excel 文件：`test_acc.xlsx`、`test_loss.xlsx`
- Weights & Biases 仪表板（如果启用）

### 防御配置实验

三个实验用于分析 BR-FL 的理论特性：

**实验A**：裁剪率 vs 攻击强度
- 测试 3 种配置 × 6 种攻击级别
- 输出：3×6 准确率热力图

**实验B**：Gamma 的影响
- 测试 4 个 gamma 值（0.3、0.5、0.7、1.0）
- 输出：准确率曲线 + 通信开销

**实验C**：阈值验证
- 测试 7 种攻击配置
- 验证基础阈值：⌊2mcP⌋ + mpP < P/2

### 引用

如果您在研究中使用此代码，请引用：

```bibtex
@article{brfl2024,
  title={Byzantine-Resilient Federated Learning with Dual-Sided Defense},
  author={[作者]},
  journal={[期刊]},
  year={2024}
}
```

### 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

### 致谢

- Weights & Biases 提供实验跟踪支持
- PyTorch 团队提供深度学习框架

---

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

## Updates

- **2024-XX**: Added GeoMed baseline defense
- **2024-XX**: Added defense configuration experiments
- **2024-XX**: Updated default parameters (alpha=0.35, beta=0.35)
- **2024-XX**: Initial release
