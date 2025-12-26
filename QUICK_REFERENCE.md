# Quick Reference Guide | 快速参考指南

[English](#commands) | [中文](#命令速查)

---

<a name="commands"></a>
## Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirement.txt

# Prepare datasets (first time only)
python create_data.py
```

### Run Experiments

#### Basic Experiments
```bash
# Run with CIFAR-10
python main.py  # Uses config/cifar10_resnet18.yaml

# Run with MNIST (edit main.py line 25-26 first)
# Uncomment: path = './config/mnist_mlp.yaml'
python main.py
```

#### Test Different Defenses
```bash
# Edit config file to change defense method
# Options: Avg, Krum, FLtrust, FedMs, GeoMed, Ours

# Example: Test GeoMed
sed -i "s/defense: 'Ours'/defense: 'GeoMed'/" config/cifar10_resnet18.yaml
python main.py
```

#### Defense Configuration Experiments
```bash
# Run specific experiment
python defense_config_experiments.py --experiment A  # Clipping vs Attack
python defense_config_experiments.py --experiment B  # Gamma impact
python defense_config_experiments.py --experiment C  # Threshold validation

# Run all experiments
python defense_config_experiments.py --experiment all

# Custom output directory
python defense_config_experiments.py --experiment A --output ./my_results
```

### Testing

```bash
# Test GeoMed implementation
python test_geomed.py

# Syntax check
python -m py_compile main.py
python -m py_compile defense_config_experiments.py
```

### Configuration Quick Edit

```bash
# Change defense method
sed -i "s/defense: 'Avg'/defense: 'GeoMed'/" config/cifar10_resnet18.yaml

# Change attack type
sed -i "s/client_attacks: 'Random'/client_attacks: 'SignFlip'/" config/cifar10_resnet18.yaml

# Change attack ratio
sed -i "s/client_attack_ratio: 0/client_attack_ratio: 0.1/" config/cifar10_resnet18.yaml
```

### View Results

```bash
# Results are saved in Excel format
ls new_ms_res/  # CIFAR-10 results
ls ms_res/      # MNIST results

# Defense config experiment results
ls defense_config_results/
```

---

<a name="命令速查"></a>
## 常用命令

### 环境设置
```bash
# 安装依赖
pip install -r requirement.txt

# 准备数据集（仅首次）
python create_data.py
```

### 运行实验

#### 基础实验
```bash
# 使用 CIFAR-10
python main.py  # 使用 config/cifar10_resnet18.yaml

# 使用 MNIST（需先编辑 main.py 第25-26行）
# 取消注释: path = './config/mnist_mlp.yaml'
python main.py
```

#### 测试不同防御方法
```bash
# 编辑配置文件更改防御方法
# 选项: Avg, Krum, FLtrust, FedMs, GeoMed, Ours

# 示例：测试 GeoMed
sed -i "s/defense: 'Ours'/defense: 'GeoMed'/" config/cifar10_resnet18.yaml
python main.py
```

#### 防御配置实验
```bash
# 运行特定实验
python defense_config_experiments.py --experiment A  # 裁剪率 vs 攻击
python defense_config_experiments.py --experiment B  # Gamma 影响
python defense_config_experiments.py --experiment C  # 阈值验证

# 运行所有实验
python defense_config_experiments.py --experiment all

# 自定义输出目录
python defense_config_experiments.py --experiment A --output ./my_results
```

### 测试

```bash
# 测试 GeoMed 实现
python test_geomed.py

# 语法检查
python -m py_compile main.py
python -m py_compile defense_config_experiments.py
```

### 快速修改配置

```bash
# 更改防御方法
sed -i "s/defense: 'Avg'/defense: 'GeoMed'/" config/cifar10_resnet18.yaml

# 更改攻击类型
sed -i "s/client_attacks: 'Random'/client_attacks: 'SignFlip'/" config/cifar10_resnet18.yaml

# 更改攻击比例
sed -i "s/client_attack_ratio: 0/client_attack_ratio: 0.1/" config/cifar10_resnet18.yaml
```

### 查看结果

```bash
# 结果以 Excel 格式保存
ls new_ms_res/  # CIFAR-10 结果
ls ms_res/      # MNIST 结果

# 防御配置实验结果
ls defense_config_results/
```

---

## Parameter Quick Reference | 参数速查

### Defense Methods | 防御方法
| Method | Description (EN) | 描述 (中文) |
|--------|------------------|-------------|
| `Avg` | Simple averaging (FedAvg baseline) | 简单平均（FedAvg 基线） |
| `Krum` | Select closest updates | 选择最接近的更新 |
| `FLtrust` | Trust-based with central dataset | 基于可信数据集的信任机制 |
| `FedMs` | Client-side trimmed mean only | 仅客户端修剪均值 |
| `GeoMed` | Geometric median (robust to outliers) | 几何中值（对异常值鲁棒） |
| `Ours` | Dual-sided trimmed mean (BR-FL) | 双侧修剪均值（BR-FL） |

### Attack Types | 攻击类型
| Attack | Description (EN) | 描述 (中文) |
|--------|------------------|-------------|
| `Noise` | Add Gaussian noise to updates | 向更新添加高斯噪声 |
| `Random` | Replace with random values | 用随机值替换 |
| `SignFlip` | Flip gradient signs | 翻转梯度符号 |
| `Backward` | Send scaled old model | 发送缩放的旧模型 |
| `LabelFlip` | Flip training labels | 翻转训练标签 |

### Key Parameters | 关键参数
| Parameter | Default | Range | Description (EN) | 描述 (中文) |
|-----------|---------|-------|------------------|-------------|
| `alpha` | 0.35 | 0-0.5 | Server-side clipping rate | 服务器端裁剪率 |
| `beta` | 0.35 | 0-0.5 | Client-side clipping rate | 客户端裁剪率 |
| `gamma` | 1.0 | 0-1 | Broadcast fraction | 广播比例 |
| `mc` | 0.1 | 0-0.5 | Client attack ratio | 客户端攻击比例 |
| `mp` | 0.1 | 0-0.5 | Server attack ratio | 服务器攻击比例 |

---

## Troubleshooting | 故障排除

### Common Issues | 常见问题

**Problem**: `CUDA out of memory`
**Solution**:
```bash
# Reduce batch size in config
sed -i 's/batch_size: 128/batch_size: 64/' config/cifar10_resnet18.yaml

# Or use CPU
sed -i 's/GPU: True/GPU: False/' config/cifar10_resnet18.yaml
```

**问题**: `CUDA 内存不足`
**解决方案**:
```bash
# 减小批量大小
sed -i 's/batch_size: 128/batch_size: 64/' config/cifar10_resnet18.yaml

# 或使用 CPU
sed -i 's/GPU: True/GPU: False/' config/cifar10_resnet18.yaml
```

---

**Problem**: `ModuleNotFoundError: No module named 'Defense.GeoMed'`
**Solution**: Make sure you've created the GeoMed.py file or switch to another defense method.

**问题**: `ModuleNotFoundError: No module named 'Defense.GeoMed'`
**解决方案**: 确保已创建 GeoMed.py 文件，或切换到其他防御方法。

---

**Problem**: Dataset not found
**Solution**:
```bash
# Re-run data preparation
python create_data.py
```

**问题**: 找不到数据集
**解决方案**:
```bash
# 重新运行数据准备
python create_data.py
```

---

## Tips | 使用技巧

### Running Long Experiments | 运行长时间实验
```bash
# Use tmux or screen to avoid interruption
tmux new -s brfl_exp
python defense_config_experiments.py --experiment all
# Detach: Ctrl+B, D
# Reattach: tmux attach -t brfl_exp
```

```bash
# 使用 tmux 或 screen 避免中断
tmux new -s brfl_exp
python defense_config_experiments.py --experiment all
# 分离: Ctrl+B, D
# 重新连接: tmux attach -t brfl_exp
```

### Quick Testing | 快速测试
```bash
# Test with fewer rounds for quick validation
# Edit max_rounds in defense_config_experiments.py to 5-10
python defense_config_experiments.py --experiment B
```

```bash
# 使用较少轮数进行快速验证
# 在 defense_config_experiments.py 中将 max_rounds 改为 5-10
python defense_config_experiments.py --experiment B
```

### Batch Experiments | 批量实验
```bash
# Test multiple defenses automatically
for defense in Avg Krum GeoMed Ours; do
    sed -i "s/defense: '.*'/defense: '$defense'/" config/cifar10_resnet18.yaml
    python main.py
done
```

```bash
# 自动测试多个防御方法
for defense in Avg Krum GeoMed Ours; do
    sed -i "s/defense: '.*'/defense: '$defense'/" config/cifar10_resnet18.yaml
    python main.py
done
```
