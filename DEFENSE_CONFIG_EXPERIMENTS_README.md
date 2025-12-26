# Defense Configuration Experiments

这个脚本实现了三个实验来分析BR-FL的防御配置特性。

## 实验说明

### 实验A: Clipping Rate vs Attack Intensity
测试不同的(alpha, beta)配置在不同攻击强度下的表现。

**参数配置**:
- Clipping rates: (0.25, 0.25), (0.35, 0.35), (0.45, 0.45)
- Attack levels: 6种不同的(mc, mp)组合
- 固定: gamma=1.0, attack='Random', dataset='CIFAR-10'

**输出**:
- `accuracy_matrix_<timestamp>.npy`: 3×6的accuracy矩阵（NumPy格式）
- `accuracy_matrix_<timestamp>.csv`: 3×6的accuracy矩阵（CSV格式，易读）
- `detailed_results_<timestamp>.csv`: 详细的实验结果

**用途**: 生成热力图，展示不同防御强度对不同攻击强度的响应

### 实验B: Gamma的影响
测试broadcast fraction (γ)对鲁棒性和通信开销的影响。

**参数配置**:
- mc=0.1, mp=0.1
- alpha=0.35, beta=0.40
- gamma values: [0.3, 0.5, 0.7, 1.0]

**输出**:
- `accuracy_curves_<timestamp>.npy`: 每个γ值的accuracy曲线
- `gamma_impact_<timestamp>.csv`: 汇总表（包含final accuracy和通信开销）

**用途**: 分析γ参数的trade-off（通信vs鲁棒性）

### 实验C: Threshold验证
验证理论threshold，测试防御在threshold内外的表现。

**参数配置**:
- 最保守配置: alpha=0.45, beta=0.45, gamma=1.0
- 7个测试点，包括应该成功和失败的案例

**输出**:
- `threshold_curves_<timestamp>.npy`: 所有测试点的accuracy曲线
- `threshold_validation_<timestamp>.csv`: 验证结果（预期vs实际）

**用途**: 验证理论推导的fundamental threshold

## 使用方法

### 运行单个实验

```bash
# 实验A: Clipping Rate vs Attack Intensity
python defense_config_experiments.py --experiment A

# 实验B: Gamma的影响
python defense_config_experiments.py --experiment B

# 实验C: Threshold验证
python defense_config_experiments.py --experiment C
```

### 运行所有实验

```bash
python defense_config_experiments.py --experiment all
```

### 自定义输出目录

```bash
python defense_config_experiments.py --experiment A --output ./my_results
```

## 输出结构

```
defense_config_results/
├── exp_a/
│   ├── accuracy_matrix_20250326_143022.npy
│   ├── accuracy_matrix_20250326_143022.csv
│   └── detailed_results_20250326_143022.csv
├── exp_b/
│   ├── accuracy_curves_20250326_144530.npy
│   └── gamma_impact_20250326_144530.csv
└── exp_c/
    ├── threshold_curves_20250326_150145.npy
    └── threshold_validation_20250326_150145.csv
```

## 实验时间估计

- **实验A**: 约1-2小时（3×6=18次实验，每次40轮）
- **实验B**: 约20-30分钟（4次实验，每次40轮）
- **实验C**: 约30-40分钟（7次实验，每次40轮）
- **全部实验**: 约2-3小时

## 配置要求

脚本默认使用 `config/cifar10_resnet18.yaml` 配置文件。确保：

1. 数据集已准备好（运行过 `python create_data.py`）
2. GPU可用（或修改config中的GPU设置）
3. 有足够的磁盘空间保存结果

## 结果解读

### 实验A的预期结果
根据CLAUDE_CODE_GUIDE.md：
- α=0.25时，mc=0.1, mp=0.1应该**失败**（accuracy低）
- α=0.35时，mc=0.1, mp=0.1应该**成功**（accuracy高）
- α=0.45时，所有轻中度攻击应该**成功**

### 实验C的成功标准
- Accuracy > 70%: 视为成功
- Accuracy ≤ 70%: 视为失败
- 验证是否符合理论threshold: ⌊2mcP⌋ + mpP < P/2

## 注意事项

1. **长时间运行**: 完整实验需要2-3小时，建议使用 `tmux` 或 `screen`
2. **GPU内存**: 如果遇到OOM错误，减小batch_size或使用CPU
3. **随机性**: 结果可能因随机种子而略有不同
4. **保存**: 所有结果带时间戳，不会覆盖之前的实验结果

## 后续分析

结果保存后，可以使用以下方法进行可视化：

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取实验A的结果（热力图）
matrix = np.load('defense_config_results/exp_a/accuracy_matrix_XXX.npy')
sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn')
plt.title('Clipping Rate vs Attack Intensity')
plt.show()

# 读取实验B的结果（曲线图）
curves = np.load('defense_config_results/exp_b/accuracy_curves_XXX.npy')
for i, gamma in enumerate([0.3, 0.5, 0.7, 1.0]):
    plt.plot(curves[i], label=f'γ={gamma}')
plt.legend()
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('Impact of Gamma on Robustness')
plt.show()
```

## 故障排除

### 问题: "No module named 'Defense.GeoMed'"
**解决**: 确保已实现GeoMed defense（如果使用），或修改脚本使用其他defense

### 问题: "CUDA out of memory"
**解决**:
1. 减小 `batch_size` 在config中
2. 设置 `GPU: False` 使用CPU
3. 减小 `num_workers`

### 问题: 实验中断
**解决**:
- 使用 `tmux` 或 `screen` 运行
- 减少 `max_rounds` 进行快速测试

## 引用

如果使用这些实验，请引用原始BR-FL论文。
