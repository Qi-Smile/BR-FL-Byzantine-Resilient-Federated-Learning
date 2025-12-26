# BR-FL 项目修改指南（给 Claude Code）

## 项目背景

这是一个Byzantine-Resilient Federated Learning (BR-FL)的研究项目，正在为IEEE TC期刊投稿进行修改。

### 核心算法
BR-FL使用trimmed mean在server端和client端进行双侧防御：
- **Server端**：每个server随机选择C/P个clients，使用trimmed mean（clipping rate α）聚合
- **Client端**：每个client从γP个servers接收结果，使用trimmed mean（clipping rate β）聚合

### 关键参数
- `C`: client总数 (默认50)
- `P`: server总数 (默认10)  
- `mc`: Byzantine client比例
- `mp`: Byzantine server比例
- `alpha`: server端clipping rate (0, 0.5)
- `beta`: client端clipping rate (0, 0.5)
- `gamma`: broadcast fraction (0, 1]

---

## 需要完成的修改

### 任务1：添加GeoMed Baseline

**需求**：实现Geometric Median聚合方法作为新的baseline

**算法**：
```python
# Weiszfeld算法求解geometric median
# argmin_y Σ ||x_i - y||_2

def geometric_median(updates, max_iter=100, eps=1e-6):
    median = np.mean(updates, axis=0)
    for _ in range(max_iter):
        distances = np.linalg.norm(updates - median, axis=1)
        distances = np.maximum(distances, eps)
        weights = 1.0 / distances
        weights = weights / np.sum(weights)
        new_median = np.sum(updates * weights[:, np.newaxis], axis=0)
        if np.linalg.norm(new_median - median) < eps:
            break
        median = new_median
    return median
```

**集成方式**：
- 每个server独立使用GeoMed聚合其收到的client updates
- Client端直接平均收到的server结果（无client-side defense）
- 这与Krum/FLTrust的适配方式相同

---

### 任务2：更新实验参数配置

**旧参数**：
```python
alpha = 0.25
beta = 0.25
gamma = 1.0
```

**新参数**：
```python
alpha = 0.35
beta = 0.35
gamma = 1.0
```

**理由**：当mc=0.1, mp=0.1时，α=0.35是满足理论safe region的最宽松配置。

需要更新的实验：
- Main Results (所有攻击类型)
- Hybrid Attack
- Data Distribution
- Ablation Study

---

### 任务3：新增Defense Configuration实验

这是一个全新的实验模块，用于验证BR-FL的理论特性。

#### 实验A：Clipping Rate vs Attack Intensity

```python
# 配置
configs = [
    {'alpha': 0.25, 'beta': 0.25},
    {'alpha': 0.35, 'beta': 0.35},
    {'alpha': 0.45, 'beta': 0.45},
]

# 攻击强度
attack_levels = [
    {'mc': 0.05, 'mp': 0.05},
    {'mc': 0.10, 'mp': 0.10},
    {'mc': 0.15, 'mp': 0.10},
    {'mc': 0.10, 'mp': 0.15},
    {'mc': 0.15, 'mp': 0.15},
    {'mc': 0.20, 'mp': 0.20},
]

# gamma固定为1.0
# attack固定为'Random'
# dataset: 'CIFAR-10'
```

输出：3×6的accuracy矩阵，用于生成热力图

#### 实验B：γ的影响

```python
mc, mp = 0.1, 0.1
alpha, beta = 0.35, 0.40
gamma_values = [0.3, 0.5, 0.7, 1.0]
```

输出：每个γ值的accuracy曲线 + 通信开销

#### 实验C：Threshold验证

```python
# 最保守配置
alpha, beta, gamma = 0.45, 0.45, 1.0

# 测试点
test_points = [
    (0.10, 0.10),  # 应该成功
    (0.15, 0.10),  # 应该成功
    (0.10, 0.20),  # 应该成功
    (0.15, 0.15),  # 应该成功
    (0.20, 0.15),  # 应该失败
    (0.15, 0.20),  # 应该失败
    (0.20, 0.20),  # 应该失败
]
```

输出：每个测试点的accuracy曲线，标注哪些在threshold内/外

---

## 代码结构期望

请先了解现有代码结构，然后：

1. **找到defense/aggregation相关代码**，添加GeoMed
2. **找到实验配置文件**，更新默认参数
3. **找到实验脚本**，添加defense_config实验
4. **找到plotting代码**，添加热力图绘制功能

---

## 注意事项

1. **保持接口一致**：GeoMed的接口应该与现有defense方法一致
2. **参数传递**：确保alpha, beta, gamma可以从命令行或配置文件传入
3. **结果保存**：实验结果需要保存为可复现的格式（.npy或.csv）
4. **日志记录**：打印关键信息方便调试

---

## 验证标准

实验结果应该符合以下预期：

1. **GeoMed性能**：应该比FedAvg好，但比BR-FL差（尤其在双侧攻击下）
2. **Defense Config实验A**：
   - α=0.25时，mc=0.1, mp=0.1应该失败
   - α=0.35时，mc=0.1, mp=0.1应该成功
   - α=0.45时，所有轻中度攻击应该成功
3. **Threshold验证**：超出⌊2mcP⌋ + mpP < P/2的测试点应该失败
