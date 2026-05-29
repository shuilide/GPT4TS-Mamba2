# 实验3：预训练知识迁移验证 (Pre-trained Knowledge Transfer Study)

## 📌 实验概述

本实验旨在验证加载 NLP 领域预训练的 GPT-2 权重对于时间序列分类任务的实际帮助，证明模型性能提升不仅仅是因为架构设计优秀，而是预训练知识确实提供了有价值的先验信息。

## 🎯 实验目的

**核心问题**：预训练的 GPT-2 权重对时间序列任务是否真的有帮助？

**实验假设**：即使 GPT-2 的主体参数被冻结（不更新），预训练权重仍然能提供比随机初始化更好的特征表示能力，从而带来：
- ✅ 更快的收敛速度
- ✅ 更高的最终准确率
- ✅ 更稳定的训练过程

## 🔬 实验设计

### 对比方案

| 实验组 | GPT-2 初始化方式 | 冻结策略 | 训练参数比例 |
|--------|------------------|----------|--------------|
| **Pre-trained** | 加载 NLP 预训练权重 | 冻结 90% 参数 | 只训练 10%（LayerNorm、Mamba、分类头） |
| **From Scratch** | 随机初始化 | 冻结 90% 参数 | 只训练 10%（LayerNorm、Mamba、分类头） |

### 关键控制变量

为了确保对比的公平性，以下参数在两组实验中**完全相同**：

- ✅ **冻结策略**：都冻结 GPT-2 的 Attention 和 MLP 层（约 90% 参数）
- ✅ **可训练参数**：都只训练 LayerNorm、位置编码、Mamba 适配器、Gate、分类头等（约 10% 参数）
- ✅ **训练速度**：由于训练参数相同，两组的训练速度几乎一致
- ✅ **优化器配置**：学习率、优化器类型、权重衰减等
- ✅ **训练轮数**：100 epochs
- ✅ **随机种子**：seed=42
- ✅ **数据集划分**：相同的训练集、验证集、测试集

**唯一变量**：GPT-2 的初始化方式（预训练 vs 随机）

### 为什么这样设计？

1. **排除架构因素干扰**：通过冻结 90% 参数，确保性能差异不是来自架构设计
2. **控制训练成本**：只训练 10% 参数，训练速度快，实验周期短
3. **验证预训练价值**：即使不更新参数，预训练权重是否仍然有用？

## 📊 数据集

实验选择 5 个 UCR 时间序列分类数据集：

| 数据集 | 序列长度 | 特征维度 | 类别数 | 特殊配置 |
|--------|----------|----------|--------|----------|
| **ArticularyWordRecognition** | 144 | 9 | 25 | 默认参数 |
| **AtrialFibrillation** | 640 | 2 | 3 | 默认参数 |
| **BasicMotions** | 100 | 6 | 4 | 默认参数 |
| **CharacterTrajectories** | 182 | 3 | 20 | 默认参数 |
| **Cricket** | 1197 | 3 | 12 | lr=0.001, patch=8, stride=8 |

## ️ 实现细节

### 1. 代码修改

#### `gpt4ts.py` - 模型初始化逻辑

```python
# 实验3：支持随机初始化对比（Pre-trained vs From Scratch）
use_pretrained = not config.get('no_pretrained', False)

if use_pretrained:
    # 加载预训练权重
    self.gpt2 = GPT2Model.from_pretrained('gpt2', ...)
    print("✓ Loading pre-trained GPT-2 weights (Frozen)")
else:
    # 随机初始化
    from transformers import GPT2Config
    gpt2_config = GPT2Config(
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072
    )
    self.gpt2 = GPT2Model(gpt2_config)
    print("✓ Using randomly initialized GPT-2 (From Scratch, Frozen)")
```

#### `options.py` - 新增参数

```python
self.parser.add_argument('--no_pretrained', action='store_true', 
                        help='Disable pre-trained GPT-2 weights (use random initialization)')
```

### 2. 参数冻结策略

当前模型的参数冻结逻辑（第 187-191 行）：

```python
for name, param in self.gpt2.named_parameters():
    if 'ln' in name or 'wpe' in name or 'mamba' in name.lower() or 'gate' in name.lower():
        param.requires_grad = True  # 可训练
    else:
        param.requires_grad = False  # 冻结
```

**冻结的参数（90%）**：
- ❄️ GPT-2 Attention 层（Q/K/V/Output projections）
- ❄️ GPT-2 MLP 层
- ❄️ Word Token Embedding (wte)

**可训练的参数（10%）**：
- 🔓 LayerNorm 参数（ln）
- 🔓 位置编码（wpe）
- 🔓 Mamba 适配器层
- 🔓 Gate 参数
- 🔓 Prompt Generator
- 🔓 分类头（out_layer）
- 🔓 Attention Pooling 组件

### 3. 文件结构

```
Classification2/
├── src/
│   ├── models/
│   │   └── gpt4ts.py           # ✅ 修改：添加 no_pretrained 支持
│   └── options.py              # ✅ 修改：添加 --no_pretrained 参数
└── exp/
    └── exp3/
        ├── run_exp3_pretrained.sh      # 📄 实验脚本
        ├── visualize_convergence.py    # 📊 可视化脚本
        └── README.md                   # 📖 本文档
```

## 🚀 使用方法

### 运行完整实验

```bash
cd Classification2/exp/exp3
bash run_exp3_pretrained.sh
```

脚本会自动：
1. 遍历 5 个数据集
2. 对每个数据集运行两组实验（Pre-trained vs From Scratch）
3. 训练完成后自动生成可视化图表

### 运行单个数据集

```bash
# Pre-trained 版本
python ../../src/main.py \
    --output_dir results \
    --name ArticularyWordRecognition_pretrained \
    --data_dir ../../datasets/ArticularyWordRecognition \
    --gpu 2 --seed 42 --epochs 100 \
    --gpt_layers 6 --num_mamba_layers 2 \
    --no_attn_pooling

# From Scratch 版本（添加 --no_pretrained）
python ../../src/main.py \
    --output_dir results \
    --name ArticularyWordRecognition_scratch \
    --data_dir ../../datasets/ArticularyWordRecognition \
    --gpu 2 --seed 42 --epochs 100 \
    --gpt_layers 6 --num_mamba_layers 2 \
    --no_attn_pooling \
    --no_pretrained  # ← 关键参数
```

### 生成可视化图表

```bash
# 生成所有数据集的图表
python visualize_convergence.py --data_dir results

# 只生成指定数据集的图表
python visualize_convergence.py --data_dir results --dataset Cricket
```

## 📈 可视化输出

实验完成后，在 `results/exp3_plots/` 目录下会生成：

### 1. 收敛曲线对比图

每个数据集生成一张图（如 `Cricket_convergence.png`），包含：

- **左图**：Loss 收敛曲线
  - 蓝色：Pre-trained (Train Loss + Val Loss)
  - 红色：From Scratch (Train Loss + Val Loss)
  
- **右图**：Accuracy 收敛曲线
  - 蓝色：Pre-trained Validation Accuracy
  - 红色：From Scratch Validation Accuracy

### 2. 最终 Accuracy 对比柱状图

`all_datasets_comparison.png`：
- 横轴：5 个数据集
- 纵轴：最佳验证集准确率
- 蓝色柱子：Pre-trained 最佳准确率
- 红色柱子：From Scratch 最佳准确率
- 柱子上标注具体数值

## 📊 预期结果分析

### 收敛速度对比

| 指标 | Pre-trained | From Scratch | 原因 |
|------|-------------|--------------|------|
| **初期 Loss** | 较低 | 较高 | 预训练权重已有良好的特征提取能力 |
| **收敛速度** | 较快 | 较慢 | 预训练知识加速了优化过程 |
| **稳定 epoch** | 较早（~50） | 较晚（~80） | 预训练提供了更好的初始化点 |

### 最终性能对比

| 指标 | Pre-trained | From Scratch | 差异 |
|------|-------------|--------------|------|
| **最佳 Accuracy** | 较高 | 较低 | 预训练知识提升了表示能力 |
| **稳定性** | 更稳定 | 可能波动 | 预训练权重更鲁棒 |

### 典型曲线特征

**Pre-trained 曲线**：
```
Loss:    快速下降 → 平稳收敛
Accuracy: 快速上升 → 稳定在高位
```

**From Scratch 曲线**：
```
Loss:    缓慢下降 → 后期才收敛
Accuracy: 缓慢上升 → 最终略低于预训练
```

## 🔍 结果解读

### 如果 Pre-trained 显著优于 From Scratch

**结论**：预训练知识即使冻结也能提供帮助！

**原因分析**：
1. 预训练的 Attention 机制已经学会了如何关注重要信息
2. 预训练的 MLP 已经有了强大的非线性变换能力
3. 这些"冻结"的参数虽然不更新，但提供了高质量的初始特征表示

**论文贡献**：
- 证明了迁移学习在时间序列任务中的价值
- 不仅仅是架构好，预训练知识确实有用

### 如果两者性能相近

**可能原因**：
1. 时间序列任务与 NLP 任务差异较大，预训练知识迁移有限
2. 冻结策略过于严格，限制了预训练知识的发挥
3. 数据集规模较小，随机初始化也能快速学习

**后续改进**：
- 尝试解冻更多层（如 20%、30% 参数）
- 使用在时间序列上预训练的模型（而非 NLP）

## ⚙️ 参数说明

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--no_pretrained` | 禁用预训练权重，使用随机初始化 | False（加载预训练） |
| `--gpt_layers` | 使用的 GPT-2 层数 | 6 |
| `--num_mamba_layers` | Mamba 适配器层数 | 2 |
| `--epochs` | 训练轮数 | 100 |
| `--lr` | 学习率 | 0.0005（Cricket: 0.001） |
| `--seed` | 随机种子 | 42 |

### 数据集特定参数

**Cricket 数据集**：
```bash
--lr 0.001 --patch_size 8 --stride 8
```

**其他数据集**：
```bash
--lr 0.0005 --patch_size 16 --stride 2
```

## 🔒 安全性保证

### 对主实验的影响

**✅ 完全向后兼容，不影响任何现有实验！**

原因：
1. `--no_pretrained` 是新增的可选参数
2. 默认行为不变（`use_pretrained=True`）
3. 所有现有脚本都不包含 `--no_pretrained`，都走预训练分支
4. 只有实验3的新脚本使用 `--no_pretrained`

### 验证方法

运行主实验脚本（如 `all.sh`），会看到输出：
```
✓ Loading pre-trained GPT-2 weights (Frozen)
```
这与修改前的行为**完全一致**。

##  实验记录

### 实验配置

```yaml
# 实验3配置
experiment: Pre-trained Knowledge Transfer Study
gpu: 2
seed: 42
epochs: 100
datasets:
  - ArticularyWordRecognition
  - AtrialFibrillation
  - BasicMotions
  - CharacterTrajectories
  - Cricket
```

### 预期输出文件

```
Classification2/results/
├── ArticularyWordRecognition_pretrained/  # 实验结果目录
├── ArticularyWordRecognition_scratch/
├── AtrialFibrillation_pretrained/
├── AtrialFibrillation_scratch/
├── ...（其他数据集）
├── exp3_records.xls                        # Excel 记录文件
└── exp3_plots/                             # 可视化图表
    ├── ArticularyWordRecognition_convergence.png
    ├── AtrialFibrillation_convergence.png
    ├── ...
    ── all_datasets_comparison.png
```

##  学术价值

### 创新点

1. **首次系统性验证**：在时间序列分类任务中验证 NLP 预训练知识的迁移价值
2. **控制变量设计**：通过冻结策略排除架构因素干扰
3. **高效实验方案**：只训练 10% 参数，快速验证假设

### 可能的论文贡献

- 📄 证明预训练知识迁移的有效性
- 📊 提供收敛曲线和性能对比的实证数据
-  为时间序列任务中使用预训练语言模型提供理论支持

## 🐛 故障排除

### 常见问题

**Q1: 运行时出现 `NameError: name 'x_std' is not defined`**
- A: 检查 `gpt4ts.py` 第 219-224 行，确保删除了 `x_std` 的同时也修改了 `torch.cat` 列表

**Q2: 可视化脚本报错 "Experiment directory not found"**
- A: 确保先运行了训练脚本，生成了实验结果目录

**Q3: 训练速度很慢**
- A: 确认 GPU 配置正确，当前配置应该很快（只训练 10% 参数）

**Q4: 如何单独查看某个数据集的结果？**
- A: 使用 `python visualize_convergence.py --data_dir results --dataset Cricket`

## 📚 参考资料

- GPT-2 论文: "Language Models are Unsupervised Multitask Learners"
- Mamba 论文: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- UCR 时间序列数据集: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

---

**实验3完成！** 🎉

如有问题，请查看本 README 或联系实验维护者。
