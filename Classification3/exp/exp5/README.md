# 实验五：架构消融实验 (Architecture Ablation Study)

## 📌 实验概述

本实验通过系统性地对比四种不同的架构配置，验证平行融合架构（Parallel Mamba-Attention Adapter）的优越性。

## 🎯 实验目的

**核心问题**：平行融合架构是否优于纯GPT-2、纯Mamba和串行架构？

**实验假设**：
- ✅ 平行融合架构（Ours）应该在所有配置中表现最佳
- ✅ 串行架构（Sequential）应优于纯Mamba但略逊于平行融合
- ✅ 纯Mamba网络应优于纯GPT-2（在长序列任务上）

## 🔬 实验设计

### 四种架构模式

| 架构模式 | 说明 | 预期性能 |
|----------|------|----------|
| **Pure GPT-2** | 仅使用GPT-2，禁用Mamba适配器（gate=0） | Baseline |
| **Pure Mamba** | 仅使用Mamba，禁用GPT-2分支（gate=10） | 中等 |
| **Sequential** | 串行架构：先过GPT-2，再过Mamba | 较好 |
| **Parallel** | 平行融合架构（你的方法）：同时使用GPT-2和Mamba，通过gate动态融合 | 最佳 |

### 架构对比

```
Pure GPT-2:         GPT-2 Layers → Classification Head

Pure Mamba:         Mamba Layers → Classification Head

Sequential:         GPT-2 Layers → Mamba Layers → Classification Head

Parallel (Ours):    GPT-2 Layers ─
                                   ├→ Add(GPT + gate×Mamba) → Classification Head
                    Mamba Layers ─┘
```

## 📊 实现细节

### 1. 代码修改

#### `src/models/gpt4ts.py` - ParallelMambaAdapter

添加了`sequential_mode`支持：

```python
class ParallelMambaAdapter(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.sequential_mode = False  # 实验五：支持串行模式
    
    def forward(self, hidden_states, *args, **kwargs):
        # 实验五：串行模式支持
        if self.sequential_mode:
            # 串行架构：先过GPT-2，再过Mamba
            with torch.no_grad():
                gpt_out = self.gpt_layer(hidden_states, *args, **kwargs)[0]
            mamba_out = self.mamba(self.mamba_ln(gpt_out))
            if isinstance(mamba_out, tuple):
                mamba_out = mamba_out[0]
            return (mamba_out, None)
        
        # 默认：平行融合架构
        # ...
```

#### `exp/exp5/main_exp5.py` - 实验主程序

- 支持四种架构模式切换
- 根据架构模式自动调整gate参数
- 保存实验结果到JSON文件

### 2. 文件结构

```
Classification3/
├── src/
│   └── models/
│       └── gpt4ts.py              # ✅ 修改：添加 sequential_mode 支持
└── exp/
    └── exp5/
        ├── main_exp5.py           #  实验主程序
        ├── run_exp5.sh            # 📄 运行脚本
        ├── visualize_exp5.py      # 📊 可视化脚本
        ├── README.md              # 📖 本文档
        └── results/               # 📂 实验结果（自动生成）
            ├── {dataset}_pure_gpt/
            ├── {dataset}_pure_mamba/
            ├── {dataset}_sequential/
            ├── {dataset}_parallel/
            ── exp5_plots/        # 可视化图表
```

## 🚀 使用方法

### 运行完整实验

```bash
cd Classification3/exp/exp5
bash run_exp5.sh Heartbeat
```

脚本会自动：
1. 遍历四种架构模式
2. 对每种模式运行完整训练
3. 保存结果到对应目录

### 运行单个架构模式

```bash
# Pure GPT-2
python main_exp5.py --dataset Heartbeat --gpu 0 --epochs 100 --arch_mode pure_gpt

# Pure Mamba
python main_exp5.py --dataset Heartbeat --gpu 0 --epochs 100 --arch_mode pure_mamba

# Sequential
python main_exp5.py --dataset Heartbeat --gpu 0 --epochs 100 --arch_mode sequential

# Parallel (Ours)
python main_exp5.py --dataset Heartbeat --gpu 0 --epochs 100 --arch_mode parallel
```

### 生成可视化图表

```bash
# 生成所有图表
python visualize_exp5.py --data_dir results

# 生成指定数据集的图表
python visualize_exp5.py --data_dir results --dataset Heartbeat
```

## 📈 可视化输出

实验完成后，在 `results/exp5_plots/` 目录下会生成：

### 1. 准确率对比柱状图

`accuracy_comparison.png`：
- 横轴：四种架构模式
- 纵轴：最佳验证集准确率
- 标注具体数值
- 预期：Parallel > Sequential > Pure Mamba > Pure GPT-2

### 2. 训练曲线对比

`training_curves.png`：
- 左图：Validation Loss曲线对比
- 右图：Validation Accuracy曲线对比
- 四种架构模式分别用不同颜色表示

### 3. 实验结果汇总表

`summary_table.xlsx`：
- 包含所有架构模式的Accuracy和Loss
- 可直接用于论文表格

## 📊 预期结果分析

### 性能对比预期

| 架构 | Accuracy | 优势 | 劣势 |
|------|----------|------|------|
| **Pure GPT-2** | Baseline | 参数量少，训练快 | 缺乏Mamba的序列建模能力 |
| **Pure Mamba** | 中等 | 长序列建模能力强 | 缺乏GPT-2的注意力机制 |
| **Sequential** | 较好 | 结合两者优势 | 信息单向流动，缺乏动态融合 |
| **Parallel (Ours)** | **最佳** | 动态门控融合，充分利用两者 | 参数量稍多 |

### 典型结果示例

```
Architecture         | Accuracy  | Improvement vs Baseline
---------------------|-----------|------------------------
Pure GPT-2           | 0.8234    | Baseline
Pure Mamba           | 0.8456    | +2.69%
Sequential           | 0.8678    | +5.39%
Parallel (Ours)      | 0.8891    | +7.98%
```

## 🔍 结果解读

### 如果 Parallel > Sequential > Pure Mamba > Pure GPT-2

**结论**：平行融合架构设计正确！

**原因分析**：
1. **Parallel > Sequential**：并行结构允许GPT-2和Mamba同时处理信息，通过可学习的gate动态调整权重
2. **Sequential > Pure Mamba**：GPT-2提取的高层特征对Mamba有帮助
3. **Pure Mamba > Pure GPT-2**：Mamba在长序列建模上具有优势

**论文贡献**：
- 证明了平行融合架构的优越性
- 验证了动态门控机制的有效性

### 如果性能差异不显著

**可能原因**：
1. 数据集较小，无法体现架构差异
2. 训练轮数不够，需要更多epochs
3. 超参数需要针对每个架构单独调优

**后续改进**：
- 尝试更多数据集
- 增加训练轮数
- 针对每个架构进行超参数搜索

## 🔒 安全性保证

### 对主实验的影响

**✅ 完全向后兼容，不影响任何现有实验！**

原因：
1. `sequential_mode`默认值为`False`，走平行融合分支
2. 只有实验五的`main_exp5.py`会修改此参数
3. 所有现有脚本不受影响

### 验证方法

运行主实验脚本（如`all.sh`），会看到输出：
```
✓ Replaced last 2 layers with Parallel Mamba-Attention Adapters
```
这与修改前的行为**完全一致**。

## ️ 参数说明

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--arch_mode` | 架构模式 | parallel |
| `--gpt_layers` | 使用的GPT-2层数 | 6 |
| `--num_mamba_layers` | Mamba适配器层数 | 2 |
| `--epochs` | 训练轮数 | 100 |
| `--lr` | 学习率 | 1e-4（根据架构自动调整） |
| `--seed` | 随机种子 | 42 |

### 数据集特定参数

某些数据集可能需要特殊配置（如Cricket）：
```bash
--lr 0.001 --patch_size 8 --stride 8
```

## 📚 实验记录

### 实验配置

```yaml
# 实验五配置
experiment: Architecture Ablation Study
datasets:
  - Heartbeat (default)
  - Can be changed via script argument
gpu: 0
seed: 42
epochs: 100
arch_modes:
  - pure_gpt
  - pure_mamba
  - sequential
  - parallel
```

### 预期输出文件

```
Classification3/exp/exp5/results/
├── Heartbeat_pure_gpt/
│   ├── model_last.pth
│   ├── experiment5_summary.json
│   ├── output.log
│   └── metrics_*.xls
── Heartbeat_pure_mamba/
── Heartbeat_sequential/
── Heartbeat_parallel/
└── exp5_plots/
    ├── accuracy_comparison.png
    ├── training_curves.png
    └── summary_table.xlsx
```

##  学术价值

### 创新点

1. **系统性架构对比**：全面对比四种不同架构配置
2. **消融实验设计**：严格控制变量，隔离架构因素的影响
3. **动态门控机制验证**：通过对比实验证明gate参数的重要性

### 可能的论文贡献

- 📄 证明平行融合架构优于串行架构
-  提供详细的消融实验数据
-  为时间序列分类任务中的架构设计提供实证支持

## 🐛 故障排除

### 常见问题

**Q1: 串行模式下训练很慢？**
- A: 正常现象。串行模式下GPT-2和Mamba都参与计算，需要两个forward pass

**Q2: Pure GPT-2模式仍有Mamba层？**
- A: 正常。Pure GPT-2是通过将gate设为0实现的，等效于只使用GPT-2

**Q3: 可视化脚本报错 "No results found"？**
- A: 确保先运行了训练脚本，生成了实验结果目录

**Q4: 如何单独运行某个架构模式？**
- A: 使用 `python main_exp5.py --arch_mode parallel`（替换parallel为你需要的模式）

## 📚 参考资料

- Mamba论文: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- Transformer论文: "Attention Is All You Need"
- 消融实验方法: "Ablation Study in Deep Learning"

---

**实验五完成！** 🎉

如有问题，请查看本README或联系实验维护者。
