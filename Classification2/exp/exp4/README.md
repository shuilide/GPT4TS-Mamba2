# 实验四：模型可解释性可视化 (Interpretability & Visualization)

## 📖 实验概述

本实验旨在验证和展示模型内部决策机制的可解释性，对应论文方法部分 **1.4 节 注意力池化分类头**。

### 核心目的
1. **注意力权重可视化 (Attention Heatmap)**：证明模型能够精准关注时间序列中的关键片段（如异常点、动作发生段），忽略噪声。
2. **特征分布可视化 (t-SNE)**：证明引入 **Attention Pooling** 后，模型提取的特征具有更好的类簇分离度，优于传统的 Flatten (Adaptive Pooling) 方法。

---

##  目录结构

```
Classification2/exp/exp4/
├── README.md                 # 本说明文档
├── extract_data.py           # 数据提取脚本：从模型中提取注意力权重和特征向量
├── visualize_ex4.py          # 可视化脚本：生成热力图和 t-SNE 散点图
└── run_exp4.sh               # 主运行脚本：自动化整个流程
```

### 生成的文件
运行脚本后，会在当前目录下生成以下文件夹：
```
exp4/
├── data_attn/                # Attention Pooling 模式提取的数据
│   ├── features.npy          # 所有样本的池化特征 [N, D]
│   ├── labels.npy            # 对应的标签 [N]
│   ├── sample_attention.npy  # 第一个样本的注意力权重 [Num_Patches]
│   └── sample_raw_data.npy   # 第一个样本的原始波形 [Length, Channels]
├── data_flat/                # Flatten (Adaptive Pooling) 模式提取的数据
│   ├── features.npy
│   └── labels.npy
└── plots/                    # 最终生成的可视化图表
    ├── attention_heatmap.png # 波形与注意力权重叠加图
    └── tsne_comparison.png   # Flatten vs Attention 的 t-SNE 对比图
```

---

## 🛠️ 前置修改

为了支持可解释性分析，核心模型文件 `src/models/gpt4ts.py` 已经进行了以下修改：

1. **捕获注意力权重**：在 `forward` 函数中，当 `self.use_attn_pooling` 为 True 时，保存 `MultiHeadAttention` 输出的 `attn_weights` 到 `self.last_attention_weights`。
2. **捕获池化特征**：保存经过 LayerNorm 后的特征向量到 `self.last_pooled_feature`。
3. **动态开关**：通过设置 `model.return_attn = True` 来激活数据捕获功能，不影响正常训练时的性能。

---

##  运行指南

### 1. 准备训练好的模型
你需要一个已经训练完成且包含权重文件（`.pth`）和配置文件（`configuration.json`）的目录。
*例如：`../../results/BasicMotions_pretrained`*

### 2. 运行主脚本
进入 `exp4` 目录，并传入模型路径：

```bash
cd Classification2/exp/exp4
bash run_exp4.sh <训练好的模型目录路径>
```

**示例命令：**
```bash
bash run_exp4.sh ../../results/BasicMotions_pretrained
```

### 3. 脚本执行流程
`run_exp4.sh` 会自动完成以下步骤：
1. **解析配置**：从模型目录的 `configuration.json` 中读取数据集路径、Patch Size、Stride 等超参数。
2. **提取 Attention 数据**：加载模型，运行 `extract_data.py` 提取开启 Attention Pooling 时的特征和权重，保存到 `data_attn/`。
3. **提取 Flatten 数据**：加载模型，添加 `--no_attn_pooling` 参数，运行 `extract_data.py` 提取仅使用 Adaptive Pooling 时的特征，保存到 `data_flat/`。
4. **生成图表**：运行 `visualize_ex4.py`，读取上述数据并生成 `plots/` 下的两张核心图表。

---

## 📊 结果解读

### 1. Attention Heatmap (`attention_heatmap.png`)
* **黑色线条**：原始时间序列波形。
* **红色填充区域**：模型计算的注意力权重（归一化后）。
* **预期现象**：红色区域应高度覆盖波形变化剧烈或具有判别性的时间段（如心电图中异常 R 波，或动作识别中加速阶段），而在平稳段权重较低。

### 2. t-SNE Comparison (`tsne_comparison.png`)
* **左图 (Flatten)**：使用传统 Adaptive Pooling 提取的特征分布。
* **右图 (Attention Pooling)**：使用本文提出的注意力池化提取的特征分布。
* **预期现象**：右图中不同类别的散点（不同颜色）应呈现更清晰的聚类效果，类与类之间的边界更明确，证明 Attention Pooling 能提取更具判别力的全局特征。

---

## ⚙️ 依赖要求

除了项目原有的依赖外，本实验可视化部分需要：
* `scikit-learn` (用于 t-SNE 降维)
* `matplotlib` (用于绘图)

通常这些库在基础 PyTorch 环境中已包含。

---

## 📝 备注
* 脚本默认分析验证集（Validation Set）中的第一个样本用于绘制 Heatmap。
* t-SNE 会分析验证集中的所有样本以展示整体分布。
* 如果模型未开启 `use_attn_pooling`（即默认使用 Adaptive Pooling），则 Heatmap 可能无法生成（因为不存在 Query-Key Attention 机制），但 t-SNE 对比依然有效。
