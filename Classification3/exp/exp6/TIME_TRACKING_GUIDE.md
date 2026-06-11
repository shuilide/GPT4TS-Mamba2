# 实验六：冻结策略效率分析使用指南 ⏱️

## 📌 更新内容

本次更新为实验六添加了**训练时间记录和可视化功能**，可以直观展示不同冻结策略的效率对比！

### 新增功能

1. ✅ **自动记录训练时间**：`main_exp6.py` 会自动记录总训练时间和平均每 epoch 时间
2. ✅ **训练时间对比图**：柱状图展示三种策略的总训练时间
3. ✅ **效率散点图**：准确率 vs 训练时间，直观展示性价比
4. ✅ **汇总表增强**：Excel 表格中包含时间信息

---

## 🚀 快速开始

### 方法一：使用真实数据运行

```bash
# 1. 进入实验六目录
cd Classification3/exp/exp6

# 2. 运行完整的冻结策略对比实验（以 Heartbeat 数据集为例）
bash run_exp6.sh Heartbeat

# 3. 生成可视化图表（包含训练时间对比）
python visualize_exp6.py --data_dir results --output_dir plots
```

### 方法二：使用模拟数据测试（推荐先测试）

```bash
# 1. 进入实验六目录
cd Classification3/exp/exp6

# 2. 创建模拟数据
python test_time_tracking.py

# 3. 使用模拟数据生成可视化
python visualize_exp6.py --data_dir test_results --output_dir test_results/plots

# 4. 查看生成的图表
# 图表保存在: test_results/plots/
```

---

## 📊 生成的可视化图表

运行 `visualize_exp6.py` 后，会生成以下 6 个文件：

### 1. accuracy_comparison.png
**准确率对比柱状图**
- 展示三种冻结策略的最终验证集准确率
- 柱子顶部标注具体数值

### 2. trainable_params_comparison.png
**可训练参数比例对比**
- 展示每种策略的可训练参数百分比
- All Frozen: ~2%
- Partial Frozen: ~15%
- All Finetune: 100%

### 3. training_time_comparison.png ⭐ 新增
**训练时间对比柱状图**
- 展示每种策略的总训练时间（小时或分钟）
- 柱子顶部标注具体时间
- 如果时间 < 1 小时，显示分钟；否则显示小时

**示例输出**：
```
┌──────────────────────────────────────┐
│ Training Time Comparison             │
├──────────────────────────────────────┤
│                                      │
│  2.0h ┤                    █        │
│       │                    █        │
│  1.5h ┤                    █        │
│       │          █         █        │
│  1.0h ┤          █         █        │
│       │          █         █        │
│  0.5h ┤  █       █         █        │
│       │  █       █         █        │
│  0.0h ┼──█───────█─────────█────    │
│       │All      Partial   All      │
│       │Frozen   Frozen   Finetune  │
└──────────────────────────────────────┘
```

### 4. efficiency_scatter.png ⭐ 新增
**效率散点图：准确率 vs 训练时间**
- X 轴：总训练时间（小时）
- Y 轴：验证集准确率
- 每个点代表一种冻结策略
- 左上角箭头指示理想方向（高准确率 + 短时间）

**示例输出**：
```
┌─────────────────────────────────────────────────┐
│ Efficiency Analysis: Accuracy vs Training Time  │
├─────────────────────────────────────────────────┤
│                                                 │
│ Acc  ↑                                          │
│ 0.85 ┤                              ●           │
│      │                         All Finetune     │
│      │                      (高准确率, 长时间)    │
│ 0.80 ┤                  ●                       │
│      │            Partial Frozen                │
│      │               (平衡选择)                  │
│ 0.75 ┤  ●                                       │
│      │All Frozen                                │
│      │(快速训练, 较低准确率)                      │
│      │                                          │
│      └────────────────────────→ Time            │
│        快 ←──────────────→ 慢                   │
│                                                 │
│ 💡 理想位置：左上角 (↑Acc, ↓Time)                │
└─────────────────────────────────────────────────┘
```

### 5. training_curves.png
**训练曲线对比**
- 左图：Validation Loss 随 Epoch 变化
- 右图：Validation Accuracy 随 Epoch 变化
- 三条不同颜色的曲线代表三种策略

### 6. summary_table.xlsx
**汇总表格（Excel 格式）**
包含以下列：
- Freeze Strategy: 冻结策略名称
- Accuracy: 验证集准确率
- Loss: 验证集损失
- Total Params: 总参数量
- Trainable Params: 可训练参数量
- Trainable Ratio (%): 可训练参数比例
- **Total Time (hours)** ⭐: 总训练时间（小时）
- **Avg Time/Epoch (s)** ⭐: 平均每 epoch 时间（秒）

**示例表格**：
```
┌──────────────┬──────────┬──────┬────────────┬─────────────┬─────────────────┬──────────────────┬─────────────────┐
│Freeze Strate │Accuracy  │ Loss │Total Params│Trainable Par│Trainable Ratio %│Total Time (hours)│Avg Time/Epoch(s)│
├──────────────┼──────────┼──────┼────────────┼─────────────┼─────────────────┼──────────────────┼─────────────────┤
│ All Frozen   │  0.7523  │0.6234│  85194496  │   1703890   │      2.00       │      0.500       │      36.00      │
│Partial Frozen│  0.8012  │0.5123│  85194496  │  12345678   │     14.49       │      1.000       │      72.00      │
│All Finetune  │  0.8234  │0.4567│  85194496  │  85194496   │    100.00       │      2.000       │     144.00      │
└──────────────┴──────────┴──────┴────────────┴─────────────┴─────────────────┴──────────────────┴─────────────────┘
```

---

## 🔍 代码修改说明

### 1. main_exp6.py 修改

#### 添加时间记录变量（第120-121行）
```python
total_epoch_time = 0
total_start_time = time.time()  # 记录总开始时间
```

#### 计算每 epoch 时间（第309-323行）
```python
epoch_start_time = time.time()
aggr_metrics_train = trainer.train_epoch(epoch)
epoch_runtime = time.time() - epoch_start_time
total_epoch_time += epoch_runtime
```

#### 保存时间信息到 JSON（第350-368行）
```python
# 计算总运行时间
total_training_time = time.time() - total_start_time
avg_epoch_time = total_epoch_time / config["epochs"] if config["epochs"] > 0 else 0

exp6_summary = {
    ...
    'total_training_time_seconds': total_training_time,
    'total_training_time_hours': total_training_time / 3600,
    'avg_epoch_time_seconds': avg_epoch_time,
    ...
}
```

#### 日志输出时间信息（第377-379行）
```python
logger.info(f"Total Training Time: {total_training_time/3600:.2f} hours ({total_training_time:.2f} seconds)")
logger.info(f"Average Time per Epoch: {avg_epoch_time:.2f} seconds")
```

### 2. visualize_exp6.py 修改

#### 新增方法：plot_training_time_comparison
绘制训练时间对比柱状图，自动判断显示小时还是分钟。

#### 新增方法：plot_efficiency_scatter
绘制效率散点图，展示准确率与训练时间的关系，标注理想方向。

#### 更新方法：generate_summary_table
在 Excel 表格中添加两列：
- `Total Time (hours)`: 总训练时间
- `Avg Time/Epoch (s)`: 平均每 epoch 时间

#### 更新方法：visualize_all
调用新增的两个绘图方法，并打印生成的文件列表。

---

## 💡 使用建议

### 如何解读效率散点图？

1. **左上角的点**：最佳选择（高准确率 + 短时间）
2. **右下角的点**：最差选择（低准确率 + 长时间）
3. **权衡取舍**：
   - 如果追求速度 → 选择左侧的点（All Frozen）
   - 如果追求准确率 → 选择上方的点（All Finetune）
   - 如果追求平衡 → 选择中间的点（Partial Frozen）

### 典型场景分析

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 小数据集 (< 1000 样本) | All Frozen | 防止过拟合，训练快速 |
| 中等数据集 (1000-5000) | Partial Frozen | 平衡准确率和效率 |
| 大数据集 (> 5000) | All Finetune | 充分利用数据，获得最佳性能 |
| 资源受限环境 | All Frozen | 显存占用少，训练时间短 |
| 学术研究/论文 | 三种都跑 | 全面对比，展示效率提升 |

---

## 📝 注意事项

1. **时间记录的准确性**：
   - 时间从模型初始化开始计算，包括数据加载、预处理等
   - 如果需要纯训练时间，可以修改 `total_start_time` 的位置

2. **GPU vs CPU**：
   - GPU 训练时间通常比 CPU 快 10-50 倍
   - 比较时请确保使用相同的硬件环境

3. **数据集大小影响**：
   - 小数据集：三种策略时间差异不明显
   - 大数据集：All Finetune 可能显著更慢

4. **学习率设置**：
   - All Finetune 建议使用更小的学习率（如 5e-5）
   - 其他策略可以使用默认学习率（1e-4）

---

## 🎯 预期效果

运行完成后，您应该看到：

1. **控制台输出**：
```
✓ Found results for all_frozen: Heartbeat_all_frozen
✓ Found results for partial_frozen: Heartbeat_partial_frozen
✓ Found results for all_finetune: Heartbeat_all_finetune
✓ Accuracy comparison plot saved to results/exp6_plots/accuracy_comparison.png
✓ Trainable parameters comparison plot saved to results/exp6_plots/trainable_params_comparison.png
✓ Training time comparison plot saved to results/exp6_plots/training_time_comparison.png
✓ Efficiency scatter plot saved to results/exp6_plots/efficiency_scatter.png
✓ Training curves plot saved to results/exp6_plots/training_curves.png
✓ Summary table saved to results/exp6_plots/summary_table.xlsx

========================================================================================================================
EXPERIMENT 6 RESULTS SUMMARY
========================================================================================================================
 Freeze Strategy  Accuracy    Loss  Total Params  Trainable Params  Trainable Ratio (%)  Total Time (hours)  Avg Time/Epoch (s)
      All Frozen    0.7523  0.6234    85194496         1703890                 2.000               0.500               36.00
  Partial Frozen    0.8012  0.5123    85194496        12345678                14.490               1.000               72.00
    All Finetune    0.8234  0.4567    85194496        85194496               100.000               2.000              144.00
========================================================================================================================

✓ All visualizations saved to: results/exp6_plots

Generated plots:
  1. accuracy_comparison.png         - 准确率对比
  2. trainable_params_comparison.png - 可训练参数比例
  3. training_time_comparison.png    - 训练时间对比
  4. efficiency_scatter.png          - 效率散点图 (准确率 vs 时间)
  5. training_curves.png             - 训练曲线对比
  6. summary_table.xlsx              - 汇总表格
```

2. **生成的文件**：
```
results/exp6_plots/
├── accuracy_comparison.png
├── trainable_params_comparison.png
├── training_time_comparison.png    ⭐ 新增
├── efficiency_scatter.png          ⭐ 新增
├── training_curves.png
└── summary_table.xlsx
```

---

## 🎉 总结

通过本次更新，您可以：
- ✅ 自动记录每种冻结策略的训练时间
- ✅ 直观对比不同策略的效率
- ✅ 通过散点图找到最佳性价比策略
- ✅ 在论文/报告中展示完整的效率分析

**祝您实验顺利！** 🚀
