# 实验二：Gate 参数训练轨迹追踪

## 实验目标

追踪 Mamba Adapter 中可学习参数 g（gate）在训练过程中的变化轨迹，验证以下假设：

- **长序列数据集**：g 值会显著变大（模型自主学会依赖 Mamba 处理长时序）
- **短序列数据集**：g 值可能较小（GPT attention 已足够处理）

## 文件说明

### 核心代码

1. **`gate_tracker.py`** - Gate 参数追踪工具
   - `GateTracker` 类：记录每个 epoch 的 gate 值
   - 自动保存 JSON 和 CSV 格式数据
   - 生成统计信息（初始值、最终值、变化幅度等）

2. **`main_exp2.py`** - 实验二专用训练脚本
   - 基于原有 `main.py` 扩展
   - 集成 gate 参数追踪功能
   - 初始化 gate 值为 0.1

3. **`visualize_gates.py`** - 可视化工具
   - 绘制 gate 参数随 epoch 的变化曲线
   - 生成初始值 vs 最终值对比柱状图
   - 支持命令行调用

### 运行脚本

4. **`run_exp2_gate_tracking.sh`** - 批量运行脚本
   - 自动运行 4 个数据集（短/中/长序列）
   - 用法: `bash run_exp2_gate_tracking.sh [gpu] [epochs] [data_dir]`

5. **`run_single_dataset.sh`** - 单数据集快速测试
   - 用法: `bash run_single_dataset.sh <dataset_name> [gpu] [epochs] [data_dir]`

## 快速开始

### 1. 单个数据集测试

```bash
# 测试短序列数据集
bash run_single_dataset.sh Heartbeat 0 50

# 测试长序列数据集
bash run_single_dataset.sh EthanolConcentration 0 50

# 自定义参数
bash run_single_dataset.sh JapaneseVowels 0 100 ../datasets/UCRArchive_2018
```

### 2. 批量运行多个数据集

```bash
# 使用默认配置运行所有 4 个数据集
bash run_exp2_gate_tracking.sh 0 50

# 指定 GPU 和 epoch 数
bash run_exp2_gate_tracking.sh 1 100 ../datasets/UCRArchive_2018
```

### 3. 手动运行

```bash
python main_exp2.py \
    --config ../../src/config.json \
    --data_dir ../../datasets/UCRArchive_2018 \
    --data_class Heartbeat \
    --task classification \
    --gpu 0 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --d_model 768 \
    --patch_size 64 \
    --stride 64 \
    --gpt_layers 6 \
    --num_mamba_layers 2 \
    --val_ratio 0.2 \
    --normalization standardization \
    --optimizer Adam \
    --key_metric accuracy \
    --name "exp2_heartbeat" \
    --track_gate \
    --no_timestamp \
    --output_dir ./results/heartbeat \
    --seed 42
```

## 输出结果

每个实验会在输出目录下生成 `gate_tracking/` 文件夹，包含：

```
gate_tracking/
── gate_history.json              # 完整追踪数据（JSON 格式）
├── gate_history.csv               # 追踪数据（CSV 格式，可导入 Excel）
├── gate_tracking_final.json       # 最终结果 + 统计信息
├── plots/
│   ├── gate_evolution.png         # Gate 参数变化曲线
│   ── gate_comparison.png        # 初始值 vs 最终值对比
└── experiment2_summary.json       # 实验总结报告
```

## 可视化

训练完成后自动生成可视化图表：

```bash
# 查看特定实验的可视化结果
python visualize_gates.py --data_dir ./results/heartbeat/exp2_heartbeat

# 直接显示图表（不保存）
python visualize_gates.py --data_dir ./results/heartbeat/exp2_heartbeat --show
```

## 数据分析

### JSON 数据结构

`gate_tracking_final.json` 包含以下信息：

```json
{
  "epochs": [0, 1, 2, ...],
  "gate_values": {
    "h.4": [0.1, 0.12, 0.15, ...],
    "h.5": [0.1, 0.11, 0.14, ...]
  },
  "statistics": {
    "h.4": {
      "initial": 0.1,
      "final": 0.85,
      "change": 0.75,
      "change_percent": 750.0,
      "min": 0.1,
      "max": 0.87,
      "mean": 0.52,
      "std": 0.23
    }
  }
}
```

### 预期结果分析

| 数据集类型 | 序列长度 | 预期 gate 变化 | 理论解释 |
|-----------|---------|--------------|---------|
| Heartbeat | 短 (405) | 较小 (0.1 → 0.2~0.4) | GPT attention 已足够 |
| JapaneseVowels | 中 (1280) | 中等 (0.1 → 0.5~0.7) | 部分依赖 Mamba |
| EthanolConcentration | 长 (1751) | 较大 (0.1 → 0.7~1.0) | 主要依赖 Mamba |
| PEMS-SF | 长 (1440) | 较大 (0.1 → 0.6~0.9) | 主要依赖 Mamba |

## 论文写作建议

### 理论深度增强点

1. **自适应性证明**：gate 参数从 0.1 增长到较大值，证明模型能够自主学习何时依赖 Mamba
2. **序列长度相关性**：gate 值与序列长度呈正相关，验证 Mamba 对长序列的优势
3. **动态融合机制**：gate 参数的变化轨迹展示了模型在不同训练阶段的策略调整

### 可视化建议

- 使用折线图展示 gate 参数随 epoch 的变化趋势
- 使用柱状图对比不同数据集的最终 gate 值
- 在论文中展示 gate 值与序列长度的关系图

## 注意事项

1. **显存要求**：实验二与正常训练相同，需要约 8-12GB 显存
2. **训练时间**：50 epochs 约需 30-60 分钟（取决于数据集大小）
3. **随机种子**：建议设置 `--seed 42` 保证实验可复现性
4. **参数初始化**：所有 gate 参数初始化为 0.1（90% GPT, 10% Mamba）

## 故障排查

### Gate 值不变化

- 检查 `--track_gate` 参数是否启用
- 确认 Mamba adapter 层已正确替换（查看日志）
- 验证 `num_mamba_layers` 参数设置

### 可视化失败

- 安装 matplotlib: `pip install matplotlib`
- 检查数据文件是否完整生成
- 查看日志文件确认训练是否正常完成

## 参考文献

- 公式 (17): 平行 Mamba-Attention 适配器模块
- Gate 参数初始化: 0.1（偏向 GPT，避免初期过度依赖 Mamba）
- 动态融合: `output = gpt_out + gate * mamba_out`
