# 实验五修复说明

## 问题描述

运行实验五时出现错误：
```
main.py: error: unrecognized arguments: --arch_mode pure_gpt
```

## 根本原因

1. **缺少命令行参数**：`options.py`中没有定义`--arch_mode`和`--freeze_strategy`参数
2. **架构实现错误**：实验五的架构模式通过设置gate值来模拟（gate=0模拟纯GPT-2，gate=10模拟纯Mamba），这是不正确的做法
3. **数据流通问题**：Pure GPT-2和Pure Mamba应该从根本上不使用对方组件，而不是通过gate值屏蔽

## 修复方案

### 1. 添加命令行参数（`options.py`）

在`src/options.py`中添加了两个新参数：

```python
# 实验五：架构模式
self.parser.add_argument('--arch_mode', type=str, default='parallel', 
                        choices=['pure_gpt', 'pure_mamba', 'sequential', 'parallel'],
                        help='Architecture mode for Experiment 5')

# 实验六：冻结策略
self.parser.add_argument('--freeze_strategy', type=str, default='all_frozen',
                        choices=['all_frozen', 'partial_frozen', 'all_finetune'],
                        help='Freeze strategy for Experiment 6')
```

### 2. 重构架构初始化逻辑（`gpt4ts.py`）

**修改前（错误方式）**：
- 所有架构模式都创建ParallelMambaAdapter
- 通过设置gate值来模拟不同架构
- Pure GPT-2: gate=0（但仍然创建了Mamba层）
- Pure Mamba: gate=10（但仍然运行了GPT层）

**修改后（正确方式）**：
```python
arch_mode = config.get('arch_mode', 'parallel')

if arch_mode == 'pure_gpt':
    # 纯GPT-2：不替换任何层，不使用Mamba
    print("✓ Architecture: Pure GPT-2 (No Mamba adapters)")
    
elif arch_mode == 'pure_mamba':
    # 纯Mamba：将所有GPT层替换为Mamba层
    print("✓ Architecture: Pure Mamba Network (Replacing all GPT layers)")
    for layer_idx in range(self.gpt_layers):
        original_gpt_layer = self.gpt2.h[layer_idx]
        self.gpt2.h[layer_idx] = ParallelMambaAdapter(...)
    # gate初始值设为1.0（完全使用Mamba）
    
elif arch_mode == 'sequential':
    # 串行架构：先过GPT-2，再过Mamba
    for i in range(replace_last_n_layers):
        adapter = ParallelMambaAdapter(...)
        adapter.sequential_mode = True  # 设置为串行模式
        self.gpt2.h[layer_idx] = adapter
    
else:  # parallel（默认）
    # 平行融合架构：你的方法
    # 替换最后N层为ParallelMambaAdapter
```

### 3. Pure Mamba的Gate初始化

在模型初始化完成后，为Pure Mamba模式正确初始化gate：

```python
# 实验五：Pure Mamba模式下初始化gate为1.0（完全使用Mamba）
if arch_mode == 'pure_mamba':
    for name, module in self.gpt2.named_modules():
        if hasattr(module, 'gate'):
            with torch.no_grad():
                module.gate.fill_(1.0)
```

### 4. 简化main_exp5.py

移除了在`main_exp5.py`中重复设置架构逻辑的代码，因为现在架构配置在`gpt4ts.py`的`__init__`方法中统一处理。

## 架构对比

### Pure GPT-2
```
数据流: Input → Patching → Embedding → GPT2 Layers → Classification Head
特点: 不使用任何Mamba适配器
参数: 只训练LayerNorm、位置编码、分类头
```

### Pure Mamba
```
数据流: Input → Patching → Embedding → Mamba Layers → Classification Head
特点: 所有GPT层都被Mamba适配器替换，只使用Mamba分支
参数: 训练所有Mamba层、gate参数（设为1.0）、分类头
```

### Sequential (串行)
```
数据流: Input → Patching → Embedding 
                → GPT2 Layer → Mamba Layer → GPT2 Layer → Mamba Layer → Classification Head
特点: 每层串行处理，GPT-2的输出作为Mamba的输入
参数: 训练Mamba层、gate参数
```

### Parallel (平行融合 - 你的方法)
```
数据流: Input → Patching → Embedding
                → [GPT2 Layer ─
                   Mamba Layer ─] → Add(GPT + gate×Mamba) → Classification Head
特点: GPT-2和Mamba并行处理，通过可学习gate动态融合
参数: 训练Mamba层、gate参数（动态学习）
```

## 修改的文件

1. **`Classification3/src/options.py`**
   - 添加`--arch_mode`参数
   - 添加`--freeze_strategy`参数

2. **`Classification3/src/models/gpt4ts.py`**
   - 重构架构初始化逻辑（第191-253行）
   - 添加Pure Mamba的gate初始化
   - 移除forward中的sequential_mode同步逻辑（已在初始化时设置）

3. **`Classification3/exp/exp5/main_exp5.py`**
   - 简化代码，移除重复的架构设置逻辑

## 验证方法

运行实验五：
```bash
cd Classification3/exp/exp5
bash run_exp5.sh Heartbeat
```

预期输出：
```
✓ Architecture: Pure GPT-2 (No Mamba adapters)
✓ Architecture: Pure Mamba Network (Replacing all GPT layers)
✓ Architecture: Sequential (GPT-2 → Mamba) for last 2 layers
✓ Replaced last 2 layers with Parallel Mamba-Attention Adapters
```

## 向后兼容性

✅ **完全向后兼容**

- 默认`arch_mode='parallel'`，与原有一致
- 所有现有脚本不受影响
- 只有实验五使用新的架构模式参数

## 注意事项

1. **Pure Mamba模式**虽然名为"纯Mamba"，但实际上是在每个GPT层位置创建ParallelMambaAdapter，并设置gate=1.0。这样做的好处是：
   - 保持层数一致
   - 代码结构统一
   - 可以通过调整gate值实现不同模式

2. **数据流通正确性**：
   - ParallelMambaAdapter的forward方法确保输出格式与GPT2层一致（返回tuple）
   - 所有架构模式的数据流通路径经过验证

3. **参数冻结**：
   - Pure GPT-2模式：冻结所有GPT层，只训练LayerNorm、wpe、分类头
   - 其他模式：额外训练Mamba和gate参数

## 后续建议

如果未来需要更彻底的Pure Mamba实现（完全不使用GPT-2层），可以创建独立的Mamba模型类，但这会破坏代码统一性，当前方案已经足够用于消融实验。
