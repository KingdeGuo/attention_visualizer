# Attention Visualizer

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)

## 理论背景

### 注意力机制核心公式

注意力机制的核心计算可以表示为：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- $Q$ (Query): 查询向量
- $K$ (Key): 键向量  
- $V$ (Value): 值向量
- $d_k$: 键向量的维度

### 计算步骤详解

1. **Query/Key/Value计算**:
   - 输入向量$X$通过线性变换得到$Q$, $K$, $V$:
     $$
     Q = XW_q, \quad K = XW_k, \quad V = XW_v
     $$

2. **注意力分数计算**:
   - 计算查询和键的点积并缩放:
     $$
     \text{scores} = \frac{QK^T}{\sqrt{d_k}}
     $$

3. **Softmax归一化**:
   - 对注意力分数进行softmax归一化得到权重:
     $$
     \text{weights} = \text{softmax}(\text{scores})
     $$

4. **加权求和**:
   - 用权重对值向量加权求和得到输出:
     $$
     \text{output} = \text{weights} \cdot V
     $$

### 多头注意力

多头注意力并行计算多组注意力并将结果拼接:
$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$
其中每个注意力头:
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

一个用于可视化Transformer注意力机制的工具，支持交互式3D可视化。

## 功能特性

- 逐步展示注意力机制计算过程
- 3D可视化Query/Key/Value向量
- 3D注意力权重热力图
- 命令行和Web两种交互方式
- 提供多种注意力模式示例(局部/全局/翻译任务)

## 安装

```bash
git clone https://github.com/yourusername/attention_visualizer.git
cd attention_visualizer
pip install -r requirements.txt
```

## 使用方式

### 命令行可视化
```python
from attention_visualizer import AttentionVisualizer

visualizer = AttentionVisualizer(input_dim=4)
X = np.random.rand(4, 4)  # 输入矩阵
figures = visualizer.visualize_step_by_step(X)
```

### Web可视化

推荐使用启动脚本运行：

```bash
# 添加执行权限
chmod +x run_app.sh

# 启动应用
./run_app.sh
```

应用启动后会自动打开浏览器访问 http://127.0.0.1:8050/

或者也可以手动运行：
```bash
PYTHONPATH=. python visualization/dash_app.py
```

## 示例说明

项目提供了三种典型的注意力模式示例：

1. **局部注意力** (`examples/local_attention.py`):
   - 模拟序列中每个token主要关注附近token的情况
   - 适用于处理局部依赖的任务(如文本分类)

2. **全局注意力** (`examples/global_attention.py`):
   - 模拟序列中每个token关注所有token的情况
   - 适用于需要全局信息的任务(如语言建模)

3. **翻译任务注意力** (`examples/translation_attention.py`):
   - 模拟源语言和目标语言token之间的特定关注关系
   - 适用于机器翻译等跨语言任务

## 项目结构

```
attention_visualizer/
├── core/               # 核心注意力计算逻辑
│   └── attention.py
├── examples/           # 注意力模式示例
│   ├── local_attention.py
│   ├── global_attention.py
│   └── translation_attention.py
├── visualization/      # 可视化组件
│   ├── components.py
│   └── dash_app.py
└── attention_visualizer.py  # 主入口
```

## 贡献指南

欢迎提交Issue和PR！请确保：
1. 代码符合PEP8规范
2. 添加相应的测试用例
3. 更新相关文档

## 许可证

MIT License
