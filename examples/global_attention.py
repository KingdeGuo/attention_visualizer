import numpy as np

# 全局注意力模式示例
# 模拟序列中每个token关注所有token的情况

def get_global_attention_example():
    """生成全局注意力模式示例数据"""
    # 输入矩阵 (4 tokens, 4 dimensions)
    X = np.array([
        [1.0, 0.5, 0.2, 0.1],  # Token 0
        [0.8, 1.0, 0.3, 0.2],   # Token 1
        [0.2, 0.3, 1.0, 0.7],   # Token 2
        [0.1, 0.2, 0.8, 1.0]    # Token 3
    ])
    
    # 均匀分布的全局注意力
    uniform_weights = np.full((4, 4), 0.25)
    
    # 特定模式的全局注意力(如关注特定token)
    focused_weights = np.array([
        [0.1, 0.1, 0.1, 0.7],   # Token 0主要关注Token 3
        [0.1, 0.1, 0.7, 0.1],   # Token 1主要关注Token 2
        [0.7, 0.1, 0.1, 0.1],   # Token 2主要关注Token 0
        [0.1, 0.7, 0.1, 0.1]    # Token 3主要关注Token 1
    ])
    
    return {
        'input': X,
        'uniform_weights': uniform_weights,
        'focused_weights': focused_weights,
        'description': '全局注意力模式: 每个token可以关注序列中的所有token'
    }
