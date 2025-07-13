import numpy as np

# 局部注意力模式示例
# 模拟序列中每个token主要关注附近token的情况

def get_local_attention_example():
    """生成局部注意力模式示例数据"""
    # 输入矩阵 (4 tokens, 4 dimensions)
    X = np.array([
        [1.0, 0.5, 0.2, 0.1],  # Token 0
        [0.8, 1.0, 0.3, 0.2],   # Token 1
        [0.2, 0.3, 1.0, 0.7],   # Token 2
        [0.1, 0.2, 0.8, 1.0]    # Token 3
    ])
    
    # 预期的注意力权重模式
    expected_weights = np.array([
        [0.7, 0.3, 0.0, 0.0],   # Token 0主要关注自己和Token 1
        [0.3, 0.5, 0.2, 0.0],   # Token 1关注附近token
        [0.0, 0.2, 0.5, 0.3],   # Token 2关注附近token
        [0.0, 0.0, 0.3, 0.7]    # Token 3主要关注自己和Token 2
    ])
    
    return {
        'input': X,
        'expected_weights': expected_weights,
        'description': '局部注意力模式: 每个token主要关注其附近的token'
    }
