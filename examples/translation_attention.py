import numpy as np

# 机器翻译任务注意力模式示例
# 模拟源语言和目标语言token之间的特定关注关系

def get_translation_attention_example():
    """生成机器翻译注意力模式示例数据"""
    # 输入矩阵 (6 tokens, 4 dimensions)
    # 前3个是源语言token，后3个是目标语言token
    X = np.array([
        [1.0, 0.5, 0.2, 0.1],  # 源语言Token 0
        [0.8, 1.0, 0.3, 0.2],   # 源语言Token 1
        [0.2, 0.3, 1.0, 0.7],   # 源语言Token 2
        [0.1, 0.2, 0.8, 1.0],   # 目标语言Token 0
        [0.3, 0.4, 0.9, 0.8],   # 目标语言Token 1
        [0.5, 0.6, 0.7, 0.9]    # 目标语言Token 2
    ])
    
    # 预期的注意力权重模式
    # 目标语言token关注对应的源语言token
    expected_weights = np.array([
        [0.0, 0.0, 0.0, 0.8, 0.1, 0.1],  # 源Token 0主要被目标Token 0关注
        [0.0, 0.0, 0.0, 0.1, 0.8, 0.1],  # 源Token 1主要被目标Token 1关注
        [0.0, 0.0, 0.0, 0.1, 0.1, 0.8],  # 源Token 2主要被目标Token 2关注
        [0.7, 0.2, 0.1, 0.0, 0.0, 0.0],  # 目标Token 0主要关注源Token 0
        [0.2, 0.7, 0.1, 0.0, 0.0, 0.0],  # 目标Token 1主要关注源Token 1
        [0.1, 0.2, 0.7, 0.0, 0.0, 0.0]   # 目标Token 2主要关注源Token 2
    ])
    
    return {
        'input': X,
        'expected_weights': expected_weights,
        'description': '机器翻译注意力模式: 目标语言token关注对应的源语言token'
    }
