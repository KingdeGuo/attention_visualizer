import numpy as np

class Attention:
    def __init__(self, input_dim=4, num_heads=1):
        """初始化注意力机制
        
        参数:
            input_dim (int): 输入维度
            num_heads (int): 注意力头数
        """
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # 随机初始化query, key, value矩阵
        self.Wq = np.random.randn(input_dim, input_dim)
        self.Wk = np.random.randn(input_dim, input_dim)
        self.Wv = np.random.randn(input_dim, input_dim)
        
    def compute_attention(self, X):
        """计算注意力机制各步骤
        
        参数:
            X (np.array): 输入矩阵 shape=(seq_len, input_dim)
            
        返回:
            dict: 包含各步骤结果的字典
        """
        seq_len = X.shape[0]
        
        # 1. 计算query, key, value
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        
        # 2. 计算注意力分数
        attention_scores = Q @ K.T / np.sqrt(self.input_dim)
        
        # 3. softmax归一化
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)
        
        # 4. 加权求和
        output = attention_weights @ V
        
        return {
            'input': X,
            'query': Q,
            'key': K,
            'value': V,
            'attention_scores': attention_scores,
            'attention_weights': attention_weights,
            'output': output
        }
