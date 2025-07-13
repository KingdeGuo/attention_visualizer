import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.attention import Attention

class TestAttention:
    def setup_method(self):
        self.attention = Attention(input_dim=4)
        self.X = np.random.rand(4, 4)  # 4个token，每个4维

    def test_compute_attention_shapes(self):
        """测试注意力计算各步骤输出的形状"""
        results = self.attention.compute_attention(self.X)
        
        assert results['input'].shape == (4, 4)
        assert results['query'].shape == (4, 4)
        assert results['key'].shape == (4, 4)
        assert results['value'].shape == (4, 4)
        assert results['attention_scores'].shape == (4, 4)
        assert results['attention_weights'].shape == (4, 4)
        assert results['output'].shape == (4, 4)

    def test_attention_weights_normalization(self):
        """测试注意力权重是否归一化"""
        results = self.attention.compute_attention(self.X)
        weights = results['attention_weights']
        
        # 检查每行是否和为1
        row_sums = np.sum(weights, axis=1)
        assert np.allclose(row_sums, np.ones(4))

    def test_output_range(self):
        """测试输出值范围是否合理"""
        results = self.attention.compute_attention(self.X)
        output = results['output']
        
        # 简单检查输出值是否在合理范围内
        assert np.all(output >= -10) and np.all(output <= 10)
