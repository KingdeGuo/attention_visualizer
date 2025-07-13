import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention_visualizer import AttentionVisualizer

class TestVisualizerIntegration:
    def test_visualize_step_by_step(self):
        """测试完整的可视化流程"""
        visualizer = AttentionVisualizer(input_dim=4)
        X = np.random.rand(4, 4)  # 测试输入
        
        # 执行完整可视化流程
        figures = visualizer.visualize_step_by_step(X)
        
        # 验证返回的图形数量和步骤对应
        assert len(figures) == 6  # 6个步骤
        for fig in figures:
            assert fig is not None  # 每个图形都应有效
            
    def test_compute_attention_integration(self):
        """测试计算注意力与可视化的集成"""
        visualizer = AttentionVisualizer(input_dim=4)
        X = np.random.rand(4, 4)
        
        # 计算注意力结果
        results = visualizer.compute_attention(X)
        
        # 验证所有步骤结果都存在
        assert 'input' in results
        assert 'query' in results
        assert 'key' in results
        assert 'value' in results
        assert 'attention_scores' in results
        assert 'attention_weights' in results
        assert 'output' in results
