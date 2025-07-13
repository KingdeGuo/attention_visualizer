import numpy as np
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.components import add_3d_vectors, add_3d_attention_weights

class TestVisualizationComponents:
    def test_add_3d_vectors(self):
        """测试3D向量可视化函数"""
        fig = go.Figure()
        vectors = np.random.rand(3, 3)  # 3个3维向量
        add_3d_vectors(fig, vectors, "测试向量")
        
        # 验证图形是否包含预期的trace类型
        assert any(trace.type == 'scatter3d' and 'markers' in trace.mode for trace in fig.data)  # 点
        assert sum(trace.type == 'scatter3d' and 'lines' in trace.mode for trace in fig.data) >= 3  # 线
        assert fig.layout.scene.xaxis.title.text == "测试向量 X"

    def test_add_3d_attention_weights(self):
        """测试3D注意力权重可视化函数"""
        fig = go.Figure()
        weights = np.random.rand(3, 3)  # 3x3注意力权重
        add_3d_attention_weights(fig, weights)
        
        # 验证图形是否包含surface trace
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Surface)
        assert fig.layout.scene.xaxis.title.text == "Query Position"
