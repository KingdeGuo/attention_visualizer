import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AttentionVisualizer:
    def __init__(self, input_dim=4, num_heads=1):
        """初始化注意力可视化器
        
        参数:
            input_dim (int): 输入维度
            num_heads (int): 注意力头数
        """
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.fig = None
        
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
    
    def visualize_step_by_step(self, X):
        """逐步可视化注意力机制
        
        参数:
            X (np.array): 输入矩阵 shape=(seq_len, input_dim)
            
        返回:
            list: 包含各步骤图形的列表
        """
        results = self.compute_attention(X)
        seq_len = X.shape[0]
        figures = []
        
        # 步骤1: 展示输入embeddings
        fig = go.Figure()
        self._add_3d_vectors(fig, results['input'], title='1. 输入Embeddings')
        fig.update_layout(title_text="第一步: 输入表示", title_x=0.5)
        fig.add_annotation(text="这是输入的词向量表示，每个点代表一个token",
                          xref="paper", yref="paper",
                          x=0.5, y=-0.15, showarrow=False)
        figures.append(fig)
        
        # 步骤2: 展示Query向量
        fig = go.Figure()
        self._add_3d_vectors(fig, results['query'], title='2. Query向量')
        fig.update_layout(title_text="第二步: 计算Query向量", title_x=0.5)
        fig.add_annotation(text="Query向量表示当前token想要关注什么",
                          xref="paper", yref="paper",
                          x=0.5, y=-0.15, showarrow=False)
        figures.append(fig)
        
        # 步骤3: 展示Key向量
        fig = go.Figure()
        self._add_3d_vectors(fig, results['key'], title='3. Key向量')
        fig.update_layout(title_text="第三步: 计算Key向量", title_x=0.5)
        fig.add_annotation(text="Key向量表示其他token可以提供什么信息",
                          xref="paper", yref="paper",
                          x=0.5, y=-0.15, showarrow=False)
        figures.append(fig)
        
        # 步骤4: 展示Value向量
        fig = go.Figure()
        self._add_3d_vectors(fig, results['value'], title='4. Value向量')
        fig.update_layout(title_text="第四步: 计算Value向量", title_x=0.5)
        fig.add_annotation(text="Value向量包含实际要传递的信息",
                          xref="paper", yref="paper",
                          x=0.5, y=-0.15, showarrow=False)
        figures.append(fig)
        
        # 步骤5: 展示注意力权重
        fig = go.Figure()
        self._add_3d_attention_weights(fig, results['attention_weights'])
        fig.update_layout(title_text="第五步: 计算注意力权重", title_x=0.5)
        fig.add_annotation(text="通过Query和Key的点积计算注意力分数，再经过softmax归一化",
                          xref="paper", yref="paper",
                          x=0.5, y=-0.15, showarrow=False)
        figures.append(fig)
        
        # 步骤6: 展示最终输出
        fig = go.Figure()
        self._add_3d_vectors(fig, results['output'], title='6. 输出')
        fig.update_layout(title_text="第六步: 加权求和得到输出", title_x=0.5)
        fig.add_annotation(text="用注意力权重对Value向量加权求和，得到最终输出",
                          xref="paper", yref="paper",
                          x=0.5, y=-0.15, showarrow=False)
        figures.append(fig)
        
        return figures
        
        # 1. 可视化输入embeddings
        self._add_3d_vectors(self.fig, results['input'], row=1, col=1, title='Input Embeddings')
        
        # 2. 可视化query向量
        self._add_3d_vectors(self.fig, results['query'], row=1, col=2, title='Query Vectors')
        
        # 3. 可视化key向量
        self._add_3d_vectors(self.fig, results['key'], row=1, col=3, title='Key Vectors')
        
        # 4. 可视化value向量
        self._add_3d_vectors(self.fig, results['value'], row=2, col=1, title='Value Vectors')
        
        # 5. 可视化注意力权重(3D柱状图)
        self._add_3d_attention_weights(self.fig, results['attention_weights'], row=2, col=2)
        
        # 6. 可视化输出
        self._add_3d_vectors(self.fig, results['output'], row=2, col=3, title='Output')
        
        self.fig.update_layout(
            height=1000,
            width=1500,
            title_text="Attention Mechanism Visualization",
            showlegend=False
        )
        
        return self.fig
    
    def _add_3d_vectors(self, fig, vectors, title=None):
        """添加3D向量可视化"""
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+text',
                marker=dict(size=8, color=np.arange(len(x))),
                text=[f"Token {i}" for i in range(len(x))],
                textposition="top center"
            )
        )
        
        # 添加从原点到每个向量的线
        for i in range(len(x)):
            fig.add_trace(
                go.Scatter3d(
                    x=[0, x[i]], y=[0, y[i]], z=[0, z[i]],
                    mode='lines',
                    line=dict(width=2, color='gray'),
                    showlegend=False
                )
            )
        
        if title:
            fig.update_layout(scene=dict(
                xaxis_title=f'{title} X',
                yaxis_title=f'{title} Y',
                zaxis_title=f'{title} Z'
            ))
    
    def _add_3d_attention_weights(self, fig, weights):
        """添加3D注意力权重可视化"""
        seq_len = weights.shape[0]
        x, y = np.meshgrid(range(seq_len), range(seq_len))
        
        fig.add_trace(
            go.Surface(
                z=weights,
                x=x,
                y=y,
                colorscale='Viridis',
                colorbar=dict(title='Attention Weight')
            )
        )
        
        fig.update_layout(scene=dict(
            xaxis_title='Query Position',
            yaxis_title='Key Position',
            zaxis_title='Weight'
        ))

# 示例用法
if __name__ == "__main__":
    # 创建可视化器
    visualizer = AttentionVisualizer(input_dim=4)
    
    # 生成随机输入 (4个token，每个token是4维向量)
    X = np.random.rand(4, 4)
    
    # 逐步可视化
    figures = visualizer.visualize_step_by_step(X)
    
    # 展示每个步骤
    for i, fig in enumerate(figures):
        print(f"\n正在展示第 {i+1} 步...")
        fig.show()
        input("按Enter键继续下一步...")
