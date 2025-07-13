import plotly.graph_objects as go
import numpy as np

def add_3d_vectors(fig, vectors, title=None, step=None):
    """添加3D向量可视化
    
    参数:
        fig (go.Figure): Plotly图形对象
        vectors (np.array): 向量矩阵
        title (str): 可选标题
        step (str): 当前计算步骤说明
    """
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    
    # 添加向量点
    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(size=8, color=np.arange(len(x))),
            text=[f"Token {i}" for i in range(len(x))],
            textposition="top center",
            hoverinfo='text',
            hovertext=[f"""
                <b>向量值</b>: {vectors[i].round(2)}<br>
                <b>步骤</b>: {step if step else '向量表示'}<br>
                <b>计算公式</b>: {'Q = XW<sub>q</sub>' if title=='Query' else 
                                'K = XW<sub>k</sub>' if title=='Key' else 
                                'V = XW<sub>v</sub>' if title=='Value' else 
                                '线性变换'}
            """ for i in range(len(x))]
        )
    )
    
    # 添加从原点到每个向量的线
    for i in range(len(x)):
        fig.add_trace(
            go.Scatter3d(
                x=[0, x[i]], y=[0, y[i]], z=[0, z[i]],
                mode='lines',
                line=dict(width=2, color='gray'),
                showlegend=False,
                hoverinfo='none'
            )
        )
    
    # 更新布局
    layout_updates = dict(
        scene=dict(
            xaxis_title=f'{title} X' if title else 'X',
            yaxis_title=f'{title} Y' if title else 'Y', 
            zaxis_title=f'{title} Z' if title else 'Z'
        ),
        title=f"{title} Vectors" if title else "3D Vectors"
    )
    
    if step:
        layout_updates['annotations'] = [dict(
            x=0.5,
            y=1.05,
            xref='paper',
            yref='paper',
            text=f"<b>当前步骤</b>: {step}",
            showarrow=False,
            font=dict(size=14)
        )]
    
    fig.update_layout(**layout_updates)

def add_3d_attention_weights(fig, weights, step=None):
    """添加3D注意力权重可视化
    
    参数:
        fig (go.Figure): Plotly图形对象
        weights (np.array): 注意力权重矩阵
        step (str): 当前计算步骤说明
    """
    seq_len = weights.shape[0]
    x, y = np.meshgrid(range(seq_len), range(seq_len))
    
    # 添加表面图
    fig.add_trace(
        go.Surface(
            z=weights,
            x=x,
            y=y,
            colorscale='Viridis',
            colorbar=dict(title='Attention Weight'),
            hoverinfo='z',
            hovertext=[[f"""
                <b>Query位置</b>: {i}<br>
                <b>Key位置</b>: {j}<br>
                <b>注意力权重</b>: {weights[i,j]:.4f}<br>
                <b>计算公式</b>: softmax(QK<sup>T</sup>/√d<sub>k</sub>)
            """ for j in range(seq_len)] for i in range(seq_len)]
        )
    )
    
    # 更新布局
    layout_updates = dict(
        scene=dict(
            xaxis_title='Query Position',
            yaxis_title='Key Position', 
            zaxis_title='Weight'
        ),
        title="Attention Weights Heatmap"
    )
    
    if step:
        layout_updates['annotations'] = [dict(
            x=0.5,
            y=1.05,
            xref='paper',
            yref='paper',
            text=f"<b>当前步骤</b>: {step}",
            showarrow=False,
            font=dict(size=14)
        )]
    
    fig.update_layout(**layout_updates)
