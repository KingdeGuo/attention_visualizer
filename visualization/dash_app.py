from dash import Dash, html, dcc, Input, Output, callback_context
import plotly.graph_objects as go
import numpy as np
from core.attention import Attention
from visualization.components import add_3d_vectors, add_3d_attention_weights

# 初始化Dash应用
app = Dash(__name__)

# 创建注意力机制实例
attention = Attention(input_dim=4)
X = np.random.rand(4, 4)  # 示例输入
results = attention.compute_attention(X)

# 定义步骤说明
step_descriptions = [
    "1. 输入Embeddings: 这是输入的词向量表示，每个点代表一个token",
    "2. Query向量: 表示当前token想要关注什么",
    "3. Key向量: 表示其他token可以提供什么信息",
    "4. Value向量: 包含实际要传递的信息",
    "5. 注意力权重: 通过Query和Key的点积计算注意力分数，再经过softmax归一化",
    "6. 输出: 用注意力权重对Value向量加权求和，得到最终输出"
]

# 应用布局
app.layout = html.Div([
    html.H1("Transformer注意力机制可视化"),
    html.Div(id='step-description', style={'margin': '20px'}),
    dcc.Graph(id='attention-graph'),
    html.Button('下一步', id='next-button', n_clicks=0),
    html.Button('重置', id='reset-button', n_clicks=0),
])

# 回调函数
@app.callback(
    [Output('attention-graph', 'figure'),
     Output('step-description', 'children')],
    [Input('next-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')]
)
def update_graph(next_clicks, reset_clicks):
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'reset-button'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'reset-button':
        step = 0
    else:
        step = next_clicks % len(step_descriptions)
    
    fig = go.Figure()
    
    if step == 0:
        add_3d_vectors(fig, results['input'], '输入Embeddings')
    elif step == 1:
        add_3d_vectors(fig, results['query'], 'Query向量')
    elif step == 2:
        add_3d_vectors(fig, results['key'], 'Key向量')
    elif step == 3:
        add_3d_vectors(fig, results['value'], 'Value向量')
    elif step == 4:
        add_3d_attention_weights(fig, results['attention_weights'])
    else:
        add_3d_vectors(fig, results['output'], '输出')
    
    fig.update_layout(height=600, width=800)
    return fig, step_descriptions[step]

if __name__ == '__main__':
    app.run(debug=True)
