import numpy as np
import matplotlib.pyplot as plt
from local_attention import get_local_attention_example
from global_attention import get_global_attention_example
from translation_attention import get_translation_attention_example

def visualize_attention(input_matrix, weights, title):
    """可视化注意力权重"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示输入矩阵
    ax1.imshow(input_matrix, cmap='viridis')
    ax1.set_title('输入矩阵')
    ax1.set_xlabel('维度')
    ax1.set_ylabel('Token')
    
    # 显示注意力权重
    ax2.imshow(weights, cmap='hot', interpolation='nearest')
    ax2.set_title('注意力权重')
    ax2.set_xlabel('Key Token')
    ax2.set_ylabel('Query Token')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def run_local_demo():
    """运行局部注意力demo"""
    data = get_local_attention_example()
    print("\n" + "="*50)
    print("局部注意力模式演示")
    print(data['description'])
    print("="*50)
    
    visualize_attention(data['input'], data['expected_weights'], 
                      "局部注意力模式")

def run_global_demo():
    """运行全局注意力demo"""
    data = get_global_attention_example()
    print("\n" + "="*50)
    print("全局注意力模式演示")
    print(data['description'])
    print("="*50)
    
    print("\n均匀分布全局注意力:")
    visualize_attention(data['input'], data['uniform_weights'],
                      "均匀分布全局注意力")
    
    print("\n特定模式全局注意力:")
    visualize_attention(data['input'], data['focused_weights'],
                      "特定模式全局注意力")

def run_translation_demo():
    """运行翻译注意力demo"""
    data = get_translation_attention_example()
    print("\n" + "="*50)
    print("机器翻译注意力模式演示")
    print(data['description'])
    print("="*50)
    
    visualize_attention(data['input'], data['expected_weights'],
                      "机器翻译注意力模式")

if __name__ == "__main__":
    print("Transformer注意力机制演示程序")
    print("请选择要运行的演示:")
    print("1. 局部注意力模式")
    print("2. 全局注意力模式") 
    print("3. 机器翻译注意力模式")
    print("4. 全部演示")
    
    choice = input("请输入选项(1-4): ")
    
    if choice == '1':
        run_local_demo()
    elif choice == '2':
        run_global_demo()
    elif choice == '3':
        run_translation_demo()
    elif choice == '4':
        run_local_demo()
        run_global_demo()
        run_translation_demo()
    else:
        print("无效输入，请输入1-4之间的数字")
