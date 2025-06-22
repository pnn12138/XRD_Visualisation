import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


# 假设 origin_df 包含以下列：
# 'xrd_tsne_1': t-SNE 第一维度
# 'xrd_tsne_2': t-SNE 第二维度
# 'formation_energy': 形成能数据

# 示例数据生成（你可以替换为实际数据）
# origin_df = pd.DataFrame({
#     'xrd_tsne_1': np.random.randn(100),
#     'xrd_tsne_2': np.random.randn(100),
#     'formation_energy': np.random.uniform(-5, 5, 100)
# })

# 绘制二维散点图，并根据形成能映射颜色
origin_df=pd.read_pickle(r"C:\code\XRD_Visualisation\data\mp_20_xrd_sinc_gau\test_csv_Q_tsne_visualization.csv")



print(origin_df.columns)
xrd_data = np.vstack(origin_df['xrdd'].values)

plt.figure(figsize=(8, 6))




# 使用形成能作为颜色映射
plt.plot(xrd_data[1040])

# 图表标题和坐标轴标签
plt.title('t-SNE Visualization of XRD Data (Colored by Formation Energy)', fontsize=14)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.grid(True)

# 保存图像到文件
output_path = 'tsne_visualization_colored.png'  # 你可以根据需要修改路径和文件名
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# 显示图像
plt.show()