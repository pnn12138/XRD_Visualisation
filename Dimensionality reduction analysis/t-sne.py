import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 绘制二维散点图，并根据形成能映射颜色
origin_df=pd.read_csv(r"../data/mp_20_xrd/train.csv_Q_tsne_visualization.csv")
#根据各种值绘制
"""plt.figure(figsize=(8, 6))

# 使用形成能作为颜色映射
scatter = plt.scatter(
    origin_df['xrd_tsne_1'],
    origin_df['xrd_tsne_2'],
    c=origin_df['Si_100_mismatch'],  # 颜色映射到形成能
    cmap='coolwarm',                  # 使用 coolwarm 色带（蓝色到红色）
    alpha=0.7,                        # 设置透明度
    s=50                               # 设置点的大小
)
# 添加颜色条以显示形成能的范围
colorbar = plt.colorbar(scatter)
colorbar.set_label('Formation Energy', fontsize=12)
# 图表标题和坐标轴标签
plt.title('t-SNE Visualization of XRD Data (Colored by Formation Energy)', fontsize=14)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.grid(True)
# 保存图像到文件
output_path = 'tsne_visualization_colored.png'  # 你可以根据需要修改路径和文件名
plt.savefig(output_path, dpi=300, bbox_inches='tight')
# 显示图像
plt.show()"""

colors = plt.cm.tab10.colors  # 使用 tab10 色带
markers = ['o', 's', '^', 'D', 'P']
unique_space_groups = origin_df['lattice_type'].unique()
for i, space_group in enumerate(unique_space_groups):
    subset = origin_df[origin_df['lattice_type'] == space_group]
    plt.scatter(
        subset['xrd_tsne_1'],
        subset['xrd_tsne_2'],
        color=colors[i % len(colors)],  # 循环使用颜色
        marker=markers[i % len(markers)],  # 循环使用形状
        label=space_group,  # 添加图例标签
        alpha=0.7,
        s=5
    )
# 图例
plt.figure(figsize=(8, 6))
plt.legend(title='lattice_type', fontsize=10, title_fontsize=12)

# 图表标题和坐标轴标签
plt.title('t-SNE Visualization of XRD Data (Colored by lattice_type)', fontsize=14)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.grid(True)

# 保存图像到文件
out_path= '../fig/t-sne'
os.makedirs(out_path, exist_ok=True)
output_path = '../fig/t-sne/tsne_train_visualization_by_lattice_type.png'  # 输出路径
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# 显示图像
plt.show()
