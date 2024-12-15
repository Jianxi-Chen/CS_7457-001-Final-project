import pandas as pd
import matplotlib.pyplot as plt
import itertools

# 读取CSV文件
file_path = 'bandwidth/upload_bandwidth_test.csv'  # 请将路径替换为实际的CSV文件路径
data = pd.read_csv(file_path)

# 将数据转换为长格式
data_long = pd.melt(data, id_vars='protocol', value_vars=['Max Bandwidth (Mbps)', 'Min Bandwidth (Mbps)', 'Avg Bandwidth (Mbps)'], 
                    var_name='Bandwidth Type', value_name='Bandwidth (Mbps)')

# 获取按箱线图中显示顺序的协议名称
protocols_in_order = sorted(data['protocol'].unique())

# 创建颜色列表，每个协议使用不同颜色
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink', 'lightyellow', 'lightgrey']

# 如果协议数量超过颜色数量，循环使用颜色
if len(protocols_in_order) > len(colors):
    colors = list(itertools.islice(itertools.cycle(colors), len(protocols_in_order)))

# 创建协议到颜色的映射
protocol_color_mapping = dict(zip(protocols_in_order, colors))

# 创建一个图形对象，并设置图的大小
fig, ax = plt.subplots(figsize=(15, 6))

# 绘制箱图，并获取当前的axes对象
box = data_long.boxplot(column='Bandwidth (Mbps)', by='protocol', patch_artist=True, grid=False, ax=ax)

# 给不同的协议箱体上色
for patch, protocol in zip(ax.artists, protocols_in_order):
    patch.set_facecolor(protocol_color_mapping[protocol])

# 设置图例
handles = [plt.Rectangle((0,0),1,1, color=protocol_color_mapping[protocol]) for protocol in protocols_in_order]
plt.legend(handles, protocols_in_order, title="Protocols", loc='upper right')

# 设置标题和标签
plt.title('Bandwidth Comparison by Protocol')
plt.suptitle('')  # 去掉子标题
plt.xlabel('Protocol')
plt.ylabel('Bandwidth (Mbps)')

# 调整图形的布局
fig.tight_layout()

# 显示图表
plt.show()