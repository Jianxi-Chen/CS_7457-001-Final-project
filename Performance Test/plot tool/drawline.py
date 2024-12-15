import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置文件夹路径
folder_path = 'nonpayload'  # 替换为你的csv文件夹路径

# 初始化数据存储
latency_data = []
jitter_data = []

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # 读取CSV文件
        df = pd.read_csv(file_path, header=None)
        protocol_name = df.iloc[0, 0]  # 第1行是协议名称
        latency = df.iloc[:, 1]  # 第2列是延迟
        jitter = df.iloc[:, 2]  # 第3列是抖动

        # 将协议名称、延迟、抖动分别存储
        latency_data.append((protocol_name, latency))
        jitter_data.append((protocol_name, jitter))

# 绘制延迟图
plt.figure(figsize=(20, 5))
for protocol, latency in latency_data:
    plt.plot(latency, label=protocol)
plt.title('Latency over Time')
plt.xlabel('Number of Samples')
plt.ylabel('Latency (ms)')
plt.legend()
plt.grid(True)
plt.show()

# 绘制抖动图
plt.figure(figsize=(20, 5))
for protocol, jitter in jitter_data:
    plt.plot(jitter, label=protocol)
plt.title('Jitter over Time')
plt.xlabel('Number of Samples')
plt.ylabel('Jitter (ms)')
plt.legend()
plt.grid(True)
plt.show()
