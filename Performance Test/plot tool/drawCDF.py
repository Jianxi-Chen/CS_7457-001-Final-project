import os
import pandas as pd
import numpy as np
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

# 绘制延迟的CDF图
plt.figure(figsize=(10, 8))
for protocol, latency in latency_data:
    sorted_latency = np.sort(latency)  # 排序延迟
    cdf = np.linspace(0, 1, len(sorted_latency))  # 生成CDF
    plt.plot(sorted_latency, cdf, label=protocol)
plt.title('CDF of Latency')
plt.xlabel('Latency (ms)')
plt.ylabel('CDF')
plt.legend()
plt.grid(True)
plt.show()

# 绘制抖动的CDF图
plt.figure(figsize=(10, 8))
for protocol, jitter in jitter_data:
    sorted_jitter = np.sort(jitter)  # 排序抖动
    cdf = np.linspace(0, 1, len(sorted_jitter))  # 生成CDF
    plt.plot(sorted_jitter, cdf, label=protocol)
plt.title('CDF of Jitter')
plt.xlabel('Jitter (ms)')
plt.ylabel('CDF')
plt.legend()
plt.grid(True)
plt.show()
