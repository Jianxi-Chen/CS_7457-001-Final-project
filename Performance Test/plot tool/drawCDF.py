import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder_path = 'nonpayload'  

latency_data = []
jitter_data = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        df = pd.read_csv(file_path, header=None)
        protocol_name = df.iloc[0, 0]  
        latency = df.iloc[:, 1]  
        jitter = df.iloc[:, 2]  

        latency_data.append((protocol_name, latency))
        jitter_data.append((protocol_name, jitter))

plt.figure(figsize=(10, 8))
for protocol, latency in latency_data:
    sorted_latency = np.sort(latency)  
    cdf = np.linspace(0, 1, len(sorted_latency))  
    plt.plot(sorted_latency, cdf, label=protocol)
plt.title('CDF of Latency')
plt.xlabel('Latency (ms)')
plt.ylabel('CDF')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
for protocol, jitter in jitter_data:
    sorted_jitter = np.sort(jitter)
    cdf = np.linspace(0, 1, len(sorted_jitter)) 
    plt.plot(sorted_jitter, cdf, label=protocol)
plt.title('CDF of Jitter')
plt.xlabel('Jitter (ms)')
plt.ylabel('CDF')
plt.legend()
plt.grid(True)
plt.show()
