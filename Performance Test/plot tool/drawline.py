import os
import pandas as pd
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

plt.figure(figsize=(20, 5))
for protocol, latency in latency_data:
    plt.plot(latency, label=protocol)
plt.title('Latency over Time')
plt.xlabel('Number of Samples')
plt.ylabel('Latency (ms)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(20, 5))
for protocol, jitter in jitter_data:
    plt.plot(jitter, label=protocol)
plt.title('Jitter over Time')
plt.xlabel('Number of Samples')
plt.ylabel('Jitter (ms)')
plt.legend()
plt.grid(True)
plt.show()
