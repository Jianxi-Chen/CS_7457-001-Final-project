import pandas as pd
import matplotlib.pyplot as plt
import itertools

file_path = 'bandwidth/upload_bandwidth_test.csv'  
data = pd.read_csv(file_path)

data_long = pd.melt(data, id_vars='protocol', value_vars=['Max Bandwidth (Mbps)', 'Min Bandwidth (Mbps)', 'Avg Bandwidth (Mbps)'], 
                    var_name='Bandwidth Type', value_name='Bandwidth (Mbps)')

protocols_in_order = sorted(data['protocol'].unique())

colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink', 'lightyellow', 'lightgrey']

if len(protocols_in_order) > len(colors):
    colors = list(itertools.islice(itertools.cycle(colors), len(protocols_in_order)))

protocol_color_mapping = dict(zip(protocols_in_order, colors))

fig, ax = plt.subplots(figsize=(15, 6))

box = data_long.boxplot(column='Bandwidth (Mbps)', by='protocol', patch_artist=True, grid=False, ax=ax)

for patch, protocol in zip(ax.artists, protocols_in_order):
    patch.set_facecolor(protocol_color_mapping[protocol])

handles = [plt.Rectangle((0,0),1,1, color=protocol_color_mapping[protocol]) for protocol in protocols_in_order]
plt.legend(handles, protocols_in_order, title="Protocols", loc='upper right')

plt.title('Bandwidth Comparison by Protocol')
plt.suptitle('')  
plt.xlabel('Protocol')
plt.ylabel('Bandwidth (Mbps)')

fig.tight_layout()
plt.show()
