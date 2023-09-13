import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

time_interval = "7D"
start_time = "2023-05-01"
end_time = "2023-09-15"

time_col = pd.date_range(start='2023-01-01', end='2023-12-30', freq='D').to_list() * 3
group_col = np.random.choice(['A', 'B', 'C'], size=len(time_col))
value = np.random.randint(6, 11, size=len(time_col))

data = {
    'time_column': time_col,
    'group_column': group_col,
    'value': value
}
df = pd.DataFrame(data)

df = df[(df['time_column'] >= start_time) & (df['time_column'] <= end_time)] 

grouped = df.groupby([pd.Grouper(key='time_column', freq='D'), 'group_column'])
aggregated = grouped.sum().reset_index()

final_data = {}
for name, group in aggregated.groupby('group_column'):
    resampled = group.set_index('time_column').resample(time_interval).sum()
    final_data[name] = resampled

for name, data in final_data.items():
    plt.plot(data.index, data['value'], label=name)

for name, group in df.groupby('group_column'):
    plt.scatter(group['time_column'], group['value'], label=f'Original Data {name}', alpha=0.6, s=10)


plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(title='Groups')
plt.title(f'Sum of Different Groups time: {start_time} - {end_time} Interval: {time_interval}')
plt.show()
