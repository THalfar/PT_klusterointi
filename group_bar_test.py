import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample DataFrame with duplicate time entries and random worker hours per day for four value columns
data = {
    'time_column': pd.date_range(start='2023-01-01', end='2023-02-28', freq='D').to_list() * 2,
    'group_column': np.random.choice(['A', 'B', 'C'], size=2*59),
    'value1': np.random.randint(6, 11, size=2*59),
    'value2': np.random.randint(10, 15, size=2*59),
    'value3': np.random.randint(1, 5, size=2*59),
    'value4': np.random.randint(15, 20, size=2*59)
}

df = pd.DataFrame(data)

start_date = '2023-01-15'
end_date = '2023-02-15'
filtered_df = df[(df['time_column'] >= start_date) & (df['time_column'] <= end_date)]

grouped_value4 = filtered_df.groupby(['group_column']).agg({'value1':'sum', 'value2':'sum', 'value3':'sum', 'value4':'sum'})
grouped_value4.plot(kind='bar', figsize=(12, 7))

plt.xlabel('Group')
plt.ylabel('Values')
plt.title(f'Between {start_date} and {end_date} groups values')
plt.tight_layout()
plt.show()

