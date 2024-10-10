import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 这里仅尝试使用Arial，你可能需要根据你的系统环境修改或添加字体路径
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 18  # 设置全局字体大小为16pt

# Load the data
data = pd.read_csv('D:\grad\MLMD\本科生\Data\剔除离群点后的数据.csv')  # 注意在Windows中路径需要使用双反斜杠或原始字符串

# Define the hardness intervals and corresponding colors
intervals = [(30, 45), (45, 55), (55, 65), (65, 75)]
colors = ['blue', 'green', 'orange', 'red']

# Prepare the plot
plt.figure(figsize=(8, 6))
for (low, high), color in zip(intervals, colors):
    # Filter data within the current interval
    mask = (data['Hardness'] >= low) & (data['Hardness'] < high)
    plt.scatter(data.loc[mask, 'Time'], data.loc[mask, 'Temperature'], color=color, label=f'{low}-{high}')

# Enhance the plot
plt.title('Hardness by Time and Temperature', fontsize=18)
plt.xlabel('Time(h)',fontsize=18)
plt.ylabel('Temperature(°C)',fontsize=18)
plt.legend(title='Hardness Interval',fontsize=14)
plt.grid(True)

# 保存图片，DPI设置为300
plt.savefig('D:\grad\MLMD\本科生\图tif格式\二维视角图.tif', dpi=300)