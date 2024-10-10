import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 定义输入文件路径和输出文件路径
input_csv_file = r'D:\grad\MLMD\本科生\Data\硬度变化data.csv' # 更改为你的CSV文件路径
output_folder = os.path.expanduser(r'D:\grad\MLMD\本科生\Data')  # 将文件保存到桌面

# 读取CSV文件
data = pd.read_csv(input_csv_file)

# 使用StandardScaler标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
# 删除包含NaN的行
data_cleaned = data.dropna()

# 现在你可以继续应用StandardScaler和DBSCAN
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_cleaned)

# 使用DBSCAN密度聚类算法进行聚类处理
dbscan = DBSCAN(eps=0.5, min_samples=5)  # 根据数据特性调整eps和min_samples参数
labels = dbscan.fit_predict(scaled_data)

# 找出非离群点的索引
non_outlier_indices = np.where(labels != -1)[0]

# 提取非离群点数据
filtered_data = data.iloc[non_outlier_indices]

# 保存处理后的数据文件
output_csv_file = os.path.join(output_folder, 'filtered_data1.csv')
filtered_data.to_csv(output_csv_file, index=False)

print(f'处理后的数据已保存至：{output_csv_file}')
