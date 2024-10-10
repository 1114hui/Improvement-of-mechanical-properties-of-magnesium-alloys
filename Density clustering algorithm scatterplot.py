import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib as mpl

# 加载CSV文件，并指定编码（这里假设是GBK，但您可以根据需要更改为其他编码）
file_path = r'D:\grad\MLMD\本科生\Data\剔除离群点后的数据.csv'
data = pd.read_csv(file_path, encoding='gbk')  # 修改这里，指定编码为'gbk'

# 提取温度和时间数据
X = data[['Temperature', 'Time']]

# 确保Time列是数值类型，如果不是，则需要进行转换（这里假设已经是数值类型）

# 执行DBSCAN聚类以去除离群点
dbscan = DBSCAN(eps=0.5, min_samples=5)  # 根据实际情况调整eps和min_samples
clusters = dbscan.fit_predict(X)

# 绘制2D散点图
plt.figure(figsize=(8, 6))

# 设置全局字体为Arial，字号为18pt
mpl.rcParams['font.family'] = 'Arial'  # 注意：如果系统中没有Arial字体，这可能会失败
mpl.rcParams['font.size'] = 18

# 绘制离群点（红色）
plt.scatter(X.loc[clusters == -1, 'Temperature'], X.loc[clusters == -1, 'Time'], c='red', label='Outliers', s=40)

# 绘制其他数据点（根据聚类着色）
plt.scatter(X.loc[clusters != -1, 'Temperature'], X.loc[clusters != -1, 'Time'], c=clusters[clusters != -1],
            cmap='viridis', label='Clusters', s=40)

# 设置坐标轴标签和标题
plt.xlabel('Temperature (°C)', fontsize=18)
plt.ylabel('Time (h)', fontsize=18)
plt.title('DBSCAN Clustering Result', fontsize=18)

# 显示图例和颜色条
plt.colorbar(label='Cluster')
plt.legend()

# 保存图片，DPI设置为600
plt.savefig('D:\grad\MLMD\本科生\图chz\DBSCAN密度聚类.tif', dpi=300)