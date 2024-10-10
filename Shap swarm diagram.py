import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 设置matplotlib的字体样式和大小
plt.rcParams['font.family'] = 'Arial'  # 尝试设置为'Arial'，如果不可用则改为其他字体
plt.rcParams['font.size'] = 16  # 设置全局字体大小

# 读取CSV文件
data = pd.read_csv(r"D:\grad\MLMD\本科生\Data\剔除离群点new.csv")
X = data[['Mg', 'Zn', 'Ca', 'Time', 'Temperature']]  # 特征变量
y = data['Hardness']  # 目标变量

# 检查数据是否正确读取
print("数据读取成功，数据头部：")
print(data.head())

# 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 初始化SHAP的TreeExplainer
explainer = shap.TreeExplainer(model)

# 计算训练集的SHAP值
shap_values = explainer.shap_values(X)

# 创建一个具有所需大小的图形
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制SHAP摘要图
shap.summary_plot(shap_values, X, plot_type="dot", show=False)

# 确保所有内容都可见
plt.tight_layout()

# 保存图片，DPI设置为300
plt.savefig(r"D:\grad\MLMD\本科生\图tif格式\蜂群图.tif", dpi=300)

print("图片保存成功")
