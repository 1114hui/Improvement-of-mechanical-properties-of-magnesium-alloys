import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# 设置matplotlib的字体样式和大小
plt.rcParams['font.family'] = 'Arial'  # 尝试设置为'Arial'，如果不可用则改为其他字体
plt.rcParams['font.size'] = 14  # 设置全局字体大小

# CSV文件路径
csv_file_path = r"D:\grad\MLMD\本科生\Data\剔除离群点new.csv"  # 注意使用双反斜杠或原始字符串

# 读取CSV文件
data = pd.read_csv(csv_file_path)

# 分离特征和目标变量
X = data.drop('Hardness', axis=1)  # 确保目标变量名称正确
y = data['Hardness']

# 划分训练集和测试集（这里仅用于示例，实际中SHAP值通常只在训练集上计算）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 初始化SHAP的TreeExplainer
explainer = shap.TreeExplainer(model)

# 计算训练集的SHAP值
shap_values = explainer.shap_values(X_train)

# 计算特征级别的平均绝对SHAP值，以表示特征重要性
shap_feature_importances = np.abs(shap_values).mean(0)

# 绘制条形图展示特征重要性
plt.figure(figsize=(8, 4))
bar_width = 0.7  # 条形的宽度
y_pos = np.arange(len(X_train.columns))  # 特征的索引位置

# 绘制条形图
bars = plt.barh(y_pos, shap_feature_importances, bar_width, color='Crimson')

# 为每个条形添加数值标签
for bar, val in zip(bars, shap_feature_importances):
    # 计算条形图的x位置（即其值），并稍微向右偏移以避免与条形重叠
    x_val = bar.get_x() + bar.get_width() -0.01 * max(shap_feature_importances)
    y_val = bar.get_y() + bar.get_height() / 2  # 垂直居中
    plt.text(x_val + 0.01 * max(shap_feature_importances), y_val, '+{:.2f}'.format(val),fontsize=14, va='center')


plt.xlim(0, max(shap_feature_importances) * 1.1)

# 修改x轴标签
plt.xlabel('Mean(|SHAP value|)',fontsize=16)
plt.ylabel('Features',fontsize=16)
plt.yticks(y_pos, X_train.columns)  # 确保y轴标签与特征名称匹配
plt.tight_layout()

# 保存图片，DPI设置为300
plt.savefig('D:\grad\MLMD\本科生\图tif格式\条形图.tif', dpi=300)