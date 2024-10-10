import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体和字号
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 18

# MAE计算函数
def calculate_mae(actual, prediction):
    return np.mean(np.abs(actual - prediction))

# 绘制条形图函数
def plot_bar_chart(model_names, train_maes, test_maes):
    bar_width = 0.25
    index = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(8, 6))
    train_bars = ax.bar(index, train_maes, bar_width, label='Train MAE', color='Orange')
    test_bars = ax.bar(index + bar_width, test_maes, bar_width, label='Test MAE', color='SteelBlue')

    ax.set_xlabel('Model', fontsize=20)
    ax.set_ylabel('MAE')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.xticks(rotation=0)
    plt.tight_layout()

    # 保存图片，DPI设置为600
    plt.savefig('D:\grad\MLMD\本科生\图tif格式\model_comparison.tif', dpi=300)

# 文件路径
file_paths = [
    ['D:\\grad\\MLMD\\本科生\\Data\\随机森林\\随机森林\\训练.csv', 'D:\\grad\\MLMD\\本科生\\Data\\随机森林\\随机森林\\测试.csv'],
    ['D:\\grad\\MLMD\\本科生\\Data\\SVR\\SVR\\训练svr.csv', 'D:\\grad\\MLMD\\本科生\\Data\\SVR\\SVR\\测试svr.csv'],
    ['D:\\grad\\MLMD\\本科生\\Data\\XGB\\XGB\\训练xgb.csv', 'D:\\grad\\MLMD\\本科生\\Data\\XGB\\XGB\\测试xgb.csv'],
    ['D:\\grad\\MLMD\\本科生\\Data\\AdaB\\AdaB\\训练adab.csv', 'D:\\grad\\MLMD\\本科生\\Data\\AdaB\\AdaB\\测试adab.csv']
]

model_names = ['RF', 'SVR', 'XGBoost', 'AdaBoost']
train_maes = []
test_maes = []

# 遍历每个模型的CSV文件
for i, (train_file, test_file) in enumerate(file_paths):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    train_mae = calculate_mae(train_data['actual'], train_data['prediction'])
    test_mae = calculate_mae(test_data['actual'], test_data['prediction'])

    train_maes.append(train_mae)
    test_maes.append(test_mae)

# 绘制条形图
plot_bar_chart(model_names, train_maes, test_maes)