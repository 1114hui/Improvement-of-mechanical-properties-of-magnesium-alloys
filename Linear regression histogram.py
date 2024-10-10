import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件并计算MAE和MAPE
def calculate_metrics(csv_files):
    mae_values = []
    mape_values = []

    for file in csv_files:
        df = pd.read_csv(file)
        actual = df['actual']
        prediction = df['prediction']
        absolute_errors = np.abs(actual - prediction)
        mae = np.mean(absolute_errors)
        mape = np.mean(absolute_errors / actual) * 100
        mae_values.append(mae)
        mape_values.append(mape)

    return mae_values, mape_values

# 三个CSV文件的路径
csv_files = ['D:\grad\MLMD\本科生\Data\线性回归\Linear.csv',
             'D:\grad\MLMD\本科生\Data\线性回归\Ridge.csv'
             'D:\grad\MLMD\本科生\Data\线性回归\Lasso.csv',]

# 计算MAE和MAPE
mae_values, mape_values = calculate_metrics(csv_files)

# 模型名称
models = ['Linear', 'Ridge', 'Lasso']

# 设置柱状图的宽度
bar_width = 0.35

# 绘制柱状图
fig, ax1 = plt.subplots()

# 左边y轴表示MAE
color1 = 'tab:blue'
ax1.set_xlabel('Model')
ax1.set_ylabel('MAE', color=color1)
ax1.bar(np.arange(len(models)), mae_values, color=color1, width=bar_width, label='MAE')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(np.arange(len(models)))
ax1.set_xticklabels(models)

# 右边y轴表示MAPE
ax2 = ax1.twinx()
color2 = 'tab:purple'
ax2.set_ylabel('MAPE (%)', color=color2)
ax2.bar(np.arange(len(models)) + bar_width, mape_values, color=color2, width=bar_width, label='MAPE')
ax2.tick_params(axis='y', labelcolor=color2)

# 设置y轴刻度
ax1.set_yticks(np.linspace(0, 10, 5))
ax2.set_yticks(np.linspace(0, 14, 5))

# 设置图例
fig.tight_layout()
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), shadow=True, ncol=2)
plt.show()

