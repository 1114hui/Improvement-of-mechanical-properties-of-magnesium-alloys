import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为Arial，字号为14pt
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 18


# Function to calculate MAPE
def calculate_mape(actual, prediction):
    return np.mean(np.abs((actual - prediction) / actual)) * 100


# Function to plot bar chart
def plot_bar_chart(model_names, train_mapes, test_mapes):
    bar_width = 0.25
    index = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(8, 6))
    train_bars = ax.bar(index, train_mapes, bar_width, label='Train MAPE', color='Orange')
    test_bars = ax.bar(index + bar_width, test_mapes, bar_width, label='Test MAPE', color='SteelBlue')

    ax.set_xlabel('Model', fontsize=20)
    ax.set_ylabel('MAPE (%)')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.xticks(rotation=0)
    plt.tight_layout()

    # Save the figure instead of showing it
    plt.savefig('D:\grad\MLMD\本科生\图tif格式\model_comparison.tif', dpi=300)

# File paths for the uploaded CSV files
file_paths = [
    ['D:\\grad\\MLMD\\本科生\\Data\\随机森林\\随机森林\\训练.csv','D:\\grad\\MLMD\\本科生\\Data\\随机森林\\随机森林\\测试.csv'],
    ['D:\\grad\\MLMD\\本科生\\Data\\SVR\\SVR\\训练svr.csv', 'D:\\grad\\MLMD\\本科生\\Data\\SVR\\SVR\\测试svr.csv'],
    ['D:\\grad\\MLMD\\本科生\\Data\\XGB\\XGB\\训练xgb.csv', 'D:\\grad\\MLMD\\本科生\\Data\\XGB\\XGB\\测试xgb.csv'],
    ['D:\\grad\\MLMD\\本科生\\Data\\AdaB\\AdaB\\训练adab.csv', 'D:\\grad\\MLMD\\本科生\\Data\\AdaB\\AdaB\\测试adab.csv']
]

model_names = ['RF', 'SVR', 'XGBoost', 'AdaBoost']
train_mapes = []
test_mapes = []

# Iterate through each model's CSV files
for i, (train_file, test_file) in enumerate(file_paths):
    # Load CSV files
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Calculate MAPE for train and test sets
    train_mape = calculate_mape(train_data['actual'], train_data['prediction'])
    test_mape = calculate_mape(test_data['actual'], test_data['prediction'])

    train_mapes.append(train_mape)
    test_mapes.append(test_mape)

# Plotting the bar chart
plot_bar_chart(model_names, train_mapes, test_mapes)