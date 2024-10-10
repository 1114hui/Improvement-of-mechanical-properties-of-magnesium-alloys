import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 定义输入文件路径和输出文件路径
input_csv_file = r'D:\grad\MLMD\本科生\Data\硬度变化chz\硬度变化剔除离群点.csv'  # 更改为你的CSV文件路径
output_folder = os.path.expanduser(r'D:\grad\MLMD\本科生\Data\硬度变化chz\训练集和测试集')  # 将文件保存到桌面
# 定义分割比例
split_ratio = 0.5  # 例如：80%作为第一个文件，20%作为第二个文件

# 读取CSV文件
data = pd.read_csv(input_csv_file)

# 随机分割数据
train_data, test_data = train_test_split(data, test_size=1-split_ratio, random_state=42)

# 保存分割后的数据文件
train_file_path = os.path.join(output_folder, 'train_data5.5.csv')
test_file_path = os.path.join(output_folder, 'test_data5.5.csv')

train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

print(f'数据已分割并保存到：\n{train_file_path}\n{test_file_path}')