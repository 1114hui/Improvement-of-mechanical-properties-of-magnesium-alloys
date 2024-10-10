import numpy as np
import pandas as pd
import csv
#创建函数来进行插值操作
def interpolatevalues(elementrange, step):
    elementvalues = np.arange(elementrange[0], elementrange[1], step)
    return elementvalues
#后面的小数0.1要小于步长0.5，保证取到右边界
Mgrange = (90, 100)
Carange = (0, 1)
Znrange = (0, 3)
Temprange = (150, 161)
Timerange = (10, 100)
#调用函数生成插值后的数据点
Mgvalues = interpolatevalues(Mgrange, 0.5)
Cavalues = interpolatevalues(Carange, 0.1)
Znvalues = interpolatevalues(Znrange, 0.1)
Tempvalues = interpolatevalues(Temprange, 1)
Timevalues = interpolatevalues(Timerange, 1)
#生成成分特征组合
combinations = [(m, c, z, t, s)
                for m in Mgvalues
                for c in Cavalues
                for z in Znvalues
                for t in Tempvalues
                for s in Timevalues]
#将结果写入CSV文件
with open('combinations.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Mg', 'Ca', 'Zn', 'Temp', 'Time'])  # 写入表头
    for comp in combinations:
        m, c, z, t, s = comp
        if m + c + z == 100:  # 判断成分的和
            writer.writerow(comp)  # 写入csv文件