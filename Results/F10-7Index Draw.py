import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates

# 数据
# 第一组数据
doy1 = [334, 335, 336, 337, 338, 339, 340]
altimetry = [53247, 54391, 54423, 54765, 55403, 54467, 54112]
doris = [48746, 47258, 48096, 48222, 46684, 47972, 48217]

# 第二组数据
doy2 = [129, 130, 131, 132, 133, 134, 135]
altimetry2 = [56605, 55797, 55927, 55427, 56074, 57619, 56032]
doris2 = [35503, 33748, 32588, 33009, 33702, 33056, 33358]

# 创建子图
plt.subplots(1, 2, figsize=(10, 6))  # 2行1列的子图
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20})       
# 第一个子图
plt.subplot(121)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.plot(doy1, altimetry, marker='o', label='Altimetry', color='blue', linewidth=2.5)
plt.plot(doy1, doris, marker='o', label='DORIS', color='orange', linewidth=2.5)
plt.xlabel('DOY 2019', fontsize=20, fontname='Times New Roman')
plt.ylim(25000, 60000)
plt.ylabel('Observation Numbers', fontsize=20, fontname='Times New Roman')
plt.yticks(fontsize=20, fontname='Times New Roman')
plt.xticks(doy1, fontsize=20, fontname='Times New Roman')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # 使用虚线和透明度设置
plt.legend(fontsize=20,loc='upper left')

plt.subplot(122)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.plot(doy2, altimetry2, marker='o', label='Altimetry', color='blue', linewidth=2.5)
plt.plot(doy2, doris2, marker='o', label='DORIS', color='orange', linewidth=2.5)
plt.xlabel('DOY 2024', fontsize=20, fontname='Times New Roman')
plt.ylim(25000, 60000)
plt.yticks(fontsize=20, fontname='Times New Roman')
plt.xticks(doy2, fontsize=20, fontname='Times New Roman')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # 使用虚线和透明度设置
# 显示图形
plt.show()

# Read the text file (assuming columns are space-separated)
df = pd.read_csv('./Results/fluxtable.txt', delim_whitespace=True)
df['datetime'] = pd.to_datetime(df['fluxdate'][1:].astype(str) + df['fluxtime'][1:].astype(str), format='%Y%m%d%H%M%S')
fluxobs = df['fluxobsflux'][1:].astype(float)
index = [ind+1 for ind,flux in enumerate(fluxobs) if flux <300]
plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = 22
plt.plot(df['datetime'][index], fluxobs[index], linestyle='-', label='Observed Flux')
plt.xlabel('Time[year]')
plt.ylabel('F10.7 Index[sfu]')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
start_calm = datetime(2019, 11, 30, 0, 0)      # Start datetime
end_calm = datetime(2019, 12, 6, 23, 59) 
index_calm = [ind for ind,date in enumerate(df['datetime']) if start_calm<date <end_calm]

start_fierce = datetime(2024, 5, 8, 0, 0)      # Start datetime
end_fierce = datetime(2024, 5, 14, 23, 59) 
index_fierce = [ind for ind,date in enumerate(df['datetime']) if start_fierce<date <end_fierce]



plt.figure(figsize=(10, 6))

plt.rcParams['font.size'] = 16
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('% DOY'))

plt.subplot(121)

plt.grid(axis='y', linestyle='--', alpha=0.7)  # 使用虚线和透明度设置
bars = plt.bar(df['datetime'][index_calm[::3]],fluxobs[index_calm[::3]] , color='green', alpha=0.7)
plt.xlabel('Date (Year DOY)')
plt.xticks(ticks=df['datetime'][index_calm[::3]], labels=[f"{date.dayofyear}" for date in df['datetime'][index_calm[::3]]])
plt.ylim([0,300])
plt.ylabel('F10.7index[sfu]')
for bar in bars:
    yval = bar.get_height()  # 获取柱子的高度
    plt.text(bar.get_x() + bar.get_width() / 2, yval,  # 确定文本位置
             yval,  # 显示的数值
             ha='center',  # 水平对齐方式
             va='bottom')  # 垂直对齐方式



plt.subplot(122)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # 使用虚线和透明度设置
bars = plt.bar(df['datetime'][index_fierce[::3]],fluxobs[index_fierce[1::3]] , color='red', alpha=0.7)
plt.xlabel('Date (Year DOY)')

plt.ylim([0,300])
# 设置 x 轴的刻度
plt.xticks(ticks=df['datetime'][index_fierce[::3]], labels=[f"{date.dayofyear}" for date in df['datetime'][index_fierce[::3]]])
for bar in bars:
    yval = bar.get_height()  # 获取柱子的高度
    plt.text(bar.get_x() + bar.get_width() / 2, yval,  # 确定文本位置
             yval,  # 显示的数值
             ha='center',  # 水平对齐方式
             va='bottom')  # 垂直对齐方式
plt.show()