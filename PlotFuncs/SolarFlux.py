import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取数据
df = pd.read_csv('fluxtable.txt', delim_whitespace=True, skiprows=[1])
df['datetime'] = pd.to_datetime(df['fluxdate'].astype(str) + df['fluxtime'].astype(str), format='%Y%m%d%H%M%S')
df['fluxadjflux'] = df['fluxadjflux'].astype(float)
df.set_index('datetime', inplace=True)

# 时间段1：2019-11-30 到 2019-12-29
start1 = pd.Timestamp('2019-11-30')
end1 = start1 + pd.Timedelta(days=30)
df1 = df.loc[start1:end1]
xticks1 = pd.date_range(start=start1, end=end1, freq='6D')

# 时间段2：2024-05-08 到 2024-06-06
start2 = pd.Timestamp('2024-05-08')
end2 = start2 + pd.Timedelta(days=30)
df2 = df.loc[start2:end2]
xticks2 = pd.date_range(start=start2, end=end2, freq='6D')

# 创建子图
fig, axs = plt.subplots(2, 1, figsize=(14, 8))

# 子图 1（2019）
axs[0].plot(df1.index, df1['fluxadjflux'], color='blue', label='Adjusted F10.7 Flux')
axs[0].set_title('F10.7 Adjusted Flux — Year 2019')
axs[0].set_ylabel('F10.7 Flux')
axs[0].grid(True, axis='y')
axs[0].set_xticks(xticks1)
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
axs[0].legend()

# 子图 2（2024）
axs[1].plot(df2.index, df2['fluxadjflux'], color='blue', label='Adjusted F10.7 Flux')
axs[1].set_title('F10.7 Adjusted Flux — Year 2024')
axs[1].set_xlabel('Date (MM-DD)')
axs[1].set_ylabel('F10.7 Flux')
axs[1].grid(True, axis='y')
axs[1].set_xticks(xticks2)
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
axs[1].legend()

# 自动布局
plt.tight_layout()

# 保存图像（PNG，300 dpi 分辨率）
plt.savefig("f107_flux_comparison.png", dpi=300)

# 显示图像（可选）
plt.show()
