import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 读取和处理数据
df = pd.read_csv('fluxtable.txt', delim_whitespace=True, skiprows=[1])
df['filtered'] = df['fluxadjflux'].rolling(window=3, center=True).median()
df['datetime'] = pd.to_datetime(df['fluxdate'].astype(str) + df['fluxtime'].astype(str), format='%Y%m%d%H%M%S')
df['filtered'] = df['filtered'].astype(float)
df.set_index('datetime', inplace=True)

# 主时间段：2019 - 2025
start1 = pd.Timestamp('2019-01-01')
end1 = start1 + pd.Timedelta(days=2000)
df1 = df.loc[start1:end1]
xticks1 = pd.date_range(start=start1, end=end1, freq='YS')

# 矩形标注的子区域
zoom_start = pd.Timestamp('2019-11-30')
zoom_end = pd.Timestamp('2019-12-29')
df_zoom = df.loc[zoom_start:zoom_end]

# 绘图
plt.rcParams['font.family'] = 'Times New Roman'
fig, axs = plt.subplots(1, 1, figsize=(17, 9))
plt.suptitle('Solar Activity - F10.7 Adjusted Flux', fontsize=20)

# 主图
axs.plot(df1.index, df1['filtered'], color='C0', label='F10.7 Flux (Filtered)')
axs.set_ylabel('F10.7 Flux [s.f.u]', fontsize=20)
axs.grid(True, axis='y', linestyle = '--')
axs.set_xticks(xticks1)
axs.set_ylim(50, 350)
axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs.tick_params(labelsize=20)

# 设置起点和终点（数据点 → inset 小图）
arrow_start_time = pd.Timestamp(2019,12,10,18,0,0)  # 箭头起点时间（zoom 范围中部）
arrow_start_y = df1.loc[arrow_start_time, 'filtered']

# inset 图位置（以 axes 相对坐标表示），可根据 inset 的 bbox_to_anchor 调整
arrow_end_x = 0.15   # 横向位置，单位是 axes fraction
arrow_end_y = 0.22   # 纵向位置，靠近 inset 图的中下部

# 在主图上画注释箭头
axs.annotate(
    '',  # 不显示文字
    xy=(arrow_end_x, arrow_end_y), xycoords='axes fraction',         # 目标：小图位置
    xytext=(arrow_start_time, arrow_start_y), textcoords='data',    # 起点：主图数据坐标
    arrowprops=dict(
        arrowstyle="->",
        color='C3',
        lw=2
    )
)
# 添加嵌入放大图（inset axes）
ax_inset = inset_axes(
    axs,
    width=3.5, height=2.0,       # 以英寸为单位控制大小
    bbox_to_anchor=(0.05, 0.55), # 控制左下角位置：x, y（单位是 axes 相对坐标）
    bbox_transform=axs.transAxes,
    loc='upper left',
    borderpad=0
)
ax_inset.plot(df_zoom.index, df_zoom['filtered'], color='C1', label = '2019')
ax_inset.set_xticks(pd.date_range(zoom_start, zoom_end, freq='7D'))
ax_inset.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax_inset.tick_params(labelsize=14)
ax_inset.set_ylim([65,75])
ax_inset.set_yticks([65,70,75])
ax_inset.legend(fontsize=14)


# 设置起点和终点（数据点 → inset 小图）
arrow_start_time = pd.Timestamp(2024,5,8,17,0,0)  # 箭头起点时间（zoom 范围中部）
arrow_start_y = df1.loc[arrow_start_time, 'filtered']

# inset 图位置（以 axes 相对坐标表示），可根据 inset 的 bbox_to_anchor 调整
arrow_end_x = 0.35  # 横向位置，单位是 axes fraction
arrow_end_y = 0.75   # 纵向位置，靠近 inset 图的中下部

# 在主图上画注释箭头
axs.annotate(
    '',  # 不显示文字
    xy=(arrow_end_x, arrow_end_y), xycoords='axes fraction',         # 目标：小图位置
    xytext=(arrow_start_time, arrow_start_y), textcoords='data',    # 起点：主图数据坐标
    arrowprops=dict(
        arrowstyle="->",
        color='C4',
        lw=2
    )
)
zoom_start = pd.Timestamp('2024-05-08')
zoom_end = pd.Timestamp('2024-06-06')
df_zoom = df.loc[zoom_start:zoom_end]
ax_inset_high = inset_axes(
    axs,
    width=3.5, height=2.0,       # 以英寸为单位控制大小
    bbox_to_anchor=(0.05, 0.93), # 控制左下角位置：x, y（单位是 axes 相对坐标）
    bbox_transform=axs.transAxes,
    loc='upper left',
    borderpad=0
)
ax_inset_high.plot(df_zoom.index, df_zoom['filtered'], color='C2', label = '2024')
ax_inset_high.set_xticks(pd.date_range(zoom_start, zoom_end, freq='7D'))
ax_inset_high.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax_inset_high.tick_params(labelsize=14)
ax_inset_high.set_ylim([150,250])
ax_inset_high.set_yticks([150,200,250])
ax_inset_high.legend(fontsize=14)
plt.savefig("f107_flux_plot_zoom.png", dpi=600, bbox_inches='tight')
plt.show()

