import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib import cm

date = pd.Timestamp(2024, 5, 8)
year, month, day = date.year, date.month, date.day
doy = date.dayofyear
proc_sate = 'ja3'

with open(f'./DORISVTECStorage/{proc_sate}/{year}/DOY{doy:03d}.pickle', 'rb') as path:
    df_doris = pickle.load(path)

with open(f'./AltimetryVTECStorage/{proc_sate}/{year}/DOY{doy:03d}.pickle', 'rb') as path:
    df_altimetry = pickle.load(path)

def find_ith_circle(df_altimetry, i):
    # 计算ipp_lon的差值（下一个比前一个小则认为新一圈开始）
    lon_diff = df_altimetry['ipp_lon'].diff()
    # 圈的开始点（diff小于-180，意味着从大跳到小，比如359->1）
    circle_start_idx = np.where(lon_diff < -180)[0]
    
    # 圈的起止索引列表
    circle_starts = np.concatenate(([0], circle_start_idx + 1))  # 第一圈从0开始
    circle_ends = np.concatenate((circle_start_idx, [len(df_altimetry)-1]))  # 最后一圈到结尾

    # 检查i是否超界
    if i >= len(circle_starts):
        raise IndexError(f"df_altimetry中只有{len(circle_starts)}圈，无法获取第{i}圈")

    # 获取第i圈区间
    start_idx = circle_starts[i]
    end_idx = circle_ends[i]
    df_circle = df_altimetry.iloc[start_idx:end_idx+1]

    return df_circle

# 添加NSWE方向和°符号
def lon_formatter(x, pos):
    if abs(x) == 180:
        return "180°"
    elif x < 0:
        return f"{abs(int(x))}°W"
    elif x > 0:
        return f"{int(x)}°E"
    else:
        return "0°"

def lat_formatter(y, pos):
    if y < 0:
        return f"{abs(int(y))}°S"
    elif y > 0:
        return f"{int(y)}°N"
    else:
        return "0°"

i = 8
df_circle = find_ith_circle(df_altimetry, i)
altimetry_ipp_lon = df_circle['ipp_lon']
altimetry_ipp_lat = df_circle['ipp_lat']
# 获取obs_epoch范围
start_time = df_circle['obs_epoch'].min()
end_time = df_circle['obs_epoch'].max()

df_doris_selected = df_doris[(df_doris['obs_epoch'] >= start_time) & (df_doris['obs_epoch'] <= end_time)]
doris_ipp_lon = df_doris_selected['ipp_lon']
doris_ipp_lat = df_doris_selected['ipp_lat']

### Big alt/DORIS figure
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.figure(figsize=(17, 9))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_global()
# ax.coastlines(linewidth=1)
# ax.add_feature(cfeature.LAND, facecolor='#93e29b')
# ax.add_feature(cfeature.OCEAN, facecolor='#aed7f3')
# ax.gridlines(draw_labels=False, linestyle=':', color='k', alpha=0.6)

# # IPP点绘制
# ax.scatter(doris_ipp_lon, doris_ipp_lat, s=20, label='DORIS IPP', alpha=0.9, zorder=5)
# ax.scatter(altimetry_ipp_lon, altimetry_ipp_lat, s=20, label='Altimetry IPP', alpha=0.9, zorder=5)
# ax.scatter(altimetry_ipp_lon[40200], altimetry_ipp_lat[40200], s=50,alpha=0.9, zorder=5)
# # 设置经纬度主刻度
# xticks = range(-180, 181, 30)
# yticks = range(-90, 91, 30)
# ax.set_xticks(xticks, crs=ccrs.PlateCarree())
# ax.set_yticks(yticks, crs=ccrs.PlateCarree())

# ax.xaxis.set_major_formatter(mticker.FuncFormatter(lon_formatter))
# ax.yaxis.set_major_formatter(mticker.FuncFormatter(lat_formatter))

# # 刻度字体大小与字体
# ax.tick_params(labelsize=20)

# # 标题
# plt.title('IPP Location for Altimetry and DORIS', fontsize=20, fontname='Times New Roman', pad=15)
# plt.tight_layout()
# # 图例
# plt.legend(fontsize=20, loc='upper right', frameon=True)
# # plt.savefig('IPP_map.png', dpi=600, bbox_inches='tight')

# plt.show()



### small window illustration
# df_alt, df_doris_selected, df_circle 里字段分别为 lat/lon, ipp_lat/ipp_lon

# 1. 获取中心点
center_lat = df_circle['ipp_lat'][40200]  # 以实际列名为准
center_lon = df_circle['ipp_lon'][40200]
center_lon = ((center_lon + 180) % 360) - 180
# 2. 矩形边界
lon_min, lon_max = center_lon - 12, center_lon + 12
lat_min, lat_max = center_lat - 6, center_lat + 6

# 3. DORIS筛选
mask = (
    (df_doris_selected['ipp_lon'] >= lon_min) & (df_doris_selected['ipp_lon'] <= lon_max) &
    (df_doris_selected['ipp_lat'] >= lat_min) & (df_doris_selected['ipp_lat'] <= lat_max)
)
doris_in_rect = df_doris_selected[mask]
doris_out_rect = df_doris_selected[~mask]

# 4. Altimetry42000点
alt40200_lat = center_lat
alt40200_lon = center_lon

### zoom-in data selection illustration
# # 5. 坐标轴范围（适当留2°边界）
# extra_lon = 40
# extra_lat = 20
# xlim = (-70,10)
# ylim = (20, 60)

# plt.rcParams['font.family'] = 'Times New Roman'
# plt.figure(figsize=(17, 9))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=ccrs.PlateCarree())
# ax.coastlines(linewidth=1)
# ax.add_feature(cfeature.LAND, facecolor='#93e29b')
# ax.add_feature(cfeature.OCEAN, facecolor='#aed7f3')
# ax.gridlines(draw_labels=False, linestyle=':', color='k', alpha=0.6)
# # 6. 绘制 Altimetry 曲线与所有点
# # ax.plot(df_circle['lon'], df_circle['lat'], color='#ff9300', linewidth=2, alpha=0.7, label='Altimetry Track', zorder=3)
# ax.scatter(df_circle['ipp_lon'], df_circle['ipp_lat'], s=30, color='C0', alpha=0.8, zorder=4, label='Altimetry IPP')
# # 7. alt42000点
# ax.scatter(alt40200_lon, alt40200_lat, s=100, color='C2', edgecolor = 'black', zorder=10, label='Interpolation Point')

# # 8. DORIS轨迹与点
# # ax.plot(df_doris_selected['ipp_lon'], df_doris_selected['ipp_lat'], color='#4da6ff', linewidth=2, alpha=0.7, label='DORIS Track', zorder=2)
# ax.scatter(doris_out_rect['ipp_lon'], doris_out_rect['ipp_lat'], s=30, color='C1', alpha=0.8, zorder=3, label='DORIS IPP')
# # 9. DORIS框内红色点
# ax.scatter(doris_in_rect['ipp_lon'], doris_in_rect['ipp_lat'], s=70, color='red', zorder=8, label='DORIS Data Points')

# # 10. 画黑色矩形
# rect_lon = [lon_min, lon_max, lon_max, lon_min, lon_min]
# rect_lat = [lat_min, lat_min, lat_max, lat_max, lat_min]
# ax.plot(rect_lon, rect_lat, color='black', linewidth=2, zorder=20)

# # 11. 坐标刻度和格式化
# xticks = range(xlim[0],xlim[1]+1,10)
# yticks = range(ylim[0],ylim[1]+1,10)
# ax.set_xticks(xticks, crs=ccrs.PlateCarree())
# ax.set_yticks(yticks, crs=ccrs.PlateCarree())

# ax.xaxis.set_major_formatter(mticker.FuncFormatter(lon_formatter))
# ax.yaxis.set_major_formatter(mticker.FuncFormatter(lat_formatter))
# ax.tick_params(labelsize=20)
# plt.tight_layout()
# plt.title('Illustration of DORIS Data Selection', fontsize=20, pad=15)
# plt.legend(fontsize=20, loc='best', frameon=True)
# plt.savefig('zoomed_alt_doris.png', dpi=600, bbox_inches='tight')
# plt.show()

### zoom-zoom-in data interpolation
extra_lon = 40
extra_lat = 20
xlim = (-47,-17)
ylim = (34, 49)

def haversine(lat1, lon1, lat2, lon2):
    # 角度转弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # 地球半径，单位：km
    return c * r
distances = haversine(
    doris_in_rect['ipp_lat'].values, doris_in_rect['ipp_lon'].values,
    center_lat, center_lon
)

# 避免除零
distances = np.where(distances == 0, 1e-6, distances)

# IDW权重
idw_weights = 1 / distances
idw_weights_norm = (idw_weights - idw_weights.min()) / (idw_weights.max() - idw_weights.min())  # 归一化

# 颜色映射
cmap = cm.get_cmap('viridis')



colors = cmap(idw_weights_norm)
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(17, 9))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=ccrs.PlateCarree())
ax.coastlines(linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='#93e29b')
ax.add_feature(cfeature.OCEAN, facecolor='#aed7f3')
ax.gridlines(draw_labels=False, linestyle=':', color='k', alpha=0.6)
ax.scatter(df_circle['ipp_lon'], df_circle['ipp_lat'], s=30, color='C0', alpha=0.8, zorder=4, label='Altimetry IPP')

ax.scatter(alt40200_lon, alt40200_lat, s=200, color='C2', edgecolor = 'black', zorder=10, label='Interpolation Point')
ax.scatter(doris_out_rect['ipp_lon'], doris_out_rect['ipp_lat'], s=30, color='C1', alpha=0.8, zorder=3, label='DORIS IPP')

# 绘图部分——只画矩形内的doris点用此colors参数
ax.scatter(
    doris_in_rect['ipp_lon'], doris_in_rect['ipp_lat'],
    s=50, c=colors, zorder=8,
    label='DORIS in Rectangle'
)

# rect_lon = [lon_min, lon_max, lon_max, lon_min, lon_min]
# rect_lat = [lat_min, lat_min, lat_max, lat_max, lat_min]
# ax.plot(rect_lon, rect_lat, color='#183b9a', linewidth=2, zorder=20)
# 11. 坐标刻度和格式化
xticks = range(-50,-20+1,10)
yticks = range(35,46,5)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())
rect_lon = [lon_min, lon_max, lon_max, lon_min, lon_min]
rect_lat = [lat_min, lat_min, lat_max, lat_max, lat_min]
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lon_formatter))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lat_formatter))
ax.tick_params(labelsize=20)
plt.tight_layout()
plt.title('Illustration of DORIS Data Interpolation', fontsize=20, pad=15)
plt.legend(fontsize=20, loc='best', frameon=True)
plt.savefig('zoomed_alt_doris.png', dpi=600, bbox_inches='tight')
plt.show()