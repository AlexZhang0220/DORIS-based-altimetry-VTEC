import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import h5py

insti = 'DORIS'
yr = 2023
day = 221
hour = '05'

doris_vtec = []
ipp_lat = []
ipp_lon = []

with h5py.File('vtec/DOY221.h5', 'r') as file:

    for pass_index in file['/y2023/d221/ele_cut_0']:
        pass_index_folder = file[f'/y2023/d221/ele_cut_0/{pass_index}']


        doris_vtec += pass_index_folder['vtec']
        ipp_lat += pass_index_folder['ipp_lat']
        ipp_lon += pass_index_folder['ipp_lon']

# The relevant IGS data range

# 设置地图全局属性
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置整体的字体为Times New Roman
plt.figure(figsize=(10, 6), dpi=600)  # 设置大小和分辨率

# 创建底图，设置地图投影为World Plate Carrée，分辨率为高分辨率，地图范围为全球
m = Basemap(projection='cyl', resolution='h', llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90)

# 设置地图经纬线，并只在左端和底端显示
m.drawmeridians(np.arange(-180, 181, 30), labels=[0, 0, 0, 1], fontsize=10, linewidth=0.3, color='grey', )  # 经线
m.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0], fontsize=10, linewidth=0.3, color='grey')  # 纬线
m.drawmapboundary(fill_color='lightblue')  # 海洋填充颜色
m.fillcontinents(color='lightgreen', lake_color='lightblue')  # 陆地填充颜色
m.drawcoastlines(linewidth=0.1)  # 绘制海岸线

plt.scatter(ipp_lon, ipp_lat, c = doris_vtec, cmap = 'jet', s = 5)
plt.title('Global TEC Map of ' + str(insti), fontsize = 10)

# 设置图例
cb = m.colorbar(location='right', pad=0.1, size = 0.2)  # 图例在右侧显示
cb.set_label('TECU', fontsize=10)  # 设置图例名称和字体大小
fig_name = 'TEC Map of ' + str(insti)+'.png'

# 保存图片并显示
plt.savefig(fig_name, dpi=600, bbox_inches='tight',
            pad_inches=0.1)  # 输出地图，并设置边框空白紧密
# plt.show()  # 显示地图
