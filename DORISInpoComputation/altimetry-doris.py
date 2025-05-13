import json
import os
import numpy as np
from astropy.time import Time
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from readFile import read_ionFile
import h5py

process_epoch = datetime(2023, 8, 9) # the data in user-defined DAY will be processed
year = process_epoch.year
month = process_epoch.month
day = process_epoch.day
doy = process_epoch.timetuple().tm_yday

# elev_mask = 15

lat_range = 7
lon_range = 14

time_range = 60 # second
int_strat = 'IDW'
limit = 'location' # time or location
obs_least = 10
elev_mask_list = [15]
num_obs_list = [80]
init_epoch = datetime(1985, 1, 1, 0, 0, 0)

jason3_freq = 13.575e9
current_path = os.getcwd()
orbit_filename = current_path + '/Orbit'
epoch_filename = current_path + '/Epoch'
dion_filename = current_path + '/Dion'

# IGS data
# ion_file = './passdetectionTest/igsion/igsg'+str(doy)+'0.'+str(year)[-2:]+'i'
ion_file = './IGSGIM/IGS0OPSFIN_'+str(year)+str(doy)+'0000_01D_02H_GIM.INX'
tec_data = read_ionFile(ion_file)
tec_data = tec_data*0.1 

with open(orbit_filename + '/'+str(month)+str(day)+'glon.json', 'r') as file:
    glon_list = json.load(file)
with open(orbit_filename + '/'+str(month)+str(day)+'glat.json', 'r') as file:
    glat_list = json.load(file)
with open(epoch_filename + '/'+str(month)+str(day)+'sec.json', 'r') as file:
    sec_list = json.load(file)
with open(epoch_filename + '/'+str(month)+str(day)+'msec.json', 'r') as file:
    micsec_list = json.load(file)
with open(dion_filename + '/'+str(month)+str(day)+'dion.json', 'r') as file:
    dion_list = json.load(file)

vtec_list = []
epoch_list = []

for i in range(len(dion_list)):
    non_nan_indices  = np.where(~np.isnan(np.array(dion_list[i])))
    glon_list[i] = np.array(glon_list[i])[non_nan_indices]
    glon_list[i] = np.array([glon - 360 if glon >= 180 else glon for glon in glon_list[i]])
    glat_list[i] = np.array(glat_list[i])[non_nan_indices]
    sec_list[i] = np.array(sec_list[i])[non_nan_indices]
    micsec_list[i] = np.array(micsec_list[i])[non_nan_indices]
    vtec_list.append(-jason3_freq ** 2 / 40.3 * np.array(dion_list[i])[non_nan_indices] / 1e16)
    epoch_list.append([Time(init_epoch + timedelta(seconds=sec.item())).mjd for sec in sec_list[i]])
    
# DORIS data in 
elev_doris_list = []
ipp_lat_doris_list = []
ipp_lon_doris_list= []
vtec_doris_list = []
epoch_doris_list = []
with h5py.File('./vtec/DOY'+str(doy)+'.h5', 'r') as file:
    for pass_index in file['/y'+str(year)+'/d'+str(doy)+'/ele_cut_0']:
        pass_index_folder = file['/y'+str(year)+'/d'+str(doy)+f'/ele_cut_0/{pass_index}']

        vtec_doris_list += pass_index_folder['vtec']
        ipp_lat_doris_list += pass_index_folder['ipp_lat']
        ipp_lon_doris_list += pass_index_folder['ipp_lon']
        epoch_doris_list += pass_index_folder['epoch']
        elev_doris_list += pass_index_folder['elevation']

ipp_lat_int_list = []
ipp_lon_int_list = []
ipp_lon_doris_list1 = []
ipp_lat_doris_list1 = []
epoch_alt_list = []
for pass_num in [15,16]:
    for i, epoch_int in enumerate(epoch_list[pass_num]):
    # for i in data[pass_num]:
        # the following code is preparing reference vtec at altimetry ipp
        ipp_lat_int = glat_list[pass_num][i]
        ipp_lon_int = glon_list[pass_num][i]
        ipp_pos_int = np.stack((ipp_lon_int, ipp_lat_int), axis=0)
        ipp_vtec_int = vtec_list[pass_num][i]
        ipp_lat_int_list.append(ipp_lat_int)
        ipp_lon_int_list.append(ipp_lon_int)
    ## selection of data using time limit

indices1 = np.where(epoch_list[15][0]<epoch_doris_list)[0]
indices2 = np.where(epoch_list[16][-1]>epoch_doris_list)[0]
indices = list(set(indices1) & set(indices2))


# compensation in longitude for data taken in different time (very small)
ipp_lon_doris2 = np.array(ipp_lon_doris_list)[indices]
ipp_lat_doris2 = np.array(ipp_lat_doris_list)[indices]
ipp_vtec_doris2 = np.array(vtec_doris_list)[indices] 
epoch_doris2 = np.array(epoch_doris_list)[indices] 

indices3 = np.where(abs(epoch_list[15][1500]-epoch_doris2)<120/24/60/60)[0]
indices4 = np.where(abs(epoch_list[15][1500]-epoch_doris2)<120/24/60/60)[0]
indices_pos = list(set(indices3) & set(indices4))

ipp_lon_doris_ex = np.array(ipp_lon_doris2)[indices_pos]
ipp_lat_doris_ex = np.array(ipp_lat_doris2)[indices_pos]
ipp_vtec_doris_ex = np.array(ipp_vtec_doris2)[indices_pos]  

insti = 'DORIS'

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置整体的字体为Times New Roman
plt.figure(figsize=(10, 6), dpi=1000)  # 设置大小和分辨率

# 创建底图，设置地图投影为World Plate Carrée，分辨率为高分辨率，地图范围为全球

m = Basemap(
    projection='cyl',  # 使用等距圆柱投影
    llcrnrlon=-90,  # 左下角经度
    llcrnrlat=-30,  # 左下角纬度
    urcrnrlon=30,    # 右上角经度
    urcrnrlat=30,   # 右上角纬度
    resolution='f'
)

# 设置地图经纬线，并只在左端和底端显示
m.drawparallels(range(-30, 31, 30), labels=[1, 0, 0, 0], fontsize=10)  # 纬度
m.drawmeridians(range(-90, 31, 30), labels=[0, 0, 0, 1], fontsize=10)  # 经度
m.drawmapboundary(fill_color='lightblue')  # 海洋填充颜色
m.fillcontinents(color='lightgreen', lake_color='lightblue')  # 陆地填充颜色
m.drawcoastlines(linewidth=0.1)  # 绘制海岸线

plt.scatter(ipp_lon_int_list, ipp_lat_int_list, s = 20, c = '#ff7f0e', label = 'Altimetry IPP')
plt.scatter(ipp_lon_doris2, ipp_lat_doris2, s = 20, c = '#1f77b4', label = 'DORIS IPP')

rectangle = plt.Rectangle((ipp_lon_int_list[1500]-10.5, ipp_lat_int_list[1500]-5.5), 21, 11, 
                          edgecolor='black', facecolor='none', linewidth=2)
# plt.gca().add_patch(rectangle)
plt.scatter(ipp_lon_int_list[1500], ipp_lat_int_list[1500], s = 20, c = 'blue')
plt.scatter(ipp_lon_doris_ex, ipp_lat_doris_ex, s = 20, c = 'red')
plt.title('Illustration of Time Division Strategy', fontsize = 10)
plt.legend(loc='upper right')
fig_name = 'TEC Map of a' + str(insti)+'.png'
# 保存图片并显示
plt.savefig(fig_name, dpi=1000, bbox_inches='tight',
            pad_inches=0.1)  # 输出地图，并设置边框空白紧密
# plt.show()  # 显示地图


    