# -*- encoding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import typing as T
import tables
import os
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from readFile import read_ionFile
import h5py


def mjd_to_time(mjd):
    dt = datetime(1858, 11, 17) + timedelta(days=mjd)
    return dt

def GIMPlot(ion_file):
    # Read VTEC data from ion file
    lltec = read_ionFile(ion_file)
    lltec = lltec*0.1 # i文件单位为0.1tecu
    insti = ion_file[0:3]
    day = ion_file[15:18]
    yr = ion_file[11:15]

    for i in range(lltec.shape[0]):
        
        tec = lltec[i]
        tec = tec[::-1, :]
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.figure(figsize=(10, 6), dpi=1000)
        m = Basemap(projection='cyl', resolution='h', llcrnrlon=-180, llcrnrlat=-87.5, urcrnrlon=180, urcrnrlat=87.5)
        m.drawmeridians(np.arange(-180, 181, 30), labels=[0, 0, 0, 1], fontsize=14, linewidth=0.3, color='grey')
        m.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0], fontsize=14, linewidth=0.3, color='grey')
        m.drawcoastlines(linewidth=0.5)
        m.imshow(tec, extent=(-180, 180, 87.5, -87.5), origin='lower', cmap='jet', interpolation='bilinear',
                  vmin = np.min(lltec), vmax = np.max(lltec))
        
        cb = m.colorbar(location='right', pad=0.1, size = 0.2) 
        cb.set_label('TECU', fontsize=14) 
        fig_name = 'TEC Map.png'
        
        plt.title('GIM VTEC Map on 2023-08-10-00:00', fontsize = 14)
        plt.savefig(fig_name, dpi=600, bbox_inches='tight',
                    pad_inches=0.1)
        if i ==0:
            break
#GIMPlot('IGSGIM\IGS0OPSFIN_20232220000_01D_02H_GIM.INX')
def GIM_inpo(lltec, epoch: list[float], IPPlat: list[float], IPPlon: list[float]) -> list[float]: 
    # lltec is the data read in by read_ionFile
    # make sure the epochs of IPP data only span through a day, not more days

    lat_grid_interval = 2.5
    lon_grid_interval = 5
    lat_start = -87.5
    lon_start = -180
    lat_ind_max = int( - lat_start * 2 / lat_grid_interval + 1)
    lon_ind_max = int( - lon_start * 2 / lon_grid_interval + 1)

    # 2D coordinates
    IPPlat_ind = (IPPlat - lat_start) / lat_grid_interval
    IPPlon_ind = (IPPlon - lon_start) / lon_grid_interval
    # time 

    # ind is the index of the two-hour interval; 
    # diff is the difference with the closest even hour thats smaller, converted into fraction
    epoch_ind = []
    epoch_diff = []
    epoch = [mjd_to_time(obj) for obj in epoch]
    for i in range(len(epoch)):
        epoch_ind.append(epoch[i].hour // 2)
        epoch_diff.append(((epoch[i].hour + epoch[i].minute / 60.0 + epoch[i].second / 3600.0) % 2) / 2)
    IPP_vtec = []

    for i in range(len(epoch)):
        igs_vtec_now = lltec[epoch_ind[i]]
        igs_vtec_now = igs_vtec_now[::-1, :] # in ion file the latitude is ranging from 87.5~-87.5, while should be reversed

        igs_vtec_next = lltec[epoch_ind[i] + 1]
        igs_vtec_next = igs_vtec_next[::-1, :]
        # bilinear interpolation 
        if i == 0 or epoch_ind[i] != epoch_ind [i - 1]:
            interp_func_now = RegularGridInterpolator((list(range(0, lat_ind_max)), list(range(0, lon_ind_max))),
                igs_vtec_now, method='linear')
            interp_func_next = RegularGridInterpolator((list(range(0, lat_ind_max)), list(range(0, lon_ind_max))),
                igs_vtec_next, method='linear')
        # time interpolation
        IPP_vtec.append((1 - epoch_diff[i]) * interp_func_now([IPPlat_ind[i], IPPlon_ind[i]])[0] 
             + epoch_diff[i] * interp_func_next([IPPlat_ind[i], IPPlon_ind[i]])[0])
        
    return IPP_vtec

def split_array(arr, split_indices):
    result = []
    start_index = 0

    for index in split_indices:
        result.append(arr[start_index:index])
        start_index = index

    result.append(arr[start_index:])

    return result


# GIMPlot('IGS0OPSFIN_20232210000_01D_02H_GIM.INX')

# scale_factor_doris = 0.925 # accouting for Jason-3 orbit height difference with GNSS height

# IPPlats = []; IPPlons = []; STEC = []; STEC_c = []; elev = []; map_value = []; epoch = []
# hours = ['00', '01', '02', '03', '05']
# hours = ['01']

# for hour in hours:
#     dataDir = 'Test.h5'
#     h5file = tables.open_file(dataDir, mode = 'r')
#     group = h5file.get_node('/y2023/d221/h' + str(hour))

#     IPPlat_set = np.array(group.IPP_lats)
#     IPPlon_set = np.array(group.IPP_lons)
#     STEC_set = np.array(group.stec)
#     STEC_c_set = np.array(group.stec_c)
#     epoch_set = np.array(group.epochs)
#     elev_set = np.array(group.elevs)
#     map_value_set = np.array(group.mf)

#     IPPlats = np.concatenate((IPPlats, IPPlat_set), axis=0)
#     IPPlons = np.concatenate((IPPlons, IPPlon_set), axis=0)
#     STEC = np.concatenate((STEC, STEC_set), axis=0)
#     STEC_c = np.concatenate((STEC_c, STEC_c_set), axis=0)
#     epoch = np.concatenate((epoch, epoch_set), axis=0)
#     elev = np.concatenate((elev, elev_set), axis=0)
#     map_value = np.concatenate((map_value, map_value_set), axis=0)
    
#     h5file.close()

# epoch = [mjd_to_time(mjd) for mjd in epoch]
# # longitude 0 ~ 360 to -180 ~ 180
# for i in range(len(IPPlons)):
#     if IPPlons[i] > 180:
#         IPPlons[i] -= 360
# # separation of different obs arc, noticing that every arc starts with a zero value in STEC
# div_ind_stec = np.where(STEC == 0)[0]
# div_ind_stec = div_ind_stec[1:]
# arc_count = len(div_ind_stec) + 1 # plus one, eg. two separations in a list can make three arcs

# igs_vtec = np.array(GIM_inpo('./passdetectionTest/IGS0OPSFIN_20232210000_01D_02H_GIM.INX', epoch, IPPlats, IPPlons))
# igs_stec = igs_vtec * map_value

# # separation of the observables into different obs arc
# div_elev = split_array(elev, div_ind_stec)
# div_doris_stec = split_array(STEC, div_ind_stec)
# div_igs_vtec = split_array(igs_vtec, div_ind_stec)
# div_igs_stec = split_array(igs_stec, div_ind_stec)
# div_epoch = split_array(epoch, div_ind_stec)
# div_map_value = split_array(map_value, div_ind_stec)

# ## calibration of DORIS STEC to IGS level
# # indice where elevation is maximum
# max_indices = []
# for i in range(len(div_elev)):
#     max_index = np.argmax(div_elev[i])
#     max_indices.append(max_index)
# # Difference of DORIS and IGS STEC in each arc
# Delta = []
# offset_std = []
# for i in range(arc_count):
#     Delta.append(div_igs_stec[i][max_indices[i]] * scale_factor_doris - div_doris_stec[i][max_indices[i]])
#     offset_std.append(np.std(div_igs_stec[i] * scale_factor_doris - div_doris_stec[i]))
# div_doris_vtec_cal = []
# for i in range(arc_count):
#     div_doris_vtec_cal.append((div_doris_stec[i] + Delta[i]) / div_map_value[i])

# # statistics of VTEC differences in DORIS and IGS
# vtec_mean = []
# vtec_std = []
# for i in range(arc_count):
#     if np.max(div_elev[i]) > 20 and offset_std[i] < 5: # maximum elevation < 20, or std of offset higher than 5 TECU
#         vtec_mean.append(np.mean((div_doris_vtec_cal[i] - scale_factor_doris * div_igs_vtec[i])))
#         vtec_std.append(np.std((div_doris_vtec_cal[i] - scale_factor_doris * div_igs_vtec[i])))

# mean = np.mean(vtec_mean) / scale_factor_doris # comparison in GNSS height
# std = np.mean(vtec_std) / scale_factor_doris

# # examine the relation of offset_std and elevation + VTEC level

# # div_max_elev = []
# # div_max_vtec = []
# # for i in range(arc_count):
# #     div_max_elev.append(np.max(div_elev[i]))
# #     div_max_vtec.append(np.max(div_igs_vtec[i]))
# # offset_elev_vtec = np.vstack((offset_std, div_max_elev, div_max_vtec))
# # sorted_indices = np.argsort(offset_elev_vtec[0])
# # sorted_arr = offset_elev_vtec[:, sorted_indices]

# # fig, ax = plt.subplots()
# # # TODO: 深度学习预测下std的阈值？
# # ax.scatter((sorted_arr[1] ** 2 + sorted_arr[2] ** 2) ** 0.5, sorted_arr[0], c='r', marker='o')

# # ax.set_xlabel('max_elev')
# # ax.set_ylabel('max_vtec')
# # plt.show()

# ind = 7
# fig, ax = plt.subplots(figsize=(10, 8))
# ax.plot(div_epoch[ind], div_doris_stec[ind], label='DORIS STEC', color='red', linestyle='-')
# ax.plot(div_epoch[ind], div_doris_stec[ind]+Delta[ind], label='DORIS STEC', color='red', linestyle='--')
# ax.plot(div_epoch[ind], div_igs_stec[ind]*scale_factor_doris, label='IGS GIM STEC', color='black', linestyle='-')
# # draw the point where elevation is maximum
# ax.scatter(div_epoch[ind][max_indices[ind]], div_igs_stec[ind][max_indices[ind]]*scale_factor_doris, color='black')
# ax.scatter(div_epoch[ind][max_indices[ind]], div_doris_stec[ind][max_indices[ind]], color='black')
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# fig.autofmt_xdate()
# plt.xlabel('time [h]')
# plt.ylabel('STEC [TECU]')
# plt.title('STEC Comparison in DOY221 for sarc in C++')
# plt.legend()
# ax.grid(True, which='both', linestyle='--')
# ax.set_xlim(div_epoch[ind][0]-timedelta(minutes=5), div_epoch[ind][-1]+timedelta(minutes=5))
# ax.set_ylim(-100, 200)
# ax.tick_params(axis='x', labelrotation=0)
# plt.show()


