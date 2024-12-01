import numpy as np
import matplotlib.pyplot as plt
import typing as T
import matplotlib.dates as mdates
from readFile import read_ionFile
from IGSVTECMap import GIM_inpo
from datetime import datetime, timedelta
import h5py

div_igs_vtec = []
div_igs_stec = []
div_doris_stec = []
div_elev = []
div_epoch = []
div_map_value = []


ion_file = './passdetectionTest/IGS0OPSFIN_20232210000_01D_02H_GIM.INX'

with h5py.File('Test.h5', 'r') as file:

    tec_data = read_ionFile(ion_file)
    tec_data = tec_data*0.1 # INX unit being 0.1tecu

    for pass_index in file['/y2023/d221/ele_cut_0']:
        pass_index_folder = file[f'/y2023/d221/ele_cut_0/{pass_index}']

        epoch = pass_index_folder['epoch']
        ipp_lon = np.array(pass_index_folder['ipp_lon'])
        ipp_lat = np.array(pass_index_folder['ipp_lat'])
        stec = np.array(pass_index_folder['stec'])
        elev = np.array(pass_index_folder['elevation'])
        map_value = np.array(pass_index_folder['map_value'])

        # In the following part, IGS VTEC values are used for leveling of DORIS STEC

        div_igs_vtec.append(np.array(GIM_inpo(tec_data, epoch, ipp_lat, ipp_lon)))
        div_map_value.append(map_value)
        div_igs_stec.append(div_igs_vtec[-1]*map_value)
        div_doris_stec.append(stec)
        div_elev.append(elev)
        div_epoch.append(epoch)   

arc_count = len(div_epoch)
scale_factor_doris = 0.925
max_indices = []
for i in range(len(div_elev)):
    max_index = np.argmax(div_elev[i])
    max_indices.append(max_index)
# Difference of DORIS and IGS STEC in each arc
Delta = []
offset_std = []
for i in range(arc_count):
    Delta.append(div_igs_stec[i][max_indices[i]] * scale_factor_doris - div_doris_stec[i][max_indices[i]])
    offset_std.append(np.std(div_igs_stec[i] * scale_factor_doris - div_doris_stec[i]))
div_doris_vtec_cal = []
for i in range(arc_count):
    div_doris_vtec_cal.append((div_doris_stec[i] + Delta[i]) / div_map_value[i])

# statistics of VTEC differences in DORIS and IGS
vtec_mean = []
vtec_std = []
for i in range(arc_count):
    if np.max(div_elev[i]) > 20 and offset_std[i] < 5: # maximum elevation > 20, or std of offset < 5 TECU
        vtec_mean.append(np.mean((div_doris_vtec_cal[i] - scale_factor_doris * div_igs_vtec[i])))
        vtec_std.append(np.std((div_doris_vtec_cal[i] - scale_factor_doris * div_igs_vtec[i])))

mean = np.mean(vtec_mean) / scale_factor_doris # comparison in GNSS height
std = np.mean(vtec_std) / scale_factor_doris

print(mean)
print(std)

# ind = 8
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
# plt.title('STEC Comparison in DOY221 for sarc in python')
# plt.legend()
# ax.grid(True, which='both', linestyle='--')
# ax.set_xlim(div_epoch[ind][0]-timedelta(minutes=5), div_epoch[ind][-1]+timedelta(minutes=5))
# ax.set_ylim(-100, 200)
# ax.tick_params(axis='x', labelrotation=0)
# plt.show()
    
