import json
import os
import numpy as np
from astropy.time import Time
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap
from readFile import read_ionFile
from readFile import read_ionRMSFile
from IGSVTECMap import GIM_inpo
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import Rbf, griddata
from sklearn.ensemble import RandomForestRegressor
import h5py
import warnings
import time
start_time = time.time()
warnings.filterwarnings("ignore", category=RuntimeWarning)
rms_results_np = []
def inverse_distance_weighting(points, values, target_point, power=2):

    distances = np.linalg.norm(points - target_point, axis=1)
    weights = 1.0 / (distances ** power)
    weights[distances == 0] = np.inf
    weights /= np.sum(weights)
    weighted_value = np.sum(weights * values)
    
    return weighted_value

def nearest_neighbor(points, values, target_point, power=2):

    distances = np.linalg.norm(points - target_point, axis=1)
    idx = np.argmin(distances)

    return values[idx]

# percentage of data eliminated from sigmatest; delete_point for storage of deleted indices
def sigmatest(doris_vtec_list, ipp_lat_int, ipp_lon_int, indices):
    std = np.std(doris_vtec_list) # if the TEC values are 3 std away from the base_value, not valid.
    mean = np.mean(doris_vtec_list)
    delete_point = []
    for idx in range(len(doris_vtec_list) - 1, -1, -1):
        if abs(doris_vtec_list[idx] - mean) > 3 * std:
            doris_vtec_list = np.delete(doris_vtec_list, np.s_[idx])
            ipp_lat_int = np.delete(ipp_lat_int, np.s_[idx])
            ipp_lon_int = np.delete(ipp_lon_int, np.s_[idx])
            delete_point.append(indices[idx])
        
    return doris_vtec_list, ipp_lat_int, ipp_lon_int, delete_point
for day in range(30,31):
    process_epoch = datetime(2019, 11, int(day)) # the data in user-defined DAY will be processed
    year = process_epoch.year
    month = process_epoch.month
    day = process_epoch.day
    doy = process_epoch.timetuple().tm_yday

    # elev_mask = 15

    lat_range = 7
    lon_range = lat_range * 2

    time_range = 180 #second
    int_strat = 'IDW'
    limit = 'location' # time or location
    elev_mask_list = [10]
    num_obs_list = [30]
    init_epoch = datetime(1985, 1, 1, 0, 0, 0)

    jason3_freq = 13.575e9
    current_path = os.getcwd()
    orbit_filename = current_path + '/Orbit'
    epoch_filename = current_path + '/Epoch'
    dion_filename = current_path + '/Dion'

    # IGS data
    ion_file = './passdetectionTest/igsion/igsg'+str(doy)+'0.'+str(year)[-2:]+'i'
    # ion_file = './IGSGIM/IGS0OPSFIN_'+str(year)+str(doy)+'0000_01D_02H_GIM.INX'
    
    tec_data = read_ionFile(ion_file)
    tec_data = tec_data*0.1 
    tec_RMSdata = read_ionRMSFile(ion_file) 
    tec_RMSdata = tec_RMSdata*0.1

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

    ipp_lat_list = []
    diff_doris_alt = []
    rms = []
    rms_igs_list = []
    DORIS_obs_precent = []
    std_vtec = []

    for elev_mask in elev_mask_list:
        indices_doris_ele = [index for index, ele in enumerate(elev_doris_list) if ele > elev_mask]
        vtec_doris_list = np.array(vtec_doris_list)[indices_doris_ele]
        ipp_lat_doris_list = np.array(ipp_lat_doris_list)[indices_doris_ele]
        ipp_lon_doris_list = np.array(ipp_lon_doris_list)[indices_doris_ele]
        epoch_doris_list = np.array(epoch_doris_list)[indices_doris_ele]
        elev_doris_list = np.array(elev_doris_list)[indices_doris_ele]
        for num_obs_least in num_obs_list:
            ipp_vtec_igs_inpo = []
            vtec_int_doris = []
            vtec_altimetry = []
            vtec_int_igs = []
            ipp_doris_lat = []
            ipp_doris_lon = []
            min_elev = []
            num_obs = []
            result_indices = []
            vtec_int_igs_inpo = []
            delete_count = [] # number of data deleted in the sigmatest

            # pass_num = 10
            for pass_num in range(len(epoch_list)):
                for i, epoch_int in enumerate(epoch_list[pass_num]):
                # for i in data[pass_num]:
                #     epoch_int = epoch_list[pass_num][i]
                    # the following code is preparing reference vtec at altimetry ipp
                    ipp_lat_int = glat_list[pass_num][i]
                    ipp_lon_int = glon_list[pass_num][i]
                    ipp_pos_int = np.stack((ipp_lon_int, ipp_lat_int), axis=0)
                    ipp_vtec_int = vtec_list[pass_num][i]
                    
                    ## selection of data using space limit
                    if limit == 'location':
                        result_indices = []
                        indices = np.where((np.abs(glat_list[pass_num][i]-ipp_lat_doris_list) < lat_range) \
                                            & (np.abs(glon_list[pass_num][i]-ipp_lon_doris_list) < lon_range)
                                                & (np.abs(epoch_int-epoch_doris_list) < 10/24/60) )[0]
                        for idx in indices:
                            ipp_lon = ipp_lon_doris_list[idx] + (epoch_doris_list[idx] - epoch_int)*360
                            if abs(ipp_lon - glon_list[pass_num][i]) < lon_range or abs(ipp_lon - glon_list[pass_num][i]) > 360 - lon_range:
                                result_indices.append(idx)
                        indices = result_indices
                        ipp_lon_doris = np.array(ipp_lon_doris_list)[indices]

                    ## selection of data using time limit
                    if limit == 'time':
                        indices = np.where(np.abs(epoch_int-epoch_doris_list) < time_range/24/3600)[0]
                        # compensation in longitude for data taken in different time (very small)
                        ipp_lon_doris = np.array(ipp_lon_doris_list)[indices] + (epoch_doris_list[indices] - epoch_int)*360

                    ipp_lat_doris = np.array(ipp_lat_doris_list)[indices]
                    ipp_vtec_doris = np.array(vtec_doris_list)[indices]   
                    ipp_elev_doris = np.array(elev_doris_list)[indices]  

                    # the following code gets rid of outliers in the doris vtec set
                    # base value is set to mean value in this epoch        
                    # ipp_vtec_doris, ipp_lat_doris, ipp_lon_doris, delete_point = sigmatest(ipp_vtec_doris, ipp_lat_doris, ipp_lon_doris, indices)
                    ipp_pos_doris = np.column_stack((ipp_lon_doris, ipp_lat_doris))
                    # if delete_point != []:
                    #     delete_count += delete_point

                    if len(ipp_vtec_doris) < num_obs_least: 

                        vtec_int_doris.append(None)
                        vtec_altimetry.append(None)
                        vtec_int_igs.append(None)
                        min_elev.append(None) 
                        num_obs.append(None)
                        vtec_int_igs_inpo.append(None)
                        continue
                    



                    # the following code is preparing igs data
                    igs_vtec = GIM_inpo(tec_data, tec_RMSdata,[epoch_int], [ipp_lat_int], [ipp_lon_int])[0]
                    # the following gets the doris interpolation results and igs results

                    # cal of igs_interpolation
                    # for i in range(len(indices)):
                    #     ipp_vtec_igs_inpo.append(GIM_inpo(tec_data, tec_RMSdata,[epoch_doris_list[indices][i]], [ipp_lat_doris[i]], [ipp_lon_doris[i]])[0])
                    
        
                    
                    ## IDW
                    if int_strat == 'IDW':
                        vtec_int_doris.append(inverse_distance_weighting(ipp_pos_doris, ipp_vtec_doris, ipp_pos_int))
                        # cal of igs interpolation
                        # vtec_int_igs_inpo.append(inverse_distance_weighting(ipp_pos_doris, np.array(ipp_vtec_igs_inpo)*0.925, ipp_pos_int))
                    ipp_vtec_igs_inpo = []
                    std_vtec.append(np.std(ipp_vtec_doris))


                    # model = RandomForestRegressor(n_estimators=100, random_state=42)
                    # model.fit(ipp_pos_doris, ipp_vtec_doris)
                    # vtec_int_doris.append(model.predict([[ipp_lon_int, ipp_lat_int]]))

                    ## krige
                    if int_strat == 'krige':
                        OK = OrdinaryKriging(
                        np.array(ipp_lon_doris),
                        np.array(ipp_lat_doris), 
                        np.array(ipp_vtec_doris),
                        variogram_model='linear',
                        verbose=False,
                        enable_plotting=False
                        )
                        vtec_at_target, _ = OK.execute('points', np.array([ipp_lon_int]), np.array([ipp_lat_int]))
                        vtec_int_doris.append(vtec_at_target[0])

                    vtec_int_igs.append(igs_vtec * 0.925)
                    vtec_altimetry.append(ipp_vtec_int)
                    # if abs(ipp_vtec_int - vtec_int_doris[-1]) >10:
                    #     print('')
                    min_elev.append(np.min(ipp_elev_doris)) 
                    num_obs.append(len(ipp_vtec_doris))

                    diff_doris_alt.append(vtec_int_doris[-1]-vtec_altimetry[-1])
                    ipp_lat_list.append(ipp_lat_int)
                               
            vtec_int_doris = np.where(vtec_int_doris == None, np.nan, vtec_int_doris).astype(float)
            # vtec_int_igs_inpo = np.where(vtec_int_igs_inpo == None, np.nan, vtec_int_igs_inpo).astype(float)
            vtec_int_igs = np.where(vtec_int_igs == None, np.nan, vtec_int_igs).astype(float)
            vtec_altimetry = np.where(vtec_altimetry == None, np.nan, vtec_altimetry).astype(float)
            # print(np.mean(std_vtec))

            # bins = [-90,-60, -20, 20, 60, 90]

            # # 使用numpy的digitize函数进行分组
            # indices = np.digitize(ipp_lat_list, bins)

            # # 使用列表推导式，结合numpy计算每个区间的RMS
            # rms_results_np.append([
            #     np.sqrt(np.mean(np.array(diff_doris_alt)[indices == q] ** 2)) if np.any(indices == q) else None
            #     for q in range(1, len(bins))
            # ])

            mask = ~np.isnan(vtec_int_doris)
            a = [aa for aa in mask if aa == 1]
            DORIS_obs_precent.append(len(a)/sum(len(sublist) for sublist in epoch_list))
            mean_doris = np.mean(vtec_int_doris[mask]) - np.mean(vtec_altimetry[mask])
            
            mean_igs = np.mean(vtec_int_igs[mask]) - np.mean(vtec_altimetry[mask])

            rms_doris = np.sqrt(np.mean(np.square(vtec_int_doris[mask] - vtec_altimetry[mask])))
      
            rms_igs = np.sqrt(np.mean(np.square(vtec_int_igs[mask] - vtec_altimetry[mask])))
       
            # rms_igs_inpo = np.sqrt(np.mean(np.square(vtec_int_igs_inpo[mask] - vtec_altimetry[mask])))
            # print(rms_igs_inpo)
            rms.append(rms_doris)
            rms_igs_list.append(rms_igs)



    for xi, yi, zi in zip(DORIS_obs_precent, rms, rms_igs_list):
        formatted_xi = f"{round(xi, 2):.2f}"
        formatted_yi = f"{round(yi, 2):.2f}"
        formatted_zi = f"{round(zi, 2):.2f}"
        print(f"{formatted_xi}-{formatted_yi}-{formatted_zi}")
    count = sum(len(sub_list) for sub_list in glon_list)


# 
# print(count)
# print(f'num_obs_elev{elev_mask}:{len(vtec_doris_list)}')
# print(f'delete_count:{len(np.unique(delete_count))}')
# print(f'mean_doris_diff:{mean_doris}')
# print(f'mean_igs_diff:{mean_igs}')
# print(f'rms_doris:{rms_doris}')
# print(f'rms_igs:{rms_igs}')
# print(f'num_obs:{len(a)}')

# plt.figure(figsize=(20, 10))
# plt.subplot(211)
# plt.title('Comparison of DORIS and IGS with altimetry')
# plt.plot(vtec_int_doris[mask] - vtec_altimetry[mask], label='DORIS-altimetry')
# plt.plot(vtec_int_igs[mask] - vtec_altimetry[mask], label='IGS-altimetry')
# plt.legend()
# plt.subplot(212)
# plt.title('altimetry VTEC')
# plt.plot(vtec_altimetry[mask], label='altimetry value')
# plt.legend()
# plt.show

# x=[221,222,223,224,225,226,227]
# rms_doris = [4.35, 4.84, 4.21, 4.69, 4.65, 4.51, 5.03]
# rms_igs = [4.53, 4.77, 4.48, 4.65, 4.46, 4.32, 4.9]
# plt.figure(figsize=(20, 10))
# plt.subplot(211)
# plt.title('rms value in comparison', fontsize=16)
# plt.xlabel('DOY')
# plt.plot(x, rms_doris, label='rms_doris', linewidth=3)
# plt.plot(x, rms_igs, label='rms_igs', linewidth=3)
# for i in range(len(x)):
#     plt.text(x[i], rms_doris[i], str(rms_doris[i]), fontsize=16, ha='left', va='bottom')
#     plt.text(x[i], rms_igs[i], str(rms_igs[i]), fontsize=16, ha='left', va='bottom')
# plt.legend()
# mean_doris = [3.5, 3.83, 3, 3.67, 3.89, 3.51, 4.18]
# mean_igs = [3.55, 3.68, 3.03, 3.62, 3.55, 3.33, 4.18]
# plt.subplot(212)
# plt.title('mean value in comparison', fontsize=16)
# plt.plot(x, mean_doris, label='mean_doris', linewidth=3)
# plt.plot(x, mean_igs, label='mean_igs', linewidth=3)
# for i in range(len(x)):
#     plt.text(x[i], mean_doris[i], str(mean_doris[i]), fontsize=16, ha='left', va='bottom')
#     plt.text(x[i], mean_igs[i], str(mean_igs[i]), fontsize=16, ha='left', va='bottom')
# plt.legend()

# diff_doris = vtec_int_doris[mask]-vtec_altimetry[mask]
# diff_lat = np.concatenate(glat_list)[mask]
# plt.figure(figsize=(20,10))
# plt.scatter(diff_lat,diff_doris)
# plt.xlabel('latitude[deg]', fontsize=14)
# plt.ylabel('VTEC difference for DORIS[TECU]', fontsize=14)
# plt.title(f'Impact of latitude in time-divided VTEC computation in DOY{doy}', fontsize=16)

end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.2f} seconds")