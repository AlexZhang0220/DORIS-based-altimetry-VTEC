import json
import numpy as np
import h5py
import time
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
from astropy.time import Time
from datetime import datetime, timedelta
from readFile import read_ionFile
from tools import haversine_vec, compute_lat_lon_distances

# inverse distance weighting
def idw(dist, values):
    # elevation weighting is not better
    dist[dist == 0] = 1e-12  # Avoid division by zero
    weights = 1 / dist 
    weights /= weights.sum(axis=0)
    return np.dot(values, weights).item()

# GIM VTEC interpolation function
def interpolate_gim_vtec(gim_vtec, ipp_epoch, ipp_lat, ipp_lon):
    max_lat_index = 71
    max_lon_index = 73
    ipp_epoch = Time(ipp_epoch, format='mjd').to_datetime()
    interp_funcs = [
        RegularGridInterpolator((list(range(0, max_lat_index)), list(range(0, max_lon_index))), gim_vtec[i], method='linear')
        for i in range(13)
    ]

    interpolated_vtec = []
    for i, epoch in enumerate(ipp_epoch):
        lat_idx = (ipp_lat[i] + 87.5) / 2.5
        lon_idx = (ipp_lon[i] + 180) / 5
        time_index = epoch.hour // 2
        time_scale = ((epoch.hour + epoch.minute / 60 + epoch.second / 3600) % 2) / 2
        vtec = (
            (1 - time_scale) * interp_funcs[time_index]([lat_idx, lon_idx])[0] +
            time_scale * interp_funcs[time_index + 1]([lat_idx, lon_idx])[0]
        )
        interpolated_vtec.append(vtec)
    return interpolated_vtec

# for altimetry data where the 2D list has various columns for each row
def load_and_flatten(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return np.array([item for sublist in data for item in sublist])

# Load and preprocess altimetry data
def load_altimetry_data(month, day):    
    alt_lon = load_and_flatten(f'./AltimetryData/Orbit/{month}{day}glon.json')
    alt_lat = load_and_flatten(f'./AltimetryData/Orbit/{month}{day}glat.json')
    alt_epoch = load_and_flatten(f'./AltimetryData/Epoch/{month}{day}sec.json')
    alt_vtec = load_and_flatten(f'./AltimetryData/Dion/{month}{day}dion.json')

    valid_indices = ~np.isnan(alt_vtec)
    alt_lon, alt_lat, alt_epoch, alt_vtec = (
        arr[valid_indices] for arr in (alt_lon, alt_lat, alt_epoch, alt_vtec)
    )
    alt_lon = np.array([lon - 360 if lon >= 180 else lon for lon in alt_lon])
    base_time = np.datetime64('1985-01-01T00:00:00')
    mjd_epoch = np.datetime64('1858-11-17T00:00:00')
    alt_epoch = ((base_time + alt_epoch.astype('timedelta64[s]')) - mjd_epoch) / np.timedelta64(1, 'D')
    alt_vtec = -13.575e9**2 / 40.3 * alt_vtec / 1e16
    return alt_lon, alt_lat, alt_epoch, alt_vtec

# Load and preprocess doris data
def load_doris_data(year, doy):

    longitudes = []
    latitudes = []
    epochs = []
    vtec_values = []
    elevations = []
    file_path = f'./DORISVTEC/{year}/DOY{doy:03d}.h5'
    # file_path = f'./DOY{doy:03d}.h5'a

    with h5py.File(file_path, 'r') as file:
        # passes_group = file[f'/y{year}/d{doy:03d}/ele_cut_0']
        passes_group = file[f'/y{year}/{doy:03d}/count{30}']

        for pass_index in passes_group:
            # Access the data for each pass
            pass_data = passes_group[pass_index]
            longitudes.extend(pass_data['ipp_lon'][:])
            latitudes.extend(pass_data['ipp_lat'][:])
            epochs.extend(pass_data['epoch'][:])
            vtec_values.extend(pass_data['vtec'][:])
            elevations.extend(pass_data['elevation'][:])

    return (
        np.array(longitudes),
        np.array(latitudes),
        np.array(epochs),
        np.array(vtec_values),
        np.array(elevations),
    )


def doris_npy2csv(file_name: str, out_file):
    data = np.load(file_name)
    with open(out_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        ele_count, min_obs_count, lat_range_count, _ = data.shape
        
        # 遍历第一维（ele）
        for ele_idx in range(ele_count):
            ele_header = f"Ele {ele_idx + 1}"  # Ele 标识
            
            for metric_idx, metric_name in enumerate(["RMS", "Percent"]):  # 遍历第四维
                # 写入表格标题
                writer.writerow([f"{ele_header} - {metric_name}"])
                
                # 写入列标头（lat_range）
                lat_headers = [f"Lat {j+1}" for j in range(lat_range_count)]
                writer.writerow(["Min Obs \\ Lat Range"] + lat_headers)
                
                # 写入表格数据
                for min_obs_idx in range(min_obs_count):
                    row_data = [f"Min Obs {min_obs_idx + 1}"]  # 行标头
                    row_data.extend(data[ele_idx, min_obs_idx, :, metric_idx])  # 取出一行的数据
                    writer.writerow(row_data)
                
                # 空行分隔不同表格
                writer.writerow([])

    print(f"Data successfully saved to {out_file}.")

def gim_npy2csv(file_name: str,out_file):
    data = np.load(file_name)
    with open(out_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        ele_count, min_obs_count, lat_range_count = data.shape
        
        # 遍历第一维（ele）
        for ele_idx in range(ele_count):
            ele_header = f"Ele {ele_idx + 1}"  # Ele 标识
            
            # 写入表格标题
            writer.writerow([f"{ele_header} - RMS"])
            
            # 写入列标头（lat_range）
            lat_headers = [f"Lat {j+1}" for j in range(lat_range_count)]
            writer.writerow(["Min Obs \\ Lat Range"] + lat_headers)
            
            # 写入表格数据
            for min_obs_idx in range(min_obs_count):
                row_data = [f"Min Obs {min_obs_idx + 1}"]  # 行标头
                row_data.extend(data[ele_idx, min_obs_idx, :])  # 取出一行的数据
                writer.writerow(row_data)
            
            # 空行分隔不同表格
            writer.writerow([])

    print(f"Data successfully saved to {out_file}.")

def ROTI(epoch_segments, vtec_segments):
    all_rates = []

    for epochs, vtecs in zip(epoch_segments, vtec_segments):
        if len(epochs) < 2:
            continue  # 忽略不满足计算条件的段
        
        time_diff = np.diff(epochs) * 1440  # 天 -> 分钟
        vtec_diff = np.diff(vtecs)
        rate = vtec_diff / time_diff

        all_rates.extend(rate)

    if len(all_rates) == 0:
        return np.nan  # 如果没有可用数据

    return np.std(all_rates)

def find_doris_indices(doris_lon_time, doris_lat_time, min_obs_count, 
                       lat_range, max_lat_gap, alt_lat_idx, alt_lon_idx):
    """
    根据经纬度逐步扩大搜索窗口，寻找至少 min_obs_count 个满足条件的 DORIS 点。
    条件为：纬度差 <= lat_threshold，且经度方向距离 <= lon_distance_threshold_km。
    如果纬度差超过max_lat_gap仍找不到足够点，则返回空列表。
    """
    target_lat = alt_lat_idx
    target_lon = alt_lon_idx

    lat_threshold = lat_range
    lon_distance_threshold_km = lat_threshold * 2 * 111.0  # 初始窗口

    while lat_threshold <= max_lat_gap:
        lat_diff = np.abs(doris_lat_time - target_lat)
        lat_mask = lat_diff <= lat_threshold

        lon_diff_deg = np.abs((doris_lon_time - target_lon + 180) % 360 - 180)
        lon_km_per_deg = np.cos(np.radians(target_lat)) * 111.0
        lon_diff_km = lon_diff_deg * lon_km_per_deg
        lon_mask = lon_diff_km <= lon_distance_threshold_km

        combined_mask = lat_mask & lon_mask
        matching_indices = np.where(combined_mask)[0]

        if len(matching_indices) >= min_obs_count:
            return matching_indices.tolist()

        lat_threshold += 1.0
        lon_distance_threshold_km += 2 * 111.0  

    return []

def apply_earth_rotation_correction(lon, epoch, ref_epoch):
    """
    对经度坐标应用地球自转改正，使其对应于 ref_epoch 时刻。
    
    参数:
        lon (np.ndarray): 原始经度数组（单位：度）
        epoch (np.ndarray): 对应观测时刻（单位：天，如 Julian Day）
        ref_epoch (float): 目标参考时刻（单位：天）

    返回:
        np.ndarray: 经过改正并归一化到 [-180, 180] 的经度数组
    """
    omega = 7.2921159e-5  # 地球自转角速度 [rad/s]
    delta_t = (epoch - ref_epoch) * 86400  # 时间差 [秒]
    delta_lon_deg = np.degrees(omega * delta_t)  # 转换为角度

    lon_corrected = lon + delta_lon_deg
    lon_normalized = (lon_corrected + 180) % 360 - 180  # 归一化到 [-180, 180]

    return lon_normalized

def average_vector_magnitude(lat0, lon0, lat1, lon1, R=6371 + 450):
    """
    计算从参考点 (lat0, lon0) 到一组点之间的向量之和的平均向量的模长。
    
    参数：
    - lat0, lon0: 参考点的纬度和经度（单位：度）
    - lat1, lon1: 多个点的纬度和经度（单位：度）
    - R: 球半径，单位为 km，默认为 6371 + 450 km
    
    返回：
    - 平均向量的模长（单位：km）
    """
    # 将角度转为弧度
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    
    # 将参考点和目标点转为三维直角坐标
    def sph2cart(lat, lon, r):
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        return np.stack((x, y, z), axis=-1)
    
    P0 = sph2cart(lat0_rad, lon0_rad, R)         # (3,)
    Pn = sph2cart(lat1_rad, lon1_rad, R)           # (N, 3)
    
    vectors = Pn - P0                             # 从参考点指向其他点的向量 (N, 3)
    sum_vector = np.sum(vectors, axis=0)          # 向量求和 (3,)
    avg_vector = sum_vector / len(lon1_rad)         # 平均向量 (3,)
    
    magnitude = np.linalg.norm(avg_vector)        # 平均向量的模长
    return magnitude

# Main processing
if __name__ == '__main__':
    
    start_time = time.time()
    # inpo_results_plot(2024,5,8,30,['obs-30','ele-10','lat-8'])    
    # Settings
    start_year = 2024
    start_month = 5
    start_day = 13
    process_days = 5
    start_epoch = datetime(start_year, start_month, start_day)
    start_doy = start_epoch.timetuple().tm_yday

    division_strategy = 'Space'  # 'Time' or 'Space' 
    if division_strategy == 'Time':
        ele_mask_list = [10]
        min_obs_count_list = [30]
        time_gap_list = range(120,241,60)
        results_shape = (len(ele_mask_list), len(min_obs_count_list), len(time_gap_list), 4)
        division_str = 'sec-'+'-'.join(map(str, time_gap_list))
    else:
        weighting_method = 'Window'
        ele_mask_list = [10]
        min_obs_count_list = [30]
        lat_range_list = [1]
        lon_range_list = [lat * 2 for lat in lat_range_list]
        max_lat_gap = 10
        roti_threshold = 1

        results_shape = (len(ele_mask_list), len(min_obs_count_list), len(lon_range_list), 4)
        division_str = 'lat-'+'-'.join(map(str, lat_range_list))

    ele_str = 'ele-'+'-'.join(map(str, ele_mask_list))
    min_obs_str = 'obs-'+'-'.join(map(str, min_obs_count_list))

    # Initialize results arrays
    doris_results = np.zeros(results_shape)
    gim_rms_results = np.zeros(results_shape[:-1])
    doris_ipp_results = np.zeros(results_shape[:-1])

    # Loop over processing interval
    for day_idx, day_offset in enumerate(range(process_days)):

        doris_alt_vtec_mean = []
        process_date = start_epoch + timedelta(days=day_offset)
        year, month, day = process_date.year, process_date.month, process_date.day
        doy = process_date.timetuple().tm_yday
        
        # Load altimetry, DORIS and GIM data 
        alt_lon, alt_lat, alt_epoch, alt_vtec = load_altimetry_data(month, day)
        doris_lon, doris_lat, doris_epoch, doris_vtec, doris_ele = load_doris_data(year, doy)
        
        ion_file = (
            f"./DORISInput/IGSGIM/{year}/igsg{doy:03d}0.{str(year)[-2:]}i"
            if year <= 2022 else
            f"./DORISInput/IGSGIM/{year}/IGS0OPSFIN_{year}{doy:03d}0000_01D_02H_GIM.INX"
        )
        gim_vtec = read_ionFile(ion_file) * 0.1
        gim_vtec = gim_vtec[:, ::-1, :]
        gim_vtec_alt_ipp = interpolate_gim_vtec(gim_vtec, alt_epoch, alt_lat, alt_lon)

        for ele_mask_idx, ele_mask in enumerate(ele_mask_list):
            valid_indices = np.where(doris_ele > ele_mask)
            # one needs to make sure that the ele mask list is increasing
            doris_lon, doris_lat, doris_epoch, doris_vtec, doris_ele = (
                arr[valid_indices] for arr in (doris_lon, doris_lat, doris_epoch, doris_vtec, doris_ele)
            )        
            for min_obs_idx, min_obs_count in enumerate(min_obs_count_list):            
                # Process results based on strategy
                if division_strategy == 'Time':
                    for time_gap_idx, time_gap in enumerate(time_gap_list):
                        doris_vtec_alt_ipp = []
                        doris_std_alt_ipp = [] # std of the interpolation points
                        doris_points_alt_ipp = [] # number of the interpolation points
                        doris_dist_alt_ipp = []
                        gim_vtec_alt_ipp_doris = []
                        alt_vtec_alt_ipp_doris = []
                        doris_lat_alt_ipp = []
                        doris_lon_alt_ipp = []

                        for alt_epoch_idx, alt_epoch_proc in enumerate(alt_epoch):
                            valid_indices = np.where(np.abs(doris_epoch - alt_epoch_proc) < time_gap/24/3600)
                            doris_lon_inpo, doris_lat_inpo, doris_epoch_inpo, doris_vtec_inpo, doris_ele_inpo = (
                                arr[valid_indices] for arr in (doris_lon, doris_lat, doris_epoch, doris_vtec, doris_ele)
                            )     
                            doris_lon_inpo += (doris_epoch_inpo - alt_epoch_proc) * 360
                            doris_lon_inpo = np.where(doris_lon_inpo > 180, doris_lon_inpo - 360, doris_lon_inpo)
                            valid_indices_1 = np.where(np.abs(doris_epoch - alt_epoch_proc) < 120/24/3600)
                            if len(valid_indices_1[0]) < min_obs_count: continue
                            else:
                                alt_lat_idx = alt_lat[alt_epoch_idx]
                                alt_lon_idx = alt_lon[alt_epoch_idx]
                                dists = haversine_vec(alt_lat_idx, alt_lon_idx, doris_lat_inpo, doris_lon_inpo)        
                                d_selected = np.array(dists)
                                d_selected[d_selected == 0] = 1e-12  # Avoid division by zero
                                weights = 1 / d_selected 
                                weights /= weights.sum(axis=0)
                                est_iono = np.sum(weights * doris_vtec_inpo)   
                                if est_iono < 1110:                             
                                    doris_vtec_alt_ipp.append(est_iono)
                                    doris_std_alt_ipp.append(np.std(doris_vtec_inpo))
                                    doris_points_alt_ipp.append(len(doris_vtec_inpo))
                                    doris_dist_alt_ipp.append(np.mean(dists))

                                    range_ratio = 0.925
                                    gim_vtec_alt_ipp_doris.append(gim_vtec_alt_ipp[alt_epoch_idx] * range_ratio)
                                    alt_vtec_alt_ipp_doris.append(alt_vtec[alt_epoch_idx])  
                                    doris_lat_alt_ipp.append(alt_lat[alt_epoch_idx])
                                    doris_lon_alt_ipp.append(alt_lon[alt_epoch_idx])

                        doris_vtec_alt_ipp = np.array(doris_vtec_alt_ipp)
                        doris_std_alt_ipp = np.array(doris_std_alt_ipp)
                        doris_points_alt_ipp = np.array(doris_points_alt_ipp)
                        doris_dist_alt_ipp = np.array(doris_dist_alt_ipp)
                        gim_vtec_alt_ipp_doris = np.array(gim_vtec_alt_ipp_doris)
                        alt_vtec_alt_ipp_doris = np.array(alt_vtec_alt_ipp_doris)
                                  
                        doris_results[ele_mask_idx, min_obs_idx, time_gap_idx, 0] = np.sqrt(np.mean(np.square(doris_vtec_alt_ipp - alt_vtec_alt_ipp_doris)))   
                        doris_results[ele_mask_idx, min_obs_idx, time_gap_idx, 1] = np.mean(doris_std_alt_ipp)
                        doris_results[ele_mask_idx, min_obs_idx, time_gap_idx, 2] = np.mean(doris_points_alt_ipp)
                        doris_results[ele_mask_idx, min_obs_idx, time_gap_idx, 3] = len(doris_vtec_alt_ipp) / len(gim_vtec_alt_ipp) 
                        gim_rms_results[ele_mask_idx, min_obs_idx, time_gap_idx] = np.sqrt(np.mean(np.square(gim_vtec_alt_ipp_doris - alt_vtec_alt_ipp_doris)))
                        print(doris_results[ele_mask_idx, min_obs_idx, time_gap_idx, :], np.mean(doris_dist_alt_ipp), np.sqrt(np.mean(np.square(gim_vtec_alt_ipp_doris - alt_vtec_alt_ipp_doris))))
                        # plot_bias_std_vs_vtec(
                        #         doris_vtec_alt_ipp, doris_std_alt_ipp, alt_vtec_alt_ipp_doris,
                        #         label=time_gap
                        #     )
                        # name_vtec = f'./InpoResults/DORIS/VTEC/y{year}_d{doy}_sec-{time_gap}_ele-{ele_mask}_obs-{min_obs_count}.npy'
                        # np.save(name_vtec, np.column_stack((doris_vtec_alt_ipp, doris_std_alt_ipp, doris_points_alt_ipp, alt_vtec_alt_ipp_doris))) 
                        # name_ipp = f'./InpoResults/DORIS/IPP/y{year}_d{doy}_sec-{time_gap}_ele-{ele_mask}_obs-{min_obs_count}.npy'
                        # np.save(name_ipp, np.column_stack((doris_lat_alt_ipp, doris_lon_alt_ipp)))                        

                elif division_strategy == 'Space' and weighting_method == 'Window':
                    for lat_range_idx, lat_range in enumerate(lat_range_list):
                        vtec_estimates, vtec_stds, obs_counts, obs_dists = [], [], [], []
                        gim_scaled_list, alt_vtec_list = [], []
                        ipp_lats, ipp_lons, roti_values = [], [], []
                        avg_vec_dis = []
                        for epoch_idx, epoch_time in enumerate(alt_epoch):
                            lat = alt_lat[epoch_idx]
                            lon = alt_lon[epoch_idx]
                            time_diff_sec = np.abs(doris_epoch - epoch_time) * 86400
                            time_mask = time_diff_sec < 400
                            if np.count_nonzero(time_mask) < min_obs_count: continue       
                            d_lon, d_lat, d_epoch, d_vtec, d_ele = (
                                arr[time_mask] for arr in (doris_lon, doris_lat, doris_epoch, doris_vtec, doris_ele))   
                            
                            d_lon = apply_earth_rotation_correction(d_lon, d_epoch, epoch_time)

                            # # write a func for all these
                            time_mask_roti = time_diff_sec < 250
                            d_epoch_roti = doris_epoch[time_mask_roti]
                            d_vtec_roti = doris_vtec[time_mask_roti]
                            diff_idx = np.diff(np.where(time_mask_roti)[0])
                            seg_breaks = np.where(diff_idx != 1)[0] + 1
                            split_points = np.concatenate(([0], seg_breaks, [len(np.where(time_mask_roti)[0])]))
                            epoch_segs = [d_epoch_roti[s:e] for s, e in zip(split_points[:-1], split_points[1:])]
                            vtec_segs = [d_vtec_roti[s:e] for s, e in zip(split_points[:-1], split_points[1:])]
                            roti = ROTI(epoch_segs, vtec_segs)
                            
                            if roti < roti_threshold:                             
                                if (indices := find_doris_indices(d_lon, d_lat, min_obs_count, 10, max_lat_gap, lat, lon)):
                                    sel_lon = d_lon[indices]
                                    sel_lat = d_lat[indices]
                                    sel_vtec = d_vtec[indices]
                                    avg_vec_dis.append(average_vector_magnitude(lat, lon, sel_lat, sel_lon))
                                    dists = np.array(haversine_vec(lat, lon, sel_lat, sel_lon))
                                    dists[dists == 0] = 1e-12
                                    weights = 1 / dists
                                    weights /= weights.sum()
                                    weighted_vtec = np.sum(weights * sel_vtec)
                                    vtec_estimates.append(weighted_vtec)
                                    vtec_stds.append(np.std(sel_vtec))
                                    obs_counts.append(len(sel_vtec))
                                    obs_dists.append(np.mean(dists))
                                    gim_scaled_list.append(gim_vtec_alt_ipp[epoch_idx] * 0.925)
                                    alt_vtec_list.append(alt_vtec[epoch_idx])
                                    ipp_lats.append(lat)
                                    ipp_lons.append(lon)
                                    roti_values.append(roti)   

                            elif roti >= roti_threshold: 
                                continue
                                if (indices := find_doris_indices(d_lon, d_lat, min_obs_count, 6, 6, lat, lon)):
                                    sel_lon = d_lon[indices]
                                    sel_lat = d_lat[indices]
                                    sel_vtec = d_vtec[indices]
                                    avg_vec_dis.append(average_vector_magnitude(lat, lon, sel_lat, sel_lon))
                                    dists = np.array(haversine_vec(lat, lon, sel_lat, sel_lon))
                                    dists[dists == 0] = 1e-12
                                    weights = 1 / dists
                                    weights /= weights.sum()
                                    weighted_vtec = np.sum(weights * sel_vtec)
                                    
                                    vtec_estimates.append(weighted_vtec)
                                    vtec_stds.append(np.std(sel_vtec))
                                    obs_counts.append(len(sel_vtec))
                                    obs_dists.append(np.mean(dists))
                                    gim_scaled_list.append(gim_vtec_alt_ipp[epoch_idx] * 0.925)
                                    alt_vtec_list.append(alt_vtec[epoch_idx])
                                    ipp_lats.append(lat)
                                    ipp_lons.append(lon)
                                    roti_values.append(roti)

                                else:
                                    
                                    # lat_diff = np.abs(d_lat - lat)
                                    # lat_mask = lat_diff <= 8

                                    # lon_diff_deg = np.abs((d_lon - lon + 180) % 360 - 180)
                                    # lon_km_per_deg = np.cos(np.radians(lat)) * 111.0
                                    # lon_diff_km = lon_diff_deg * lon_km_per_deg
                                    # lon_mask = lon_diff_km <= 8 * 2 * 111

                                    # combined_mask = lat_mask & lon_mask
                                    # matching_indices = np.where(combined_mask)[0]

                                    # if len(matching_indices) < min_obs_count:
                                    #     continue

                                    if (indices := find_doris_indices(d_lon, d_lat, min_obs_count, 10, 10, lat, lon)):
                                        sel_lon = d_lon[indices]
                                        sel_lat = d_lat[indices]
                                        sel_vtec = d_vtec[indices]
                                        avg_vec_dis.append(average_vector_magnitude(lat, lon, sel_lat, sel_lon))
                                        dists = np.array(haversine_vec(lat, lon, sel_lat, sel_lon))
                                        dists[dists == 0] = 1e-12
                                        weights = 1 / dists
                                        weights /= weights.sum()
                                        weighted_vtec = np.sum(weights * sel_vtec)
                                        
                                        vtec_estimates.append(weighted_vtec)
                                        vtec_stds.append(np.std(sel_vtec))
                                        obs_counts.append(len(sel_vtec))
                                        obs_dists.append(np.mean(dists))
                                        gim_scaled_list.append(gim_vtec_alt_ipp[epoch_idx] * 0.925)
                                        alt_vtec_list.append(alt_vtec[epoch_idx])
                                        ipp_lats.append(lat)
                                        ipp_lons.append(lon)
                                        roti_values.append(roti)

                        vtec_estimates = np.array(vtec_estimates)
                        vtec_stds = np.array(vtec_stds)
                        obs_counts = np.array(obs_counts)
                        obs_dists = np.array(obs_dists)
                        gim_scaled_list = np.array(gim_scaled_list)
                        alt_vtec_list = np.array(alt_vtec_list)
                        
                        doris_results[ele_mask_idx, min_obs_idx, lat_range_idx] = [
                            np.sqrt(np.mean((vtec_estimates - alt_vtec_list) ** 2)),
                            np.mean(vtec_stds),
                            np.mean(obs_counts),
                            len(vtec_estimates) / len(gim_vtec_alt_ipp)
                        ]
                        gim_rms_results[ele_mask_idx, min_obs_idx, lat_range_idx] = np.sqrt(np.mean(np.square(gim_scaled_list - alt_vtec_list)))
                        print(doris_results[ele_mask_idx, min_obs_idx, lat_range_idx], np.mean(obs_dists), gim_rms_results[ele_mask_idx, min_obs_idx, lat_range_idx], np.mean(avg_vec_dis))                       
                        
                elif weighting_method == 'Distance':
                    for dist_range_idx, dist_range in enumerate(dist_range_list):
                        doris_vtec_alt_ipp = []
                        doris_std_alt_ipp = [] # std of the interpolation points
                        doris_points_alt_ipp = [] # number of the interpolation points
                        doris_dist_alt_ipp = []
                        gim_vtec_alt_ipp_doris = []
                        alt_vtec_alt_ipp_doris = []
                        doris_lat_alt_ipp = []
                        doris_lon_alt_ipp = []
                        for alt_epoch_idx, alt_epoch_proc in enumerate(alt_epoch):
                            alt_lat_idx = alt_lat[alt_epoch_idx]
                            alt_lon_idx = alt_lon[alt_epoch_idx]
                            time_diff_sec = np.abs(doris_epoch - alt_epoch_proc)*24*60*60
                            time_mask = time_diff_sec < 1000
                            
                            doris_lon_inpo, doris_lat_inpo, doris_epoch_inpo, doris_vtec_inpo, doris_ele_inpo = (
                                arr[time_mask] for arr in (doris_lon, doris_lat, doris_epoch, doris_vtec, doris_ele))     
                            doris_lon_inpo += (doris_epoch_inpo - alt_epoch_proc) * 360
                            doris_lon_inpo = np.where(doris_lon_inpo > 180, doris_lon_inpo - 360, doris_lon_inpo)                           
                            lat_diff = np.abs(doris_lat_inpo - alt_lat_idx)
                            lon_diff = np.abs(doris_lon_inpo - alt_lon_idx)
                    
                            # valid_indices_1 = np.where(np.abs(doris_epoch_inpo - alt_epoch_proc) < 60/24/3600)

                            dists = haversine_vec(alt_lat_idx, alt_lon_idx, doris_lat_inpo, doris_lon_inpo)
                            spatial_mask = (dists <= dist_range)
                            doris_lon_inpo, doris_lat_inpo, doris_epoch_inpo, doris_vtec_inpo = (
                            arr[spatial_mask] for arr in (doris_lon_inpo, doris_lat_inpo, doris_epoch_inpo, doris_vtec_inpo)) 

                        
                            if doris_lon_inpo.size < min_obs_count: 
                                continue
                            else:
                                dists = dists[spatial_mask]
                                # 获取最小的50个距离的索引
                                min_indices = np.argsort(dists)[:29]
                                # 提取对应的距离和函数值
                                dists = dists[min_indices]
                                doris_vtec_inpo = doris_vtec_inpo[min_indices]

                                d_selected = np.array(dists)
                                d_selected[d_selected == 0] = 1e-12  
                                weights = 1 / d_selected 
                                weights /= weights.sum(axis=0)
                                est_iono = np.sum(weights * doris_vtec_inpo)      
                                # if est_iono > 15: continue                               
                                doris_vtec_alt_ipp.append(est_iono)
                                doris_std_alt_ipp.append(np.std(doris_vtec_inpo))
                                doris_points_alt_ipp.append(len(doris_vtec_inpo))
                                doris_dist_alt_ipp.append(np.mean(dists))
                                range_ratio = 0.925
                                gim_vtec_alt_ipp_doris.append(gim_vtec_alt_ipp[alt_epoch_idx] * range_ratio)
                                alt_vtec_alt_ipp_doris.append(alt_vtec[alt_epoch_idx]) 
                                doris_lat_alt_ipp.append(alt_lat[alt_epoch_idx])
                                doris_lon_alt_ipp.append(alt_lon[alt_epoch_idx])
                        # we have negative values for altimetry ionospheric delay. why?
                        doris_vtec_alt_ipp = np.array(doris_vtec_alt_ipp)
                        doris_std_alt_ipp = np.array(doris_std_alt_ipp)
                        doris_points_alt_ipp = np.array(doris_points_alt_ipp)
                        doris_dist_alt_ipp = np.array(doris_dist_alt_ipp)
                        gim_vtec_alt_ipp_doris = np.array(gim_vtec_alt_ipp_doris)
                        alt_vtec_alt_ipp_doris = np.array(alt_vtec_alt_ipp_doris)
                                
                        doris_results[ele_mask_idx, min_obs_idx, dist_range_idx] = [
                            np.sqrt(np.mean((doris_vtec_alt_ipp - alt_vtec_alt_ipp_doris) ** 2)),
                            np.mean(doris_std_alt_ipp),
                            np.mean(doris_points_alt_ipp),
                            len(doris_vtec_alt_ipp) / len(gim_vtec_alt_ipp)
                        ]
                        gim_rms_results[ele_mask_idx, min_obs_idx, dist_range_idx] = np.sqrt(np.mean(np.square(gim_vtec_alt_ipp_doris - alt_vtec_alt_ipp_doris)))
    
                        print(doris_results[ele_mask_idx, min_obs_idx, dist_range_idx], np.mean(doris_dist_alt_ipp), np.sqrt(np.mean(np.square(gim_vtec_alt_ipp_doris - alt_vtec_alt_ipp_doris))))

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
