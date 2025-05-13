import json
import numpy as np
import h5py
import time
import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from readFile import read_ionFile
from tools import haversine_vec

# inverse distance weighting
def inverse_distance_weighting(distances, values):
    distances[distances == 0] = 1e-12  
    weights = 1 / distances
    weights /= weights.sum(axis=0)
    return np.dot(values, weights).item()

# GIM VTEC interpolation function
def interpolate_gim_vtec(gim_data, mjd_times, lats, lons):
    from scipy.interpolate import RegularGridInterpolator
    from astropy.time import Time

    max_lat_idx, max_lon_idx = 71, 73
    datetimes = Time(mjd_times, format='mjd').to_datetime()
    
    interpolators = [
        RegularGridInterpolator(
            (range(max_lat_idx), range(max_lon_idx)), gim_data[i], method='linear'
        )
        for i in range(13)
    ]
    
    vtec_values = []
    for dt, lat, lon in zip(datetimes, lats, lons):
        lat_idx = (lat + 87.5) / 2.5
        lon_idx = (lon + 180) / 5
        hour_idx = dt.hour // 2
        time_frac = ((dt.hour + dt.minute / 60 + dt.second / 3600) % 2) / 2
        vtec = (
            (1 - time_frac) * interpolators[hour_idx]([lat_idx, lon_idx])[0] +
            time_frac * interpolators[hour_idx + 1]([lat_idx, lon_idx])[0]
        )
        vtec_values.append(vtec)
    
    return vtec_values

# for altimetry data passes are recorded in different columns. convert them into one
def load_and_flatten_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return np.array([item for sublist in data for item in sublist])

# Load and preprocess altimetry data
def load_altimetry_data(month, day):
    base_time = np.datetime64('1985-01-01T00:00:00')
    mjd_ref = np.datetime64('1858-11-17T00:00:00')

    lon = load_and_flatten_json(f'./AltimetryData/Orbit/{month}{day}glon.json')
    lat = load_and_flatten_json(f'./AltimetryData/Orbit/{month}{day}glat.json')
    sec = load_and_flatten_json(f'./AltimetryData/Epoch/{month}{day}sec.json')
    dion = load_and_flatten_json(f'./AltimetryData/Dion/{month}{day}dion.json')

    valid = ~np.isnan(dion)
    lon, lat, sec, dion = (arr[valid] for arr in (lon, lat, sec, dion))
    lon = np.where(lon >= 180, lon - 360, lon)
    mjd = ((base_time + sec.astype('timedelta64[s]')) - mjd_ref) / np.timedelta64(1, 'D')
    ion_delay = -13.575e9**2 / 40.3 * dion / 1e16
    return lon, lat, mjd, ion_delay

# Load and preprocess doris data
# Change of doris file name (with the presence of cycle slip detection) -- check that part 
def load_doris_data(year, doy):
    filepath = f'./DORISVTEC/{year}/DOY{doy:03d}.h5'
    lon, lat, epoch, vtec, elev = [], [], [], [], []

    with h5py.File(filepath, 'r') as f:
        group = f[f'/y{year}/{doy:03d}']
        for pass_id in group:
            data = group[pass_id]
            lon.extend(data['ipp_lon'][:])
            lat.extend(data['ipp_lat'][:])
            epoch.extend(data['epoch'][:])
            vtec.extend(data['vtec'][:])
            elev.extend(data['elevation'][:])

    return map(np.array, (lon, lat, epoch, vtec, elev))

def save_doris_results_to_csv(data_path, output_path):
    data = np.load(data_path)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        ele_n, obs_n, lat_n, _ = data.shape

        for ele in range(ele_n):
            for metric_idx, metric in enumerate(["RMS", "Percent"]):
                writer.writerow([f"Ele {ele + 1} - {metric}"])
                writer.writerow(["Min Obs \\ Lat Range"] + [f"Lat {j+1}" for j in range(lat_n)])
                for obs in range(obs_n):
                    row = [f"Min Obs {obs + 1}"] + data[ele, obs, :, metric_idx].tolist()
                    writer.writerow(row)
                writer.writerow([])

def save_gim_results_to_csv(data_path, output_path):
    data = np.load(data_path)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        ele_n, obs_n, lat_n = data.shape

        for ele in range(ele_n):
            writer.writerow([f"Ele {ele + 1} - RMS"])
            writer.writerow(["Min Obs \\ Lat Range"] + [f"Lat {j+1}" for j in range(lat_n)])
            for obs in range(obs_n):
                row = [f"Min Obs {obs + 1}"] + data[ele, obs, :].tolist()
                writer.writerow(row)
            writer.writerow([])

def compute_roti(epoch_segments, vtec_segments):
    rates = []

    for epochs, vtecs in zip(epoch_segments, vtec_segments):
        if len(epochs) < 2:
            continue
        dt_min = np.diff(epochs) * 1440
        d_vtec = np.diff(vtecs)
        rates.extend(d_vtec / dt_min)

    return np.std(rates) if rates else np.nan

def find_nearby_doris_indices(lon_d, lat_d, min_obs, lat_start_gap, lat_max_gap, lat_ref, lon_ref):
    threshold = lat_start_gap
    max_dist_km = lat_start_gap * 2 * 111

    while threshold <= lat_max_gap:
        lat_mask = np.abs(lat_d - lat_ref) <= threshold
        lon_deg_diff = np.abs((lon_d - lon_ref + 180) % 360 - 180)
        lon_km = np.cos(np.radians(lat_ref)) * 111.0 * lon_deg_diff
        combined = lat_mask & (lon_km <= max_dist_km)

        if np.count_nonzero(combined) >= min_obs:
            return np.where(combined)[0].tolist()

        threshold += 1.0
        max_dist_km += 2 * 111

    return []

def correct_for_earth_rotation(lons, epochs, ref_epoch):
    omega = 7.2921159e-5  # rad/s
    delta_t = (epochs - ref_epoch) * 86400
    delta_lon_deg = np.degrees(omega * delta_t)
    corrected = lons + delta_lon_deg
    return (corrected + 180) % 360 - 180

def mean_vector_magnitude(ref_lat, ref_lon, target_lats, target_lons, radius=6371+450):
    def sph2cart(lat, lon, r):
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        return np.stack((x, y, z), axis=-1)

    ref_lat, ref_lon = np.radians(ref_lat), np.radians(ref_lon)
    tgt_lat, tgt_lon = np.radians(target_lats), np.radians(target_lons)

    vec_ref = sph2cart(ref_lat, ref_lon, radius)
    vec_tgt = sph2cart(tgt_lat, tgt_lon, radius)
    
    vectors = vec_tgt - vec_ref
    mean_vec = vectors.mean(axis=0)
    
    return np.linalg.norm(mean_vec)

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
