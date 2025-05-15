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

def compute_roti(doris_epoch, doris_vtec, time_diff_sec, time_gap=250):

    time_mask = time_diff_sec < time_gap
    d_epoch = doris_epoch[time_mask]
    d_vtec = doris_vtec[time_mask]

    diff_idx = np.diff(np.where(time_mask)[0])
    seg_breaks = np.where(diff_idx != 1)[0] + 1
    split_points = np.concatenate(([0], seg_breaks, [len(d_epoch)]))

    epoch_segs = [d_epoch[s:e] for s, e in zip(split_points[:-1], split_points[1:])]
    vtec_segs = [d_vtec[s:e] for s, e in zip(split_points[:-1], split_points[1:])]

    rates = []
    for epochs, vtecs in zip(epoch_segs, vtec_segs):
        if len(epochs) < 2:
            continue
        dt_min = np.diff(epochs) * 1440  # convert days to minutes
        d_vtec = np.diff(vtecs)
        rates.extend(d_vtec / dt_min)

    return np.std(rates) if rates else np.nan

def find_nearby_doris_indices(lat_d, lon_d, min_obs, lat_start_gap, lat_max_gap, lat_ref, lon_ref):
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

def process_valid_point(lat, lon, epoch_idx, d_lat, d_lon, d_vtec, indices,
                        vtec_estimates, vtec_stds, obs_counts, obs_dists, avg_vec_dis,
                        gim_vtec_interp, alt_vtec, gim_vtec_filtered, alt_vtec_filtered,
                        lat_ipp_filtered, lon_ipp_filtered):

    sel_lon, sel_lat, sel_vtec = (arr[indices] for arr in (d_lon, d_lat, d_vtec))
    distances = np.array(haversine_vec(lat, lon, sel_lat, sel_lon))
    weighted_vtec = inverse_distance_weighting(distances, sel_vtec)

    vtec_estimates.append(weighted_vtec)
    vtec_stds.append(np.std(sel_vtec))
    obs_counts.append(len(sel_vtec))
    obs_dists.append(np.mean(distances))
    avg_vec_dis.append(mean_vector_magnitude(lat, lon, sel_lat, sel_lon))

    gim_vtec_filtered.append(gim_vtec_interp[epoch_idx] * 0.925)
    alt_vtec_filtered.append(alt_vtec[epoch_idx])
    lat_ipp_filtered.append(lat)
    lon_ipp_filtered.append(lon)

# Main processing
if __name__ == '__main__':
    start_time = time.time()

    start_date = datetime(2024, 5, 8)
    num_days = 30
    params = {
        'ele_mask': 10,
        'min_obs': 30,
        'lat_win': 1,
        'lon_win': 2,
        'max_lat_gap': 8,
        'roti_gt1_lat_win': 6,
        'roti_gt1_lat_win_2': 10,
        'roti_threshold': 1
    }

    for day_offset in range(num_days):
        date = start_date + timedelta(days=day_offset)
        year, month, day = date.year, date.month, date.day
        doy = date.timetuple().tm_yday

        alt_lon, alt_lat, alt_epoch, alt_vtec = load_altimetry_data(month, day)
        doris_lon, doris_lat, doris_epoch, doris_vtec, doris_ele = load_doris_data(year, doy)

        ion_file = (
            f"./DORISInput/IGSGIM/{year}/igsg{doy:03d}0.{str(year)[-2:]}i"
            if year <= 2022 else
            f"./DORISInput/IGSGIM/{year}/IGS0OPSFIN_{year}{doy:03d}0000_01D_02H_GIM.INX"
        )

        gim_vtec_raw = read_ionFile(ion_file)
        gim_vtec_scaled = gim_vtec_raw[:, ::-1, :] * 0.1
        gim_vtec_interp = interpolate_gim_vtec(gim_vtec_scaled, alt_epoch, alt_lat, alt_lon)

        mask = doris_ele > params['ele_mask']
        doris_lon, doris_lat, doris_epoch, doris_vtec, doris_ele = (
            arr[mask] for arr in (doris_lon, doris_lat, doris_epoch, doris_vtec, doris_ele)
        )

        vtec_estimates, vtec_stds, obs_counts, obs_dists, avg_vec_dis = [], [], [], [], []
        lat_ipp_filtered, lon_ipp_filtered = [], [] 
        gim_vtec_filtered, alt_vtec_filtered = [], []
         
        for epoch_idx, epoch_time in enumerate(alt_epoch):
            lat, lon = alt_lat[epoch_idx], alt_lon[epoch_idx]
            time_diff_sec = np.abs(doris_epoch - epoch_time) * 86400
            time_mask = time_diff_sec < 400

            if np.count_nonzero(time_mask) < params['min_obs']:
                continue

            d_lon, d_lat, d_epoch, d_vtec = (
                arr[time_mask] for arr in (doris_lon, doris_lat, doris_epoch, doris_vtec)
            )
            d_lon = correct_for_earth_rotation(d_lon, d_epoch, epoch_time)

            # computation of roti: a smaller time gap
            roti_val = compute_roti(doris_epoch, doris_vtec, time_diff_sec, time_gap=150)
           
            # small roti means small iono fluctuation and flexible growing window
            if roti_val < params['roti_threshold']:

                indices = find_nearby_doris_indices(
                    d_lat, d_lon,
                    params['min_obs'], params['lat_win'],
                    params['max_lat_gap'], lat, lon
                )

                if indices:
                    process_valid_point(
                        lat, lon, epoch_idx, d_lat, d_lon, d_vtec, indices,
                        vtec_estimates, vtec_stds, obs_counts, obs_dists, avg_vec_dis,
                        gim_vtec_interp, alt_vtec, gim_vtec_filtered, alt_vtec_filtered,
                        lat_ipp_filtered, lon_ipp_filtered
                    )

            elif roti_val >= params['roti_threshold']:

                indices = find_nearby_doris_indices(
                    d_lat, d_lon,
                    params['min_obs'], params['roti_gt1_lat_win'],
                    params['roti_gt1_lat_win'], lat, lon
                )

                if indices:
                    process_valid_point(
                        lat, lon, epoch_idx, d_lat, d_lon, d_vtec, indices,
                        vtec_estimates, vtec_stds, obs_counts, obs_dists, avg_vec_dis,
                        gim_vtec_interp, alt_vtec, gim_vtec_filtered, alt_vtec_filtered,
                        lat_ipp_filtered, lon_ipp_filtered
                    )

                else:

                    lat_diff = np.abs(d_lat - lat)
                    lat_mask = lat_diff <= 10

                    lon_diff_deg = np.abs((d_lon - lon + 180) % 360 - 180)
                    lon_km_per_deg = np.cos(np.radians(lat)) * 111.0
                    lon_diff_km = lon_diff_deg * lon_km_per_deg
                    lon_mask = lon_diff_km <= 10 * 2 * 111

                    combined_mask = lat_mask & lon_mask
                    matching_indices = np.where(combined_mask)[0]

                    if len(matching_indices) < 30:
                        continue
                    indices = find_nearby_doris_indices(
                        d_lat, d_lon,
                        params['min_obs'], params['roti_gt1_lat_win_2'],
                        params['roti_gt1_lat_win_2'], lat, lon
                    )

                    if indices:
                        process_valid_point(
                            lat, lon, epoch_idx, d_lat, d_lon, d_vtec, indices,
                            vtec_estimates, vtec_stds, obs_counts, obs_dists, avg_vec_dis,
                            gim_vtec_interp, alt_vtec, gim_vtec_filtered, alt_vtec_filtered,
                            lat_ipp_filtered, lon_ipp_filtered
                        )

        vtec_estimates = np.array(vtec_estimates)
        gim_vtec_filtered = np.array(gim_vtec_filtered)
        alt_vtec_filtered = np.array(alt_vtec_filtered)
        
        rms_doris =  np.sqrt(np.mean((vtec_estimates - alt_vtec_filtered) ** 2))
        rms_gim = np.sqrt(np.mean((gim_vtec_filtered - alt_vtec_filtered) ** 2))
        cov_doris = len(vtec_estimates) / len(gim_vtec_interp)

        print(rms_doris, rms_gim, cov_doris, np.mean(vtec_stds), np.mean(obs_counts), np.mean(obs_dists), np.mean(avg_vec_dis))                           
            
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
