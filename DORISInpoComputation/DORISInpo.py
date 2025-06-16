import numpy as np
from numpy.lib import recfunctions as rfn
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tools import haversine_vec, idw, get_igs_vtec
from readFile import read_ionFile
import time
import pickle
import pandas as pd
import constant as const
import matplotlib.pyplot as plt

def compute_roti(ns_doris_time, ref_epoch, time_gap=150):

    time_diff_sec = np.abs((ns_doris_time['obs_epoch'] - ref_epoch) / np.timedelta64(1, 's'))
    time_mask = time_diff_sec < time_gap

    d_epoch, d_vtec, d_station, d_passID = (
            ns_doris_time['obs_epoch'][time_mask],
            ns_doris_time['VTEC'][time_mask],
            ns_doris_time['station_code'][time_mask],
            ns_doris_time['pass_id'][time_mask]
    )
    station_encoded, _ = pd.factorize(d_station)
    combined = np.stack([station_encoded, d_passID], axis=1)

    unique_pairs, inverse_indices = np.unique(combined, axis=0, return_inverse=True)

    rates = []
    for idx, _ in enumerate(unique_pairs):
        mask = inverse_indices == idx
        # make sure the list is epoch-sequenced
        group_epoch = d_epoch[mask]
        group_vtec = d_vtec[mask]

        if len(group_epoch) < 2:
            continue
        dt_min = np.diff(group_epoch) / np.timedelta64(1, 'm')  # convert days to minutes
        diff_vtec = np.diff(group_vtec)
        rates.extend(diff_vtec / dt_min)

    return np.std(rates) if rates else np.nan

def main_processing_pipeline(df_altimetry: pd.DataFrame, df_doris: pd.DataFrame, settings):

    df_doris = df_doris[df_doris['elevation'] > settings['ele_mask']].reset_index(drop=True)

    ns_altimetry = df_altimetry.to_records(index=False) # from dataframe to numpy stuructured array
    ns_doris = df_doris.to_records(index=False) 
    length = len(ns_altimetry)

    doris_vtec = np.full(length, np.nan)
    doris_points_count = np.full(length, -1, dtype=int)
    doris_points_std = np.full(length, np.nan)
    window_size = np.full(length, np.nan)
    roti_list = np.full(length, np.nan)

    for idx, row in enumerate(ns_altimetry):

        # if row['ascend']: continue # bool type, true means the pass is asceding

        lat, lon = row['ipp_lat'], row['ipp_lon'] % 360

        # --- Time filtering ---
        time_diff_sec = np.abs((df_doris['obs_epoch'] - row['obs_epoch']) / np.timedelta64(1, 's'))
        time_mask = time_diff_sec < 400

        if np.count_nonzero(time_mask) < settings['min_obs_count']:
            continue

        ns_doris_time = ns_doris[time_mask]

        # --- Earth rotation correction ---
        omega = const.omega
        delta_t = (ns_doris_time['obs_epoch'] - row['obs_epoch']) / np.timedelta64(1, 's')
        delta_lon_deg = np.degrees(omega * delta_t)
        corrected = ns_doris_time['ipp_lon'] + delta_lon_deg
        ns_doris_time['ipp_lon'] = corrected % 360

        # --- ROTI calculation ---
        roti = compute_roti(ns_doris_time, row['obs_epoch'], settings['roti_sec_gap'])
        roti_list[idx] = roti

        # --- Window search & interpolation ---
        if roti < settings['roti_threshold']:
            lat_gap = settings['lat_gap_lf']
            lon_gap_km = lat_gap * 2 * 111
            while lat_gap <= settings['max_lat_gap_lf']:
                lon_deg_diff = np.abs((ns_doris_time['ipp_lon'] - lon + 180) % 360 - 180)
                lon_km = np.cos(np.radians(ns_doris_time['ipp_lat'])) * 111.0 * lon_deg_diff
                combined = (np.abs(ns_doris_time['ipp_lat'] - lat) <= lat_gap) & (lon_km <= lon_gap_km)
                if np.count_nonzero(combined) >= settings['min_obs_count']:
                    ns_doris_inpo = ns_doris_time[combined]
                    distances = np.array(haversine_vec(lat, lon, ns_doris_inpo['ipp_lat'], ns_doris_inpo['ipp_lon']))
                    weighted_vtec = idw(distances, ns_doris_inpo['VTEC'])
                    doris_vtec[idx] = weighted_vtec
                    doris_points_std[idx] = np.std(ns_doris_inpo['VTEC'])
                    window_size[idx] = lat_gap            
                    break
                lat_gap += 1
                lon_gap_km += 2 * 111
            doris_points_count[idx] = np.count_nonzero(combined)

        elif roti >= settings['roti_threshold']:
            continue
            lat_gap = settings['lat_gap_hf']
            lon_gap_km = lat_gap * 2 * 111
            lon_deg_diff = np.abs((ns_doris_time['ipp_lon'] - lon + 180) % 360 - 180)
            lon_km = np.cos(np.radians(ns_doris_time['ipp_lat'])) * 111.0 * lon_deg_diff
            combined = (np.abs(ns_doris_time['ipp_lat'] - lat) <= lat_gap) & (lon_km <= lon_gap_km)
            if np.count_nonzero(combined) >= settings['min_obs_count']:
                ns_doris_inpo = ns_doris_time[combined]
                distances = np.array(haversine_vec(lat, lon, ns_doris_inpo['ipp_lat'], ns_doris_inpo['ipp_lon']))
                weighted_vtec = idw(distances, ns_doris_inpo['VTEC'])
                doris_vtec[idx] = weighted_vtec
                doris_points_std[idx] = np.std(ns_doris_inpo['VTEC'])
                window_size[idx] = lat_gap

            else:
                lat_gap = settings['lat_gap_hf_big']
                lon_gap_km = lat_gap * 2 * 111
                lon_deg_diff = np.abs((ns_doris_time['ipp_lon'] - lon + 180) % 360 - 180)
                lon_km = np.cos(np.radians(ns_doris_time['ipp_lat'])) * 111.0 * lon_deg_diff
                combined = (np.abs(ns_doris_time['ipp_lat'] - lat) <= lat_gap) & (lon_km <= lon_gap_km)
                if np.count_nonzero(combined) >= settings['min_obs_count']:
                    ns_doris_inpo = ns_doris_time[combined]
                    distances = np.array(haversine_vec(lat, lon, ns_doris_inpo['ipp_lat'], ns_doris_inpo['ipp_lon']))
                    weighted_vtec = idw(distances, ns_doris_inpo['VTEC'])
                    doris_vtec[idx] = weighted_vtec
                    doris_points_std[idx] = np.std(ns_doris_inpo['VTEC'])
                    window_size[idx] = lat_gap

            doris_points_count[idx] = np.count_nonzero(combined)

    ns_altimetry_updated = rfn.append_fields(
        ns_altimetry,
        names=['doris_vtec', 'doris_points_count', 'doris_points_std', 'window_size', 'roti'],
        data=[doris_vtec, doris_points_count, doris_points_std, window_size, roti_list],
        dtypes=['f8', 'i4', 'f8', 'f8', 'f8'],
        usemask=False
    )
    return pd.DataFrame.from_records(ns_altimetry_updated)

satellite_list = ['ja2', 'ja3', 's3a', 's3b', 'srl']
range_ratio_list = [0.925]

def split_df_by_time(df, hours_per_chunk=3):
    df['hour'] = df['obs_epoch'].dt.hour
    chunks = []
    for start_hour in range(0, 24, hours_per_chunk):
        end_hour = start_hour + hours_per_chunk
        chunk = df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)].copy()
        chunks.append(chunk)
    df.drop(columns='hour', inplace=True) 
    return chunks

# Main processing
if __name__ == '__main__':
    start_time = time.time()

    proc_sate = satellite_list[1]
    range_ratio = range_ratio_list[0]

    start_date = datetime(2024, 5, 8)
    num_days = 1
    settings = {
        'ele_mask': 10,
        'roti_threshold': 1, 
        'roti_sec_gap': 150,
        'min_obs_count': 30,
        'lat_gap_lf': 1,
        'max_lat_gap_lf': 10,
        'lat_gap_hf': 1,
        'lat_gap_hf_big': 10
    }

    for day_offset in range(num_days):
        date = start_date + timedelta(days=day_offset)
        year, month, day = date.year, date.month, date.day
        doy = date.timetuple().tm_yday

        with open(f'./DORISVTECStorage/{proc_sate}/{year}/DOY{doy:03d}.pickle', 'rb') as path:
            df_doris = pickle.load(path)

        with open(f'./AltimetryVTECStorage/{proc_sate}/{year}/DOY{doy:03d}.pickle', 'rb') as path:
            df_altimetry = pickle.load(path)

        ion_file = (
            f"./DORISInput/IGSGIM/{year}/igsg{doy:03d}0.{str(year)[-2:]}i"
            if year <= 2022 else
            f"./DORISInput/IGSGIM/{year}/IGS0OPSFIN_{year}{doy:03d}0000_01D_02H_GIM.INX"
        )

        # --- GIM VTEC computation & comparison with Altimetry VTEC ---
        gim_vtec_raw = read_ionFile(ion_file)
        gim_vtec_scaled = gim_vtec_raw[:, ::-1, :] * 0.1
        gim_vtec_interp = get_igs_vtec(gim_vtec_scaled, df_altimetry)
        gim_vtec_diff = df_altimetry['VTEC'] - gim_vtec_interp * range_ratio
        gim_vtec_rms = np.sqrt(np.mean(gim_vtec_diff ** 2))

        # --- DORIS VTEC computation & comparison with Altimetry VTEC ---
        chunks = split_df_by_time(df_altimetry)
        results = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(main_processing_pipeline, chunk, df_doris, settings) for chunk in chunks]
            for future in as_completed(futures):
                result_df = future.result()
                results.append(result_df)
        df_doris_result = pd.concat(results, ignore_index=True)
        df_doris_result = df_doris_result.sort_values('obs_epoch').reset_index(drop=True)
        lon_adjusted = (df_doris_result['ipp_lon'] + 180) % 360 - 180 
        df_doris_result['local_time'] = df_doris_result['obs_epoch'] + pd.to_timedelta(lon_adjusted / 15, unit='h')
        df_doris_result.drop(columns='hour', inplace=True)

        df_doris_result['gim_vtec'] = gim_vtec_interp

        df_doris_result.to_csv(f"./DORISInpoOutput/{year}/DOY{doy}Ele{settings['ele_mask']}.csv")


        roti_mean = np.mean(df_doris_result['roti'])
        vtec_mean = np.mean(df_doris_result['doris_vtec'])
        doris_vtec_diff = df_doris_result['VTEC'] - df_doris_result['doris_vtec']

        non_nan_indices = np.where(~np.isnan(doris_vtec_diff.values))[0]
        doris_vtec_rms = np.sqrt(np.mean(doris_vtec_diff[non_nan_indices] ** 2))
        gim_vtec_rms_doris = np.sqrt(np.mean(gim_vtec_diff[non_nan_indices] ** 2))

        print(doris_vtec_rms, gim_vtec_rms_doris, len(doris_vtec_diff[non_nan_indices]), len(doris_vtec_diff))


        
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
