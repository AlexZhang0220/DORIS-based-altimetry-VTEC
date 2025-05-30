import numpy as np
from numpy.lib import recfunctions as rfn
import time
from datetime import datetime, timedelta
import pickle
from tools import haversine_vec, idw, get_igs_vtec
from readFile import read_ionFile
import pandas as pd
import constant as const
import json
# import matplotlib.pyplot as plt

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

def load_and_flatten_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return np.array([item for sublist in data for item in sublist])

def load_altimetry_data_as_dataframe(month, day):

    base_time = np.datetime64('1985-01-01T00:00:00')

    lon = load_and_flatten_json(f'./AltimetryData/Orbit/{month}{day}glon.json')
    lat = load_and_flatten_json(f'./AltimetryData/Orbit/{month}{day}glat.json')
    sec = load_and_flatten_json(f'./AltimetryData/Epoch/{month}{day}sec.json')
    dion = load_and_flatten_json(f'./AltimetryData/Dion/{month}{day}dion.json')

    valid = ~np.isnan(dion)
    lon, lat, sec, dion = (arr[valid] for arr in (lon, lat, sec, dion))

    lon = lon % 360

    obs_epoch = base_time + sec.astype('timedelta64[s]')

    ion_delay = (-13.575e9**2) / 40.3 * dion / 1e16

    df = pd.DataFrame({
        'ipp_lon': lon,
        'ipp_lat': lat,
        'obs_epoch': obs_epoch,
        'VTEC': ion_delay
    })

    return df

def main_processing_pipeline(df_altimetry: pd.DataFrame, df_doris: pd.DataFrame, settings):

    ns_altimetry = df_altimetry.to_records(index=False) # from dataframe to numpy stuructured array
    ns_doris = df_doris.to_records(index=False) # from dataframe to numpy stuructured array

    length = len(ns_altimetry)

    doris_vtec = np.full(length, np.nan)
    doris_points_count = np.full(length, -1, dtype=int)
    doris_points_std = np.full(length, np.nan)
    window_size = np.full(length, np.nan)
    roti_list = np.full(length, np.nan)

    for idx, row in enumerate(ns_altimetry):
            lat, lon = row['ipp_lat'], row['ipp_lon'] % 360

            # --- Time filtering ---
            time_diff_sec = np.abs((df_doris['obs_epoch'] - row['obs_epoch']) / np.timedelta64(1, 's'))
            time_mask = time_diff_sec < 350

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
                        if np.max((ns_doris_inpo['obs_epoch'] - row['obs_epoch']) / np.timedelta64(1, 's')) > 350:
                            print('aa')
                        doris_vtec[idx] = weighted_vtec
                        doris_points_std[idx] = np.std(ns_doris_inpo['VTEC'])
                        window_size[idx] = lat_gap
                        break
                    lat_gap += 1
                    lon_gap_km += 2 * 111
                doris_points_count[idx] = len(ns_doris_inpo)
            elif roti >= settings['roti_threshold']:
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
                    continue
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
                doris_points_count[idx] = len(ns_doris_inpo)

    ns_altimetry_updated = rfn.append_fields(
        ns_altimetry,
        names=['doris_vtec', 'doris_points_count', 'doris_points_std', 'window_size', 'roti'],
        data=[doris_vtec, doris_points_count, doris_points_std, window_size, roti_list],
        dtypes=['f8', 'i4', 'f8', 'f8', 'f8'],
        usemask=False
    )
    return ns_altimetry_updated

satellite_list = ['ja2', 'ja3', 's3a', 's3b', 'srl']
range_ratio_list = [0.925]
    
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
        'max_lat_gap_lf': 6,
        'lat_gap_hf': 6,
        'lat_gap_hf_big': 10
    }

    for day_offset in range(num_days):
        date = start_date + timedelta(days=day_offset)
        year, month, day = date.year, date.month, date.day
        doy = date.timetuple().tm_yday

        with open(f'./DORISVTECStorage/{proc_sate}/{year}/DOY{doy:03d}.pickle', 'rb') as path:
            df_doris = pickle.load(path)

        df_altimetry = load_altimetry_data_as_dataframe(month, day)

        ion_file = (
            f"./DORISInput/IGSGIM/{year}/igsg{doy:03d}0.{str(year)[-2:]}i"
            if year <= 2022 else
            f"./DORISInput/IGSGIM/{year}/IGS0OPSFIN_{year}{doy:03d}0000_01D_02H_GIM.INX"
        )

        gim_vtec_raw = read_ionFile(ion_file)
        gim_vtec_scaled = gim_vtec_raw[:, ::-1, :] * 0.1
        gim_vtec_interp = get_igs_vtec(gim_vtec_scaled, df_altimetry)

        gim_vtec_diff = df_altimetry['VTEC'] - gim_vtec_interp * range_ratio

        main_processing_pipeline(df_altimetry, df_doris, settings)
            
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
