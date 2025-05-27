import pandas as pd
import numpy as np
import constant as const
import time
import pickle
import json
from tools import get_igs_vtec
from readFile import read_ionFile
# import matplotlib.pyplot as plt

def haversine(lon1, lat1, lon2, lat2):

    R = const.AE84 / 1000 # 地球半径 (km)
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def idw_interpolation(distances, values, power=2):
    weights = 1 / np.power(distances, power)
    weights[distances == 0] = 1e12 
    return np.sum(weights * values) / np.sum(weights)

def compute_roti_for_alt_point(df_doris_nearby):
    rates = []
    grouped = df_doris_nearby.groupby(['station_code', 'pass_id'])
    
    for (station, pass_id), group in grouped:
        group_sorted = group.sort_values('obs_epoch')
        vtec_values = group_sorted['VTEC'].values
        times = group_sorted['obs_epoch'].values.astype('datetime64[s]').astype(float)
        
        if len(vtec_values) < 2:
            continue  # 不够点跳过
        
        dvtec = np.abs(np.diff(vtec_values))
        dt_min = np.diff(times) / 60  # 秒转分钟
        rate = dvtec / dt_min
        rates.extend(rate)
    
    if len(rates) > 0:
        roti = np.std(rates)
    else:
        roti = np.nan
    
    return roti

def match_and_interpolate_altimetry_vtec_idw(
    df_altimetry,
    df_doris,
    elevation_threshold=20,
    initial_lat_window_deg=1.0,
    min_doris_count=30,
    lat_window_step=0.5,
    max_expansions=10,
    idw_power=2
):
    results = []
    
    df_doris_elevthres = df_doris[df_doris['elevation'] >= elevation_threshold].copy()
    
    for _, alt_row in df_altimetry.iterrows():
        alt_lat = alt_row['ipp_lat']
        alt_lon = alt_row['ipp_lon'] % 360
        alt_vtec = alt_row['VTEC']
        igs_vtec = alt_row['igs_vtec']
        alt_time = alt_row['obs_epoch']

        
        delta_t_doris = (df_doris_elevthres['obs_epoch'] - alt_time).dt.total_seconds()
        df_doris_filtered = df_doris_elevthres[np.abs(delta_t_doris) < 500]
        
        roti = compute_roti_for_alt_point(df_doris_elevthres[np.abs(delta_t_doris) < 150])

        doris_time = df_doris_filtered['obs_epoch']
        delta_t_doris = (doris_time - alt_time).dt.total_seconds()
        delta_lon_doris = np.degrees(const.omega * delta_t_doris)
        df_doris_filtered['corrected_lon'] = (df_doris_filtered['ipp_lon'] + delta_lon_doris + 360) % 360

        lat_window = initial_lat_window_deg
        matched_doris = pd.DataFrame()
        
        if roti > 1:
            continue

        for _ in range(max_expansions + 1):
            cos_lat = np.cos(np.deg2rad(df_doris_filtered['ipp_lat']))
            lon_window_deg = 2 * lat_window / cos_lat  
            
            lat_diffs = np.abs(df_doris_filtered['ipp_lat'] - alt_lat)
            diff = np.abs(alt_lon - df_doris_filtered['corrected_lon']) % 360
            lon_diffs = np.minimum(diff, 360 - diff)
            
            spatial_doris = df_doris_filtered[
                (lat_diffs <= lat_window) &
                (lon_diffs <= lon_window_deg)
            ]
            
            if len(spatial_doris) >= min_doris_count:
                matched_doris = spatial_doris
                break
            else:
                lat_window += lat_window_step
        
        if len(matched_doris) >= min_doris_count:

            distances = haversine(
                np.full(len(matched_doris), alt_lon),
                np.full(len(matched_doris), alt_lat),
                matched_doris['ipp_lon'].values,
                matched_doris['ipp_lat'].values
            )
            
            # IDW 插值
            doris_interp_vtec = idw_interpolation(distances, matched_doris['VTEC'].values, power=idw_power)
            
            vtec_difference = doris_interp_vtec - alt_vtec
            
            results.append({
                'obs_epoch': alt_time,
                'alt_ipp_lon': alt_lon,
                'alt_ipp_lat': alt_lat,
                'alt_vtec': alt_vtec,
                'igs_vtec': igs_vtec,
                'interp_vtec': doris_interp_vtec,
                'doris_vtec_difference': vtec_difference,
                'igs_vtec_difference': igs_vtec - alt_vtec,
                'roti': roti,
                'doris_count': len(matched_doris)
            })
        else:
            # 没找到足够 DORIS 点，标记为 NaN
            results.append({
                'obs_epoch': alt_time,
                'alt_ipp_lon': alt_lon,
                'alt_ipp_lat': alt_lat,
                'alt_vtec': alt_vtec,
                'igs_vtec': igs_vtec,
                'interp_vtec': np.nan,
                'doris_vtec_difference': np.nan,
                'igs_vtec_difference': igs_vtec - alt_vtec,
                'roti': roti,
                'doris_count': len(matched_doris)
            })
    
    result_df = pd.DataFrame(results)
    return result_df

def load_and_flatten_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return np.array([item for sublist in data for item in sublist])

def load_altimetry_data_as_dataframe(month, day):

    base_time = np.datetime64('1985-01-01T00:00:00')
    mjd_ref = np.datetime64('1858-11-17T00:00:00')

    lon = load_and_flatten_json(f'./AltimetryData/Orbit/{month}{day}glon.json')
    lat = load_and_flatten_json(f'./AltimetryData/Orbit/{month}{day}glat.json')
    sec = load_and_flatten_json(f'./AltimetryData/Epoch/{month}{day}sec.json')
    dion = load_and_flatten_json(f'./AltimetryData/Dion/{month}{day}dion.json')

    valid = ~np.isnan(dion)
    lon, lat, sec, dion = (arr[valid] for arr in (lon, lat, sec, dion))

    # 经度调整到 0~360
    lon = lon % 360

    # 计算 MJD（modified julian date）
    mjd = ((base_time + sec.astype('timedelta64[s]')) - mjd_ref) / np.timedelta64(1, 'D')

    # 转换为 pandas.Timestamp（基于 MJD）
    mjd_days = mjd.astype(float)
    obs_epoch = pd.to_datetime(mjd_days - 40587, unit='D')  # MJD to UNIX epoch (days since 1970-01-01)

    # 计算 ionospheric delay or VTEC
    ion_delay = (-13.575e9**2) / 40.3 * dion / 1e16

    # 生成 DataFrame
    df = pd.DataFrame({
        'ipp_lon': lon,
        'ipp_lat': lat,
        'obs_epoch': obs_epoch,
        'VTEC': ion_delay
    })

    return df

satellite_list = ['ja2', 'ja3', 's3a', 's3b', 'srl']
range_ratio_list = [0.925] # 这两个做成一个dict

if __name__ == '__main__':
    start_time = time.time()
    
    year, month, day = 2024, 5, 9
    proc_days = 30
    proc_sate = satellite_list[1]
    min_obs_count = 30
    range_ratio = range_ratio_list[0]

    for i in range(proc_days):
        process_epoch = pd.Timestamp(year, month, day) + pd.Timedelta(days=i)
        doy = process_epoch.dayofyear

        ## read in rinex-reading results
        with open(f'./DORISVTECStorage/{proc_sate}/{year}/DOY{doy:03d}.pickle', 'rb') as path:
            df_doris = pickle.load(path)

        df_altimetry = load_altimetry_data_as_dataframe(process_epoch.month, process_epoch.day)

        ion_file = (
            f"./DORISInput/IGSGIM/{year}/igsg{doy:03d}0.{str(year)[-2:]}i"
            if year <= 2022
            else f"./DORISInput/IGSGIM/{year}/IGS0OPSFIN_{year}{doy:03d}0000_01D_02H_GIM.INX"
        )

        gim_vtec = read_ionFile(ion_file) * 0.1
        gim_vtec = gim_vtec[:, ::-1, :]
        df_altimetry['igs_vtec'] = get_igs_vtec(gim_vtec, df_altimetry)

        doris_inpo_results = match_and_interpolate_altimetry_vtec_idw(df_altimetry, df_doris, elevation_threshold=10)

        valid_rows = doris_inpo_results[doris_inpo_results['doris_vtec_difference'].notna()]
        
        cvg = valid_rows.sum() / len(doris_inpo_results)

        doris_diff_values = valid_rows['doris_vtec_difference'].values
        doris_rms = np.sqrt(np.mean(doris_diff_values ** 2))

        igs_diff_values = valid_rows['igs_vtec_difference'].values
        igs_rms = np.sqrt(np.mean(igs_diff_values ** 2))

        print(f"doris_cvg:{cvg} | doris_rms:{doris_rms} | igs_rms:{igs_rms}")
