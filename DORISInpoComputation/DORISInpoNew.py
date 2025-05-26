import pandas as pd
import numpy as np
import constant as const
import time
import pickle

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
    
    df_doris_filtered = df_doris[df_doris['elevation'] >= elevation_threshold].copy()
    
    for idx, alt_row in df_altimetry.iterrows():
        alt_lat = alt_row['ipp_lat']
        alt_lon = alt_row['ipp_lon']
        alt_vtec = alt_row['VTEC']
        alt_time = alt_row['obs_epoch']
        
        lat_window = initial_lat_window_deg
        matched_doris = pd.DataFrame()
        
        for expansion in range(max_expansions + 1):
            cos_lat = np.cos(np.deg2rad(alt_lat))
            lon_window_deg = 2 * lat_window / cos_lat  
            
            spatial_doris = df_doris_filtered[
                (np.abs(df_doris_filtered['ipp_lat'] - alt_lat) <= lat_window) &
                (np.abs(df_doris_filtered['ipp_lon'] - alt_lon) <= lon_window_deg)
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
            interp_vtec = idw_interpolation(distances, matched_doris['VTEC'].values, power=idw_power)
            
            vtec_difference = alt_vtec - interp_vtec
            
            results.append({
                'altimetry_index': idx,
                'obs_epoch': alt_time,
                'alt_ipp_lon': alt_lon,
                'alt_ipp_lat': alt_lat,
                'alt_vtec': alt_vtec,
                'interp_vtec': interp_vtec,
                'vtec_difference': vtec_difference,
                'doris_count': len(matched_doris)
            })
        else:
            # 没找到足够 DORIS 点，标记为 NaN
            results.append({
                'altimetry_index': idx,
                'obs_epoch': alt_time,
                'alt_ipp_lon': alt_lon,
                'alt_ipp_lat': alt_lat,
                'alt_vtec': alt_vtec,
                'interp_vtec': np.nan,
                'vtec_difference': np.nan,
                'doris_count': len(matched_doris)
            })
    
    result_df = pd.DataFrame(results)
    return result_df

satellite_list = ['ja2', 'ja3', 's3a', 's3b', 'srl']
range_ratio_list = [0.925]
if __name__ == '__main__':
    start_time = time.time()
    
    year, month, day = 2024, 5, 8
    proc_days = 30
    proc_sate = satellite_list[1]
    min_obs_count = 30
    range_ratio = range_ratio_list[0]

    for i in range(proc_days):
        process_epoch = pd.Timestamp(year, month, day) + pd.Timedelta(days=i)
        doy = process_epoch.dayofyear

        ## read in rinex-reading results
        with open(f'./DORISVTECStorage/pandas/{proc_sate}/{year}/DOY{doy:03d}.pickle', 'rb') as path:
            df_doris = pickle.load(path)

        with open(f'./AltVTECStorage/pandas/{proc_sate}/{year}/DOY{doy:03d}.pickle', 'rb') as path:
            df_altimetry = pickle.load(path)

    match_and_interpolate_altimetry_vtec_idw(df_altimetry, df_doris)
