from CycleSlipDetection import detect_passes
from tools import get_igs_vtec
from readFile import read_ionFile
from pandas import Timestamp, Timedelta
import time
import pandas as pd
import pickle
# import matplotlib.pyplot as plt

def correct_pass_vtec(pass_df, max_row, igs_vtec, range_ratio=1.0):
    igs_stec = igs_vtec * max_row['map_value']
    delta = igs_stec * range_ratio - max_row['STEC']
    corrected_stec = pass_df['STEC'] + delta
    corrected_vtec = corrected_stec / pass_df['map_value']
    
    pass_df = pass_df.copy()
    pass_df['STEC'] = corrected_stec
    pass_df['VTEC'] = corrected_vtec
    return pass_df

satellite_list = ['ja2', 'ja3', 's3a', 's3b', 'srl']
range_ratio_list = [0.925]

if __name__ == '__main__':

    start_time = time.time()
    
    year, month, day = 2024, 5, 8
    proc_days = 30
    proc_sate = satellite_list[1]
    min_obs_count = 30
    range_ratio = range_ratio_list[0]

    columns_to_keep = [
        'obs_epoch', 'ipp_lat', 'ipp_lon',
        'STEC', 'elevation', 'map_value',
        'station_code', 'PRN'
    ]

    for i in range(proc_days):
        process_epoch = Timestamp(year, month, day) + Timedelta(days=i)
        doy = process_epoch.dayofyear

        ## read in rinex-reading results
        with open(f'./DORISObsStorage/pandas/{proc_sate}/{year}/DOY{doy:03d}.pickle', 'rb') as path:
            obs = pickle.load(path)

        # sat_clock_offset = compute_sat_clock_corrections(process_epoch, obs)
        
        ## Detection and remove of cycle slip 
        pass_all_station= []
        for station_code, grouped_obs in obs.storage.groupby('station_code'):
            pass_all_station.append(detect_passes(grouped_obs, min_obs_count, columns_to_keep))
            
        pass_all_station = pd.concat(pass_all_station, ignore_index=True)
        
        ion_file = (
            f"./DORISInput/IGSGIM/{year}/igsg{doy:03d}0.{str(year)[-2:]}i"
            if year <= 2022
            else f"./DORISInput/IGSGIM/{year}/IGS0OPSFIN_{year}{doy:03d}0000_01D_02H_GIM.INX"
        )

        ## Leveling of STEC using IGS GIM
        gim_vtec = read_ionFile(ion_file) * 0.1
        gim_vtec = gim_vtec[:, ::-1, :]

        max_elev_idx = pass_all_station.groupby(['station_code', 'pass_id'])['elevation'].idxmax()
        df_max = pass_all_station.loc[max_elev_idx].reset_index(drop=True)
        
        df_max['igs_vtec'] = get_igs_vtec(gim_vtec, df_max)

        max_row_dict = {
            (row['station_code'], row['pass_id']): row for _, row in df_max.iterrows()
        }

        corrected_passes = []
        for (station_code, pass_id), pass_df in pass_all_station.groupby(['station_code', 'pass_id']):
            max_row = max_row_dict[(station_code, pass_id)]
            igs_vtec = max_row['igs_vtec']
            corrected_pass = correct_pass_vtec(pass_df, max_row, igs_vtec, range_ratio)
            corrected_passes.append(corrected_pass)

        pass_all_station_processed = pd.concat(corrected_passes, ignore_index=True)

        ## storage of pass with VTEC into file
        pass_file_name = f'./DORISVTECStorage/pandas/{proc_sate}/{year}/DOY{doy:03d}.pickle'
        with open(pass_file_name, "wb") as f:
            pickle.dump(pass_all_station_processed, f)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")