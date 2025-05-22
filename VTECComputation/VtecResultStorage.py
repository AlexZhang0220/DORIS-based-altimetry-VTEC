from ObjectClasses import DORISObs, Thresholds
from ReceiverOffsetFast import compute_sat_clock_corrections
from CycleSlipDetection import detect_passes
from ObjectClasses import PassObj
from tools import GIMInpo
from readFile import read_ionFile
from joblib import Parallel, delayed
from pandas import Timestamp, Timedelta
from astropy.time import Time
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import groupby, cycle
import pickle

def process_pass_df(pass_df: pd.DataFrame, GIMVTEC, range_ratio: float) -> pd.DataFrame:

    igs_vtec = np.array(GIMInpo(GIMVTEC, pass_df['obs_epoch'], pass_df['ipp_lat'], pass_df['ipp_lon']))
    igs_stec = igs_vtec * pass_df['map_value'].values
    max_idx = pass_df['elevation'].idxmax()
    delta = igs_stec[max_idx] * range_ratio - pass_df['STEC'].values[max_idx]
    corrected_stec = pass_df['STEC'] + delta
    vtec = corrected_stec / pass_df['map_value']
    
    pass_df = pass_df.copy()
    pass_df['STEC'] = corrected_stec
    pass_df['VTEC'] = vtec
    
    return pass_df

if __name__ == '__main__':

    start_time = time.time()
    
    year, month, day = 2024, 5, 8
    proc_days = 30
    min_obs_count = 30

    for i in range(proc_days):
        process_epoch = Timestamp(year, month, day) + Timedelta(days=i)
        doy = process_epoch.dayofyear

        ## read in rinex-reading results
        with open(f'./DORISObsStorage/pandas/{year}/DOY{doy:03d}.pickle', 'rb') as path:
            obs = pickle.load(path)

        # sat_clock_offset = compute_sat_clock_corrections(process_epoch, obs)
        
        ## Detection and remove of cycle slip 
        pass_all_station= []
        for station_code, grouped_obs in obs.storage.groupby('station_code'):
            pass_all_station.append(detect_passes(grouped_obs, min_obs_count))
            
        pass_all_station = pd.concat(pass_all_station, ignore_index=True)
        

        ion_file = (
            f"./DORISInput/IGSGIM/{year}/igsg{doy:03d}0.{str(year)[-2:]}i"
            if year <= 2022
            else f"./DORISInput/IGSGIM/{year}/IGS0OPSFIN_{year}{doy:03d}0000_01D_02H_GIM.INX"
        )

        ## Computation of VTEC with IGS GIM
        gim_vtec = read_ionFile(ion_file) * 0.1
        gim_vtec = gim_vtec[:, ::-1, :]
        range_ratio = 0.925

        processed_passes = []

        for station_code, station_df in pass_all_station.groupby('station_code'):
            for pass_id, pass_df in station_df.groupby('pass_id'):
                pass_df = pass_df.reset_index(drop=True)
                processed_df = process_pass_df(pass_df, gim_vtec, range_ratio)
                processed_passes.append(processed_df)

        pass_all_station_processed = pd.concat(processed_passes, ignore_index=True)

        ## storage of pass with VTEC into file
        pass_file_name = f'./DORISVTEC/{year}/DOY{doy:03d}.pickle'
        with open(pass_file_name, "wb") as f:
            pickle.dump(pass_all_station_processed, f)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")