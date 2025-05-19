from ObjectClasses import Thresholds
from ObsStorage import DORISStorage
from OrbitStorage import OrbitStorage
from StationStorage import StationStorage
from pandas import Timestamp, Timedelta
from pathlib import Path
import os
import re
import time
import pandas as pd



def find_covering_sp3_files_from_dir(start_time: pd.Timestamp, end_time: pd.Timestamp, folder_path: str) -> list:
    
    def yydoy_to_timestamp(yydoy: int) -> pd.Timestamp:
        year = 2000 + yydoy // 1000
        doy = yydoy % 1000
        return pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1)
    
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Provided path {folder_path} is not a valid directory.")
    
    matching_files = []
    for sp3_file in folder.glob("*.sp3"):
        match = re.search(r'\.b(\d{5})\.e(\d{5})\.', sp3_file.name)
        if match:
            b_yydoy = int(match.group(1))
            e_yydoy = int(match.group(2))
            file_start = yydoy_to_timestamp(b_yydoy)
            file_end = yydoy_to_timestamp(e_yydoy)
            if file_end >= start_time and file_start <= end_time:
                matching_files.append(folder_path + '/' + sp3_file.name)
    return sorted(matching_files)

if __name__ == '__main__':

    start_time = time.time()

    year = 2024
    month = 5
    day = 8
    proc_days = 30
    start_dt = Timestamp(year, month, day)
    end_dt = Timestamp(year, month, day) + Timedelta(days=proc_days)

    settings = Thresholds(max_dion_gap=None, max_obs_epoch_gap=None, min_obs_count=None, ele_cut_off=0) 

    file = './DORISInput/sinex/dpod2020_031.snx'
    stations = StationStorage()
    stations.read_sinex(file, start_dt, end_dt)

    sp3_dir = "./DORISInput/sp3"
    matching_sp3 = find_covering_sp3_files_from_dir(start_dt, end_dt, sp3_dir)
    orbit = OrbitStorage(matching_sp3)
    
    for i in range(proc_days):
        process_epoch = Timestamp(year, month, day) + Timedelta(days=i)
        doy = process_epoch.dayofyear

        obs = DORISStorage()
        file = f'./DORISInput/rinexobs/ja3rx{str(year)[-2:]}{doy:03d}.001'
        obs.read_rinex_300(file, orbit, stations)
        obs.storage.to_parquet(f'./DORISObsStorage/pandas/{year}/DOY{doy:03d}.parquet', engine="pyarrow", index=False)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

