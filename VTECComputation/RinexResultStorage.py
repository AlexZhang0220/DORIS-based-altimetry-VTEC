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

def regenerate_daily_obs_with_margin(
    start_time: Timestamp,
    end_time: Timestamp,
    data_dir: Path,
    margin_minutes: int = 30,
):
    year = start_time.year
    margin = Timedelta(minutes=margin_minutes)

    def load_day_data(doy_val: int) -> pd.DataFrame:
        path = data_dir / f"{year}/DOY{doy_val:03d}.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        return df

    for day in pd.date_range(start=start_time, end=end_time, freq="D"):
        doy = day.dayofyear
        df_prev = load_day_data(doy - 1)
        df_curr = load_day_data(doy)
        df_next = load_day_data(doy + 1)

        if df_curr.empty:
            continue 

        df_all = pd.concat([df_prev, df_curr, df_next], ignore_index=True)
        df_all = df_all.sort_values("obs_epoch")

        start_window = day - margin
        end_window = day + Timedelta(days=1) + margin

        df_with_margin = df_all[(df_all["obs_epoch"] >= start_window) & (df_all["obs_epoch"] < end_window)]

        save_path = data_dir / f"{year}/DOY{doy:03d}.parquet"
        df_with_margin.to_parquet(save_path, index=False, engine="pyarrow")

if __name__ == '__main__':

    start_time = time.time()

    year = 2024
    month = 5
    day = 8
    proc_days = 30
    start_dt = Timestamp(year, month, day)
    end_dt = Timestamp(year, month, day) + Timedelta(days=proc_days - 1)

    settings = Thresholds(max_dion_gap=None, max_obs_epoch_gap=None, min_obs_count=None, ele_cut_off=0) 

    file = './DORISInput/sinex/dpod2020_031.snx'
    stations = StationStorage()
    stations.read_sinex(file, start_dt, end_dt)

    sp3_dir = "./DORISInput/sp3"
    matching_sp3 = find_covering_sp3_files_from_dir(start_dt, end_dt, sp3_dir)
    orbit = OrbitStorage(matching_sp3)
    
    obs_dir = Path("./DORISObsStorage/pandas")
    for i in range(proc_days):
        process_epoch = Timestamp(year, month, day) + Timedelta(days=i)
        doy = process_epoch.dayofyear

        obs = DORISStorage()
        file = f'./DORISInput/rinexobs/ja3rx{str(year)[-2:]}{doy:03d}.001'
        obs.read_rinex_300(file, orbit, stations)
        obs.storage.to_parquet(obs_dir / f"{year}/DOY{doy:03d}.parquet", engine="pyarrow", index=False)

    regenerate_daily_obs_with_margin(start_dt, end_dt, obs_dir, margin_minutes=30)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

