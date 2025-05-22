from ObjectClasses import Thresholds
from ObsStorage import DORISStorage
from OrbitStorage import OrbitStorage
from StationStorage import StationStorage
from pandas import Timestamp, Timedelta
from pathlib import Path
import re
import time
import pandas as pd
import pickle

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
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    data_dir: Path,
    margin_minutes: int = 30,
):
    year = start_time.year
    margin = pd.Timedelta(minutes=margin_minutes)

    def load_day_data(doy: int) -> pd.DataFrame:
        path = data_dir / f"{year}/DOY{doy:03d}.pickle"
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    for day in pd.date_range(start=start_time, end=end_time, freq="D"):
        
        doy = day.dayofyear
        prev_obs = load_day_data(doy - 1)
        curr_obs = load_day_data(doy)
        next_obs = load_day_data(doy + 1)

        if curr_obs is None or curr_obs.storage.empty:
            continue

        all_data = pd.concat([
            prev_obs.storage if prev_obs else pd.DataFrame(),
            curr_obs.storage,
            next_obs.storage if next_obs else pd.DataFrame()
        ], ignore_index=True)

        df_window = all_data[
            (all_data["obs_epoch"] >= day - margin) &
            (all_data["obs_epoch"] < day + pd.Timedelta(days=1) + margin)
        ].sort_values(by=["station_code", "obs_epoch"])

        new_obs = DORISStorage()
        new_obs.storage = df_window
        new_obs.stations = curr_obs.stations

        out_path = data_dir / f"{year}/DOY{doy:03d}.pickle"
        with open(out_path, "wb") as f:
            pickle.dump(new_obs, f)


if __name__ == '__main__':

    start_time = time.time()

    year, month, day = 2024, 5, 8
    proc_days = 30
    
    start_dt = Timestamp(year, month, day)
    end_dt = Timestamp(year, month, day) + Timedelta(days=proc_days - 1)

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
        with open(obs_dir / f"{year}/DOY{doy:03d}.pickle", "wb") as f:
            pickle.dump(obs, f)

    regenerate_daily_obs_with_margin(start_dt, end_dt, obs_dir, margin_minutes=30)


    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

