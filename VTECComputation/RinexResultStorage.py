from ObjectClasses import Thresholds
from ObsStorageFast import DORISStorage
from OrbitStorage import OrbitStorage
from StationStorage import StationStorage
from pandas import Timestamp, Timedelta
import os
import re
import time
import pickle

def find_sp3_files(folder_path, year, doy):
    matching_files = []
    # match files with names like ssaja320.b19330.e19340.DG_.sp3
    pattern = re.compile(r"\.b(\d{2})(\d{3})\.e(\d{2})(\d{3})\.")

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.sp3'):
            match = pattern.search(file_name)
            if match:
                start_year, start_doy, end_year, end_doy = map(int, match.groups())

                if (start_year < year or (start_year == year and start_doy <= doy)) and \
                    (end_year > year or (end_year == year and end_doy >= doy)):
                    matching_files.append(os.path.join(folder_path, file_name))

                    continue

    return matching_files

if __name__ == '__main__':

    start_time = time.time()

    year = 2024
    month = 5
    day = 8
    proc_days = 9
    start_dt = Timestamp(year, month, day)
    end_dt = Timestamp(year, month, day) + Timedelta(days=proc_days)

    settings = Thresholds(max_dion_gap=None, max_obs_epoch_gap=None, min_obs_count=None, ele_cut_off=0) 

    file = './DORISInput/sinex/dpod2020_031.snx'
    stations = StationStorage()
    stations.read_sinex(file, start_dt, end_dt)

    for i in range(proc_days):
        process_epoch = Timestamp(year, month, day) + Timedelta(days=i)
        doy = process_epoch.dayofyear

        sp3_dir = "./DORISInput/sp3"
        matching_sp3 = find_sp3_files(sp3_dir, year%100, doy)
        orbit = OrbitStorage(matching_sp3)

        obs = DORISStorage()
        file = f'./DORISInput/rinexobs/ja3rx{str(year)[-2:]}{doy:03d}.001'
        obs.read_rinex_300(file, orbit, stations)
        with open(f'./DORISObsStorage/{year}/DOY{doy:03d}.pkl', 'wb') as file:
            pickle.dump(obs, file)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

