from typing import List, Dict, Tuple
from ObjectClasses import DORISObs, Thresholds, PassObj
from ObsStorage import DORISStorage
from OrbitStorage import OrbitStorage
from StationStorage import StationStorage
from datetime import datetime, timedelta
import numpy as np
from PassDetection import passdetection
import h5py
import time

start_time = time.time()

process_epoch = datetime(2019, 12, 26) # the data in user-defined DAY will be processed
year = process_epoch.year
doy = process_epoch.timetuple().tm_yday

settings = Thresholds(max_dion_gap=0.2, max_obs_epoch_gap=120, min_obs_count=50, ele_cut_off=0) 

orbit_file = "./STEC Leveling/sp3/ssaja320.b19350.e19360.DG_.sp3"
orbit = OrbitStorage()
orbit.read_sp3(orbit_file, 'L39') # L39 is Jason-3

station_file = './STEC Leveling/sinex/ids22d01.snx' # station coord file can be found at https://ids-doris.org/documents/BC/stations/ids.snx
stations = StationStorage()
stations.read_sinex(station_file)

obs = DORISStorage()       


file = './STEC Leveling/rinexobs/ja3rx'+str(year)[-2:]+str(doy)
obs.read_rinex_300(process_epoch, file, orbit, stations, settings)

passes = []
for station_obs in obs.storage:
    if station_obs == []:
        continue
    else:
        passes += passdetection(station_obs, settings)
file_path = './vtec/DOY'+str(doy)+'.h5' 

with h5py.File(file_path, 'a') as f:
    try:
        year_group = f.create_group(f'y{year}')
    except ValueError:
        year_group = f[f'y{year}']
    
    try:
        doy_group = year_group.create_group(f'd{doy}')
    except ValueError:
        doy_group = year_group[f'd{doy}']

    try:
        setting_group = doy_group.create_group(f'ele_cut_{settings.ele_cut_off}')
    except ValueError:
        setting_group = doy_group[f'ele_cut_{settings.ele_cut_off}']

    for index, arc in enumerate(passes):
        pass_group = setting_group.require_group(f'pass{index}-{arc.station_code}')

        pass_group.create_dataset('epoch', data=arc.epoch)
        pass_group.create_dataset('map_value', data=arc.map_value)
        pass_group.create_dataset('ipp_lat', data=arc.ipp_lat)
        pass_group.create_dataset('ipp_lon', data=arc.ipp_lon)
        pass_group.create_dataset('elevation', data=arc.elevation)
        pass_group.create_dataset('stec', data=arc.STEC)
        pass_group.create_dataset('station_code', data=arc.station_code)

end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.2f} seconds")
