from ObjectClasses import DORISObs, Thresholds
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
from itertools import groupby, cycle
import pickle

def split_and_filter_by_time_gap(station_obs: list[DORISObs], settings: Thresholds) -> list[PassObj]:
    passes = []
    obs = [station_obs[0]]
    station_code = station_obs[0].station_code
    station_id = station_obs[0].station_id
    sentinel = DORISObs(0, 0, 0, [0], Timestamp(2020, 1, 1), [0], 0, 0)
    station_obs.append(sentinel)
    def create_pass(obs: list[DORISObs]) -> PassObj:
        STEC_mod = [o.STEC - obs[0].STEC for o in obs if o.elevation>10]
        ipp_lat = [o.ipp_lat for o in obs if o.elevation>10]
        ipp_lon = [o.ipp_lon for o in obs if o.elevation>10]
        epoch = [Time(o.obs_epoch).mjd for o in obs if o.elevation>10]
        elevation = [o.elevation for o in obs if o.elevation>10]
        map_value = [o.map_value for o in obs if o.elevation>10]
        return PassObj(epoch, ipp_lat, ipp_lon, STEC_mod, elevation, map_value, station_code, station_id)

    for i in range(1, len(station_obs)):
        time_gap = (station_obs[i].obs_epoch - station_obs[i - 1].obs_epoch).total_seconds()

        if time_gap < settings.max_obs_epoch_gap:
            obs.append(station_obs[i])
        else:
            if len(obs) >= settings.min_obs_count:
                epoch_dion = np.array([o.dion for o in obs])
                dd_elevation = np.array([o.elevation for o in obs][2:])
                delta_dion = np.diff(epoch_dion)
                delta_epoch_1 = round((obs[1].obs_epoch - obs[0].obs_epoch).total_seconds())
                base_pattern = [1, 7/3] if delta_epoch_1 == 7 else [7/3, 1]
                pattern = cycle(base_pattern)
                sequence = np.array([next(pattern) for _ in range(len(delta_dion))])
                delta_dion_mod = sequence * delta_dion
                delta_dion_mod_2 = np.diff(delta_dion_mod)
                
                cond1 = (dd_elevation > 30) & (np.abs(delta_dion_mod_2) > 0.015)
                cond2 = (dd_elevation <= 30) & (np.abs(delta_dion_mod_2) > 0.020)
                index_mod = np.where(cond1|cond2)[0]
                # locate consecutive indices (that is introduced by double epoch difference)
                def find_continuous_ranges(nums):
                    ranges = []
                    for _, group in groupby(enumerate(nums), lambda x: x[1] - x[0]):
                        group = list(group)
                        ranges.append([group[0][1], group[-1][1]])
                    return ranges

                def split_by_ranges(input, dd_index, min_count): # dd_index is the double difference index
                    result = []
                    start = 0  
                    for r in dd_index:
                        if r[0] != r[1]:
                            if (r[0] + 1- start) > min_count:
                                result.append(input[start:r[0]+2])
                            start = r[1] + 1 
                        else:
                            if (r[0] + 1- start) > min_count:
                                result.append(input[start:r[0]+1])
                            start = r[1] + 2
                    if start < len(input) and (len(input)- start) > min_count:
                        result.append(input[start:])
                    return result   

                # Use the remove_consecutive function with np.where
                index_mod = find_continuous_ranges(index_mod)

                if index_mod:
                    obs_subsets = split_by_ranges(obs, index_mod, settings.min_obs_count)
                    for subset in obs_subsets:
                        passes.append(create_pass(subset))
                else:
                    passes.append(create_pass(obs))

            # Reset observation list
            obs = [station_obs[i]]

    return passes

def process_arc(arc: PassObj, GIMVTEC, range_ratio: float):
    igs_vtec = np.array(GIMInpo(GIMVTEC, arc.epoch, arc.ipp_lat, arc.ipp_lon))
    igs_stec = igs_vtec * np.array(arc.map_value)
    max_index = np.argmax(arc.elevation)
    delta = igs_stec[max_index] * range_ratio - arc.STEC[max_index]
    arc.STEC += delta
    arc.vtec = (arc.STEC / np.array(arc.map_value)).tolist()
    return arc

def elevation_noise(station_obs: list[DORISObs], settings: Thresholds) -> list[PassObj]:
    elevation = []
    dd_noise = []
    obs = [station_obs[0]]
    station_code = station_obs[0].station_code
    station_id = station_obs[0].station_id
    sentinel = DORISObs(0, 0, 0, [0], Timestamp(1, 1, 1), [0], 0)
    station_obs.append(sentinel)

    for i in range(1, len(station_obs)):
        time_gap = (station_obs[i].obs_epoch - station_obs[i - 1].obs_epoch).total_seconds()

        if time_gap < settings.max_obs_epoch_gap:
            obs.append(station_obs[i])
        else:
            if len(obs) >= settings.min_obs_count:
                epoch_dion = np.array([o.dion for o in obs])
                dd_elevation = np.array([o.elevation for o in obs][2:])
                delta_dion = np.diff(epoch_dion)
                delta_epoch_1 = round((obs[1].obs_epoch - obs[0].obs_epoch).total_seconds())
                base_pattern = [1, 7/3] if delta_epoch_1 == 7 else [7/3, 1]
                pattern = cycle(base_pattern)
                sequence = np.array([next(pattern) for _ in range(len(delta_dion))])
                delta_dion_mod = sequence * delta_dion
                delta_dion_mod_2 = np.diff(delta_dion_mod)
                
                condition = np.abs(delta_dion_mod_2) < 1
                index_mod = np.where(condition)[0]
                elevation.extend(dd_elevation[index_mod])
                dd_noise.extend(delta_dion_mod_2[index_mod])
            obs = [station_obs[i]]
            
    return elevation, dd_noise

if __name__ == '__main__':

    start_time = time.time()

    year = 2024
    month = 5
    day = 8
    proc_days = 1
    max_dion_gap = 0.015
    min_obs_count = 30
    max_obs_epoch_gap = 9
    settings = Thresholds(max_dion_gap, max_obs_epoch_gap, min_obs_count, ele_cut_off=None) 
    elevation = []
    doppler = []
    for i in range(proc_days):
        process_epoch = Timestamp(year, month, day) + Timedelta(days=i)
        doy = process_epoch.dayofyear

        with open(f'./DORISObsStorage/{year}/DOY{doy:03d}.pkl', 'rb') as path:
            obs = pickle.load(path)

 
        
        pass_list= []
        for grouped_obs in obs.storage:
            passes, elev, diff_res = detect_passes(grouped_obs, min_obs_count)
            pass_list.extend(passes)
        
        print(np.sum([len(passes.STEC) for passes in pass_list if len(passes.STEC) >= 30]))

        ion_file = (
            f"./DORISInput/IGSGIM/{year}/igsg{doy:03d}0.{str(year)[-2:]}i"
            if year <= 2022
            else f"./DORISInput/IGSGIM/{year}/IGS0OPSFIN_{year}{doy:03d}0000_01D_02H_GIM.INX"
        )

        gim_vtec = read_ionFile(ion_file) * 0.1
        gim_vtec = gim_vtec[:, ::-1, :]

        range_ratio = 0.925
        pass_list = Parallel(n_jobs=-1)(delayed(process_arc)(arc, gim_vtec, range_ratio) for arc in pass_list)
        # Vtec Std of the window Number of obs in the window
        Passh5 = f'./DORISVTEC/{year}/DOY{doy:03d}.h5'
        with h5py.File(Passh5, 'w') as f:
            setting_group = f.require_group(f'y{year}/{doy:03d}')
            
            for index, arc in enumerate(pass_list):
                group_name = f'pass{index}-{arc.station_code}'
                pass_group = setting_group.require_group(group_name)

                datasets = {
                    'epoch': arc.epoch,
                    'map_value': arc.map_value,
                    'ipp_lat': arc.ipp_lat,
                    'ipp_lon': arc.ipp_lon,
                    'elevation': arc.elevation,
                    'stec': arc.STEC,
                    'station_code': arc.station_code,
                    'vtec': arc.vtec
                }

                for name, data in datasets.items():
                    if name in pass_group: 
                        del pass_group[name]
                    pass_group.create_dataset(name, data=data)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")