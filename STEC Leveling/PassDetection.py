from typing import List, Dict, Tuple
from ObjectClasses import DORISObs, Thresholds, PassObj
import numpy as np
from astropy.time import Time

def sigmatest(station_obs: List[DORISObs]) -> List[DORISObs]:
    dion = [obj.dion for obj in station_obs]
    std = np.std(dion)
    mean = np.mean(dion)
    for idx in range(len(station_obs) - 1, -1, -1):
        if abs(station_obs[idx].dion - mean) > 3 * std:
            station_obs.pop(idx)
        #可能添加两行检查始末epoch？
    return station_obs

def passdetection(station_obs: List[DORISObs], settings: Thresholds) -> List[PassObj]:

    passes = []
    obs = [station_obs[0]]
    station_code = station_obs[0].station_code
    station_id = station_obs[0].station_id
    for i in range(1,len(station_obs)):
        if (station_obs[i].obs_epoch - station_obs[i-1].obs_epoch).total_seconds() < settings.max_obs_epoch_gap \
            and abs(station_obs[i].dion - station_obs[i-1].dion) < settings.max_dion_gap:
            obs.append(station_obs[i])
        else:
             
            if len(obs) >= settings.min_obs_count:
                # dion_mod = [obs_this_epoch.dion - obs[0].dion for obs_this_epoch in obs]
                STEC_mod = [obs_this_epoch.STEC - obs[0].STEC for obs_this_epoch in obs]
                ipp_lat = [obs_this_epoch.ipp_lat for obs_this_epoch in obs]
                ipp_lon = [obs_this_epoch.ipp_lon for obs_this_epoch in obs]
                epoch = [Time(obs_this_epoch.obs_epoch).mjd for obs_this_epoch in obs]
                elevation = [obs_this_epoch.elevation for obs_this_epoch in obs]
                map_value = [obs_this_epoch.map_value for obs_this_epoch in obs]

                passes.append(PassObj(epoch, ipp_lat, ipp_lon, STEC_mod, elevation, map_value, station_code, station_id))
            obs = [station_obs[i]]
    # append the last pass
    obs = sigmatest(obs)
    if len(obs) >= settings.min_obs_count:
        
        STEC_mod = [obs_this_epoch.STEC - obs[0].STEC for obs_this_epoch in obs]
        ipp_lat = [obs_this_epoch.ipp_lat for obs_this_epoch in obs]
        ipp_lon = [obs_this_epoch.ipp_lon for obs_this_epoch in obs]
        epoch = [Time(obs_this_epoch.obs_epoch).mjd for obs_this_epoch in obs]
        elevation = [obs_this_epoch.elevation for obs_this_epoch in obs]
        map_value = [obs_this_epoch.map_value for obs_this_epoch in obs]

        passes.append(PassObj(epoch, ipp_lat, ipp_lon, STEC_mod, elevation, map_value, station_code, station_id))
    return passes





