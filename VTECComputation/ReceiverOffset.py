from pandas import Timestamp
from ObsStorageFast import DORISStorage
from StationOffset import process_station_observations
import constant as const
import numpy as np

def compute_sat_clock_corrections(process_epoch: Timestamp, obs: DORISStorage):
    _, T0_sec_stamp = process_station_observations(obs.storage[0])
    doy, year = process_epoch.dayofyear, process_epoch.year   
    # Extract reference station data
    time_ref_station, station_bias, station_shift = zip(*[
        (s.station_code, s.time_bias * 1e-6, s.time_shift * 1e-14)
        for s in obs.stations if s.time_ref_bit
    ]) if obs.stations else ([], [], [])

    equation_left, equation_right_epoch = [], []
    for station_obs in obs.storage:
        if not station_obs or station_obs[0].station_code not in time_ref_station:
            continue
        
        index = time_ref_station.index(station_obs[0].station_code)
        station_clock_bias = station_bias[index]
        
        for obs_this_epoch in station_obs:
            if abs(obs_this_epoch.C1 - obs_this_epoch.C2) > 10000 or obs_this_epoch.obs_epoch.second % 10 != T0_sec_stamp:
                continue

            station_pseudo_iono = (obs_this_epoch.C1 * const.iono_coeff - obs_this_epoch.C2) / (const.iono_coeff - 1)
            sec_from_start_to_epoch_rinex = (obs_this_epoch.obs_epoch - process_epoch).total_seconds()

            epoch_tai = obs_this_epoch.obs_epoch
            elev = obs_this_epoch.elevation

            geom_dis = (
                np.linalg.norm(obs_this_epoch.pos_sat_cele - obs_this_epoch.pos_sta_cele)
                - const.starec_iono_comb_pco * np.sin(np.deg2rad(elev))
                + 2.47 / (np.sin(np.deg2rad(elev)) + 0.0121)  # Tropospheric delay
            )
            clock_station_bias = station_clock_bias + station_shift[index] * (sec_from_start_to_epoch_rinex - station_pseudo_iono / const.c)

            equation_left.append((station_pseudo_iono - geom_dis - const.c * clock_station_bias) / const.c)
            equation_right_epoch.append((epoch_tai - process_epoch).total_seconds())
      
    sat_sec_shift, sat_shift, sat_bias = np.polyfit(equation_right_epoch, equation_left, 2)
    return sat_bias, sat_shift, sat_sec_shift
