from pandas import Timestamp
from ObsStorage import DORISStorage
import constant as const
import numpy as np

def compute_sat_clock_corrections(process_epoch: Timestamp, obs: DORISStorage):

    prn = obs.storage["PRN"].values[0]
    if prn == "L39":
        pco = const.starec_ja3_iono_comb_pco

    station_time_dict = {
        s.station_code: (s.time_bias * 1e-6, s.time_shift * 1e-14)
        for s in obs.stations if s.time_ref_bit
    } if obs.stations else {}

    df_obs = obs.storage
    df_filtered = df_obs[df_obs["station_code"].isin(station_time_dict.keys())].sort_values(by=["station_code", "obs_epoch"]).reset_index(drop=True)

    df_filtered["epoch_diff_sec"] = (
        df_filtered.groupby("station_code")["obs_epoch"]
        .diff().dt.total_seconds()
    )

    df_7s_gap = df_filtered[
        (df_filtered["epoch_diff_sec"] >= 6) & (df_filtered["epoch_diff_sec"] <= 8)
    ]

    df_pseudo_iono_good = df_7s_gap[(df_7s_gap["C1"] - df_7s_gap["C2"]).abs() < 10000]

    ds_pseudo_iono = (
        df_pseudo_iono_good["C1"] * const.iono_coeff - df_pseudo_iono_good["C2"]
    ) / (const.iono_coeff - 1)

    ds_sec_diff = (df_pseudo_iono_good["obs_epoch"] - process_epoch).dt.total_seconds()

    ds_station_shift_tuples = df_pseudo_iono_good["station_code"].map(station_time_dict)
    ds_time_bias_values = ds_station_shift_tuples.str[0]
    ds_time_shift_values = ds_station_shift_tuples.str[1]

    ds_clock_station_bias = ds_time_bias_values + ds_time_shift_values * (
        ds_sec_diff - ds_pseudo_iono / const.c
    )

    geom_vec = np.sqrt(
        (df_pseudo_iono_good["sat_x"] - df_pseudo_iono_good["sta_x"])**2 +
        (df_pseudo_iono_good["sat_y"] - df_pseudo_iono_good["sta_y"])**2 +
        (df_pseudo_iono_good["sat_z"] - df_pseudo_iono_good["sta_z"])**2
    )
    elev_rad = np.deg2rad(df_pseudo_iono_good["elevation"])
    ds_geom_dis = (
        geom_vec
        - pco * np.sin(elev_rad)
        + 2.47 / (np.sin(elev_rad) + 0.0121) # tropospheric delay model
    )
   
    ds_clock_time = (ds_pseudo_iono - ds_geom_dis - const.c * ds_clock_station_bias) / const.c
 
    sat_sec_shift, sat_shift, sat_bias = np.polyfit(ds_sec_diff, ds_clock_time, 2)

    return sat_bias, sat_shift, sat_sec_shift
