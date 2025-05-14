from ObjectClasses import DORISObs, PassObj
import numpy as np
from astropy.time import Time
from pandas import Timestamp, Timedelta
import constant as const
from OrbitStorage import OrbitStorage
from ObjectClasses import SatPos
from scipy.stats import zscore
import matplotlib.pyplot as plt

def group_observations_by_time(station_obs: list[DORISObs], min_obs_count):
    """
    Groups observations into satellite passes based on time gaps.
    """
    if len(station_obs) < 10:
        return [], 0, ""
    
    passes = []
    current_pass = [station_obs[0]]
    reference_time = None
    
    for i in range(1, len(station_obs)):
        time_gap = (station_obs[i].obs_epoch - station_obs[i - 1].obs_epoch).total_seconds()
        if time_gap < 9:
            current_pass.append(station_obs[i])
            if not reference_time and time_gap > 5:
                reference_time = station_obs[i].obs_epoch.second % 10
                antenna_type = station_obs[i].ant_type[:6]
        else:
            if len(current_pass) >= min_obs_count:
                passes.append(current_pass)
            current_pass = [station_obs[i]]
    
    if len(current_pass) >= min_obs_count:
        passes.append(current_pass)
    
    return passes, reference_time, antenna_type

def create_pass_object(observations: list[DORISObs]) -> PassObj:
    """
    Constructs a PassObj from a list of observations.
    """
    return PassObj(
        epoch=[Time(obs.obs_epoch).mjd for obs in observations],
        ipp_lat=[obs.ipp_lat for obs in observations],
        ipp_lon=[obs.ipp_lon for obs in observations],
        STEC=[obs.STEC - observations[0].STEC for obs in observations],
        elevation=[obs.elevation for obs in observations],
        map_value=[obs.map_value for obs in observations],
        station_code=observations[0].station_code,
        station_id=observations[0].station_id
    )

def detect_passes(station_obs: list[DORISObs], satellite_clock_offsets: list[float], min_obs_count) -> list[PassObj]:
    """
    Detects satellite passes and applies cycle slip detection.
    """
    passes, elev_list, diff_res_list = [], [], []
    # sat_bias, sat_shift, sat_sec_shift = satellite_clock_offsets
    grouped_obs, reference_time, antenna_type = group_observations_by_time(station_obs, min_obs_count)
    
    if not grouped_obs:
        return [], [], []

    # observation_start = grouped_obs[0][0].obs_epoch.replace(hour=0, minute=0, second=0, microsecond=0)
    
    if antenna_type == 'STAREC':
        L1_pco_offset = const.jason3_L1_pco + const.starec_L1_pco
        L2_pco_offset = const.jason3_L2_pco + const.starec_L2_pco
    elif antenna_type == 'ALCATEL':
        L1_pco_offset = const.jason3_L1_pco + const.alcatel_L1_pco
        L2_pco_offset = const.jason3_L2_pco + const.alcatel_L2_pco
    else:
        return [], [], []
    
    if grouped_obs[0][0].station_shift != 0:
        d_lambda1 = const.c / (5e6 * 407.25 * (1 + grouped_obs[0][0].station_shift * const.d_p_multik))
        d_lambda2 = const.c / (5e6 * 80.25 * (1 + grouped_obs[0][0].station_shift * const.d_p_multik))
    else:
        d_lambda1 = const.d_lambda1
        d_lambda2 = const.d_lambda2

    cycle_slip_pco_offset = (5 / d_lambda1 * L1_pco_offset) - (1 / d_lambda2 * L2_pco_offset)
    geom_factor = 5 / d_lambda1 - 1 / d_lambda2
    time_step = 10  # seconds
    
    for obs_pass in grouped_obs:
        elevations = np.array([obs.elevation for obs in obs_pass])
        valid_indices = elevations > 10
        if valid_indices.sum() < 10:
            continue
        
        filtered_pass = np.array(obs_pass)[valid_indices]
        elevations = elevations[valid_indices]
        condition_A = filtered_pass[0].obs_epoch.second % 10 == reference_time
        # delta_sec = np.array([(obs.obs_epoch - observation_start).total_seconds() for obs in filtered_pass])
        # satellite_offsets = sat_bias + sat_shift * delta_sec + sat_sec_shift * delta_sec ** 2
        L1_phase = np.array([obs.L1 for obs in filtered_pass])
        L2_phase = np.array([obs.L2 for obs in filtered_pass])
        if np.max(abs(np.diff(L1_phase)))> 5e5:
        # abnormal phase observation; large jumps between epoches
            continue
        cycle_slip_phase = 5 * L1_phase - L2_phase
        geom_distances = np.linalg.norm([obs.pos_sat_cele - obs.pos_sta_cele for obs in filtered_pass], axis=1)
        corrected_geom_distances = (geom_factor * geom_distances 
                                    - cycle_slip_pco_offset * np.sin(np.deg2rad(elevations)))                         
        mapping_func = 1 / np.sin(np.deg2rad(elevations))
        estimated_cycle_slip = cycle_slip_phase - corrected_geom_distances      
        indices = np.arange(len(estimated_cycle_slip))
        result_array = np.where(indices % 2 == 0, indices // 2, indices // 2 + (0.3 if condition_A else 0.7))
 
        X = np.column_stack((np.ones(len(estimated_cycle_slip)), result_array * time_step, mapping_func))
        beta, _, _, _ = np.linalg.lstsq(X, estimated_cycle_slip, rcond=None)
        residuals = estimated_cycle_slip - (X @ beta)
        diff_residuals = np.diff(residuals)
        z_scores = np.abs(zscore(diff_residuals))
        split_points = np.where(z_scores > 3)[0] + 1 if np.max(np.abs(diff_residuals)) > 10 else np.array([], dtype=int)
        split_points = np.hstack(([0], split_points, [len(residuals)]))
        if len(split_points) == 2:
            sub_passes = np.split(filtered_pass, np.where(np.abs(diff_residuals) > 3.5)[0] + 1)
            elev_list.extend(elevations[0:-1])
            diff_res_list.extend(diff_residuals)
        else:
            sub_passes = []
            for i in range(len(split_points) - 1):
                start, end = split_points[i], split_points[i + 1]
                if end - start >= min_obs_count:
                    beta_sub, _, _, _ = np.linalg.lstsq(X[start:end], estimated_cycle_slip[start:end], rcond=None)
                    residual_sub = estimated_cycle_slip[start:end] - (X[start:end] @ beta_sub)
                    diff_residual_sub = np.diff(residual_sub)
                    local_splits = np.where(diff_residual_sub > 3.5)[0] + 1
                    local_splits = np.hstack(([0], local_splits, [len(residual_sub)]))
                    sub_passes.extend(np.split(filtered_pass[start:end], local_splits)[1:-1])                  
                    elev_list.extend(elevations[start:end-1])
                    diff_res_list.extend(diff_residual_sub)
        
        for sub_pass in sub_passes:
            if len(sub_pass) >= min_obs_count:
                passes.append(create_pass_object(sub_pass.tolist()))
                
    return passes, elev_list, diff_res_list
