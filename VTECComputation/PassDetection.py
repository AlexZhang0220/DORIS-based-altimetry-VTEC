from ObjectClasses import DORISObs, Thresholds, PassObj
import numpy as np
from astropy.time import Time
import constant as const
from scipy.stats import zscore

def process_station_observations(station_obs: list[DORISObs]):
    """
    Groups observations into passes based on time gaps.
    """
    if len(station_obs) < 10:
        return [], 0
    
    grouped_obs = []
    current_group = [station_obs[0]]
    flag = True
    
    for i in range(1, len(station_obs)):
        time_gap = (station_obs[i].obs_epoch - station_obs[i - 1].obs_epoch).total_seconds()
        if time_gap < 9:
            current_group.append(station_obs[i])
            if flag and time_gap > 5:
                T0_sec_stamp = station_obs[i].obs_epoch.second % 10
                ant_type = station_obs[i].ant_type[0:6]
                flag = False
        else:
            if len(current_group) >= 10:
                grouped_obs.append(current_group)
            current_group = [station_obs[i]]
    
    if len(current_group) >= 10:
        grouped_obs.append(current_group)
    
    return grouped_obs, T0_sec_stamp, ant_type

def create_pass(obs: list[DORISObs]) -> PassObj:
    """
    Creates a PassObj from observations.
    """
    return PassObj(
        epoch=[Time(o.obs_epoch).mjd for o in obs],
        ipp_lat=[o.ipp_lat for o in obs],
        ipp_lon=[o.ipp_lon for o in obs],
        STEC_mod=[o.STEC - obs[0].STEC for o in obs],
        elevation=[o.elevation for o in obs],
        map_value=[o.map_value for o in obs],
        station_code=obs[0].station_code,
        station_id=obs[0].station_id
    )

def detect_passes(station_obs: list[DORISObs], sat_clock_offset: list[float]) -> list[PassObj]:
    """
    Detects satellite passes and applies cycle slip detection.
    """
    passes = []
    sat_bias, sat_shift, sat_sec_shift = sat_clock_offset
    pass_obs_list, T0_sec_stamp, ant_type = process_station_observations(station_obs)
    if not pass_obs_list:
        return []
    process_epoch = pass_obs_list[0][0].obs_epoch.replace(hour=0, minute=0, second=0, microsecond=0)

    if ant_type[0:6] == 'STAREC':
        L1_pco_offset = const.jason3_L1_pco + const.starec_L1_pco
        L2_pco_offset = const.jason3_L2_pco + const.starec_L2_pco
    elif ant_type[0:7] == 'ALCATEL':
        L1_pco_offset = const.jason3_L1_pco + const.alcatel_L1_pco
        L2_pco_offset = const.jason3_L2_pco + const.alcatel_L2_pco

    cycle_slip_offset = 5 / const.d_lambda1 * L1_pco_offset - 1 / const.d_lambda2 * L2_pco_offset
    geom_coeff = 5 / const.d_lambda1 - 1 / const.d_lambda2
    delta_t = 10

    for pass_obs in pass_obs_list:
        elevations = np.array([obs.elevation for obs in pass_obs])
        valid_indices = elevations > 10
        if valid_indices.sum() < 10:
            continue
        pass_obs_elev10 = np.array(pass_obs)[valid_indices]
        condition_A = pass_obs_elev10[0].obs_epoch.second % 10 == T0_sec_stamp
        valid_elevation = elevations[valid_indices]
        # receiver_offset = np.array([obs.clock_offset for obs in pass_obs_elev10])
        cycle_slip_comb_phase = 5 * np.array([obs.L1 for obs in pass_obs_elev10]) - np.array([obs.L2 for obs in pass_obs_elev10])
        geom_dis = np.linalg.norm([obs.pos_sat_cele - obs.pos_sta_cele for obs in pass_obs_elev10], axis=1)
        delta_sec = np.array([(obs.obs_epoch-process_epoch).total_seconds() for obs in pass_obs_elev10])
        sat_offsets = sat_bias + sat_shift * delta_sec + sat_sec_shift * delta_sec ** 2
        geom_dis_cycle_slip = geom_coeff * geom_dis - cycle_slip_offset * np.sin(np.deg2rad(valid_elevation)) - geom_coeff * sat_offsets * const.c
        mapping_func = 1 / np.sin(np.deg2rad(valid_elevation))
        Est_cycle_slip_comb = cycle_slip_comb_phase - geom_dis_cycle_slip

        indices = np.arange(len(Est_cycle_slip_comb))
        result_array = np.where(indices % 2 == 0, indices // 2, indices // 2 + (0.3 if condition_A else 0.7))
        X = np.column_stack((np.ones(len(Est_cycle_slip_comb)), result_array * delta_t, mapping_func))
        beta, _, _, _ = np.linalg.lstsq(X, Est_cycle_slip_comb, rcond=None)
        residuals = Est_cycle_slip_comb - (X @ beta)
        diff_residuals = np.diff(residuals)
        z_scores = np.abs(zscore(diff_residuals))
        split_indices = np.where(z_scores > 3)[0] + 1 if np.max(np.abs(diff_residuals)) > 10 else np.array([], dtype=int)
            
        if len(split_indices) <= 2: 
            indice_cycle_slip = np.where(np.abs(diff_residual_comb) > 3)[0] + 1
            if len(indice_cycle_slip) != 0:
                print('')
            sub_passes = np.split(pass_obs_elev10, indice_cycle_slip)
            for sub_pass in sub_passes:
                passes.append(create_pass(sub_pass.tolist()))

        else:
            diff_residual_seg_comb, diff_elev_seg = [], []
            remaining_start_indices = []
            for i in range(len(split_indices) - 1):
                start, end = split_indices[i], split_indices[i + 1]
                if end - start > 9:
                    X_segment = X[start:end]
                    y2_segment = Est_cycle_slip_comb[start:end]
                    elev_segment = elevations[start:end]

                    beta_cycle_slip_comb, _, _, _ = np.linalg.lstsq(X_segment, y2_segment, rcond=None)
                    residual_seg_comb = y2_segment - X_segment @ beta_cycle_slip_comb

                    diff_residual_seg_comb.append(np.diff(residual_seg_comb))
                    diff_elev_seg.append(elev_segment[1:]) 
                    remaining_start_indices.append(start)

            for diff_ind, diff_residual_comb in enumerate(diff_residual_seg_comb):

                local_indices = np.where(diff_residual_comb > 3.5)[0] + 1 
                if len(local_indices) != 0:
                    print('')
                local_indices = np.hstack(([0], local_indices, [len(diff_residual_comb)+1])) 
                original_indices = local_indices + remaining_start_indices[diff_ind]
                splited_passes = np.split(pass_obs_elev10, original_indices)[1:-1]
                for splited_pass in splited_passes:
                    passes.append(create_pass(splited_pass.tolist()))
    return passes
            



