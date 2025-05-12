
from ObjectClasses import DORISObs
from OrbitStorage import OrbitStorage
from pandas import Timestamp, Timedelta
from RinexResultStorage import find_sp3_files
import constant as const
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import zscore
from PassDetection import create_pass

def process_station_observations(station_obs: list[DORISObs]):

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
                flag = False
        else:
            grouped_obs.append(current_group)
            current_group = [station_obs[i]]
    
    if current_group and len(current_group) >= 10:
        grouped_obs.append(current_group)
    
    return grouped_obs, T0_sec_stamp

if __name__ == "__main__":
    L1_pco_offset = (const.jason3_L1_pco + const.starec_L1_pco)
    L2_pco_offset = (const.jason3_L2_pco + const.starec_L2_pco)
    cycle_slip_comb_pco_offset =  5 / const.d_lambda1 * L1_pco_offset - 1 / const.d_lambda2 * L2_pco_offset
    year, month, day = 2019, 12, 30
    process_epoch = Timestamp(year, month, day)
    doy = process_epoch.dayofyear
    orbit = OrbitStorage()
    sp3_dir = "./DORISInput/sp3"
    matching_sp3 = find_sp3_files(sp3_dir, year%100, doy)
    for sp3_file in matching_sp3:
        orbit.read_sp3(sp3_file, 'L39')

    with open(f'./DORISObsStorage/{year}/DOY{doy:03d}.pkl', 'rb') as file:
        obs = pickle.load(file)
    from ReceiverOffset import compute_sat_clock_corrections
    sat_bias, sat_shift, sat_sec_shift = compute_sat_clock_corrections(process_epoch, obs)
    cycle_slip_geom_coeff = 5 / const.d_lambda1 - 1 / const.d_lambda2

    delta_t = 10 # second


    data_dict = {}

    for station_obs in obs.storage:
        pass_obs_list, T0_sec_stamp = process_station_observations(station_obs)

        passes = []

        for j, pass_obs in enumerate(pass_obs_list):

            L1_phase, cycle_slip_comb_phase = [], []
            geom_dis_cycle_slip_comb = []
            Est_cycle_slip_comb = []
            mapping_func = []
            elev_list = []
            flag = True
            receicer_offset = []
            elevations = np.array([obs.elevation for obs in pass_obs])
            indices = np.where(elevations > 10)[0]

            if indices.size < 10:
                continue
            pass_obs_elev10 = pass_obs[indices[0]:indices[-1]]
            condition_A = pass_obs_elev10[0].obs_epoch.second % 10 == T0_sec_stamp
            for ind, obs_this_epoch in enumerate(pass_obs_elev10):   
                receicer_offset.append(obs_this_epoch.clock_offset)         
                epoch_rinex = obs_this_epoch.obs_epoch
                sec_from_start_to_epoch_rinex = (epoch_rinex - process_epoch).total_seconds()
                sat_offset = sat_bias + sat_shift * sec_from_start_to_epoch_rinex + sat_sec_shift * (sec_from_start_to_epoch_rinex ** 2)
                epoch_tai = epoch_rinex - Timedelta(seconds=sat_offset)
                if obs_this_epoch.elevation > 10:
                    if flag:
                        condition_A = epoch_rinex.second % 10 == T0_sec_stamp
                        flag = False
                    elev_list.append(obs_this_epoch.elevation)
                    L1_phase.append(obs_this_epoch.L1)
                    cycle_slip_comb_phase.append(5*obs_this_epoch.L1-obs_this_epoch.L2)
                    geom_dis_no_corr = np.linalg.norm(np.array(obs_this_epoch.pos_sat_cele) - np.array(obs_this_epoch.pos_sta_cele))                               
                    geom_dis_cycle_slip_comb.append(cycle_slip_geom_coeff*geom_dis_no_corr
                                                    - cycle_slip_comb_pco_offset * np.sin(np.deg2rad(obs_this_epoch.elevation))
                                                    - cycle_slip_geom_coeff * sat_offset * const.c)               
                    mapping_func.append(1/(np.sin(np.deg2rad(obs_this_epoch.elevation))))
            
            if len(L1_phase) > 1 and np.max(abs(np.diff(L1_phase)))> 5e5:
                # abnormal phase observation; large jumps between epoches
                continue
            Est_cycle_slip_comb = cycle_slip_comb_phase - np.array(geom_dis_cycle_slip_comb)
            elev_list = np.array(elev_list)      

            if Est_cycle_slip_comb.size >= 10:  
                indices = np.arange(len(Est_cycle_slip_comb))
                if condition_A:
                    result_array = np.where(indices % 2 == 0, indices // 2, indices // 2 + 0.3)
                else:
                    result_array = np.where(indices % 2 == 0, indices // 2, indices // 2 + 0.7)
                station_code = pass_obs[0].station_code          
                N = len(Est_cycle_slip_comb)
                X1 = np.ones(N)
                X2 = result_array * delta_t 
                X3 = np.array(mapping_func)
                X = np.column_stack((X1, X2, X3))
                y2 = Est_cycle_slip_comb

                beta_cycle_slip_comb, _, _, _ = np.linalg.lstsq(X, y2, rcond=None)
                y_fit_cycle_slip_comb = X @ beta_cycle_slip_comb
                residual_cycle_slip_comb = y2 - y_fit_cycle_slip_comb

                diff_residual_comb = np.diff(residual_cycle_slip_comb)
                diff_elev_list = elev_list[1:]

                z_scores = np.abs(zscore(diff_residual_comb))
                if np.max(np.abs(diff_residual_comb)) > 8:
                    split_indices = np.where(z_scores > 3)[0] + 1
                else:
                    split_indices = np.array([], dtype=int) 
                split_indices = np.hstack(([0], split_indices, [len(diff_residual_comb) + 1]))
                
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
                        start, end = int(split_indices[i]), int(split_indices[i + 1])
                        if end - start > 9:
                            X_segment = X[start:end]
                            y2_segment = y2[start:end]
                            elev_segment = elev_list[start:end]

                            beta_cycle_slip_comb, _, _, _ = np.linalg.lstsq(X_segment, y2_segment, rcond=None)
                            y_fit_cycle_slip_comb = X_segment @ beta_cycle_slip_comb
                            residual_seg_comb = y2_segment - y_fit_cycle_slip_comb

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
    
    with open('data_dict_5_8.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

                    