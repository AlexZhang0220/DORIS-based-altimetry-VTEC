import numpy as np
import pandas as pd
import constant as const
from scipy.stats import zscore
import matplotlib.pyplot as plt

# def elev_noise(elevation, residuals, bin_size=5):
#     import pickle
#     from scipy.ndimage import gaussian_filter1d
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.figure(figsize=(17, 9))  # 宽高适配PPT
#     ax1 = plt.gca()  # 获取当前的坐标轴
#     with open(f'./DORISObsStorage/ja3/2024/DOY129.pickle', 'rb') as path:
#         obs = pickle.load(path)
#         all_elevation = obs.storage['elevation']
#         all_elevation = all_elevation[all_elevation > 10]
#     # 绘制原始的散点图
#     ax1.scatter(elevation, np.array(residuals) / 26**0.5, alpha=0.6, s=30, c='tab:blue')

#     # 设置左侧y轴的标签
#     ax1.set_xlabel('Elevation [degree]', fontsize=20)
#     ax1.set_ylabel('Residuals [cycle]', fontsize=20)
#     ax1.set_title('Relation between Phase Noise and Elevation for Station COBB', fontsize=20)
#     ax1.tick_params(labelsize=20)
#     ax1.grid(axis='y', linestyle='--', alpha=0.5)

#     # 创建右侧y轴，用于绘制百分比曲线
#     ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
#     ax2.set_ylabel('Percentage of Observations [%]', fontsize=20)  # 设置右侧y轴标签

#     # 计算每个elevation范围内的观测点数量
#     bins = np.arange(min(all_elevation), max(all_elevation) + bin_size, bin_size)
#     binned_counts, _ = np.histogram(all_elevation, bins=bins)

#     # 获取第一个bin的数量作为100%
#     first_bin_count = binned_counts[0]
    
#     # 计算百分比（后续bin的百分比基于第一个bin的数量）
#     percentages = 100 * binned_counts / first_bin_count

#     # 绘制百分比曲线
#     bin_centers = (bins[:-1] + bins[1:]) / 2  # 每个bin的中心
#     ax2.plot(bin_centers, gaussian_filter1d(percentages, sigma=2), color = 'C1',alpha=0.8, label='Percentage', linewidth=2)

#     # 设置右侧y轴的刻度
#     ax2.tick_params(labelsize=20)

#     # 调整布局，确保不会有重叠
#     plt.tight_layout()

#     # 保存图像
#     plt.savefig('elevation_vs_residuals.png', dpi=600)
#     plt.show()


def split_dataframe_by_time_gap(df, time_col='obs_epoch', elev_col='elevation', elev_thres=10.0, max_gap_seconds=9, min_obs_count=30) -> list[pd.DataFrame]:

    df = df.copy()
    df = df.sort_values(by=time_col).reset_index(drop=True)
    
    time_diff = df[time_col].diff().dt.total_seconds().fillna(0)
    
    segment_id = (time_diff > max_gap_seconds).cumsum()
    df['segment_id'] = segment_id
    
    final_groups = []
    for _, group in df.groupby('segment_id'):
        group_filtered = group[group[elev_col] > elev_thres]
        if len(group_filtered) >= min_obs_count:
            final_groups.append(group_filtered.reset_index(drop=True))
            
    return final_groups

def detect_passes(obs_per_station:pd.DataFrame, min_obs_count, elev_thres, columns_to_keep) -> pd.DataFrame:

    pass_counter = 1
    pass_per_station = []
    elev_list, residuals_list = [], []
    # divide the observations to groups according to max_gap_seconds
    grouped_obs_per_station = split_dataframe_by_time_gap(obs_per_station, elev_thres=elev_thres, max_gap_seconds=9, min_obs_count=min_obs_count)
    
    if not grouped_obs_per_station:
        return pd.DataFrame()
    
    sta_ant_type = obs_per_station['sta_ant_type'].values[0]
    prn = obs_per_station['PRN'].values[0]
    station_freq_shift = obs_per_station['station_freq_shift'].values[0]
    
    if sta_ant_type[0:6] == 'STAREC' and prn == "L39":
        L1_pco_offset = const.jason3_L1_pco + const.starec_L1_pco
        L2_pco_offset = const.jason3_L2_pco + const.starec_L2_pco
    elif sta_ant_type[0:7] == 'ALCATEL' and prn == "L39":
        L1_pco_offset = const.jason3_L1_pco + const.alcatel_L1_pco
        L2_pco_offset = const.jason3_L2_pco + const.alcatel_L2_pco
    else:
        return pd.DataFrame()
    
    d_lambda1 = const.c / (5e6 * 407.25 * (1 + station_freq_shift * const.d_p_multik))
    d_lambda2 = const.c / (5e6 * 80.25 * (1 + station_freq_shift * const.d_p_multik))

    cycle_slip_pco_offset = (5 / d_lambda1 * L1_pco_offset) - (1 / d_lambda2 * L2_pco_offset)
    geom_factor = 5 / d_lambda1 - 1 / d_lambda2
    
    for grouped_obs in grouped_obs_per_station:    

        if grouped_obs["L1"].diff().abs().max(skipna=True) > 5e5 and len(grouped_obs["obs_epoch"].dt.date.unique()) == 1:
        # abnormal phase observation: large jumps between epoches
        # a jump in phase obs is universal in cross-day epoch, only in this case the jump can be tolerated (only one single jump at that one epoch)
            continue

        elev_rad = np.deg2rad(grouped_obs['elevation'].values)
        relative_sec = (grouped_obs['obs_epoch'] - grouped_obs['obs_epoch'].iloc[0]).dt.total_seconds()

        cycle_slip_phase = 5 * grouped_obs["L1"] - grouped_obs["L2"]
        geom_distances = np.sqrt(
            (grouped_obs["sat_x"] - grouped_obs["sta_x"])**2 +
            (grouped_obs["sat_y"] - grouped_obs["sta_y"])**2 +
            (grouped_obs["sat_z"] - grouped_obs["sta_z"])**2
        )
        corrected_geom_distances = (geom_factor * geom_distances 
                                    - cycle_slip_pco_offset * np.sin(elev_rad))                         
        mapping_func = 1 / np.sin(elev_rad)
        estimated_cycle_slip = cycle_slip_phase - corrected_geom_distances      
 
        X = np.column_stack((
            np.ones(len(corrected_geom_distances)),
            relative_sec,
            mapping_func
        ))
        
        beta, _, _, _ = np.linalg.lstsq(X, estimated_cycle_slip, rcond=None)
        residuals = estimated_cycle_slip - (X @ beta)
        diff_residuals = np.diff(residuals)
        z_scores = np.abs(zscore(diff_residuals))
    
        if np.max(np.abs(diff_residuals)) > 10:
            split_points = np.where(z_scores > 3)[0] + 1 
            # We do not detect where diff_residuals > 5 here,
            # because a single abnormal cycle slip can shift the entire residual series away from zero.

        else:
            split_points = np.array([], dtype=int)

        split_points = np.hstack(([0], split_points, [len(estimated_cycle_slip)]))
                
        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i+1]
            if end - start < min_obs_count:
                continue

            X_sub = np.column_stack([
                np.ones(end - start),
                relative_sec[start:end],
                mapping_func[start:end]
            ])
            y_sub = estimated_cycle_slip[start:end]
            beta_sub, _, _, _ = np.linalg.lstsq(X_sub, y_sub, rcond=None)
            residuals_sub = y_sub - X_sub @ beta_sub
            diff_res_sub = np.diff(residuals_sub)
            jump_idx = np.where(np.abs(diff_res_sub) > 3.5)[0] + 1
            jump_idx = np.hstack(([0], jump_idx, [len(y_sub)]))
            

            elev_list.extend(grouped_obs['elevation'].values[start:end-1])
            residuals_list.extend(diff_res_sub)

            for j in range(len(jump_idx) - 1):
                seg_start = start + jump_idx[j]
                seg_end = start + jump_idx[j + 1]
                if seg_end - seg_start >= min_obs_count:
                    seg = grouped_obs.iloc[seg_start:seg_end][columns_to_keep].copy()
                    seg['pass_id'] = pass_counter
                    pass_per_station.append(seg)
                    pass_counter += 1

    if pass_per_station:
        # elev_noise(elev_list, residuals_list,0.1)
        return pd.concat(pass_per_station, ignore_index=True)
    else:
        return pd.DataFrame()     