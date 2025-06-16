import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

folder_path = './DORISInpoOutput/2024'  
range_ratio = 0.925
doris_vtec_diff_list, gim_vtec_diff_list = [], []
alt_vtec_list = []
doris_vtec_rms_list_2024, gim_vtec_rms_list_2024, alt_vtec_mean_list = [], [], []
doris_vtec_std = [] 
# for filename in os.listdir(folder_path):
#     if filename[6:11] == 'Ele10':
        
#         df_doris_result = pd.read_csv(folder_path+'/'+filename)  
#         # 确保 obs_epoch 是 datetime 类型
#         df_doris_result['obs_epoch'] = pd.to_datetime(df_doris_result['obs_epoch'])

#         # 取前12小时数据
#         start_time = pd.Timestamp(2024,5,8,12,0,0) # 向下取整到整点，确保从00:00开始
#         end_time = pd.Timestamp(2024,5,9)
#         df_12h = df_doris_result[(df_doris_result['obs_epoch'] >= start_time)]
        

#         # 设置全局字体大小
#         plt.rcParams['font.family'] = 'Times New Roman'

#         # 创建大尺寸图
#         fig, ax = plt.subplots(figsize=(17, 4))

#         # 绘制曲线
#         ax.scatter(df_12h['obs_epoch'], df_12h['VTEC'], s=8, label='Altimetry', alpha=0.8)
#         ax.scatter(df_12h['obs_epoch'], df_12h['doris_vtec'], s=8, label='DORIS', alpha=0.8)
#         # ax.scatter(df_12h['obs_epoch'], df_12h['gim_vtec'], s=8, label='GIM', alpha=0.8, color='C2')

#         # 设置横坐标刻度：从00:00到12:00，每2小时一格
#         start_time = df_12h['obs_epoch'].min()
#         end_time = df_12h['obs_epoch'].max()
#         xticks = pd.date_range(start=start_time, end=end_time, freq='2H')
#         ax.set_xticks(xticks)
#         ax.set_xticklabels(xticks.strftime('%H:%M'))
#         ax.tick_params(labelsize=16)
#         # 设置横轴标签格式为 "2024 DOY129"
#         year = start_time.year
#         doy = start_time.dayofyear
#         ax.set_xlabel(f'UTC in {year} DOY{doy:03d}', fontsize=20)

#         # 设置纵轴标签和标题
#         ax.set_ylabel('VTEC [TECU]', fontsize=20)
#         ax.set_title('VTEC Comparison between Altimetry and DORIS', fontsize=20)
#         ax.legend(fontsize=16)

#         # 添加仅横向网格线
#         ax.grid(True, axis='y', linestyle='--')

#         # 保存图像
#         plt.savefig("vtec_comparison-nogim.png", dpi=600, bbox_inches='tight')
#         plt.show()
#         doris_vtec_diff = np.abs(df_doris_result['VTEC'] - df_doris_result['doris_vtec'])
#         gim_vtec_diff = np.abs(df_doris_result['VTEC'] - df_doris_result['gim_vtec'])

#         alt_vtec_list.append(df_doris_result['VTEC'])
#         doris_vtec_diff_list.append(doris_vtec_diff)
#         gim_vtec_diff_list.append(gim_vtec_diff)
#         # Calculate min and max values of VTEC

#         gim_vtec_interp = df_doris_result['gim_vtec']
#         gim_vtec_diff = df_doris_result['VTEC'] - gim_vtec_interp * range_ratio
#         gim_vtec_rms = np.sqrt(np.mean(gim_vtec_diff ** 2))

#         non_nan_indices = np.where(~np.isnan(doris_vtec_diff.values))[0]
#         doris_vtec_rms = np.sqrt(np.mean(doris_vtec_diff[non_nan_indices] ** 2))
#         gim_vtec_rms_doris = np.sqrt(np.mean(gim_vtec_diff[non_nan_indices] ** 2))

#         doris_vtec_rms_list_2024.append(doris_vtec_rms)
#         gim_vtec_rms_list_2024.append(gim_vtec_rms_doris)
#         alt_vtec_mean_list.append(np.mean(df_doris_result['VTEC'][non_nan_indices]))
#         doris_vtec_std.append(np.mean(df_doris_result['doris_points_std'][non_nan_indices]))

# # for lower activity
# folder_path = './DORISInpoOutput/2019'
# range_ratio = 0.925
# doris_vtec_diff_list_2019, gim_vtec_diff_list_2019 = [], []
# alt_vtec_list_2019 = []
# doris_vtec_rms_list_2019, gim_vtec_rms_list_2019, alt_vtec_mean_list_2019 = [], [], []
# doris_vtec_std_2019 = []

# for filename in os.listdir(folder_path):
#     if filename[6:11] == 'Ele10':
        
#         df_doris_result = pd.read_csv(os.path.join(folder_path, filename))
#         doris_vtec_diff = np.abs(df_doris_result['VTEC'] - df_doris_result['doris_vtec'])

#         alt_vtec_list_2019.append(df_doris_result['VTEC'])

#         gim_vtec_interp = df_doris_result['gim_vtec']
#         gim_vtec_diff = df_doris_result['VTEC'] - gim_vtec_interp * range_ratio
#         gim_vtec_rms = np.sqrt(np.mean(gim_vtec_diff ** 2))

#         non_nan_indices = np.where(~np.isnan(doris_vtec_diff.values))[0]
#         doris_vtec_rms= np.sqrt(np.mean(doris_vtec_diff[non_nan_indices] ** 2))
#         gim_vtec_rms_doris= np.sqrt(np.mean(gim_vtec_diff[non_nan_indices] ** 2))

#         doris_vtec_rms_list_2019.append(doris_vtec_rms)
#         gim_vtec_rms_list_2019.append(gim_vtec_rms_doris)
#         alt_vtec_mean_list_2019.append(np.mean(df_doris_result['VTEC'][non_nan_indices]))
#         doris_vtec_std_2019.append(np.mean(df_doris_result['doris_points_std'][non_nan_indices]))

doris_vtec_rms_list_2019_enh = [
    3.302461121, 3.078516875, 3.152217353, 3.21085834, 3.271417531, 
    3.256342764, 3.010113094, 3.071374249, 3.25036057, 3.232093231, 
    3.144505697, 3.243435903, 3.145178198, 3.284785334, 3.27581293, 
    3.24998911, 3.041816484, 3.254419969, 3.36473316, 3.203957091, 
    3.094604344, 3.135585393, 3.076762351, 2.969452479, 3.253090873, 
    2.878390114, 3.277033447, 3.32634503, 3.051886792, 3.033835301
]

gim_vtec_rms_list_2019_enh = [
    3.404206302, 3.266128477, 3.356797789, 3.285394473, 3.32410032, 
    3.298971296, 3.237055097, 3.201775275, 3.389922087, 3.485268944, 
    3.279491108, 3.34242836, 3.285266769, 3.499361407, 3.377500304, 
    3.412564983, 3.208879288, 3.379186015, 3.564853089, 3.415442447, 
    3.271837614, 3.225978064, 3.18617117, 3.153821488, 3.445781326, 
    3.056287329, 3.459998669, 3.473756108, 3.273783677, 3.2066573
]

doris_vtec_rms_list_2024_enh = [
    5.213686181, 5.433488582, 5.525955851, 4.853182603, 4.406010704, 
    4.698188671, 5.168779423, 5.108073085, 5.253821398, 5.25176816, 
    5.035412609, 4.960248832, 5.459780226, 5.349863897, 5.283131218, 
    5.239828037, 4.911443317, 5.070381384, 4.720643875, 4.721141333, 
    5.206383194, 5.044244351, 4.902100446, 4.920666834, 4.675016204, 
    5.038899768, 5.569583781, 5.272759936, 5.169275591, 5.277426033
]

gim_vtec_rms_list_2024_enh = [
    5.35057638, 5.624420111, 5.990490711, 5.922020619, 4.52041811, 
    4.765387192, 4.995701643, 4.932866865, 5.078287615, 5.583180782, 
    5.200084234, 5.28505502, 5.45438141, 5.454953603, 5.340158499, 
    5.150925304, 5.053619474, 5.084351044, 4.79830498, 4.955572296, 
    4.994489064, 5.012984377, 4.818476822, 4.991883053, 4.794578991, 
    4.983187372, 5.741101947, 4.927660913, 5.115934536, 5.37425639
]

# 生成30天的日期数据，起始日期是已知的
dates_2024 = pd.date_range(start='2024-05-08', end='2024-06-07', freq='D')  # 每天生成日期
dates_2019 = pd.date_range(start='2019-11-30', end='2019-12-30', freq='D')  # 每天生成日期
plt.rcParams['font.family'] = 'Times New Roman'
# 创建2个子图
fig, ax = plt.subplots(2, 1, figsize=(17, 9))

# 第一子图：绘制 2024 年的数据
ax[0].plot(dates_2024[:-1], doris_vtec_rms_list_2024_enh,
           label=f'DORIS Mean: {np.mean(doris_vtec_rms_list_2024_enh):.2f}', lw=3, color='C0')
ax[0].scatter(dates_2024[:-1], doris_vtec_rms_list_2024_enh,
              color='C0', s=50, edgecolor='white', zorder=5)

# # 2. GIM 2024
# ax[0].plot(dates_2024[:-1], gim_vtec_rms_list_2024,
#            label=f'GIM Mean: {np.mean(gim_vtec_rms_list_2024):.2f}', lw=3, color='C1')
# ax[0].scatter(dates_2024[:-1], gim_vtec_rms_list_2024,
#               color='C1', s=50, edgecolor='white', zorder=5)
ax[0].set_ylabel('RMS [VTEC]', fontsize=20)
ax[0].set_title('2024 DORIS vs GIM RMS', fontsize=20)
ax[0].legend(loc='upper right', fontsize=20)
ax[0].set_xticks(dates_2024[::5])  # 每隔5天一个xticks
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax[0].tick_params(labelsize=20)
ax[0].set_yticks([4,4.5,5,5.5,6,6.5])
ax[0].grid(True, axis='y', linestyle = '--' )
# 第2子图：绘制 2019 年的数据
ax[1].plot(dates_2019[:-1], doris_vtec_rms_list_2019_enh,
           label=f'DORIS Mean: {np.mean(doris_vtec_rms_list_2019_enh):.2f}', lw=3, color='C0')
ax[1].scatter(dates_2019[:-1], doris_vtec_rms_list_2019_enh,
              color='C0', s=50, edgecolor='white', zorder=5)

# 4. GIM 2019
# ax[1].plot(dates_2019[:-1], gim_vtec_rms_list_2019,
#            label=f'GIM Mean: {np.mean(gim_vtec_rms_list_2019):.2f}', lw=3, color='C1')
# ax[1].scatter(dates_2019[:-1], gim_vtec_rms_list_2019,
#               color='C1', s=50, edgecolor='white', zorder=5)
ax[1].set_xlabel('Date [MM-DD]', fontsize=20)
ax[1].set_ylabel('RMS [VTEC]', fontsize=20)
ax[1].set_title('2019 DORIS vs GIM RMS', fontsize=20)
ax[1].legend(loc='upper right', fontsize=20)
ax[1].set_xticks(dates_2019[::5])  # 每隔5天一个xticks
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax[1].tick_params(labelsize=20)
ax[1].set_yticks([2.5,3,3.5,4])
ax[1].grid(True, axis='y', linestyle = '--' )
# 显示图像
plt.tight_layout()
plt.savefig("DORIS_GIM_Comparison_Enhanced.png", dpi=600)
plt.show()


doris_vtec_diff_list = pd.concat(doris_vtec_diff_list, ignore_index=True)
gim_vtec_diff_list = pd.concat(gim_vtec_diff_list, ignore_index=True)
alt_vtec_list = pd.concat(alt_vtec_list, ignore_index=True)

min_vtec = 0
max_vtec = int(alt_vtec_list.max())

# Define bins with a step of 10
bins = np.arange(min_vtec, 161, 20)
# Group by bin and calculate mean of doris_vtec_diff
bin_ds = pd.cut(alt_vtec_list, bins=bins)
mean_diff_per_bin = doris_vtec_diff_list.groupby(bin_ds).mean()
mean_diff_per_bin_gim = gim_vtec_diff_list.groupby(bin_ds).mean()
# Plotting
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(17, 9))
plt.suptitle('Variation of VTEC RMS w.r.t Altimetry VTEC Magnitude (30 Days Average)', fontsize=20)
plt.subplot(121)
plt.bar(mean_diff_per_bin.index.astype(str), mean_diff_per_bin.values, width=0.8, edgecolor='black', label='DORIS')
plt.xlabel('VTEC Intervals [TECU]', fontsize=20)
plt.ylabel('VTEC RMS [TECU]', fontsize=20)
plt.grid(True, axis='y', linestyle='--')
plt.ylim([0,30])
plt.legend(loc='upper right', fontsize=20)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.subplot(122)
plt.bar(mean_diff_per_bin_gim.index.astype(str), mean_diff_per_bin_gim.values, width=0.8, edgecolor='black', label='GIM',color='C1')
plt.xlabel('VTEC Intervals [TECU]', fontsize=20)
plt.grid(True, axis='y', linestyle='--')
plt.tick_params(labelsize=15)
plt.ylim([0,30])
plt.legend(loc='upper right', fontsize=20)
plt.tight_layout()
plt.savefig('zoomed_alt_doris.png', dpi=600, bbox_inches='tight')
plt.show()