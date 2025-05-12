import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def para_comparison_series(ipp_files_1, vtec_files_1, ipp_files_2, vtec_files_2):
    """
    输入：
        ipp_files_1, vtec_files_1：第一组参数的 IPP 和 VTEC 文件列表
        ipp_files_2, vtec_files_2：第二组参数的 IPP 和 VTEC 文件列表

    输出：
        rms_1, rms_2：两个参数下的总RMS
        diff_1_all, diff_2_all：两个参数下合并后的 VTEC 差值序列（doris - alt）
    """
    all_diff_1 = []
    all_diff_2 = []

    assert len(ipp_files_1) == len(vtec_files_1) == len(ipp_files_2) == len(vtec_files_2), "输入文件数量不一致"

    for i in range(len(ipp_files_1)):
        ipp_1 = np.load(ipp_files_1[i])
        vtec_1 = np.load(vtec_files_1[i])
        doris_vtec_1 = vtec_1[:, 0]
        alt_vtec_1 = vtec_1[:, 1]

        ipp_2 = np.load(ipp_files_2[i])
        vtec_2 = np.load(vtec_files_2[i])
        doris_vtec_2 = vtec_2[:, 0]
        alt_vtec_2 = vtec_2[:, 1]

        matching_indices_1 = []
        matching_indices_2 = []

        for j in range(ipp_1.shape[0]):
            if (np.abs(ipp_1[j][0]) < 10):
                matches = np.all(ipp_1[j] == ipp_2, axis=1)
                if np.any(matches):
                    matching_indices_1.append(j)
                    matching_indices_2.append(np.where(matches)[0][0])

        if matching_indices_1:
            diff_1 = doris_vtec_1[matching_indices_1] - alt_vtec_1[matching_indices_1]
            diff_2 = doris_vtec_2[matching_indices_2] - alt_vtec_2[matching_indices_2]

            all_diff_1.append(diff_1)
            all_diff_2.append(diff_2)

    # 合并所有天的差值序列
    diff_1_all = np.concatenate(all_diff_1)
    diff_2_all = np.concatenate(all_diff_2)

    # 计算RMS
    rms_1 = np.sqrt(np.mean(diff_1_all**2))
    rms_2 = np.sqrt(np.mean(diff_2_all**2))

    # return rms_1, rms_2, diff_1_all, diff_2_all
    print(rms_1, rms_2)

    # plt.figure(figsize=(10, 6))
    # plt.subplot(2, 1, 1) 
    # plt.plot(doris_vtec_1[matching_indices_1], label=f'lat{lat_part_1}-obs{obs_part_1}')
    # plt.plot(doris_vtec_2[matching_indices_2], label=f'lat{lat_part_2}-obs{obs_part_2}')
    # plt.plot(alt_vtec_1[matching_indices_1], label='altimetry reference')
    # plt.title('Original Data')
    # plt.legend()

    # diff_1 = doris_vtec_1[matching_indices_1] - alt_vtec_1[matching_indices_1]
    # diff_2 = doris_vtec_2[matching_indices_2] - alt_vtec_1[matching_indices_1]
    # plt.subplot(2, 1, 2) 
    # plt.plot(diff_1, label=f'lat{lat_part_1}-obs{obs_part_1}')
    # plt.plot(diff_2, label=f'lat{lat_part_2}-obs{obs_part_2}')
    # plt.title('Differences')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
if __name__ == '__main__':
    ipp_1_file = [f'InpoResults/DORIS/IPP/y2024_d{day:03d}_lat-2_ele-10_obs-30_1.npy' for day in range(129, 134)]
    ipp_2_file = [f'InpoResults/DORIS/IPP/y2024_d{day:03d}_lat-8_ele-10_obs-30_1.npy' for day in range(129, 134)]
    vtec_1_file = [f'InpoResults/DORIS/VTEC/y2024_d{day:03d}_lat-2_ele-10_obs-30_1.npy' for day in range(129, 134)]
    vtec_2_file = [f'InpoResults/DORIS/VTEC/y2024_d{day:03d}_lat-8_ele-10_obs-30_1.npy' for day in range(129, 134)]
    para_comparison_series(ipp_1_file, vtec_1_file, ipp_2_file, vtec_2_file)