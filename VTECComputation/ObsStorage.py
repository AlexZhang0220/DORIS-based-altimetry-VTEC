from typing import List, Dict
from ObjectClasses import DORISObs, Thresholds, PassObj, DORISBeacon
from OrbitStorage import OrbitStorage
from StationStorage import StationStorage
from tools import batch_lagrange_interp_1d
import constant as const
import numpy as np
import mmap
import pandas as pd
import xarray as xr
import warnings

class DORISStorage:
    def __init__(self) -> None:
        self.storage = pd.DataFrame()
        self.stations: List[DORISBeacon] = []


    def read_rinex_300(self, file: str, orbit_data: OrbitStorage, station_data: StationStorage):
        header = self._parse_header(file)
        obs_type = self._extract_obs_type_count(header)
        self.stations = self._extract_stations_from_header(header)
        self._extract_time_ref_station_from_header(header)
        PRN = self._extract_satellite_PRN(header)
        
        with open(file, 'r') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                # Skip header
                mmapped_file.seek(mmapped_file.find(b"END OF HEADER") + len("END OF HEADER\n"))
                lines = mmapped_file.read().decode('utf-8').splitlines()

        df_obs = _parse_lines_chunk_df(lines, obs_type, PRN, self.stations) 

        df_obs = interpolate_satellite_positions_lagrange(df_obs, orbit_data.sat_dataset, PRN)
        df_obs = merge_station_position(df_obs, station_data.storage)

        

        self.storage = df_obs

    def _parse_header(self, file: str) -> List[str]:
        header = []
        try:
            with open(file, 'r') as f:
                for line in f:
                    header.append(line)
                    if "END OF HEADER" in line:
                        break
        except IOError:
            print(f"Unable to read the file: {file}")
        return header
    
    def _extract_stations_from_header(self, header: List[str]) -> List[DORISBeacon]:
        stations = []
        for line in header:
            if line[60:80].startswith("STATION REFERENCE"):
                station = DORISBeacon(
                    int(line[1:3]), line[5:9], line[10:40].strip(),
                    line[40:50].strip(), int(line[50:52]), int(line[52:56])
                )
                stations.append(station)
        return stations

    def _extract_time_ref_station_from_header(self, header: List[str]):
        for line in header:
            if line[60:80].startswith("TIME REF STATION"):
                station_number = int(line[1:3])
                self.stations[station_number-1].time_ref_bit = True
                self.stations[station_number-1].time_bias = float(line[5:19])
                self.stations[station_number-1].time_shift = float(line[21:35])

    def _extract_obs_type_count(self, header: List[str]) -> int:
        for line in header:
            if line[60:80].startswith("SYS / # / OBS TYPES"):
                obs_type_count = int(line[3:6])
                obs_type = [line[i+6:i+10].strip() for i in range(0, 4*obs_type_count, 4)]
                return  obs_type
        return 0
    
    def _extract_satellite_PRN(self, header: List[str]) -> str:
        for line in header:
            if line[60:80].startswith("SATELLITE NAME"):
                if line[0:60].strip() == "JASON-3": 
                    return 'L39'
        return 0

def _parse_lines_chunk_df(lines: list[str], obs_type: list[str], PRN: str, stations: list[DORISBeacon]) -> pd.DataFrame:
    records = []
    current_epoch = None
    receiver_clock_offset = None

    for line in lines:
        if line[0:4].isspace():
            continue
        elif line.startswith(">"):
            current_epoch = pd.to_datetime(line[2:31], format="%Y %m %d %H %M %S.%f")
            receiver_clock_offset = float(line[44:56])
        elif line.startswith("D"):
            ID = int(line[1:3]) #station_id
            if line[50] == '7':  # elevation < 5° -> skip
                continue
            obs_fields = [float(line[j:j + 14].strip()) for j in range(3, 83, 16)]
            record = {
                "obs_epoch": current_epoch,
                "receiver_clock_offset": receiver_clock_offset,
                "station_id": ID,
                "station_code": stations[ID - 1].station_code,
                "station_freq_shift": stations[ID - 1].freq_shift,
                "PRN": PRN,
                "L1": obs_fields[obs_type.index("L1")] if "L1" in obs_type else np.nan,
                "L2": obs_fields[obs_type.index("L2")] if "L2" in obs_type else np.nan,
                "C1": obs_fields[obs_type.index("C1")] * 10 if "C1" in obs_type else np.nan,
                "C2": obs_fields[obs_type.index("C2")] * 10 if "C2" in obs_type else np.nan,
            }
            records.append(record)

    return pd.DataFrame.from_records(records)

def interpolate_satellite_positions_lagrange(
    df_obs: pd.DataFrame,
    sat_dataset: xr.Dataset,
    prn: str,
    num_neighbors: int = 8
) -> pd.DataFrame:

    sat_data = sat_dataset.sel(sv=prn)
    eph_times = pd.to_datetime(sat_data["time"].values)
    eph_seconds = eph_times.astype(np.int64) // 10**9
    eph_xyz = sat_data["position"].values * 1000  # km -> m

    obs_time = pd.to_datetime(df_obs["obs_epoch"].values) + pd.to_timedelta(df_obs["receiver_clock_offset"], unit="s")
    obs_seconds = obs_time.astype(np.int64) / 10**9

    unique_seconds = np.unique(obs_seconds)

    sat_x_list, sat_y_list, sat_z_list = [], [], []

    for t in unique_seconds:
        idx = np.searchsorted(eph_seconds, t)
        half = num_neighbors // 2
        start = max(0, idx - half)
        end = min(len(eph_seconds), start + num_neighbors)
        start = max(0, end - num_neighbors)

        xs = np.array(eph_seconds[start:end])
        ys_x = eph_xyz[start:end, 0]
        ys_y = eph_xyz[start:end, 1]
        ys_z = eph_xyz[start:end, 2]

        if len(xs) < 2:
            sat_x_list.append(np.nan)
            sat_y_list.append(np.nan)
            sat_z_list.append(np.nan)
        else:
            sat_x_list.append(batch_lagrange_interp_1d(xs, ys_x, np.array([t]))[0])
            sat_y_list.append(batch_lagrange_interp_1d(xs, ys_y, np.array([t]))[0])
            sat_z_list.append(batch_lagrange_interp_1d(xs, ys_z, np.array([t]))[0])

    interp_df_unique = pd.DataFrame({
        "obs_seconds": unique_seconds,
        "sat_x": sat_x_list,
        "sat_y": sat_y_list,
        "sat_z": sat_z_list,
    })

    df_obs = df_obs.copy()
    df_obs["obs_seconds"] = obs_seconds
    df_obs = df_obs.merge(interp_df_unique, on="obs_seconds", how="left")
    df_obs = df_obs.drop(columns=["obs_seconds"])

    return df_obs

def merge_station_position(df_obs, stations: StationStorage):

    station_records = []

    for i, sta in enumerate(stations):
        for j, epoch in enumerate(sta.soln_epochlist):
            station_records.append({
                "station_code": sta.station_code,
                "soln_epoch": pd.to_datetime(epoch),
                "ant_type": sta.antenna_types[j],
                "sta_x": sta.soln_coor[j][0],
                "sta_y": sta.soln_coor[j][1],
                "sta_z": sta.soln_coor[j][2]
            })

    df_station_records = pd.DataFrame(station_records)

    merged_groups = []

    for station, obs_group in df_obs.groupby('station_code'):
        record_group = df_station_records[df_station_records['station_code'] == station]

        if record_group.empty:
            obs_group[['sta_x', 'sta_y', 'sta_z']] = None
            merged_groups.append(obs_group)
            continue

        obs_group_sorted = obs_group.sort_values('obs_epoch')
        record_group_sorted = record_group.sort_values('soln_epoch')

        merged = pd.merge_asof(
            obs_group_sorted,
            record_group_sorted,
            by='station_code',
            left_on='obs_epoch',
            right_on='soln_epoch',
            direction='backward'
        )

        merged_groups.append(merged)

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df_merged_final = pd.concat(merged_groups, ignore_index=True)

    return df_merged_final

def compute_geom_corrected_dion(df_obs: pd.DataFrame) -> pd.DataFrame:

    c = const.c
    shift = df_obs["station_freq_shift"].values

    # 是否为有频率偏移的观测点
    shift_nonzero = shift != 0

    # 频率偏移时的波长计算
    d_lambda1 = np.where(shift_nonzero, c / (5e6 * 407.25 * (1 + shift * const.d_p_multik)), const.d_lambda1)
    d_lambda2 = np.where(shift_nonzero, c / (5e6 * 80.25  * (1 + shift * const.d_p_multik)), const.d_lambda2)

    d_k = np.where(shift_nonzero, (d_lambda2**2) / ((d_lambda2**2) - (d_lambda1**2)), const.d_k)
    d_w = np.where(shift_nonzero, (d_lambda1**2) / 40.3, const.d_w)

    L1 = df_obs["L1"].values
    L2 = df_obs["L2"].values
    elevation = df_obs["elevation"].values

    prn_map = {
        "L27": const.d_geomcorrJa2,
        "L39": const.d_geomcorrJa3,
        "L12": const.d_geomcorrCr2,
        "L45": const.d_geomcorrHy2,
        "L46": const.d_geomcorrSar,
    }
    sat_off = df_obs["PRN"].map(prn_map).fillna(0).values

    ant_off = np.where(
        df_obs["ant_type"].str.startswith("STAREC"), const.d_STAREC,
        np.where(df_obs["ant_type"].str.startswith("ALCATEL"), const.d_ALCATEL, 0)
    )

    d_geomcorr = sat_off + ant_off

    dion = -d_k * (L1 * d_lambda1 - L2 * d_lambda2)
    correction = -d_k * (-d_geomcorr * np.sin(np.deg2rad(elevation)))
    dion += correction

    stec = dion * d_w / 1e16

    df_obs["dion"] = dion
    df_obs["STEC"] = stec

    return df_obs
