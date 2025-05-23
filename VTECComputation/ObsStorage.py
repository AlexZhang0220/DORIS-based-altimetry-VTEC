from ObjectClasses import DORISBeacon
from OrbitStorage import OrbitStorage
from StationStorage import StationStorage
from tools import batch_lagrange_interp_1d
from pyproj import Transformer
import constant as const
import numpy as np
import mmap
import pandas as pd
import xarray as xr
import warnings

from dataclasses import dataclass
@dataclass
class RINEXHeaderInfo:
    obs_type: list[str]
    stations: list[DORISBeacon]
    PRN: str

class DORISStorage:
    def __init__(self) -> None:
        self.storage = pd.DataFrame()
        self.stations: list[DORISBeacon] = []

    def read_rinex_300(self, file: str, orbit_data: xr.Dataset, station_data: StationStorage):
        
        # header
        header_info = self._parse_and_extract_header_info(file)
        self.stations = header_info.stations
        
        # main obs
        with open(file, 'r') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                # Skip header, read into memory
                mmapped_file.seek(mmapped_file.find(b"END OF HEADER") + len("END OF HEADER\n"))
                lines = mmapped_file.read().decode('utf-8').splitlines()

        df_obs = (
            _parse_lines_chunk_df(lines, header_info.obs_type, header_info.PRN, self.stations)
            .pipe(interpolate_satellite_positions_lagrange, orbit_data, header_info.PRN)
            .pipe(merge_station_position, station_data.storage)
            .pipe(get_elevation_and_map_value) 
            .pipe(get_ipp_position)
            .pipe(compute_geom_corrected_dion)        
        )

        self.storage = df_obs

    def _parse_and_extract_header_info(self, file: str) -> RINEXHeaderInfo:
        obs_type = []
        stations = []
        PRN = None
        SATELLITE_NAME_TO_PRN = {
            "JASON-3":      "L39",
            "CRYOSAT-2":    "L12",
            "SARAL":        "L46",
            "SENTINEL-3A":  "L74",
            "SENTINEL-3B":  "L98",
        }
        with open(file, 'r') as f:
            for line in f:
                label = line[60:80].strip()

                if label == "SYS / # / OBS TYPES":
                    obs_type_count = int(line[3:6])
                    obs_type = [line[i+6:i+10].strip() for i in range(0, 4 * obs_type_count, 4)]

                elif label == "STATION REFERENCE":
                    station = DORISBeacon(
                        int(line[1:3]), line[5:9], line[10:40].strip(),
                        line[40:50].strip(), int(line[50:52]), int(line[52:56])
                    )
                    stations.append(station)

                elif label == "TIME REF STATION":
                    station_number = int(line[1:3])
                    stations[station_number - 1].time_ref_bit = True
                    stations[station_number - 1].time_bias = float(line[5:19])
                    stations[station_number - 1].time_shift = float(line[21:35])

                elif label == "SATELLITE NAME":
                    sat_name = line[0:60].strip()
                    if sat_name in SATELLITE_NAME_TO_PRN:
                        PRN = SATELLITE_NAME_TO_PRN[sat_name]

                elif label == "END OF HEADER":
                    break

        return RINEXHeaderInfo(obs_type, stations, PRN)
    
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
            ID = int(line[1:3]) # station id in rinex header
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

    df_obs["obs_seconds"] = obs_seconds
    df_obs = df_obs.merge(interp_df_unique, on="obs_seconds", how="left")
    df_obs = df_obs.drop(columns=["obs_seconds"])

    return df_obs

def merge_station_position(df_obs: pd.DataFrame, stations: StationStorage) -> pd.DataFrame:

    station_records = []

    for i, sta in enumerate(stations):
        for j, epoch in enumerate(sta.soln_epochlist):
            station_records.append({
                "station_code": sta.station_code,
                "soln_epoch": pd.to_datetime(epoch),
                "sta_ant_type": sta.antenna_types[j],
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
        df_merged_final = df_merged_final.drop(columns=["soln_epoch"])
    return df_merged_final

def get_elevation_and_map_value(df_obs: pd.DataFrame) -> pd.DataFrame:
    # elevation
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

    lon, lat, _ = transformer.transform(df_obs['sta_x'].values,
                                        df_obs['sta_y'].values,
                                        df_obs['sta_z'].values,
                                        radians=True)

    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)

    R = np.array([
        [-sin_lon,            cos_lon,            np.zeros_like(lon)],
        [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon,  sin_lat]
    ])  # shape: (3, 3, N)
    R = R.transpose(2, 0, 1)  # shape: (N, 3, 3)

    vec = df_obs[['sat_x', 'sat_y', 'sat_z']].values - df_obs[['sta_x', 'sta_y', 'sta_z']].values  # shape: (N, 3)

    enu = np.einsum('nij,nj->ni', R, vec)  # shape: (N, 3)

    horizontal_norm = np.linalg.norm(enu[:, 0:2], axis=1)
    total_norm = np.linalg.norm(enu, axis=1)
    elevation_rad = np.arccos(horizontal_norm / total_norm)

    df_obs['elevation'] = np.rad2deg(elevation_rad)

    # map value
    AE84 = const.AE84
    layer_height = const.ion_height
    alpha = 0.9782

    z_angle = np.pi/2 - elevation_rad
    sin_term = np.sin(alpha * z_angle)

    ratio = AE84 / (AE84 + layer_height)
    map_value = 1 / np.cos(np.arcsin(ratio * sin_term))

    df_obs['map_value'] = map_value

    return df_obs

def get_ipp_position(df_obs: pd.DataFrame, height: float=506700) -> pd.DataFrame:
    AE84 = const.AE84

    sat = df_obs[['sat_x', 'sat_y', 'sat_z']].values
    sta = df_obs[['sta_x', 'sta_y', 'sta_z']].values

    alpha = np.deg2rad(90 + df_obs['elevation'].values)

    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    sta_lonlat = np.array(transformer.transform(sta[:, 0], sta[:, 1], sta[:, 2])).T
    sta_h = sta_lonlat[:, 2]

    vec_rs = sat - sta
    Drs = np.linalg.norm(vec_rs, axis=1)

    term1 = (AE84 + sta_h) * np.cos(alpha)
    under_sqrt = np.square(term1) + (AE84 + height)**2 - (AE84 + sta_h)**2
    under_sqrt = np.maximum(under_sqrt, 0)  
    term2 = np.sqrt(under_sqrt)

    D_rP1 = term1 + term2
    D_rP2 = term1 - term2

    Scale1 = D_rP1 / Drs
    Scale2 = D_rP2 / Drs

    vec1 = sta + (vec_rs.T * Scale1).T
    vec2 = sta + (vec_rs.T * Scale2).T

    dist1 = np.linalg.norm(sat - vec1, axis=1)
    dist2 = np.linalg.norm(sat - vec2, axis=1)

    use_vec1 = dist1 <= dist2
    IPP = np.where(use_vec1[:, np.newaxis], vec1, vec2)

    ipp_lon, ipp_lat, _ = transformer.transform(IPP[:, 0], IPP[:, 1], IPP[:, 2])

    df_obs['ipp_lon'] = ipp_lon
    df_obs['ipp_lat'] = ipp_lat

    return df_obs

def compute_geom_corrected_dion(df_obs: pd.DataFrame) -> pd.DataFrame:

    c = const.c
    shift = df_obs["station_freq_shift"].values
    shift_nonzero = shift != 0

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
        df_obs["sta_ant_type"].str.startswith("STAREC"), const.d_STAREC,
        np.where(df_obs["sta_ant_type"].str.startswith("ALCATEL"), const.d_ALCATEL, 0)
    )

    d_geomcorr = sat_off + ant_off

    dion = -d_k * (L1 * d_lambda1 - L2 * d_lambda2)
    correction = -d_k * (-d_geomcorr * np.sin(np.deg2rad(elevation)))
    dion -= correction

    stec = dion * d_w / 1e16

    df_obs["dion"] = dion
    df_obs["STEC"] = stec

    return df_obs