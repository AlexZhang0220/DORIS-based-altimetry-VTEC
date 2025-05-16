from typing import List, Dict
from ObjectClasses import DORISObs, Thresholds, PassObj, DORISBeacon
from OrbitStorage import OrbitStorage
from StationStorage import StationStorage
from tools import get_elevation, get_map_value, Raypoint
from pandas import Timestamp, Timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import mmap
import pandas as pd
import xarray as xr

class DORISStorage:
    def __init__(self) -> None:
        self.storage = pd.DataFrame() # [station][epoch] = DORISObs
        self.stations: List[DORISBeacon] = []

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
                chunks = _split_lines_by_marker(lines, 1000)

        df_obs = parse_lines_in_parallel(chunks, obs_type, PRN, self.stations) # all observation in the form of data frame

        df_obs = interpolate_satellite_positions_xarray(df_obs, orbit_data.sat_dataset, PRN) # include satellite position

        df_obs = merge_station_position(df_obs, station_data.storage)

        self.storage = df_obs

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

    def parse_doris_obs_two_lines(self, line1: str, line2: str, receiver_clock_offset: float, obs_epoch, obs_type, PRN):

        ID = int(line1[1:3])
        station = self.stations[ID - 1]
        line1_fields = [line1[i:i+14].strip() for i in range(3, len(line1), 16)]
        line2_fields = [line2[i:i+14].strip() for i in range(3, len(line2), 16)]
        all_fields = line1_fields + line2_fields

        obs_values = []
        for field in all_fields:
            if field:
                obs_values.append(float(field))

        if line1[50] == '7':
            return None
        
        return DORISObs(
            station.station_code,
            ID,
            station.shift,
            receiver_clock_offset,
            obs_values,
            obs_epoch,
            obs_type,
            PRN
        )

def _split_lines_by_marker(lines: list[str], chunk_size: int = 1000, marker: str = ">") -> list[list[str]]:
        chunks = []
        start = 0

        while start < len(lines):
            end = min(start + chunk_size, len(lines))
            # Adjust to include all lines up to the next marker
            while end < len(lines) and not lines[end].startswith(marker):
                end += 1

            # Add the chunk and move the start pointer
            chunks.append(lines[start:end])
            start = end

        return chunks

def _parse_lines_chunk_df(chunk: list[str], obs_type: list[str], PRN: str, stations: list[DORISBeacon]) -> pd.DataFrame:
    records = []
    current_epoch = None
    receiver_clock_offset = None

    for line in chunk:
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
                "station_shift": stations[ID - 1].freq_shift,
                "PRN": PRN,
                "L1": obs_fields[obs_type.index("L1")] if "L1" in obs_type else np.nan,
                "L2": obs_fields[obs_type.index("L2")] if "L2" in obs_type else np.nan,
                "C1": obs_fields[obs_type.index("C1")] * 10 if "C1" in obs_type else np.nan,
                "C2": obs_fields[obs_type.index("C2")] * 10 if "C2" in obs_type else np.nan,
            }
            records.append(record)

    return pd.DataFrame.from_records(records)

def parse_worker(lines_chunk, obs_type: list[str], PRN: str, stations: list[DORISBeacon]):
    return _parse_lines_chunk_df(lines_chunk, obs_type, PRN, stations)

def parse_lines_in_parallel(all_lines_chunks, obs_type: list[str], PRN: str, stations: list[DORISBeacon], max_workers=1):

    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(parse_worker, chunk, obs_type, PRN, stations) for chunk in all_lines_chunks]

        for future in as_completed(futures):
            try:
                dfs.append(future.result())
            except Exception as e:
                print("Rinexreading Error:", e)

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all

def interpolate_satellite_positions_xarray(df_obs: pd.DataFrame, sat_dataset: xr.Dataset, prn: str) -> pd.DataFrame:

    sat_data = sat_dataset.sel(sv=prn)
    obs_time = pd.to_datetime(df_obs["obs_epoch"].values)
    unique_times = np.unique(obs_time)

    interp_result = sat_data.interp(time=unique_times)
    interp_df = pd.DataFrame({
        "obs_epoch": unique_times,
        "sat_x": interp_result["position"].sel(ECEF="x").values * 1000,
        "sat_y": interp_result["position"].sel(ECEF="y").values * 1000,
        "sat_z": interp_result["position"].sel(ECEF="z").values * 1000,
    })
    df_obs = df_obs.merge(interp_df, on="obs_epoch", how="left")

    return df_obs

def merge_station_position(df_obs, stations: StationStorage):

    def get_station_data(stations):
        station_records = []
        for i, sta in enumerate(stations):
            for j, epoch in enumerate(sta.soln_epochlist):
                if j < len(sta.soln_coor):
                    station_records.append({
                        "station_code": sta,
                        "soln_epoch": pd.Timestamp(epoch),
                        "sta_x": sta.soln_coor[j][0],
                        "sta_y": sta.soln_coor[j][1],
                        "sta_z": sta.soln_coor[j][2]
                    })
        return pd.DataFrame(station_records)

    df_station_records = get_station_data(stations)

    df_station_records = df_station_records.sort_values(by=["station_code", "soln_epoch"])
    df_obs = df_obs.sort_values(by=["station_code", "obs_epoch"])

    df_obs_with_sta = pd.merge_asof(
        df_obs,
        df_station_records,
        by="station_code",
        left_on="obs_epoch",
        right_on="soln_epoch",
        direction="backward"
    )

    # 过滤掉没有匹配的测站坐标（obs_epoch 小于最早 soln_epoch）
    df_obs = df_obs[df_obs["soln_epoch"].notna()].copy()

    return df_obs
