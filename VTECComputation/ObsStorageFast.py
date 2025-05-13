from typing import List, Dict
from ObjectClasses import DORISObs, Thresholds, PassObj, DORISBeacon
from OrbitStorage import OrbitStorage
from StationStorage import StationStorage
from tools import get_elevation, get_map_value, Raypoint
from pandas import Timestamp, Timedelta
from multiprocessing import Pool, cpu_count
import numpy as np
import mmap


class DORISStorage:
    def __init__(self) -> None:
        self.storage: List[List[DORISObs]] = [] # [station][epoch] = DORISObs
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

    def _parse_lines_chunk(self, chunk: list[str], obs_type: list[str], PRN: str, stations: list[DORISBeacon],
                           orbit_data: OrbitStorage, station_data: StationStorage, settings: Thresholds) -> list[DORISObs]:
        """
        Processes a chunk of lines, extracting observation values and performing calculations.
        """
        results = [[] for _ in range(len(stations))]
        for i, line in enumerate(chunk):
            if line[0] == ">":  # Observation epoch header
                y, m, d, h, mi, s = int(line[2:6]), int(line[7:9]), int(line[10:12]), int(line[13:15]), int(line[16:18]), int(line[18:21])
                ms = int(line[22:28])
                ns = int(line[28:31])  # Nanoseconds
                obs_epoch = Timestamp(year=y, month=m, day=d, hour=h, minute=mi, second=s, microsecond=ms, nanosecond=ns) ## check??
                receiver_clock_offset = float(line[44:56])
                obs_epoch_offset = obs_epoch + Timedelta(receiver_clock_offset, unit='s') ## ?? check

            elif line.strip() and line[0] == 'D':  # Observation data
                ID = int(line[1:3])
                station = stations[ID - 1]
                obs_values = np.array([float(line[j:j + 14].strip()) for j in range(3, 83, 16)])
                if line[50] == '7': # < 5 elevation, as the seventh channel of DGXX receiver is used
                    continue
                obs = DORISObs(station.station_code, ID, station.shift, receiver_clock_offset, obs_values, obs_epoch, obs_type, PRN)
                obs.pos_sat_cele = orbit_data.get_pos_cele_lagr_inter_fast(obs_epoch_offset, num_inter_point=7)
                obs.pos_sta_cele = station_data.get_pos_cele(obs.station_code, obs_epoch)
                obs.ant_type = station_data.get_ant_type(obs.station_code, obs_epoch)
                if len(obs.pos_sta_cele) == 0 or len(obs.pos_sat_cele) == 0:
                    continue
                obs.elevation = get_elevation(obs.pos_sat_cele, obs.pos_sta_cele)
                if obs.elevation < settings.ele_cut_off:
                    continue
                Ion_height = 506700
                obs.ipp_lon, obs.ipp_lat = Raypoint(obs.pos_sat_cele, obs.pos_sta_cele, Ion_height)
                obs.map_value = get_map_value(obs.elevation)
                obs.geom_corr()
                results[ID-1].append(obs)
        return results

    def read_rinex_300(self, file: str, orbit_data: OrbitStorage, station_data: StationStorage,
                       settings: Thresholds):
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
        # Use multiprocessing to process chunks
        # chunk_obs = [self._parse_lines_chunk(chunk, obs_type, PRN, self.stations, orbit_data, station_data, settings) for chunk in chunks
        #     ]
        with Pool(cpu_count()) as pool:
            chunk_obs = pool.starmap(self._parse_lines_chunk, [
                (chunk, obs_type, PRN, self.stations, orbit_data, station_data, settings) for chunk in chunks
            ])

        # Combine results
        self.storage = [
            [value for chunk_idx in range(len(chunk_obs)) for value in chunk_obs[chunk_idx][station_idx]]
            for station_idx in range(len(chunk_obs[0]))
        ]
        ## The code above equals the code below
        # results_reduced = []
        # for station_idx in range(len(chunk_obs[0])):  # each station
        #     station_data = []
        #     for chunk_idx in range(len(chunk_obs)):   # each chunk
        #         station_data += chunk_obs[chunk_idx][station_idx]
        #     results_reduced.append(station_data)
        # self.storage = results_reduced


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

def _split_lines_by_marker(lines: List[str], chunk_size: int = 1000, marker: str = ">") -> List[List[str]]:
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

