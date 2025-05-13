from typing import List, Dict, Tuple
from ObjectClasses import Station
from pandas import Timestamp, Timedelta
import bisect
import numpy as np

class StationStorage:
    def __init__(self) -> None:
        self.storage: List[Station] = []
    @staticmethod
    def parse_custom_time(time_str: str) -> Timestamp:
        """
        Convert YY:DOY:SECOD format to a datetime object.
        """
        yy, doy, secod = map(int, time_str.split(":"))
        year = 1900 + yy if yy >= 90 else 2000 + yy  # Assumes year range 1990-2099
        return Timestamp(year, 1, 1) + Timedelta(days=doy - 1, seconds=secod)    
    
    @staticmethod
    def find_interval_index(dt_list: list, target_dt: Timestamp) -> int:

        idx = bisect.bisect_right(dt_list, target_dt)
        
        # If target_dt is less than all elements in dt_list, return -1.
        if idx == 0:
            return -1
        # If target_dt is greater than or equal to the last element, return the last index.
        if idx >= len(dt_list):
            return len(dt_list) - 1
        # Otherwise, target_dt lies between dt_list[idx-1] and dt_list[idx], so return idx - 1.
        return idx - 1

    def get_ant_type(self, station_code: str, epoch: Timestamp) -> str:
        for station in self.storage:
            if station_code == station.site_id:
                idx = self.find_interval_index(station.soln_epochlist, epoch)
                if idx != -1:
                    return station.antenna_types[idx]
                return ''
        return ''
    
    def get_pos_cele(self, station_code: str, epoch: Timestamp):
        for station in self.storage:
            if station_code == station.site_id:
                idx = self.find_interval_index(station.soln_epochlist, epoch)
                if idx != -1:
                    return np.array(station.soln_coor[idx])
                return ''
        return ''
    
    def get_or_create_station(self, site_id: str) -> Station:
        """
        Retrieve an existing station object or create a new one if not found.
        """
        for station in self.storage:
            if station.site_id == site_id:
                return station

        # Create a new station if not found
        new_station = Station(site_id)
        self.storage.append(new_station)
        return new_station
    
    def read_solution_epochs(self, start_dt: Timestamp, end_dt: Timestamp, records: str):
        for record in records:
            site = record[1:5].strip() 
            soln_ind = int(record[9:13].strip()) 
            start_obs = self.parse_custom_time(record[15:28].strip()) 
            end_obs = self.parse_custom_time(record[28:41].strip()) 

            if not (end_obs < start_dt or start_obs > end_dt):  # Check time overlap
                station = self.get_or_create_station(site)
                station.soln_epochlist.append(start_obs)
                station.soln_coor.append([0.0, 0.0, 0.0])  # Placeholder for XYZ coordinates
                station.antenna_types.append("")  # Placeholder for antenna type
                station.soln_idx.append(soln_ind)  # Store solution index

    def read_site_antenna(self, records: list[str]):
        for record in records:
            record_site = record[1:5].strip()  
            record_soln = int(record[9:13].strip()) 
            antenna_type = record[42:62].strip() 

            for station in self.storage:
                if station.site_id == record_site:
                    for i, soln in enumerate(station.soln_idx):
                        if soln == record_soln:  # Match solution index
                            station.antenna_types[i] = antenna_type
        
    def read_solution_estimate(self, start_time: Timestamp, records: list[str]):
        """
        Parses station coordinates from records and directly stores them in the corresponding Station instance.
        Only updates stations whose solution index is in soln_idx.

        :param records: List of strings, each representing a coordinate record.
        """
        i = 0
        while i < len(records):
            # Extract data for a single station
            stax, stay, staz, velx, vely, velz = records[i:i+6]

            # Extract site ID and solution index
            site_id = stax[14:18].strip()  # SITE ID
            soln_ind = int(stax[22:26].strip())  # SOLUTION INDEX
            epoch_time = self.parse_custom_time(stax[27:39].strip())  # Epoch time (YY:DOY:SECOD)

            # Extract X, Y, Z coordinates
            x = float(stax[47:68].strip())
            y = float(stay[47:68].strip())
            z = float(staz[47:68].strip())

            # Extract velocity components
            vx = float(velx[47:68].strip())
            vy = float(vely[47:68].strip())
            vz = float(velz[47:68].strip())

            # Find the station instance in storagetorage
            for station in self.storage:
                if station.site_id == site_id:
                    for idx, soln in enumerate(station.soln_idx):
                        if soln == soln_ind:  # Only update matching solution indices
                            delta_t = (start_time - epoch_time).total_seconds() / (365.25 * 24 * 3600)  # Years
                            computed_coords = [x + vx * delta_t, y + vy * delta_t, z + vz * delta_t]
                            station.soln_coor[idx] = computed_coords  # Store directly

            i += 6  # Move to next station's block

    def read_sinex(self, file_path: str, start_time: Timestamp, end_time: Timestamp):
        """
        Load data from a file with multiple sections.
        The file contains sections identified by header and footer lines.
        Each section is processed only if its header is one of the following:
            - SOLUTION/EPOCHS: used for observation records (time filtering)
            - SITE/ANTENNA: used for antenna type records
            - SOLUTION/ESTIMATE: used for coordinate records
        """
        desired_sections = {"SOLUTION/EPOCHS", "SITE/ANTENNA", "SOLUTION/ESTIMATE"}
        found_sections = set()  # Track sections already processed
        sections = {}  # Dict[str, List[str]]
        current_section = None
        current_lines = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                # Check for header lines starting with '+'
                if line.startswith('+'):
                    # Get section name from header
                    section_name = line.lstrip('+').strip()
                    if section_name in desired_sections:
                        current_section = section_name
                        current_lines = [line]  # Include header line
                    else:
                        current_section = None  # Not a desired section
                elif line.startswith('-') and current_section is not None:
                    # Footer line encountered; complete current section
                    current_lines.append(line)
                    # Data lines are from the third line to the second-to-last line.
                    sections[current_section] = current_lines[2:-1]
                    found_sections.add(current_section)
                    current_section = None
                    current_lines = []
                    # If all desired sections have been processed, break early
                    if found_sections == desired_sections:
                        break
                else:
                    if current_section is not None:
                        current_lines.append(line)

        if "SOLUTION/EPOCHS" in sections:
            self.read_solution_epochs(start_time, end_time, sections["SOLUTION/EPOCHS"])
        if "SITE/ANTENNA" in sections:
            self.read_site_antenna(sections["SITE/ANTENNA"])
        if "SOLUTION/ESTIMATE" in sections:
            self.read_solution_estimate(start_time, sections["SOLUTION/ESTIMATE"])
    

