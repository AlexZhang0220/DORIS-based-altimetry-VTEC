from typing import List, Dict, Tuple
from ObjectClasses import Station
import math
import os
import sys

class StationStorage:
    def __init__(self) -> None:
        self.storage: List[Station] = []

    def get_ant_type(self, station_code: str) -> str:
        for station in self.storage:
            if station_code == station.station_code:
                return station.ant_type
        return ""
    
    def get_pos_cele(self, station_code: str):
        for station in self.storage:
            if station_code == station.station_code:
                return station.pos_cele
        return []
    
    def find_indices(self, station_code: str) -> List[int]:
        indices = []
        for index, station in enumerate(self.storage):
            if station.station_code == station_code:
                indices.append(index)
        return indices

    def read_sinex(self, file: str) -> None:
        pos_cele: List[float] = [0.0] * 3  
        apriori_flag = False   
        antenna_flag = False
        try:
            with open(file, 'r') as stream:
                while True:
                    temp = stream.readline()
                    if not temp or temp.strip() == "-SOLUTION/APRIORI":
                        break
                    elif temp.strip() == "+SOLUTION/APRIORI":
                        apriori_flag = True
                        stream.readline()
                        temp = stream.readline() # line of STAX
                    if apriori_flag and temp[7:11] == "STAX":
                        station_code = temp[14:18].lower()
                        pos_cele[0] = float(temp[47:68].strip()) 

                        temp = stream.readline()
                        pos_cele[1] = float(temp[47:68].strip()) 

                        temp = stream.readline()
                        pos_cele[2] = float(temp[47:68].strip())

                        station = Station(station_code, pos_cele.copy())
                        self.storage.append(station)

                stream.seek(0)  # Reset file pointer to the beginning

                while True:
                    temp = stream.readline()
                    if not temp or temp.strip() == "-SITE/ANTENNA":
                        break
                    elif temp.strip() == "+SITE/ANTENNA":
                        antenna_flag = True
                        stream.readline()
                        temp = stream.readline()
                    if antenna_flag:
                        station_code = temp[0:5].strip()
                        station_code = station_code.lower()
                        indices = self.find_indices(station_code)
                        for index in indices:
                            self.storage[index].ant_type = temp[42:62].strip()

        except IOError:
            print(f"DORISpy----StationStorage::read_sinex----Unable to open {file}")
            sys.exit(1)

