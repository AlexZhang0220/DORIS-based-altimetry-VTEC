from typing import List, Dict
from ObjectClasses import DORISObs, Thresholds, PassObj, DORISBeacon
from OrbitStorage import OrbitStorage
from StationStorage import StationStorage
from tools import get_elevation, get_map_value, Raypoint
from datetime import datetime, timedelta
import numpy as np
import math
import sys

class DORISStorage:
    def __init__(self) -> None:
        self.storage: List[Dict[str, DORISObs]] = [] # with [station][epoch] as index
        self.passes: List[PassObj] = []
        self.stations: List[DORISBeacon] = []
        # station indexing is based on two assumptions which hold for RINEX 300: 
        # 1. In STATION REFERENCE, stations are listed from 1 to station number, complete and non-redundant
        # 2. For all other station records, e.g. time reference station, the station sequence (Dxx) are the same
        # as used in STATION REFERENCE
    def read_rinex_300(self, start_time: datetime, file: str, orbit_data: OrbitStorage, station_data: StationStorage, settings: Thresholds):
        try:
            with open(file, 'r') as rnxin:
                headerfinish = False
                prn = None # str
                for line in rnxin:
                    temp = line.rstrip()

                    if not headerfinish:
                        label = temp[60:80]
                        if   label.startswith("RINEX VERSION / TYPE"): pass

                        elif label.startswith("COMMENT"): 
                            if "D = DORIS" in temp: 
                                pass
                            else: print('Not RINEX file for DORIS!')
                            
                        elif label.startswith("SATELLITE NAME"):
                            if temp[0:60].strip() == "JASON-3": prn = "L39"

                        elif label.startswith("SYS / # / OBS TYPES"):
                            obs_type_count = int(temp[3:6])
                            subtemp = temp[6:58]
                            obs_type = [subtemp[i:i+4].strip() for i in range(0, 4*obs_type_count, 4)]

                        elif label.startswith("STATION REFERENCE"):
                            station = DORISBeacon(int(temp[1:3]), temp[5:9].lower(), temp[10:40].strip(), 
                                                temp[40:50].strip(), int(temp[50:52]), int(temp[52:56]))
                            self.stations.append(station)

                        elif label.startswith("TIME REF STATION"):
                            ID = int(temp[1:3]) 
                            self.stations[ID-1].time_ref_bit = True
                            self.stations[ID-1].bias = float(temp[5:19].strip())
                            self.stations[ID-1].timeshift = float(temp[21:35].strip())
                            continue

                        elif label.startswith("END OF HEADER"):
                            headerfinish = True
                            self.storage = [[] for _ in range(len(self.stations) + 1)]
                        
                    else: # main content
                        if temp.strip() == "":
                            continue 
                        elif temp[0] == ">":
                            y = int(temp[2:6])
                            m = int(temp[7:9])
                            d = int(temp[10:12])
                            h = int(temp[13:15])
                            mi = int(temp[16:18])

                            s, ms= temp[18:31].split('.', 1)
                            s = int(s)
                            ms = int(ms[0:6]) # six digit for datetime input
                            obs_epoch = datetime(y,m,d,h,mi,s).replace(microsecond=ms)

                            receiver_clock_offset = float(temp[43:56])
                            obs_epoch += timedelta(seconds = receiver_clock_offset)

                            if start_time.day == obs_epoch.day:
                                obs_count = int(temp[34:37])
                                line_per_obs = math.ceil(obs_type_count / 5)

                                for _ in range(obs_count):
                                    obs_value: List[float] = []
                                    for line in range(line_per_obs):
                                        temp = next(rnxin).rstrip()

                                        if line == 0:
                                            ID = int(temp[1:3])
                                            station_code = self.stations[ID-1].station_code
                                            time_freq_shift = self.stations[ID-1].shift

                                        subtemp = temp[3:83]
                                        # read single observations types
                                        for j in range(5):
                                            obs_value.append(float(subtemp[j * 16:(j + 1) * 16 - 2].strip()))

                                    DORISObs.obs_type = obs_type
                                    DORISObs.PRN = prn
                                    obs = DORISObs(station_code, ID, time_freq_shift, obs_value, obs_epoch) 
                                    
                                    ## computation of satellite position
                                    num_inter_point = 7 
                                    # points used for lagrange interpolation, not consequential really
                                    obs.pos_sat_cele = orbit_data.get_pos_cele_lagr_inter(obs, num_inter_point)  
                                    if not obs.pos_sat_cele: 
                                        continue      

                                    ## computation of station position
                                    obs.pos_sta_cele = station_data.get_pos_cele(obs.station_code)
                                    if not obs.pos_sta_cele: 
                                        continue  

                                    obs.ant_type = station_data.get_ant_type(obs.station_code)
                                    if not obs.ant_type: 
                                        continue       

                                    obs.elevation = get_elevation(obs.pos_sat_cele, obs.pos_sta_cele)
                                    if obs.elevation < settings.ele_cut_off:
                                        continue

                                    Ion_height = 506700
                                    # Ion_height = 450000

                                    obs.ipp_lon, obs.ipp_lat = Raypoint(obs.pos_sat_cele, obs.pos_sta_cele, Ion_height)
                                    obs.map_value = get_map_value(obs.elevation)
                                    obs.geom_corr()
                                    obs.VTEC = obs.STEC / obs.map_value        
                                    self.storage[obs.station_id].append(obs)
                                    obs = []
                            else:
                                continue
        except IOError:
            print(f"DORISpy----ObsStorage::read_rinex_300----Unable to open {file}")
            sys.exit(1)
                    
                
   


