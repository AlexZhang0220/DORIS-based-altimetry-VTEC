from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import constant as const
import math
import numpy as np
import copy

class DORISBeacon: # why is this class not used?
    def __init__(self, ID:int, station_code:str, name:str, domes:str, type:str, shift:int) -> None:
        self.ID = ID
        self.station_code = station_code
        self.name = name
        self.domes = domes
        self.type = type
        self.shift = shift

        self.time_ref_bit = False
        self.bias = 0.0 # time beacon reference vs. TAI reference time [1e-6 s]
        self.time_shift = 0.0 # time beacon reference shift [1e-14 s/s]

class DORISObs:

    obs_type: List[str] = None
    PRN: str = None

    def __init__(self, station_code: str, station_id: int, obs_value: List[float], obs_epoch: datetime):
        self.station_code = station_code
        self.station_id = station_id
        self.obs_value = obs_value
        self.obs_epoch = obs_epoch

        self.pass_first_element_flag = False 
        # flag used in passdetection so that the first element in the pass is not assinged value twice

        self.ant_type = None

        self.L1 = None
        self.L2 = None
        self.C1 = None
        self.C2 = None

        for i in range(len(self.obs_type)):

            if   self.obs_type[i] == "L1": self.L1 = self.obs_value[i]
            elif self.obs_type[i] == "L2": self.L2 = self.obs_value[i]
            elif self.obs_type[i] == "C1": self.C1 = self.obs_value[i] / 100
            elif self.obs_type[i] == "C2": self.C2 = self.obs_value[i] / 100
            elif (self.L1 and self.L2 and self.C1 and self.C2): break
        
        self.dion = None
        self.dion_no_geom = None

        self.pos_sta_cele: List[float] = []
        self.pos_sat_cele: List[float] = []

        self.pp: List[float] = []
        self.ipp_lon = None
        self.ipp_lat = None

        self.elevation = None
        self.map_value = None
        self.STEC = None 
        self.VTEC = None
    
    def geom_corr(self):
        self.dion = (-const.d_k) * (self.L1 * const.d_lambda1 - self.L2 * const.d_lambda2)
        self.dion_no_geom = self.dion

        # Satellite offsets
        if   self.PRN == "L27": SatOff = const.d_geomcorrJa2
        elif self.PRN == "L39": SatOff = const.d_geomcorrJa3
        elif self.PRN == "L12": SatOff = const.d_geomcorrCr2
        elif self.PRN == "L45": SatOff = const.d_geomcorrHy2
        elif self.PRN == "L46": SatOff = const.d_geomcorrSar

        # Ground Transmitter offsets
        if   self.ant_type[0:6] == "STAREC": AntOff = const.d_STAREC
        elif self.ant_type == "ALCATEL": AntOff = const.d_ALCATEL

        d_geomcorr = SatOff + AntOff

        self.dion -= (-const.d_k) * (-d_geomcorr * np.deg2rad(math.sin(self.elevation)))
        self.STEC = self.dion * const.d_w / 1e16    
    
class Station:
    def __init__(self, station_code: str, pos_cele: List[float]) -> None:
        self.station_code = station_code
        self.pos_cele = pos_cele 
        
        self.ant_type = None

class SatPos:
    def __init__(self, obs_epoch: datetime, PRN: str, pos_cele: List[float]) -> None:
        self.obs_epoch = obs_epoch
        self.PRN = PRN
        self.pos_cele = pos_cele

class Thresholds:
    def __init__(self, max_dion_gap: float, max_obs_epoch_gap: float, min_obs_count: int, ele_cut_off: float) -> None:
        self.max_dion_gap = max_dion_gap
        self.max_obs_epoch_gap = max_obs_epoch_gap # unit[sec]
        # check if necessary 
        self.min_obs_count = min_obs_count
        self.ele_cut_off = ele_cut_off # unit[deg]
        # no ele_cut_off to storage all the data 
class PassObj:
    def __init__(self, epoch: List[float], ipp_lat: List[float], ipp_lon: List[float], STEC: List[float], \
                 elevation: List[float], map_value: List[float], station_code: str , station_id: str) -> None:
        
        self.epoch = epoch
        self.ipp_lat = ipp_lat
        self.ipp_lon = ipp_lon
        self.STEC = STEC
        self.elevation = elevation
        self.map_value = map_value
        self.station_code = station_code
        self.station_id = station_id

