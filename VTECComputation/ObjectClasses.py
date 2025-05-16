from typing import List, Dict, Tuple
from pandas import Timestamp, Timedelta
import constant as const
import math
import numpy as np
import constant
import copy

class DORISBeacon: 
    def __init__(self, ID:int, station_code:str, name:str, domes:str, type:str, freq_shift:int) -> None:
        self.ID = ID
        self.station_code = station_code
        self.name = name
        self.domes = domes
        self.type = type
        self.freq_shift = freq_shift # station frequency shift factor

        self.time_ref_bit = False
        self.time_bias = 0.0 # time beacon reference vs. TAI reference time [1e-6 s]
        self.time_shift = 0.0 # time beacon reference shift [1e-14 s/s]

class DORISObs:


    def __init__(self, station_code: str, station_id: int, freq_shift:int, receiver_clock_offset: float,
                 obs_value: list[float], obs_epoch: Timestamp, obs_type: list[str], PRN):
        self.station_code = station_code
        self.station_id = station_id
        self.freq_shift = freq_shift # station frequency shift factor
        self.receiver_clock_offset = receiver_clock_offset
        self.obs_value = obs_value
        self.obs_epoch = obs_epoch
        self.obs_type = obs_type
        self.PRN = PRN
        self.ant_type = None

        self.L1 = None
        self.L2 = None
        self.C1 = None
        self.C2 = None

        for i in range(len(self.obs_type)):

            if   self.obs_type[i] == "L1": self.L1 = self.obs_value[i]
            elif self.obs_type[i] == "L2": self.L2 = self.obs_value[i]
            elif self.obs_type[i] == "C1": self.C1 = self.obs_value[i] * 10
            elif self.obs_type[i] == "C2": self.C2 = self.obs_value[i] * 10 

            elif (self.L1 and self.L2 and self.C1 and self.C2): break
        
        self.dion = None
        self.dion_no_geom = None

        self.pos_sta_cele = [] # np.array
        self.pos_sat_cele = [] # np.array

        self.pp: List[float] = []
        self.ipp_lon = None
        self.ipp_lat = None

        self.elevation = None
        self.map_value = None
        self.STEC = None 
    
    def geom_corr(self):
        if self.station_shift != 0:
            d_lambda1 = const.c / (5e6 * 407.25 * (1 + self.station_shift * const.d_p_multik))
            d_lambda2 = const.c / (5e6 * 80.25 * (1 + self.station_shift * const.d_p_multik))
            d_k = (d_lambda2**2)/((d_lambda2**2)-(d_lambda1**2))
            d_w = (d_lambda1**2)/(40.3)
        else:
            d_lambda1 = const.d_lambda1
            d_lambda2 = const.d_lambda2
            d_k = const.d_k
            d_w = const.d_w

        self.dion = (-d_k) * (self.L1 * d_lambda1 - self.L2 * d_lambda2)
        self.dion_no_geom = self.dion

        # Satellite offsets
        if   self.PRN == "L27": SatOff = const.d_geomcorrJa2
        elif self.PRN == "L39": SatOff = const.d_geomcorrJa3
        elif self.PRN == "L12": SatOff = const.d_geomcorrCr2
        elif self.PRN == "L45": SatOff = const.d_geomcorrHy2
        elif self.PRN == "L46": SatOff = const.d_geomcorrSar


        # Ground Transmitter offsets
        if   self.ant_type[0:6] == "STAREC": AntOff = const.d_STAREC
        elif self.ant_type[0:7] == "ALCATEL": AntOff = const.d_ALCATEL

        d_geomcorr = SatOff + AntOff

        self.dion -= (-d_k) * (-d_geomcorr * math.sin(np.deg2rad(self.elevation)))
        self.STEC = self.dion * d_w / 1e16    
    
class Station:
    def __init__(self, site_id: str):
        self.site_id = site_id
        self.soln_epochlist = []  # List[Timestamp]: Different epochs for the station
        self.soln_coor = []  # List[List[float]]: XYZ coordinates for each epoch
        self.antenna_types = []  # List[str]: Antenna type for each epoch
        self.soln_idx = []  # List[int]: Solution indices matching epochs

class SatPos:
    def __init__(self, obs_epoch: Timestamp, PRN: str, pos_cele: List[float]) -> None:
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
        
        self.VTEC = []

