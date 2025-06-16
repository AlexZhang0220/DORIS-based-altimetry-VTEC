import georinex as gr
import numpy as np
import math
from scipy.interpolate import lagrange
from typing import List, Dict, Tuple
from ObjectClasses import SatPos, DORISObs
from pandas import Timestamp, Timedelta

class OrbitStorage:
    def __init__(self) -> None:
        self.storage: List[SatPos] = []
    
    def read_sp3(self, file: str, prn: str):
        data = gr.load(file)
        Jason3_data = data.sel(sv=prn)
        for id,epoch in enumerate(Jason3_data.time.values):
            pos_cele = Jason3_data.position.values[id] * 1000 # from km to m
            orbit_epoch = Timestamp(epoch)
            sat_position = SatPos(orbit_epoch, prn, pos_cele)
            self.storage.append(sat_position)

    def get_pos_cele_lagr_inter(self, obs_epoch: Timestamp, num_inter_point: int):
        
        ephem_start_epoch = self.storage[0].obs_epoch
        ephem_end_epoch = self.storage[-1].obs_epoch

        if (ephem_start_epoch - obs_epoch).total_seconds() > 0 or (ephem_end_epoch - obs_epoch).total_seconds() < 0:
            return []
        
        ephem_epochs = []
        ephem_pos_x = []
        ephem_pos_y = []
        ephem_pos_z = []
        out_pos = [0.0] * 3

        epoch_diff = [(sat_position.obs_epoch - obs_epoch).total_seconds() for sat_position in self.storage]

        knot_index = np.argmin(np.abs(epoch_diff)) 
        sec_to_knot = epoch_diff[knot_index]

        if abs(sec_to_knot) > 120: # WHY 120?
            return []
        # avoidance of observation epoch too close to start or end of ephemeris epoch 
        if knot_index - num_inter_point // 2 < 0:
            knot_index = num_inter_point // 2

        if knot_index + math.ceil(num_inter_point / 2) > len(self.storage):
            knot_index = len(self.storage) - math.ceil(num_inter_point / 2)
            
        # for even number, n/2 in left, n/2-1 in right; for odd number, both sides have (n-1)/2 epochs
        for i in range(knot_index - num_inter_point // 2, knot_index + math.ceil(num_inter_point / 2)):

            delta = (self.storage[i].obs_epoch - self.storage[knot_index-num_inter_point//2].obs_epoch).total_seconds()
            ephem_epochs.append(delta)
            ephem_pos_x.append(self.storage[i].pos_cele[0])
            ephem_pos_y.append(self.storage[i].pos_cele[1])
            ephem_pos_z.append(self.storage[i].pos_cele[2])

        delta = (obs_epoch - self.storage[knot_index-num_inter_point//2].obs_epoch).total_seconds()
        out_pos[0] = lagrange(ephem_epochs, ephem_pos_x)(delta)
        out_pos[1] = lagrange(ephem_epochs, ephem_pos_y)(delta)
        out_pos[2] = lagrange(ephem_epochs, ephem_pos_z)(delta)

        return out_pos

    def get_pos_cele_lagr_inter_fast(self, obs_epoch: Timestamp, num_inter_point: int):
        ephem_start_epoch = self.storage[0].obs_epoch
        ephem_end_epoch = self.storage[-1].obs_epoch
        obs_epoch_sec = obs_epoch.timestamp()

        if obs_epoch_sec < ephem_start_epoch.timestamp() or obs_epoch_sec > ephem_end_epoch.timestamp():
            return ''

        min_diff = float('inf')
        knot_index = 0
        for i, sat_position in enumerate(self.storage):
            diff = abs(sat_position.obs_epoch.timestamp() - obs_epoch_sec)
            if diff < min_diff:
                min_diff = diff
                knot_index = i

        if min_diff > 120:  # WHY 120?
            return ''

        half_n = num_inter_point // 2
        knot_index = max(half_n, min(knot_index, len(self.storage) - (num_inter_point + 1) // 2))

        base_epoch_sec = self.storage[knot_index - half_n].obs_epoch.timestamp()
        ephem_epochs = np.array([(self.storage[i].obs_epoch.timestamp() - base_epoch_sec) for i in range(knot_index - half_n, knot_index + (num_inter_point + 1) // 2)])
        ephem_pos_x = np.array([self.storage[i].pos_cele[0] for i in range(knot_index - half_n, knot_index + (num_inter_point + 1) // 2)])
        ephem_pos_y = np.array([self.storage[i].pos_cele[1] for i in range(knot_index - half_n, knot_index + (num_inter_point + 1) // 2)])
        ephem_pos_z = np.array([self.storage[i].pos_cele[2] for i in range(knot_index - half_n, knot_index + (num_inter_point + 1) // 2)])

        delta = obs_epoch_sec - base_epoch_sec

        poly_x = np.poly1d(np.polyfit(ephem_epochs, ephem_pos_x, num_inter_point - 1))
        poly_y = np.poly1d(np.polyfit(ephem_epochs, ephem_pos_y, num_inter_point - 1))
        poly_z = np.poly1d(np.polyfit(ephem_epochs, ephem_pos_z, num_inter_point - 1))

        return np.array([poly_x(delta), poly_y(delta), poly_z(delta)])

    