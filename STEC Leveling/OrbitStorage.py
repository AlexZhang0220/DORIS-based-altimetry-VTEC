import georinex as gr
import numpy as np
import math
from scipy.interpolate import lagrange
from typing import List, Dict, Tuple
from ObjectClasses import SatPos, DORISObs
from datetime import datetime, timedelta

class OrbitStorage:
    def __init__(self) -> None:
        self.storage: List[SatPos] = []
    
    def read_sp3(self, file: str, prn: str):
        data = gr.load(file)
        Jason3_data = data.sel(sv=prn)
        for id,epoch in enumerate(Jason3_data.time.values):
            pos_cele = Jason3_data.position.values[id] * 1000 # from km to m
            orbit_epoch = epoch.astype('datetime64[s]').astype(datetime)
            sat_position = SatPos(orbit_epoch, prn, pos_cele)
            self.storage.append(sat_position)

    def get_pos_cele_lagr_inter(self, obs: DORISObs, num_inter_point: int):
        obs_epoch = obs.obs_epoch
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
