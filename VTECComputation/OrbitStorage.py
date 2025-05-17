import georinex as gr
import numpy as np   

import xarray as xr
from scipy.interpolate import lagrange
from typing import List, Dict, Tuple
from ObjectClasses import SatPos, DORISObs
from pandas import Timestamp, Timedelta

class OrbitStorage:
    def __init__(self, files: list[str]) -> None:
        ds_list = []

        for file in files:
            ds = gr.load(file) 
            ds_list.append(ds)

        self.sat_dataset = xr.concat(ds_list, dim="time")



    