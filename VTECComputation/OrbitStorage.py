import georinex as gr
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import lagrange
from typing import List, Dict, Tuple
from ObjectClasses import SatPos, DORISObs
from pandas import Timestamp, Timedelta

class OrbitStorage:
    def __init__(self, files: list[str]) -> None:
        for file in files:
            self.sat_dataset = gr.load(file)


    