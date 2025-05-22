import numpy as np
import constant
from typing import List, Dict, Tuple
import math
from pandas import Timestamp
from pyproj import Transformer
from astropy.time import Time
from scipy.interpolate import RegularGridInterpolator
from numba import njit

@njit
def batch_lagrange_interp_1d(x_known, y_known, x_interp):
    n = len(x_known)
    result = np.zeros_like(x_interp)
    for j in range(len(x_interp)):
        x = x_interp[j]
        total = 0.0
        for i in range(n):
            term = y_known[i]
            for k in range(n):
                if k != i:
                    term *= (x - x_known[k]) / (x_known[i] - x_known[k])
            total += term
        result[j] = total
    return result

def GIMInpo(GIMVTEC: list[float], IPPEpoch: Timestamp, IPPLat: list[float], IPPLon: list[float]) -> list[float]: 
# function computes GIM VTEC values at (IPPlat, IPPlon) which are taken at IPPepoch
# input:
#   GIMVTEC   GIM VTEC values for the given day, note THE LAT IS ALREADY FROM -87.5 TO 87.5!!
#   IPPEpoch  Epoch in Timestamp
#   IPPLat    Latitude in degrees
#   IPPLon    Longitude in degrees, (-180,180)
# output:
#   IPPVTEC   Interpolated VTEC value at the epoch and (lat, lon)

    MaxLatIndex = 71 #int( - lat_start * 2 / lat_grid_interval + 1) 
    MaxLonIndex = 73 #int( - lon_start * 2 / lon_grid_interval + 1)
    
    # space interpolation function list for the day 
    InpoFunc = []
    for i in range(13):
        InpoFunc.append(RegularGridInterpolator((list(range(0, MaxLatIndex)), list(range(0, MaxLonIndex))),
                GIMVTEC[i], method='linear'))
        
    IPPGIMVTEC = []
    for i in range(len(IPPEpoch)):
        IPPLatIndex = (IPPLat[i] + 87.5) / 2.5
        IPPLonIndex = (IPPLon[i] + 180) / 5

        IPPEpochIndex = IPPEpoch[i].hour // 2
        IPPEpochScale = ((IPPEpoch[i].hour + IPPEpoch[i].minute / 60.0 + IPPEpoch[i].second / 3600.0) % 2) / 2
        # IPPEpochScale: the percent this epoch has went through in the 2-hour period

        # linear interpolation in time
        IPPGIMVTEC.append((1 - IPPEpochScale) * InpoFunc[IPPEpochIndex]([IPPLatIndex, IPPLonIndex])[0]\
              + IPPEpochScale * InpoFunc[IPPEpochIndex + 1]([IPPLatIndex, IPPLonIndex])[0])
        
    return IPPGIMVTEC

def trop_saa(pos, elev, humi):
    """
    Calculates tropospheric delay correction using Saastamoinen model.
    
    Inputs:
    pos: Receiver position (longitude, latitude, altitude).
    azel: Azimuth and elevation of the satellite relative to the receiver.
    humi: Relative humidity (percentage).
    
    Outputs:
    troperr: Tropospheric delay correction value.
    """
    
    # Constants
    temp0 = 15.0  # Temperature in Celsius
    
    # Check conditions for invalid inputs
    if pos[2] < -100 or pos[2] > 10000 or elev <= 0:
        troperr = 0
        return troperr
    
    # Calculate pressure (pres), temperature (temp), and water vapor pressure (e)
    if pos[2] < 0:
        hgt = 0
    else:
        hgt = pos[2]
        
    pres = 1013.25 * (1.0 - 2.2557E-5 * hgt)**5.2568
    temp = temp0 - 6.5E-3 * hgt + 273.16
    e = 6.108 * humi * math.exp((17.15 * temp - 4684.0) / (temp - 38.45))
    
    # Calculate zenith distance (z)
    z = math.pi / 2.0 - math.radians(elev)
    
    # Dry tropospheric delay correction (trph)
    trph = 0.0022768 * pres / (1.0 - 0.00266 * math.cos(2.0 * math.radians(pos[0])) - 0.00028 * hgt / 1E3) / math.cos(z)
    
    # Wet tropospheric delay correction (trpw)
    trpw = 0.002277 * (1255.0 / temp + 0.05) * e / math.cos(z)
    
    # Total tropospheric delay correction
    troperr = trph + trpw
    
    return troperr

