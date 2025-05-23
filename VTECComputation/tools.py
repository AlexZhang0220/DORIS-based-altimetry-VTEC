import numpy as np
import math
import pandas as pd
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

def get_igs_vtec(GIMVTEC: list[np.ndarray], df_max: pd.DataFrame) -> np.ndarray:
    MaxLatIndex, MaxLonIndex = 71, 73

    inpo_funcs = [
        RegularGridInterpolator((range(MaxLatIndex), range(MaxLonIndex)), GIMVTEC[i], method="linear")
        for i in range(13)
    ]

    lat_idx = (df_max['ipp_lat'] + 87.5) / 2.5
    lon_idx = (df_max['ipp_lon'] + 180) / 5

    epoch = pd.to_datetime(df_max['obs_epoch'])
    t_idx = epoch.dt.hour // 2
    t_scale = ((epoch.dt.hour + epoch.dt.minute / 60 + epoch.dt.second / 3600) % 2) / 2

    coords = np.stack([lat_idx, lon_idx], axis=1)

    vtec1 = np.array([inpo_funcs[i]([pt])[0] for i, pt in zip(t_idx, coords)])
    vtec2 = np.array([inpo_funcs[i+1]([pt])[0] for i, pt in zip(t_idx, coords)])
    
    return (1 - t_scale) * vtec1 + t_scale * vtec2

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

