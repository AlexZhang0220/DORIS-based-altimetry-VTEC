import numpy as np
import constant
from typing import List, Dict, Tuple
import math
from pyproj import Proj, transform
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

def cart2enu(sat: List[float], station: List[float]):

    # WGS84 xyz: EPSG 4978
    # WGS84 blh：EPSG 4326
    crs_geocentric = 'EPSG:4978'
    crs_geographic = 'EPSG:4326'
    transformer = Transformer.from_crs(crs_geocentric, crs_geographic, always_xy = True)
    lon, lat, _ = transformer.transform(station[0], station[1], station[2], radians=True)

    matrix = np.array([[-np.sin(lon),np.cos(lon),0], \
                           [-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon), np.cos(lat)], \
                            [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])
    vec = np.subtract(sat, station)
    enu = np.dot(matrix, vec)
    return enu

def get_elevation(sat_pos_cele: List[float], sta_pos_cele: List[float]) -> float: # in degree measure

    enu = cart2enu(sat_pos_cele, sta_pos_cele)

    elevation = np.rad2deg(np.arccos(np.linalg.norm(enu[0:2]) / np.linalg.norm(enu)))

    return elevation

def get_map_value(elevation: float) -> float:

    layer_height = 506700 # [m]
    z_angle = np.deg2rad(90 - elevation) 
    alpha = 0.9782
    map_value = 1/np.cos(np.arcsin(constant.AE84 / (constant.AE84 + layer_height) * np.sin(alpha * z_angle)))
    return map_value

def el(sat_pos_cele: List[float], sta_pos_cele: List[float]) -> float:

    p = np.subtract(sat_pos_cele, sta_pos_cele)
    numerator = np.dot(sta_pos_cele,p)
    denumerator = np.linalg.norm(sta_pos_cele) * np.linalg.norm(p)

    return np.rad2deg(numerator / denumerator) #arccos呢？？？

def Raypoint(sat_pos_cele: List[float], sta_pos_cele: List[float], height: float):

    # WGS84 xyz: EPSG 4978
    # WGS84 blh：EPSG 4326
    crs_geocentric = 'EPSG:4978'
    crs_geographic = 'EPSG:4326'
    transformer = Transformer.from_crs(crs_geocentric, crs_geographic, always_xy = True)
    sta_pos_geod = transformer.transform(sta_pos_cele[0], sta_pos_cele[1], sta_pos_cele[2])
    sat_pos_geod = transformer.transform(sat_pos_cele[0], sat_pos_cele[1], sat_pos_cele[2])
    # lon and lat in degree measure
    IPP_cele = []
    IPP_geod = []
    
    eps = 0.001
    # Distance receiver-satellite
    Drs = np.linalg.norm(np.subtract(sat_pos_cele, sta_pos_cele))
    
    # if the point is directly over the receiver
    if abs(sta_pos_geod[0] - sat_pos_geod[0]) < eps and abs(sta_pos_geod[1] - sat_pos_geod[1]) < eps:
        IPP_geod = sta_pos_geod[:]
        IPP_geod[2] = height
        IPP_cele = transformer.transform(IPP_geod[0], IPP_geod[1], IPP_geod[2], direction=reversed)
    else:
        # Computation of angle between receiver-satellite and receiver-geocenter
        alpha = np.deg2rad(90 + el(sat_pos_cele, sta_pos_cele))
        
        # Distance from geocenter to certain Point that is located on the raypath
        # from satellite to receiver (quadratic equation)
        
        # quadratic p/q formulation: term1 +/- term2 = D_rp_1,2
        term1 = (constant.AE84 + sta_pos_geod[2]) * math.cos(alpha)
        term2 = math.sqrt(math.pow(term1, 2) + math.pow((constant.AE84 + height), 2) - math.pow(constant.AE84 + sta_pos_geod[2], 2))
        
        D_rP1 = term1 + term2
        D_rP2 = term1 - term2
        
        # Now find the correct solution that is located between receiver and satellite!
        
        # Scale factor as the ratio between Distance rec-sat and Distance rec-Point
        Scale1 = D_rP1 / Drs
        Scale2 = D_rP2 / Drs
        
        P1_cele = [
            sta_pos_cele[0] + Scale1 * (sat_pos_cele[0] - sta_pos_cele[0]),
            sta_pos_cele[1] + Scale1 * (sat_pos_cele[1] - sta_pos_cele[1]),
            sta_pos_cele[2] + Scale1 * (sat_pos_cele[2] - sta_pos_cele[2])
        ]
        
        P2_cele = [
            sta_pos_cele[0] + Scale2 * (sat_pos_cele[0] - sta_pos_cele[0]),
            sta_pos_cele[1] + Scale2 * (sat_pos_cele[1] - sta_pos_cele[1]),
            sta_pos_cele[2] + Scale2 * (sat_pos_cele[2] - sta_pos_cele[2])
        ]
        
        # Decide which point is the right one
        if math.sqrt(np.linalg.norm(np.subtract(sat_pos_cele, P1_cele))) <= Drs:
            IPP_cele = P1_cele
        else:
            IPP_cele = P2_cele

    ipp_lon, ipp_lat, _ = transformer.transform(IPP_cele[0], IPP_cele[1], IPP_cele[2])
    return ipp_lon, ipp_lat

def GIMInpo(GIMVTEC: list[float], IPPEpoch: list[float], IPPLat: list[float], IPPLon: list[float]) -> list[float]: 
# function computes GIM VTEC values at (IPPlat, IPPlon) which are taken at IPPepoch
# input:
#   GIMVTEC   GIM VTEC values for the given day, note THE LAT IS ALREADY FROM -87.5 TO 87.5!!
#   IPPEpoch  Epoch in MJD
#   IPPLat    Latitude in degrees
#   IPPLon    Longitude in degrees, (-180,180)
# output:
#   IPPVTEC   Interpolated VTEC value at the epoch and (lat, lon)

    MaxLatIndex = 71 #int( - lat_start * 2 / lat_grid_interval + 1) 
    MaxLonIndex = 73 #int( - lon_start * 2 / lon_grid_interval + 1)

    # time 
    IPPEpoch = Time(IPPEpoch, format='mjd').to_datetime()

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

