import numpy as np
import constant
from typing import List, Dict, Tuple
import datetime
import math
from pyproj import Proj, transform
from pyproj import Transformer
import pymap3d as pm

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

    return np.rad2deg(numerator / denumerator)

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
