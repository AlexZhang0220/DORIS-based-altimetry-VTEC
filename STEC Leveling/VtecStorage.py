import numpy as np
import matplotlib.pyplot as plt
import typing as T
import matplotlib.dates as mdates
from readFile import read_ionFile
from IGSVTECMap import GIM_inpo
from datetime import datetime, timedelta
import h5py
import time
from astropy.time import Time

start_time = time.time()

process_epoch = datetime(2024,5,8) # the data in user-defined DAY will be processed
year = process_epoch.year
doy = process_epoch.timetuple().tm_yday
ion_file = './passdetectionTest/igsion/igsg'+str(doy)+'0.'+str(year)[-2:]+'i'
ion_file = './passdetectionTest/igsion/IGS0OPSFIN_' + str(year) + str(doy)+'0000_01D_02H_GIM.INX'
scale_factor_doris = 0.925

stec_rms = []
MLT = []
stec_max_ele = []
lat_max_ele = []
count_stec = 0
with h5py.File('./vtec/DOY'+str(doy)+'.h5', 'r+') as file:

    tec_data = read_ionFile(ion_file)
    tec_data = tec_data*0.1 # INX unit being 0.1tecu

    for pass_index in file['/y'+str(year)+'/d'+str(doy)+'/ele_cut_0']:
        pass_index_folder = file['/y'+str(year)+'/d'+str(doy)+f'/ele_cut_0/{pass_index}']

        epoch = pass_index_folder['epoch']
        ipp_lon = np.array(pass_index_folder['ipp_lon'])
        ipp_lat = np.array(pass_index_folder['ipp_lat'])
        stec = np.array(pass_index_folder['stec'])
        count_stec += len(stec)
        elev = np.array(pass_index_folder['elevation'])
        map_value = np.array(pass_index_folder['map_value'])

        ## In the following part, IGS VTEC values are used for leveling of DORIS STEC
        # get IGS VTEC and STEC at DORIS IPP location and elevation
        igs_vtec = np.array(GIM_inpo(tec_data, epoch, ipp_lat, ipp_lon))
        igs_stec = igs_vtec * map_value
        # get the index within one pass where the elevation is highest, and calculate delta of DORIS and IGS at this point
        max_index = np.argmax(elev)
        delta = igs_stec[max_index] * scale_factor_doris - stec[max_index]
        # level of DORIS STEC to IGS STEC according to the delta
        # map the leveled DORIS STEC to VTEC
        vtec = (stec / map_value).tolist()

        if 'vtec' in pass_index_folder:
            del pass_index_folder['vtec']
        pass_index_folder.create_dataset('vtec', data=vtec)
