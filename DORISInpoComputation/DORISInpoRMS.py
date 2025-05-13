import json
import numpy as np
import h5py
import time
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import distance
from astropy.time import Time
from datetime import datetime, timedelta
from readFile import read_ionFile
from itertools import product

def IDW(x, y, values, xi, yi, power=2):
    dist = distance.cdist(np.column_stack((x, y)), np.column_stack((xi, yi)), metric='euclidean')
    dist[dist == 0] = 1e-12  
    
    weights = 1 / dist**power
    weights /= weights.sum(axis=0) 
    return np.dot(values, weights).item()

def GIMInpo(GIMVTEC: list[float], IPPEpoch: list[float], IPPLat: list[float], IPPLon: list[float]) -> list[float]: 
# function computes GIM VTEC values at (IPPlat, IPPlon) which are taken at IPPepoch
# input:
#   GIMVTEC   GIM VTEC values for the given day, note THE LAT IS ALREADY FROM -87.5 TO 87.5!!
#   IPPEpoch  Epoch in MJD
#   IPPLat    Latitude in degrees
#   IPPLon    Longitude in degrees
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

StartTime = time.time()
## Process Interval Settings
StartYear = 2019
StartMonth = 11
StartDay = 30
ProcessInterval = 3 # in days
StartEpoch = datetime(StartYear, StartMonth, StartDay)
StartDoy = StartEpoch.timetuple().tm_yday

## Process Parameter Settings
DivisionStrategy = list(['Time','Space'])[1]
if DivisionStrategy == 'Time':
    EleMaskList = [10, 15] # degree
    MinObsCountList = [30, 60]
    TimeGapList = [120, 180] # The duration of time centered on IPPEpoch, seconds.
    DORISResults = np.zeros((ProcessInterval, len(EleMaskList), len(MinObsCountList), len(TimeGapList), 2))
    # last dimension: 0 is RMS, 1 is percentage
    GIMRMSResults = np.zeros((ProcessInterval, len(EleMaskList), len(MinObsCountList), len(TimeGapList)))
    
    EpochName = str(StartYear)+'_'+str(StartDoy)+'_'+str(ProcessInterval) +'d'
    EleName = '_Ele-'+'-'.join(map(str, EleMaskList))
    MinObsName = '_Obs-'+'-'.join(map(str, MinObsCountList))
    DivisionName = '_Time-'+'-'.join(map(str, TimeGapList))
    
elif DivisionStrategy == 'Space':
    EleMaskList = [10] # degree
    MinObsCountList = [10]
    LatRangeList = [6] # The range of latitude centered on IPPLocation, degrees.
    LonRangeList = [lat * 2 for lat in LatRangeList]
    DORISResults = np.zeros((ProcessInterval, len(EleMaskList), len(MinObsCountList), len(LatRangeList), 2))
    GIMRMSResults = np.zeros((ProcessInterval, len(EleMaskList), len(MinObsCountList), len(LatRangeList)))

    EpochName = str(StartYear)+'_'+str(StartDoy)+'_'+str(ProcessInterval) +'d'
    EleName = '_Ele-'+'-'.join(map(str, EleMaskList))
    MinObsName = '_Obs-'+'-'.join(map(str, MinObsCountList))
    DivisionName = '_Lat-'+'-'.join(map(str, LatRangeList))

InpoStrategy = list(['IDW','Krigin'])[0]

## Processing started!
AltInitEpoch = datetime(1985, 1, 1, 0, 0, 0)
Jason3Freq = 13.575e9

for i,j,k in product(range(ProcessInterval), range(len(EleMaskList)), range(len(MinObsCountList))):
    
    ProcessEpoch = StartEpoch + timedelta(days=i)
    year = ProcessEpoch.year
    Month = ProcessEpoch.month
    Day = ProcessEpoch.day
    doy = ProcessEpoch.timetuple().tm_yday

    ## Read in Altimetry IPP and VTEC data
    with open('./AltimetryData/Orbit/'+str(Month)+str(Day)+'glon.json', 'r') as file: AltLonList = json.load(file)
    with open('./AltimetryData/Orbit/'+str(Month)+str(Day)+'glat.json', 'r') as file: AltLatList = json.load(file)
    with open('./AltimetryData/Epoch/'+str(Month)+str(Day)+'sec.json', 'r') as file: AltEpochList = json.load(file)
    with open('./AltimetryData/Dion/'+str(Month)+str(Day)+'dion.json', 'r') as file: AltVTECList = json.load(file)
    AltLonList = [item for sublist in AltLonList for item in sublist]
    AltLatList = [item for sublist in AltLatList for item in sublist]
    AltEpochList = [item for sublist in AltEpochList for item in sublist]
    AltVTECList = [item for sublist in AltVTECList for item in sublist]

    # Remove NaN in VTEC records
    NonNaNIndices  = np.where(~np.isnan(np.array(AltVTECList)))
    AltLonList = np.array(AltLonList)[NonNaNIndices]
    # Longitude range from (0, 360) to (-180, 180)
    AltLonList = np.array([Lon - 360 if Lon >= 180 else Lon for Lon in AltLonList])
    AltLatList = np.array(AltLatList)[NonNaNIndices]
    AltEpochList = np.array(AltEpochList)[NonNaNIndices]
    # Conversion from countsec to MJD (this step is very slow)
    AltEpochList = [Time(AltInitEpoch + timedelta(seconds=sec.item())).mjd for sec in AltEpochList]
    # Conversion from Dion to VTEC
    AltVTECList = -Jason3Freq ** 2 / 40.3 * np.array(AltVTECList)[NonNaNIndices] / 1e16
        
    ## Compute GIMVTEC values at Altimetry IPP
    IonFile = (
        f"./DORISInput/IGSGIM/{year}/igsg{doy:03d}0.{str(year)[-2:]}i"
        if year <= 2022
        else f"./DORISInput/IGSGIM/{year}/IGS0OPSFIN_{doy:03d}0000_01D_02H_GIM.INX"
    )
    GIMVTEC = read_ionFile(IonFile) * 0.1 # Unit of 0.1 TECU
    GIMVTEC = GIMVTEC[:, ::-1, :] # latitude range from (87.5,-87.5) to (-87.5,87.5) 
    AltIPPGIMVTEC = GIMInpo(GIMVTEC, AltEpochList, AltLatList, AltLonList) # about 15sec in debug

    ## Read in DORIS IPP and VTEC data
    DORISLonList= []; DORISLatList = []; DORISEpochList = []; DORISVTECList = []; DORISElevList = []
    # Lat and Lon here are for IPP

    with h5py.File(f'./DORISVtec/{year}/DOY{doy:03d}.h5', 'r') as file:
        for PassIndex in file[f'/y{year}/{doy:03d}/ele_cut_0']:
            # Each pass is a consective observation between satellite and one station;
            # in this condition all the passes are needed regardless of the specific station
            PassIndexDir = file[f'/y{year}/{doy:03d}/ele_cut_0/{PassIndex}']
            DORISLonList += PassIndexDir['ipp_lon']
            DORISLatList += PassIndexDir['ipp_lat']
            DORISEpochList += PassIndexDir['epoch']
            DORISVTECList += PassIndexDir['vtec']
            DORISElevList += PassIndexDir['elevation']

    EleMask = EleMaskList[j]
    MinObsCount = MinObsCountList[k]
    
    DORISElevMaskedList = []
    ElevMaskIndices = [index for index, ele in enumerate(DORISElevList) if ele > EleMask]
    DORISLonList = np.array(DORISLonList)[ElevMaskIndices]
    DORISLatList = np.array(DORISLatList)[ElevMaskIndices]
    DORISEpochList = np.array(DORISEpochList)[ElevMaskIndices]
    DORISVTECList = np.array(DORISVTECList)[ElevMaskIndices]
    DORISElevMaskedList = np.array(DORISElevList)[ElevMaskIndices]

    if DivisionStrategy == 'Time':
        for l, TimeGap in enumerate(TimeGapList):
            # iteration for each Altimetry observation
            for m, AltEpoch in enumerate(AltEpochList):
                TimeIndices = np.where(np.abs(DORISEpochList - AltEpoch) < TimeGap/24/3600/2)
                # a /2 at last, as we have abs here
                DORISLonInpo = DORISLonList[TimeIndices]
                DORISLatInpo = DORISLatList[TimeIndices]
                DORISEpochInpo = DORISEpochList[TimeIndices]
                DORISVTECInpo = DORISVTECList[TimeIndices]
                # compensation in longitude for data taken in different time
                DORISLonInpo += (DORISEpochInpo - AltEpoch) * 360
                DORISLonInpo[DORISLonInpo > 180] -= 360

                if len(DORISVTECInpo) < MinObsCount: continue
                # computation of DORIS interpolated VTEC at altimetry IPP with IDW
                AltVTECInpoList = []
                AltVTECInpoList.append(IDW(DORISLonInpo, DORISLatInpo, DORISVTECInpo, AltLonList[m], AltLatList[m]))                        
                # Storage of GIM VTEC at altimetry IPP where DORIS can provide inpo results
                AltVTECGIMList = []
                RangeRatio = 0.925
                # RangeRatio = func(epoch, height of jason3) computed with IRI model
                AltVTECGIMList.append(AltIPPGIMVTEC[m] * RangeRatio)
                # record of Altimetry VTEC where DORIS can provide inpo results
                AltVTECPROCList = []
                AltVTECPROCList.append(AltVTECList[m])

            DORISResults[i, j, k, l, 0] = np.sqrt(np.mean(np.square(AltVTECInpoList - AltVTECPROCList)))
            DORISResults[i, j, k, l, 1] = len(AltVTECPROCList) / len(AltVTECList)
            GIMRMSResults[i, j, k, l] = np.sqrt(np.mean(np.square(AltVTECGIMList - AltVTECPROCList)))   

    elif DivisionStrategy == 'Space':
        for l, LatRange in enumerate(LatRangeList):
            # iteration for each Altimetry observation
            AltVTECInpoList = []
            AltVTECGIMList = []
            AltVTECProcList = []
            for m, AltEpoch in enumerate(AltEpochList):
                SpaceIndices = (np.abs(DORISEpochList - AltEpoch) < 30/24/60) \
                    & (np.abs(DORISLatList - AltLatList[m]) < LatRange/2) \
                        & (np.abs(DORISLonList - AltLonList[m]) < LonRangeList[l]/2)
                # add a time limit so that DORIS observations from other revolutions will not be involved
                DORISLonInpo = DORISLonList[SpaceIndices]
                DORISLatInpo = DORISLatList[SpaceIndices]
                DORISEpochInpo = DORISEpochList[SpaceIndices]
                DORISVTECInpo = DORISVTECList[SpaceIndices]
                # compensation in longitude for data taken in different time
                DORISLonInpo += (DORISEpochInpo - AltEpoch) * 360
                DORISLonInpo[DORISLonInpo > 180] -= 360
                
                if len(DORISVTECInpo) < MinObsCount: continue
                # computation of DORIS interpolated VTEC at altimetry IPP with IDW                
                AltVTECInpoList.append(IDW(DORISLonInpo, DORISLatInpo, DORISVTECInpo, AltLonList[m], AltLatList[m]))                        
                # Storage of GIM VTEC at altimetry IPP where DORIS can provide inpo results                
                RangeRatio = 0.925
                # RangeRatio = func(epoch, height of jason3) computed with IRI model
                AltVTECGIMList.append(AltIPPGIMVTEC[m] * RangeRatio)
                # record of Altimetry VTEC where DORIS can provide inpo results                
                AltVTECProcList.append(AltVTECList[m])

            AltVTECInpoList = np.array(AltVTECInpoList); AltVTECProcList = np.array(AltVTECProcList); AltVTECGIMList = np.array(AltVTECGIMList)
            DORISResults[i, j, k, l, 0] = np.sqrt(np.mean(np.square(AltVTECInpoList - AltVTECProcList)))
            DORISResults[i, j, k, l, 1] = len(AltVTECProcList) / len(AltVTECList)
            GIMRMSResults[i, j, k, l] = np.sqrt(np.mean(np.square(AltVTECGIMList - AltVTECProcList)))   
            

np.save('./InpoResults/DORIS/' + EpochName + EleName + MinObsName + DivisionName, DORISResults)

np.save('./InpoResults/GIM' + EpochName + EleName + MinObsName + DivisionName, GIMRMSResults)

EndTime = time.time()
ElapsedTime = EndTime - StartTime
print(f"Elapsed time: {ElapsedTime:.2f} seconds")