from DORISInpo import load_altimetry_data
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

year = 2019
month = 11 
day = 30
process_date = datetime(year, month, day)
doy = process_date.timetuple().tm_yday
lat_range = 8
ele_mask = 10
min_obs_count = 30
name = f'./InpoResults/DORIS/IPP/y{year}_d{doy}_lat-{lat_range}_ele-{ele_mask}_obs-{min_obs_count}.npy'
alt_lon, alt_lat, alt_epoch, alt_vtec = load_altimetry_data(month, day)
doris_ipp_alt = np.load(name)
doris_lat_alt = doris_ipp_alt[:, 0]
doris_lon_alt = doris_ipp_alt[:, 1]

doris_mask = np.isin(alt_lon, doris_lon_alt) & np.isin(alt_lat, doris_lat_alt)
base_size = 10
sizes = np.where(doris_mask, base_size * 3, base_size) 
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 6))
ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
sc = ax.scatter(alt_lon, alt_lat, c=alt_vtec, s=sizes, cmap='jet', transform=ccrs.PlateCarree())
cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.05, label='VTEC Value')
gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.7)
gl.top_labels = False 
gl.right_labels = False  
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}
plt.show()