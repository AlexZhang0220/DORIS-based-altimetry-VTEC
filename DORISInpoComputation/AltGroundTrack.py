import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def plot_ground_track(ipp_lon, ipp_lat, dion, satellite_name='Satellite'):
    """
    Plot ground track on world map, marking NaN vs non-NaN dion points.
    
    ipp_lon, ipp_lat, dion: 2D arrays (N_pass, N_point)
    satellite_name: title for the plot
    """
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    
    # 加海岸线、陆地、海洋
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.scatter(ipp_lon[0][0], ipp_lat[0][0], linewidth=20)
    # 逐 pass 绘制
    for i in range(len(ipp_lon)):
        lon_pass = np.array(ipp_lon[i])
        lat_pass = np.array(ipp_lat[i])
        dion_pass = np.array(dion[i])
        
        # 非 NaN 部分
        mask_valid = ~np.isnan(dion_pass)
        ax.scatter(lon_pass[mask_valid], lat_pass[mask_valid],
                color='red', linewidth=1, label='Valid dion' if i == 0 else "")
        
        # NaN 部分
        mask_nan = np.isnan(dion_pass)
        ax.scatter(lon_pass[mask_nan], lat_pass[mask_nan],
                color='blue', linewidth=1, label='NaN dion' if i == 0 else "")
    
    # 添加图例和标题
    ax.legend(loc='lower left')
    ax.set_title(f"{satellite_name} Ground Track with NaN and Valid dion")
    plt.show()

if __name__ == '__main__':

    def load_json_file(path):
        with open(path, 'r') as f:
            return json.load(f)
        
    # 结果：dion不存在的点位是卫星经过陆地的点位，大约占总数的30%（因为卫星总是1s读取一次数据）
    # 并不是每天开始的一段都是nan，只是这些天的起始ground track正好在陆地上
    month, day = 5, 10
    lon = load_json_file(f'./AltimetryData/Orbit/{month}{day}glon.json')
    lat = load_json_file(f'./AltimetryData/Orbit/{month}{day}glat.json')
    sec = load_json_file(f'./AltimetryData/Epoch/{month}{day}sec.json')
    dion = load_json_file(f'./AltimetryData/Dion/{month}{day}dion.json')

    plot_ground_track(lon[0:10], lat[0:10], dion[0:10], satellite_name='Jason-3')