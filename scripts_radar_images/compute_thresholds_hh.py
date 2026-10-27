import geopandas as gpd
import os
import rasterio
from rasterstats import zonal_stats
import time
import pandas as pd
from datetime import datetime, date
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance
import xarray as xr
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def lee_filter(img, size=5):
    # 'size' es el tamaño de la ventana (ej. 5x5 píxeles)
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)
    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def angle_correction(img, m):
    return img/m

id_lakes = [5, 7]

##list of dates where d[0] is start and d[1] is end
#freeze_dates = [[date(2026, 5, 5), date(2026, 5, 12)], [date(2026, 5, 1), date(2026, 5, 7)]]
#ice_dates = [[date(2026, 5, 13), date(2026, 9, 30)], [date(2026, 5, 8), date(2026, 9, 30)]]
ice_dates = [[date(2026, 5, 12), date(2026, 9, 30)], [date(2026, 5, 1), date(2026, 9, 30)]]

sigma_freeze = []
sigma_ice = []
total = []


for idx, id_lake in enumerate(id_lakes):
    shp = gpd.read_file(f"../data/mapas/lagos_irj.shp")
    shp_lago = shp[shp["id"]==id_lake]

    path = f"../data/calibracion/HH/"
    images = sorted( os.listdir(path) )
    for tif in images:
        if tif == ".DS_Store":
            continue
        tif_path =  f"{path}/{tif}"
        fecha_imagen = tif.split("_")[-3].split(".")[0]
        fecha_imagen = datetime.strptime(fecha_imagen, "%Y-%m-%d").date()

        lake = int(tif.split("_")[2])

        with rasterio.open(tif_path) as src:
            raster_crs = src.crs
            data = src.read(1)
            meta = src.meta
            affine=src.transform

        data_filtered = data 
        data_filtered[data_filtered == -np.inf] = 0  # o un valor mínimo apropiado
        img_linear = 10 ** (data_filtered / 10)
        filtered_linear = lee_filter(img_linear, size=5)
        filtered_db = 10 * np.log10(filtered_linear)

        proy_shp = shp_lago.to_crs(raster_crs)

        stats_vv = zonal_stats(
            proy_shp,
            filtered_db,
            affine=affine,
            stats=["count", "mean", "std"],
            nodata=np.nan,
            band=0,
            raster_out=True,
        )

        masked_array = stats_vv[0]["mini_raster_array"]
        promedio = stats_vv[0]["mean"]
        pixel_values = masked_array.compressed()
       #if promedio and freeze_dates[idx][0] <= fecha_imagen <= freeze_dates[idx][1] and lake == id_lake:
       #    sigma_freeze.extend( pixel_values )
    
        if promedio and ice_dates[idx][0] <= fecha_imagen <= ice_dates[idx][1] and lake == id_lake:
            for p in pixel_values:
                total.append( [p, "Ice", id_lake, fecha_imagen, "HH"] )
            #sigma_ice.extend( pixel_values )
    #print( fecha_imagen, stats_vv[0]["mean"], stats_vv[0]["std"] )
    #            dict_lagos[lake[0]].append( [fecha_imagen, stats_vv[0]["mean"], orb] )

#sigma_freeze = np.array(sigma_freeze)
#print("Sigma freeze: ", sigma_freeze.mean(), sigma_freeze.std(), len(sigma_freeze) )

sigma_ice = np.array(sigma_ice)
print("Sigma ice: ", sigma_ice.mean(), sigma_ice.std(), len(sigma_ice) )

dt = pd.DataFrame(total, columns=["Pixels", "Event", "Id_lake", "Date", "Pol"]) 
dt["Pixels"] = pd.to_numeric(dt["Pixels"], errors="coerce")
sigma_ice = np.array(dt[dt["Event"]=="Ice"]["Pixels"])

fig, ax = plt.subplots(figsize=(8, 5))

#ax.hist( sigma_water_vv, bins=30, color='steelblue', edgecolor='black', alpha=0.8, label="Water images")
ax.hist( sigma_ice, bins=100, color='red', edgecolor='black', alpha=0.8, label="Ice images")

ax.set_xlabel("Backscattering (dB)")
ax.set_ylabel("Frequency")
ax.grid(axis='y', alpha=0.3)
plt.legend(loc="upper left", ncols=1)

plt.tight_layout()
plt.savefig('HH_pixels.png', dpi=120, bbox_inches='tight')
#plt.show()
