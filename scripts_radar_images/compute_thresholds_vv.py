import geopandas as gpd
import os
import rasterio
from rasterstats import zonal_stats
import time
from datetime import datetime, date
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter, uniform_filter, variance
import xarray as xr
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def lee_filter(img, size=5, enl=4.4):
    img = np.asarray(img, dtype=float)
    valid = np.isfinite(img)

    data = np.where(valid, img, 0)
    w = valid.astype(float)

    n = uniform_filter(w, size=size, mode="reflect")
    mean = uniform_filter(data, size=size, mode="reflect") / np.maximum(n, 1e-10)
    mean2 = uniform_filter(data**2, size=size, mode="reflect") / np.maximum(n, 1e-10)

    var = np.maximum(mean2 - mean**2, 0)
    noise = mean**2 / enl
    k = np.clip((var - noise) / np.maximum(var, 1e-10), 0, 1)

    result = mean + k * (data - mean)
    result[~valid] = np.nan

    return result.astype(np.float32)


def to_grid(src_path, ref_shape, ref_transform, ref_crs, resampling=Resampling.bilinear):
    with rasterio.open(src_path) as src:
        dst = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst

id_lakes = [30, 29, 1, 5]
lake_routes = ["aysen", "aysen", "magallanes", "magallanes"]
ice_dates = [[date(2026, 7, 2), date(2026, 9, 3)], 
             [date(2026, 7, 3), date(2026, 9, 3)], 
             [date(2026, 7, 4), date(2026, 9, 18)],
             [date(2026, 7, 8), date(2026, 9, 2)]]

water_dates = [[date(2026, 4, 17), date(2026, 6, 1)], 
             [date(2026, 4, 17), date(2026, 6, 1)], 
             [date(2026, 6, 20), date(2026, 7, 2)],
             [date(2026, 6, 20), date(2026, 7, 6)]]

sigma_freeze = []
sigma_ice = []
sigma_thaw = []
sigma_water = []
total = []
c = -0.221

for idx, id_lake in enumerate(id_lakes):
    shp = gpd.read_file(f"../data/mapas/lagos_{lake_routes[idx]}.shp")
    shp_lago = shp[shp["id"]==id_lake]

    path = f"../data/calibracion/VV/{lake_routes[idx]}"
    path_angle = f"../data/calibracion/VV/{lake_routes[idx]}_angle"
    images = sorted( os.listdir(path) )
    images_angle = sorted( os.listdir(path_angle) )

    for tif, tif_angle in zip(images, images_angle):
        if tif == ".DS_Store":
            continue
        tif_path =  f"{path}/{tif}"
        tif_path_angle =  f"{path_angle}/{tif_angle}"
        fecha_imagen = tif.split("_")[-2].split(".")[0]
        fecha_imagen = datetime.strptime(fecha_imagen, "%Y-%m-%d").date()

        lake = int(tif.split("_")[2])

        if id_lake == lake:
            with rasterio.open(tif_path) as src:
                raster_crs = src.crs
                data_vv = src.read(2)
                #data_vh = src.read(2)
                meta = src.meta
                affine=src.transform
                shape = data_vv.shape

            data_angle = to_grid(tif_path_angle, shape, affine, raster_crs)
            
            data_filtered_vv = data_vv.copy()
            data_filtered_vv[data_filtered_vv == -np.inf] = np.nan
            img_linear_vv = 10 ** (data_filtered_vv / 10)
            filtered_linear_vv = lee_filter(img_linear_vv, size=5)
            filtered_db_vv = 10 * np.log10(filtered_linear_vv)

            filtered_db_vv = filtered_db_vv - c*( data_angle - 35)
            proy_shp = shp_lago.to_crs(raster_crs)

            stats_vv = zonal_stats(
                proy_shp,
                filtered_db_vv,
                affine=affine,
                stats=["count", "mean", "std"],
                nodata=np.nan,
                band=0,
                raster_out=True,
            )

            masked_array_vv = stats_vv[0]["mini_raster_array"]
            promedio_vv = stats_vv[0]["mean"]
            pixel_values_vv = masked_array_vv.compressed()
        
            if ice_dates[idx][0] <= fecha_imagen <= ice_dates[idx][1]:
                for p in pixel_values_vv:
                    total.append( [p, "Ice", id_lake, fecha_imagen, "VV"] )
            
            if water_dates[idx][0] <= fecha_imagen <= water_dates[idx][1]:
                for p in pixel_values_vv:
                    total.append( [p, "Water", id_lake, fecha_imagen, "VV"] )
            
dt = pd.DataFrame(total, columns=["Pixels", "Event", "Id_lake", "Date", "Pol"]) 
dt["Pixels"] = pd.to_numeric(dt["Pixels"], errors="coerce")

sigma_ice_vv = np.array(dt[dt["Event"]=="Ice"]["Pixels"])
sigma_water_vv = np.array(dt[dt["Event"]=="Water"]["Pixels"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist( sigma_ice_vv, bins=100, color='red', edgecolor='black', alpha=0.5, label="Ice images")
ax.hist( sigma_water_vv, bins=100, color='blue', edgecolor='black', alpha=0.8, label="Water images")

ax.set_xlabel("Backscattering (dB)")
ax.set_ylabel("Frequency")
ax.grid(axis='y', alpha=0.3)
plt.legend(loc="upper left", ncols=1)

plt.tight_layout()
plt.savefig('VH_ice_plus.png', dpi=120, bbox_inches='tight')


#fig, ax = plt.subplots(figsize=(8, 5))
#ax.hist( sigma_water_vv, bins=100, color='red', edgecolor='black', alpha=0.8, label="Water images")

#ax.set_xlabel("Backscattering (dB)")
#ax.set_ylabel("Frequency")
#ax.grid(axis='y', alpha=0.3)
#plt.legend(loc="upper left", ncols=1)

#plt.tight_layout()
#plt.savefig('VV_water_plus.png', dpi=120, bbox_inches='tight')
