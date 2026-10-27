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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

rng = np.random.default_rng(42)
warnings.filterwarnings("ignore")

def lee_filter(tif_path, band=1, size=5, enl=4.4):
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        meta = src.meta
        affine=src.transform
        data_filtered_vv = src.read(band)
        shape = data_filtered_vv.shape
        data_filtered_vv[data_filtered_vv == -np.inf] = np.nan
        img = 10 ** (data_filtered_vv / 10)
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
        result = result.astype(np.float32)
        filtered_db_vv = 10 * np.log10(result)
    return filtered_db_vv, raster_crs, meta, affine, shape
    #return data_filtered_vv, raster_crs, meta, affine, shape

#def lee_filter(img, size=5, enl=4.4):
#    img = np.asarray(img, dtype=float)
#    valid = np.isfinite(img)
#
#    data = np.where(valid, img, 0)
#    w = valid.astype(float)

#    n = uniform_filter(w, size=size, mode="reflect")
#    mean = uniform_filter(data, size=size, mode="reflect") / np.maximum(n, 1e-10)
#    mean2 = uniform_filter(data**2, size=size, mode="reflect") / np.maximum(n, 1e-10)

#    var = np.maximum(mean2 - mean**2, 0)
#    noise = mean**2 / enl
#    k = np.clip((var - noise) / np.maximum(var, 1e-10), 0, 1)

#    result = mean + k * (data - mean)
#    result[~valid] = np.nan

#    return result.astype(np.float32)


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

id_lakes = [5, 7]
lake_routes = ["irj", "irj"]
ice_dates = [[date(2026, 6, 1), date(2026, 7, 30)], 
             [date(2026, 6, 1), date(2026, 7, 30)]] 

water_dates = [[date(2026, 3, 1), date(2026, 4, 30)], 
             [date(2026, 3, 1), date(2026, 4, 30)]] 

sigma_freeze = []
sigma_ice = []
sigma_thaw = []
sigma_water = []
total = []
c = -0.221

for idx, id_lake in enumerate(id_lakes):
    shp = gpd.read_file(f"../../data/mapas/lagos_{lake_routes[idx]}.shp")
    shp_lago = shp[shp["id"]==id_lake]

    path = f"../../data/calibracion/HH/"
    images = sorted( os.listdir(path) )

    for tif in images:
        if tif == ".DS_Store":
            continue
        tif_path =  f"{path}/{tif}"
        fecha_imagen = tif.split("_")[-3].split(".")[0]
        fecha_imagen = datetime.strptime(fecha_imagen, "%Y-%m-%d").date()

        lake = int(tif.split("_")[2])

        if id_lake == lake:
            img_vv, raster_crs, meta, affine, shape = lee_filter(tif_path, band=1, size=5, enl=4.4)
            filtered_vv = img_vv #- c*( data_angle - 35)
            #img_vh, _, _, _, _ = lee_filter(tif_path, band=2, size=5, enl=4.4)
            #filtered_vh = img_vh# - c*( data_angle - 35)
            
            proy_shp = shp_lago.to_crs(raster_crs)
            stats_vv = zonal_stats(proy_shp, filtered_vv, affine=affine, stats=["count", "mean", "std"],nodata=np.nan, band=0, raster_out=True,)
            #stats_vh = zonal_stats(proy_shp, filtered_vh, affine=affine, stats=["count", "mean", "std"],nodata=np.nan, band=0, raster_out=True,)

            arr_vv = stats_vv[0]["mini_raster_array"]
            #promedio = stats_vv[0]["mean"]
            #pixel_values_vv = masked_array.compressed()
            
            #arr_vh = stats_vh[0]["mini_raster_array"]
            #promedio = stats_vh[0]["mean"]
            #pixel_values_vh = masked_array.compressed()
            
            #assert arr_vv.shape == arr_vh.shape
            #vv = arr_vv.filled(np.nan)
            #vh = arr_vh.filled(np.nan)
            #valid = np.isfinite(vv) & np.isfinite(vh)

            pixel_values_vv = arr_vv.compressed()#vv[valid]
            print( len(pixel_values_vv) )
            #pixel_values_vh = vh[valid]
            
            if len(pixel_values_vv)>0:
                indices = rng.choice(len(pixel_values_vv), size=100, replace=False)
                for idx_2 in indices:
                    if ice_dates[idx][0] <= fecha_imagen <= ice_dates[idx][1]:
                        total.append( [pixel_values_vv[idx_2], 1, id_lake, fecha_imagen] )
            
                    if water_dates[idx][0] <= fecha_imagen <= water_dates[idx][1]:
                        total.append( [pixel_values_vv[idx_2], 0, id_lake, fecha_imagen] )


dt = pd.DataFrame(total, columns=["Pixels VV", "Event", "Id_lake", "Date"]) 
dt["Pixels VV"] = pd.to_numeric(dt["Pixels VV"], errors="coerce")

sigma_ice_vv = np.array(dt[dt["Event"]==1]["Pixels VV"])
sigma_water_vv = np.array(dt[dt["Event"]==0]["Pixels VV"])

##VV
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist( sigma_water_vv, bins=100, color='blue', edgecolor='black', alpha=0.5, label="Water images")
ax.hist( sigma_ice_vv, bins=100, color='red', edgecolor='black', alpha=0.8, label="Ice images")

ax.set_xlabel("Backscattering (dB)")
ax.set_ylabel("Frequency")
ax.grid(axis='y', alpha=0.3)
plt.legend(loc="upper left", ncols=1)

plt.tight_layout()
plt.savefig('HH.png', dpi=120, bbox_inches='tight')
