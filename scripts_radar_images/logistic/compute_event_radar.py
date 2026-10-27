
import geopandas as gpd
import os
import rasterio
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from rasterstats import zonal_stats
import time
from datetime import datetime, date
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance
import xarray as xr
import numpy as np
import warnings
from rasterio.features import geometry_mask
warnings.filterwarnings("ignore")

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

def melt_threshold(s_ice, std_ice, s_melt, std_melt):
    return ((s_ice + std_ice) + (s_melt - std_melt))/2 - s_ice

def freeze_threshold(s_ice, std_ice, s_freeze, std_freeze):
    return ((s_freeze + std_freeze) + (s_ice - std_ice))/2 - s_freeze

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

def cobertura_lago(tif_path, shp_lago):
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        poly = shp_lago.to_crs(src.crs)
        mask = geometry_mask(poly.geometry, out_shape=data.shape,
                             transform=src.transform, invert=True)
    valido = np.isfinite(data) & (data != 0)
    return valido[mask].sum() / mask.sum()

id_lakes = [29, 1, 5]
shape_routes = ["aysen", "magallanes", "magallanes"]
data_routes = ["aysen_29/VV", "magallanes_01/VV", "magallanes_05/VV"]
data_angles = ["aysen_29/angle", "magallanes_01/angle", "magallanes_05/angle"]
names = ["aysen_29", "magallanes_01", "magallanes_05"]

final = {}
for idl, shape, route, angles, name in zip(id_lakes, shape_routes, data_routes, data_angles, names):
    shp = gpd.read_file(f"./data/mapas/lagos_{shape}.shp")
    shp_lago = shp[shp["id"]==idl]
    path = f"./data/test_dataset/{route}"
    path_angle = f"./data/test_dataset/{angles}"
#path_angle = "./data/aysen_01/S1 DESCENDING ANGLE"

    time_serie = []
    time_serie_std = []
    dates = []

    images_radar = sorted( os.listdir(path) )
    print( images_radar[:3] )
    images_angle = sorted( os.listdir(path_angle) )
    #images_angle = sorted( os.listdir(path) )
    images = zip(images_radar, images_angle)

    images = [im for im in images if im[0] != ".DS_Store" and cobertura_lago(f"{path}/{im[0]}", shp_lago) >= 0.99]

    for idx in range(len(images)-1):
        if images[idx][0] == ".DS_Store":
            continue
        tif_path_i =  f"{path}/{images[idx][0]}"
        tif_path_f =  f"{path}/{images[idx+1][0]}"
        tif_angle_path_i =  f"{path_angle}/{images[idx][1]}"
        tif_angle_path_f =  f"{path_angle}/{images[idx+1][1]}"

        fecha_imagen = images[idx+1][0].split("_")[-3].split(".")[0]
        fecha_imagen = datetime.strptime(fecha_imagen, "%Y-%m-%d").date()

        with rasterio.open(tif_path_i) as src:
            raster_crs_i = src.crs
            data_i = src.read(1)
            meta_i = src.meta
            affine_i=src.transform
            shape_i = data_i.shape

        data_f = to_grid(tif_path_f, shape_i, affine_i, raster_crs_i)
        angle_i = to_grid(tif_angle_path_i, shape_i, affine_i, raster_crs_i)
        angle_f = to_grid(tif_angle_path_f, shape_i, affine_i, raster_crs_i)

        data_filtered_i = data_i.copy()
        data_filtered_i[data_filtered_i == -np.inf] = np.nan
        img_linear_i = 10 ** (data_filtered_i / 10)
        filtered_linear_i = lee_filter(img_linear_i, size=5)
        filtered_db_i = 10 * np.log10(filtered_linear_i)
    #filtered_db_i = data_i.copy()

        data_filtered_f = data_f.copy()
        data_filtered_f[data_filtered_f == -np.inf] = np.nan
        img_linear_f = 10 ** (data_filtered_f / 10)
        filtered_linear_f = lee_filter(img_linear_f, size=5)
        filtered_db_f = 10 * np.log10(filtered_linear_f)
    #filtered_db_f = data_f.copy()

        diff_db = (filtered_db_f - filtered_db_i) + 0.211*( angle_i - angle_f )
        diff_db[~np.isfinite(diff_db)] = np.nan
        proy_shp = shp_lago.to_crs(raster_crs_i)
        stats_diff = zonal_stats(
            proy_shp,
            diff_db,
            affine=affine_i,
            stats=["count", "mean", "std"],
            nodata=np.nan,
            band=0,
        )

        #print( stats_diff[0]["mean"] )
        time_serie.append( stats_diff[0]["mean"] )
        time_serie_std.append( stats_diff[0]["std"] )
        dates.append(fecha_imagen)
    final[name] = [time_serie, dates]


umbral_u = np.round(freeze_threshold(-19.75, 6.01, -15.72, 2.28), 4)
umbral_l = np.round(melt_threshold(-19.75, 6.01, -18.01, 2.02), 4)
fig, ax = plt.subplots(figsize=(10, 5))

for year in range(2026, 2027):
    plt.axvspan(date(year, 6, 1), date(year, 8, 31),
                color="skyblue", alpha=0.3,
                label="Invierno" if year == 2015 else None)


#ax.plot(final["aysen_29"][1], final["aysen_29"][0], marker="o", linestyle="-", color="tab:blue", label="Las Perdices")
ax.plot(final["magallanes_01"][1], final["magallanes_01"][0], marker="o", linestyle="-", color="tab:red", label="Parrillar")
ax.plot(final["magallanes_05"][1], final["magallanes_05"][0], marker="o", linestyle="-", color="tab:green", label="Lynch")

ax.axhline(y=umbral_u, color="pink", linestyle="--", linewidth=1.5, label=f"Umbral freeze= {umbral_u}")
ax.axhline(y=umbral_l, color="cyan", linestyle="--", linewidth=1.5, label=f"Umbral melt = {umbral_l}")

ax.set_xlabel("Fecha")
ax.set_ylabel("Valor (dB)")
ax.set_title("Serie de tiempo - diferencia SAR")
ax.legend()

# Rotar las fechas del eje x para que no se amontonen
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
