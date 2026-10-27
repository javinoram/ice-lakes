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

id_lakes = [30, 29, 1, 5]
lake_routes = ["aysen", "aysen", "magallanes", "magallanes"]
ice_dates = [[date(2026, 7, 10), date(2026, 8, 20)], 
             [date(2026, 7, 10), date(2026, 8, 20)], 
             [date(2026, 7, 10), date(2026, 8, 20)],
             [date(2026, 7, 10), date(2026, 8, 20)]]

water_dates = [[date(2026, 4, 1), date(2026, 6, 15)], 
             [date(2026, 4, 1), date(2026, 6, 15)], 
             [date(2026, 4, 1), date(2026, 6, 15)],
             [date(2026, 4, 1), date(2026, 6, 15)]]

sigma_freeze = []
sigma_ice = []
sigma_thaw = []
sigma_water = []
total = []
c = -0.221

for idx, id_lake in enumerate(id_lakes):
    shp = gpd.read_file(f"../../data/mapas/lagos_{lake_routes[idx]}.shp")
    shp_lago = shp[shp["id"]==id_lake]

    path = f"../../data/calibracion/VV/{lake_routes[idx]}"
    path_angle = f"../../data/calibracion/VV/{lake_routes[idx]}_angle"
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
            img_vv, raster_crs, meta, affine, shape = lee_filter(tif_path, band=1, size=5, enl=4.4)
            data_angle = to_grid(tif_path_angle, shape, affine, raster_crs)
            filtered_vv = img_vv - c*( data_angle - 35)
            img_vh, _, _, _, _ = lee_filter(tif_path, band=2, size=5, enl=4.4)
            filtered_vh = img_vh# - c*( data_angle - 35)
            
            proy_shp = shp_lago.to_crs(raster_crs)
            stats_vv = zonal_stats(proy_shp, filtered_vv, affine=affine, stats=["count", "mean", "std"],nodata=np.nan, band=0, raster_out=True,)
            stats_vh = zonal_stats(proy_shp, filtered_vh, affine=affine, stats=["count", "mean", "std"],nodata=np.nan, band=0, raster_out=True,)

            arr_vv = stats_vv[0]["mini_raster_array"]
            #promedio = stats_vv[0]["mean"]
            #pixel_values_vv = masked_array.compressed()
            
            arr_vh = stats_vh[0]["mini_raster_array"]
            #promedio = stats_vh[0]["mean"]
            #pixel_values_vh = masked_array.compressed()
            
            assert arr_vv.shape == arr_vh.shape
            vv = arr_vv.filled(np.nan)
            vh = arr_vh.filled(np.nan)
            valid = np.isfinite(vv) & np.isfinite(vh)

            pixel_values_vv = vv[valid]
            pixel_values_vh = vh[valid]
            
            if len(pixel_values_vv)>0:
                indices = rng.choice(len(pixel_values_vv), size=500, replace=False)
                for idx_2 in indices:
                    if ice_dates[idx][0] <= fecha_imagen <= ice_dates[idx][1]:
                        total.append( [pixel_values_vv[idx_2], pixel_values_vh[idx_2], 1, id_lake, fecha_imagen] )
            
                    if water_dates[idx][0] <= fecha_imagen <= water_dates[idx][1]:
                        total.append( [pixel_values_vv[idx_2], pixel_values_vh[idx_2], 0, id_lake, fecha_imagen] )


dt = pd.DataFrame(total, columns=["Pixels VV", "Pixels VH", "Event", "Id_lake", "Date"]) 
dt["Pixels VV"] = pd.to_numeric(dt["Pixels VV"], errors="coerce")
dt["Pixels VH"] = pd.to_numeric(dt["Pixels VH"], errors="coerce")

sigma_ice_vv = np.array(dt[dt["Event"]==1]["Pixels VV"])
sigma_water_vv = np.array(dt[dt["Event"]==0]["Pixels VV"])

sigma_ice_vh = np.array(dt[dt["Event"]==1]["Pixels VH"])
sigma_water_vh = np.array(dt[dt["Event"]==0]["Pixels VH"])

##VV
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist( sigma_water_vv, bins=100, color='blue', edgecolor='black', alpha=0.5, label="Water images")
ax.hist( sigma_ice_vv, bins=100, color='red', edgecolor='black', alpha=0.8, label="Ice images")

ax.set_xlabel("Backscattering (dB)")
ax.set_ylabel("Frequency")
ax.grid(axis='y', alpha=0.3)
plt.legend(loc="upper left", ncols=1)

plt.tight_layout()
plt.savefig('VV.png', dpi=120, bbox_inches='tight')

##VH
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist( sigma_water_vh, bins=100, color='blue', edgecolor='black', alpha=0.5, label="Water images")
ax.hist( sigma_ice_vh, bins=100, color='red', edgecolor='black', alpha=0.8, label="Ice images")

ax.set_xlabel("Backscattering (dB)")
ax.set_ylabel("Frequency")
ax.grid(axis='y', alpha=0.3)
plt.legend(loc="upper left", ncols=1)

plt.tight_layout()
plt.savefig('VH.png', dpi=120, bbox_inches='tight')


##CONJUNTA HIELO
plt.figure(figsize=(9, 7))
plt.hist2d(sigma_ice_vv, sigma_ice_vh, bins=80, cmap="viridis" )

plt.colorbar(label="Número de píxeles")
plt.xlabel("VV (dB)")
plt.ylabel("VH (dB)")
plt.title("Distribución conjunta VV–VH")

plt.tight_layout()
plt.savefig('VV+VH_ice.png', dpi=120, bbox_inches='tight')

##CONJUNTA HIELO
plt.figure(figsize=(9, 7))
plt.hist2d(sigma_water_vv, sigma_water_vh, bins=80, cmap="viridis" )

plt.colorbar(label="Número de píxeles")
plt.xlabel("VV (dB)")
plt.ylabel("VH (dB)")
plt.title("Distribución conjunta VV–VH")

plt.tight_layout()
plt.savefig('VV+VH_water.png', dpi=120, bbox_inches='tight')


##TODO JUNTO
x = np.linspace(-35, -5, 100)
y = np.linspace(-45, -10, 100)

H_water, _, _ = np.histogram2d(sigma_water_vv, sigma_water_vh, bins=[x, y], density=True)
H_ice,   _, _ = np.histogram2d(sigma_ice_vv,   sigma_ice_vh,   bins=[x, y], density=True)

# Suavizado para eliminar el ruido de bins con pocas muestras
H_water = gaussian_filter(H_water, sigma=1.5)
H_ice   = gaussian_filter(H_ice,   sigma=1.5)

# Centros de los bins
X, Y = np.meshgrid((x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2)

# --- 2. Gráfico: un color por clase, niveles relativos al máximo --------------

levels_w = H_water.max() * np.array([0.1, 0.25, 0.5, 0.75])
levels_i = H_ice.max()   * np.array([0.1, 0.25, 0.5, 0.75])

fig, ax = plt.subplots(figsize=(8, 6))

# Relleno semitransparente (la zona de mezcla = superposición)
ax.contourf(X, Y, H_water.T, levels=[levels_w[0], H_water.max()], colors=["tab:blue"], alpha=0.15)
ax.contourf(X, Y, H_ice.T,   levels=[levels_i[0], H_ice.max()],   colors=["tab:red"],  alpha=0.15)

# Líneas de contorno
ax.contour(X, Y, H_water.T, levels=levels_w, colors="tab:blue", linewidths=[0.8, 1.2, 1.6, 2.0])
ax.contour(X, Y, H_ice.T,   levels=levels_i, colors="tab:red",  linewidths=[0.8, 1.2, 1.6, 2.0])

# Leyenda con proxies
ax.plot([], [], color="tab:blue", label="Agua")
ax.plot([], [], color="tab:red",  label="Hielo")
ax.legend()

ax.set_xlabel("VV (dB)")
ax.set_ylabel("VH (dB)")
ax.set_title("Distribución conjunta VV–VH")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('VV+VH_todo.png', dpi=120, bbox_inches='tight')
