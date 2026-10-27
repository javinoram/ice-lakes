import geopandas as gpd
import os
import rasterio
from rasterstats import zonal_stats
import time
from datetime import datetime, date
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter, uniform_filter, variance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import xarray as xr
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

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

ice_dates = [[date(2026, 7, 20), date(2026, 9, 10)], 
             [date(2026, 7, 20), date(2026, 9, 10)], 
             [date(2026, 7, 20), date(2026, 9, 10)],
             [date(2026, 7, 20), date(2026, 9, 10)]]

water_dates = [[date(2026, 4, 1), date(2026, 5, 20)], 
             [date(2026, 4, 1), date(2026, 5, 20)], 
             [date(2026, 4, 1), date(2026, 5, 20)],
             [date(2026, 4, 1), date(2026, 5, 20)]]

dataset_images = []
labels_images = []
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
        lake = int(tif.split("_")[2])
        fecha_imagen = tif.split("_")[-2].split(".")[0]
        fecha_imagen = datetime.strptime(fecha_imagen, "%Y-%m-%d").date()
        
        if id_lake == lake:
            if water_dates[idx][0] <= fecha_imagen <= water_dates[idx][1]:
                dataset_images.append([f"{path}/{tif}", f"{path_angle}/{tif_angle}", id_lake, 0])
                labels_images.append([0])
            if ice_dates[idx][0] <= fecha_imagen <= ice_dates[idx][1]:
                dataset_images.append([f"{path}/{tif}", f"{path_angle}/{tif_angle}", id_lake, 1])
                labels_images.append([1])

##Imagenes y clasificacion de ellas
dataset_images = np.array(dataset_images)
labels_images = np.array(labels_images)
X_train, X_test, _, _ = train_test_split(dataset_images, labels_images, test_size=0.30, random_state=42, stratify=labels_images)

train_pixels_data_vv = []
train_pixels_data_vv_vh = []
train_y_vv = []
train_y_vv_vh = []
for idx, id_lake in enumerate(id_lakes):
    shp = gpd.read_file(f"../../data/mapas/lagos_{lake_routes[idx]}.shp")
    shp_lago = shp[shp["id"]==id_lake]

    for tif, tif_angle, id_lake_train, y_label in X_train:
        img_vv, raster_crs, meta, affine, shape = lee_filter(tif, band=1, size=5, enl=4.4)
        filtered_vv = img_vv #- c*( data_angle - 35)
        img_vh, _, _, _, _ = lee_filter(tif, band=2, size=5, enl=4.4)
        filtered_vh = img_vh# - c*( data_angle - 35)
        
        lake = int(tif.split("_")[2])
        fecha_imagen = tif.split("_")[-2].split(".")[0]
        fecha_imagen = datetime.strptime(fecha_imagen, "%Y-%m-%d").date()
        
        if id_lake == lake:
            proy_shp = shp_lago.to_crs(raster_crs)
            stats_vv = zonal_stats(proy_shp, filtered_vv, affine=affine, stats=["count", "mean", "std"], nodata=np.nan, band=0, raster_out=True)
            stats_vh = zonal_stats(proy_shp, filtered_vh, affine=affine, stats=["count", "mean", "std"], nodata=np.nan, band=0, raster_out=True)
         
            arr_vv = stats_vv[0]["mini_raster_array"]
            arr_vh = stats_vh[0]["mini_raster_array"]

            assert arr_vv.shape == arr_vh.shape
            vv = arr_vv.filled(np.nan)
            vh = arr_vh.filled(np.nan)
            valid = np.isfinite(vv) & np.isfinite(vh)

            pixel_values_vv = vv[valid]
            pixel_values_vh = vh[valid]

        #print( tif, len(pixel_values_vv), pixel_values_vv.shape )
            if len(pixel_values_vv)>0:
                indices = rng.choice(len(pixel_values_vv), size=int(len(pixel_values_vv)*0.3), replace=False)
                for idx_2 in indices:
                    train_pixels_data_vv.append( [pixel_values_vv[idx_2]] )
                    train_pixels_data_vv_vh.append( [pixel_values_vv[idx_2], pixel_values_vh[idx_2]] )
                    train_y_vv.append( [y_label] )
                    train_y_vv_vh.append( [y_label] )


test_pixels_data_vv = []
test_pixels_data_vv_vh = []
test_y_vv = []
test_y_vv_vh = []
for idx, id_lake in enumerate(id_lakes):
    shp = gpd.read_file(f"../../data/mapas/lagos_{lake_routes[idx]}.shp")
    shp_lago = shp[shp["id"]==id_lake]

    for tif, tif_angle, id_lake_train, y_label in X_test:
        img_vv, raster_crs, meta, affine, shape = lee_filter(tif, band=1, size=5, enl=4.4)
        filtered_vv = img_vv #- c*( data_angle - 35)
        img_vh, _, _, _, _ = lee_filter(tif, band=2, size=5, enl=4.4)
        filtered_vh = img_vh# - c*( data_angle - 35)

        lake = int(tif.split("_")[2])
        fecha_imagen = tif.split("_")[-2].split(".")[0]
        fecha_imagen = datetime.strptime(fecha_imagen, "%Y-%m-%d").date()
        
        if id_lake == lake:
            proy_shp = shp_lago.to_crs(raster_crs)
            stats_vv = zonal_stats(proy_shp, filtered_vv, affine=affine, stats=["count", "mean", "std"], nodata=np.nan, band=0, raster_out=True)
            stats_vh = zonal_stats(proy_shp, filtered_vh, affine=affine, stats=["count", "mean", "std"], nodata=np.nan, band=0, raster_out=True)
         
            arr_vv = stats_vv[0]["mini_raster_array"]
            arr_vh = stats_vh[0]["mini_raster_array"]

            assert arr_vv.shape == arr_vh.shape
            vv = arr_vv.filled(np.nan)
            vh = arr_vh.filled(np.nan)
            valid = np.isfinite(vv) & np.isfinite(vh)

            pixel_values_vv = vv[valid]
            pixel_values_vh = vh[valid]

            if len(pixel_values_vv)>0:
                indices = rng.choice(len(pixel_values_vv), size=int(len(pixel_values_vv)*0.3), replace=False)
                for idx in indices:
                    test_pixels_data_vv.append( [pixel_values_vv[idx]] )
                    test_pixels_data_vv_vh.append( [pixel_values_vv[idx], pixel_values_vh[idx]] )
                    test_y_vv.append( [y_label] )
                    test_y_vv_vh.append( [y_label] )

        
unique, counts = np.unique(train_y_vv_vh, return_counts=True)
print(dict(zip(unique, counts)))
unique, counts = np.unique(train_y_vv, return_counts=True)
print(dict(zip(unique, counts)))

print("VV+VH")
modelo = LogisticRegression(class_weight="balanced")
modelo.fit(np.array(train_pixels_data_vv_vh), train_y_vv_vh)
y_pred = modelo.predict(test_pixels_data_vv_vh)

print("Intercepto:", modelo.intercept_)
print("Coeficientes:", modelo.coef_)

print("\nMatriz de confusión:")
print(confusion_matrix(test_y_vv_vh, y_pred))

print("\nMétricas:")
print(classification_report(test_y_vv_vh, y_pred))


print("VV")
modelo = LogisticRegression(class_weight="balanced")
modelo.fit(np.array(train_pixels_data_vv), train_y_vv)
y_pred = modelo.predict(test_pixels_data_vv)

print("Intercepto:", modelo.intercept_)
print("Coeficientes:", modelo.coef_)

print("\nMatriz de confusión:")
print(confusion_matrix(test_y_vv, y_pred))

print("\nMétricas:")
print(classification_report(test_y_vv, y_pred))
