import geopandas as gpd
import os
import rasterio
from rasterstats import zonal_stats
import time
from datetime import datetime, date
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance
from rasterio.warp import reproject, Resampling
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def bit_flag(qa_array, bit_position):
    return (qa_array >> bit_position) & 1

def bit_confidence(qa_array, start_bit):
    return (qa_array >> start_bit) & 0b11

id_lake = 1
lake_name = "Parrillar"
shp = gpd.read_file(f"../data/mapas/lagos_magallanes.shp")
shp_lago = shp[shp["id"]==id_lake]

resultados = []
path = f"../data/L5/magallanes_01"
images = sorted( os.listdir(path) )
for tif in images:
    if tif == ".DS_Store":
        continue
    tif_path =  f"{path}/{tif}"
    fecha_imagen = tif.split("_")[-1].split(".")[0].split("(")[0]
    fecha_imagen = datetime.strptime(fecha_imagen, "%Y-%m-%d").date()
    
    with rasterio.open(tif_path) as src:
        #print(src.count)  # cuántas bandas tiene el archivo
        #print(src.descriptions)
        raster_crs = src.crs
        data_b3 = src.read(2)
        data_b8 = src.read(4)
        data_b11 = src.read(5)
        data_cloud = src.read(6)
        meta = src.meta
        affine=src.transform
    

    final_mask_water = (data_b3 - data_b8)/(data_b3 + data_b8)
    final_mask_snow = (data_b3 - data_b11)/(data_b3 + data_b11)
    
    dilated_cloud = bit_flag(data_cloud, 1)
    cloud_confidence = bit_confidence(data_cloud, 8)
    cloud_shadow_confidence = bit_confidence(data_cloud, 10)
    mask_sin_nubes = ( (dilated_cloud == 0) & (cloud_confidence <= 1) & (cloud_shadow_confidence <= 1))

    # aplicar la máscara al NDWI (los que no cumplen quedan como nan)
    ndwi_filtrado = np.where(mask_sin_nubes, final_mask_water, np.nan)
    ndsi_filtrado = np.where(mask_sin_nubes, final_mask_snow, np.nan)

    proy_shp = shp_lago.to_crs(raster_crs)
    stats_vv = zonal_stats(
        proy_shp,
        ndwi_filtrado,
        affine=affine,
        stats=["count", "mean", "std"],
        nodata=np.nan,
        band=1,
        raster_out=True,
    )
    masked_array = stats_vv[0]["mini_raster_array"]
    valores_validos_agua = masked_array.compressed()
    
    stats_vv = zonal_stats(
        proy_shp,
        ndsi_filtrado,
        affine=affine,
        stats=["count", "mean", "std"],
        nodata=np.nan,
        band=1,
        raster_out=True,
    )
    masked_array = stats_vv[0]["mini_raster_array"]
    valores_validos_nieve = masked_array.compressed()

    pixeles_agua = 100*np.sum([1 if p>=0.5 else 0 for p in valores_validos_agua])/len(valores_validos_agua)
    pixeles_nieve = 100*np.sum([1 if p>=0.42 else 0 for p in valores_validos_nieve])/len(valores_validos_nieve)

    #print(fecha_imagen, pixeles_nieve, pixeles_agua)
    if not pd.isna(pixeles_agua) and not pd.isna(pixeles_nieve):
        estado = "Agua"
        if pixeles_agua <=5 and pixeles_nieve >=95:
            estado = "Hielo"
        resultados.append([fecha_imagen, pixeles_agua, pixeles_nieve, estado])


dt = pd.DataFrame(resultados, columns=["Fecha", "NDWI", "NDSI", "Estado"])
dt.to_excel(f"{lake_name}_l5.xlsx")

