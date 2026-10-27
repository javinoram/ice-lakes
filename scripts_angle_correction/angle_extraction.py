import rasterio
from rasterio.windows import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
import os
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
import numpy as np
import warnings
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance
import pandas as pd
np.random.default_rng(42)
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

def mask_by_polygon(array, transform, crs, geometry_gdf, nodata_value=np.nan):
    geometry_proy = geometry_gdf.to_crs(crs)
    geoms = [geom for geom in geometry_proy.geometry]

    # rasterio.mask.mask necesita un dataset "en memoria" con esa forma/transform
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": array.dtype,
        "crs": crs,
        "transform": transform,
    }

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(array, 1)

        with memfile.open() as dataset:
            masked_array, _ = rio_mask(
                dataset,
                geoms,
                crop=False,          # mantiene el mismo tamaño/extensión
                nodata=nodata_value,
                filled=True,
            )
    return masked_array[0]  # rio_mask devuelve (bands, rows, cols)


def align_rasters(path_ref, path_other, band=1, resampling=Resampling.bilinear):
    with rasterio.open(path_ref) as src_ref:
        ref_crs = src_ref.crs
        ref_transform = src_ref.transform
        ref_bounds = src_ref.bounds

        with rasterio.open(path_other) as src_other:
            # Bounds del segundo raster, proyectados al CRS del primero
            other_bounds_in_ref = transform_bounds(
                src_other.crs, ref_crs, *src_other.bounds
            )

            # Intersección de ambos bounds, ya en el mismo CRS
            left = max(ref_bounds.left, other_bounds_in_ref[0])
            bottom = max(ref_bounds.bottom, other_bounds_in_ref[1])
            right = min(ref_bounds.right, other_bounds_in_ref[2])
            top = min(ref_bounds.top, other_bounds_in_ref[3])

            if left >= right or bottom >= top:
                raise ValueError("Los rasters no se traslapan.")

            # Ventana de recorte SOLO sobre el raster de referencia
            window_ref = from_bounds(left, bottom, right, top, transform=ref_transform)
            window_ref = window_ref.round_offsets().round_lengths()

            data_ref = src_ref.read(band, window=window_ref)
            out_transform = src_ref.window_transform(window_ref)

            if data_ref.size == 0:
                raise ValueError("Recorte de referencia vacío (revisa traslape).")

            with WarpedVRT(
                src_other,
                crs=ref_crs,
                transform=out_transform,
                width=data_ref.shape[1],
                height=data_ref.shape[0],
                resampling=resampling,
            ) as vrt:
                data_other = vrt.read(band)

    return data_ref, data_other, out_transform

id_lake = 7
shp = gpd.read_file(f"../data/mapas/lagos_irj.shp")
shp_lago = shp[shp["id"]==id_lake]

#path_asc_angle = f"../data/regresion/tdf_01/DESCENDING/ANGLE"
#path_asc_vv = f"../data/regresion/tdf_01/DESCENDING/VV"

path_asc_angle = f"../data/regresion/antarc_07/angle"
path_asc_vv = f"../data/regresion/antarc_07/HH"

images_angles = sorted(os.listdir(path_asc_angle))
images_vv = sorted(os.listdir(path_asc_vv))
pack_images = zip(images_vv, images_angles)

resultados = []
for tif_vv, tif_angle in pack_images:
    tif_path_vv = f"{path_asc_vv}/{tif_vv}"
    tif_path_angle = f"{path_asc_angle}/{tif_angle}"

    try:
        data_vv, data_angle, transform = align_rasters(tif_path_vv, tif_path_angle)
    except:
        continue

    with rasterio.open(tif_path_vv) as src_vv:
        raster_crs = src_vv.crs

    # Enmascarar ambos con el mismo polígono
    data_filtered = data_vv 
    data_filtered[data_filtered == -np.inf] = 0  # o un valor mínimo apropiado
    img_linear = 10 ** (data_filtered / 10)
    filtered_linear = lee_filter(img_linear, size=3)
    filtered_db = 10 * np.log10(filtered_linear)
    #data_vv= lee_filter(data_vv)

    data_vv_masked = mask_by_polygon(data_vv, transform, raster_crs, shp_lago)
    data_angle_masked = mask_by_polygon(data_angle, transform, raster_crs, shp_lago)
    valid_mask = ~np.isnan(data_vv_masked)
    valid_rows, valid_cols = np.where(valid_mask)

    if len(valid_rows) == 0:
        continue
    
    for i in range(3):
        vv_value = np.nan
        angle_value = np.nan
        intento=0
        while np.isnan(vv_value) or np.isnan(angle_value):
            idx_row = np.random.randint(len(valid_rows))
            idx_col = np.random.randint(len(valid_cols))
            row, col = valid_rows[idx_row], valid_cols[idx_col]
            vv_value = data_vv_masked[row, col]
            angle_value = data_angle_masked[row, col]
            intento+=1
            if intento>100:
                break

        if intento < 100:
            resultados.append((tif_vv, row, col, vv_value, angle_value))
            print(f"VV={np.round(vv_value,3)}, angle={np.round(angle_value, 3)}")

dt = pd.DataFrame(resultados)
dt.to_excel("output.xlsx")
