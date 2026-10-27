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


path = f"../scripts_optical_images/Master/aysen_01.xlsx"
lake_name = "aysen_01"
dt = pd.read_excel(path)
dt["Fecha"] = pd.to_datetime(dt["Fecha"])
dt = dt[dt["Fecha"]>=pd.Timestamp("1995-01-01")]
dt = dt.sort_values(by="Fecha")

dt_snow = dt[dt['NDSI']>=95]
dt_snow = dt_snow[dt_snow['NDWI']<=5]

fig, ax = plt.subplots(figsize=(14,8))
plt.plot(dt["Fecha"], dt['NDWI'], color="blue", label="NDWI",  alpha=0.5, linestyle='--')
plt.plot(dt["Fecha"], dt['NDSI'], color="green", label="NDSI",  alpha=0.5, linestyle='--')

plt.scatter(dt_snow["Fecha"], dt_snow["NDSI"], marker='x', s=60, color='green', linewidths=2)
#plt.scatter(x, y, marker='x', s=60, color='red', linewidths=2)

for year in range(1995, 2026):
    plt.axvspan(date(year, 6, 1), date(year, 8, 31),
                color="skyblue", alpha=0.3,
                label="Invierno" if year == 2015 else None)

plt.title(f"{lake_name}\'s index evolution")
plt.ylabel('Percentage of pixels')
plt.xlabel('Dates')

plt.hlines(y=[5, 95], xmin=datetime.strptime("1995-01-01", "%Y-%m-%d").date(), xmax=datetime.strptime("2026-01-01", "%Y-%m-%d").date(), colors=['r', 'r'], linestyles=['--', '--'])

plt.grid(True)
plt.legend(loc="upper left", ncols=1)

plt.show()
#plt.savefig(f"{lake_name}_optica.png", dpi=120, bbox_inches='tight')

