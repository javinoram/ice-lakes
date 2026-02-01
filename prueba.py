import geopandas as gpd
from rasterstats import zonal_stats

# Cargar shapefile
shp = gpd.read_file("./data/lagos_aysen.shp")

print(shp)

# Calcular estadísticas zonales
for tif in ["cerro_castillo_inv"]:
    stats = zonal_stats(
        shp,
        f"./data/{tif}_dB.tif",
        stats=["count", "min", "max", "mean", "sum", "std"],
        nodata=None,
        band=1,
        )

    # Mostrar resultados
    for i, s in enumerate(stats):
        if s["sum"] != 0.0:
            print(f"Polígono {i}:\n{s}\n")

