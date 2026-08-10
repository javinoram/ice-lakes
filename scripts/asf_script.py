import asf_search as asf
from shapely.geometry import shape
import geopandas as gpd
import os
import time
from pathlib import Path
import shutil
from lake_polygons import lagos_aysen, lagos_antarc, lagos_magallanes, lagos_tdf

def convert_pol_to_string(l):
    final_string = "POLYGON(("
    for (long, lat) in l:
        final_string += f"{long} {lat},"
    return final_string[:len(final_string)-1] + "))"

user = ####
pass = ####
id_lake = "43"
lake_pol = lagos_aysen[id_lake]
pol = convert_pol_to_string(lake_pol)
print(pol)

results = asf.geo_search(
    #platform=[asf.PLATFORM.RADARSAT],
    dataset="RADARSAT-1",
    intersectsWith= pol,
    start='1995-01-01T00:00UTC',
    end='2014-01-01T00:00UTC',
    maxResults=10000,
    processingLevel="L1",
    beamMode="Standard",
)

print( len(results) )
if len(results) == 0:
    print("No results")

else:

    session = asf.ASFSession().auth_with_creds(user, pass)

    output_dir = './radarsat_data'
    os.makedirs(output_dir, exist_ok=True)

    for r in results:
        print(r.properties['sceneName'])
        r.download(path=output_dir, session=session)
        time.sleep(2)
#
        original = Path(output_dir) / Path(r.properties["url"]).name
        fecha = r.properties["startTime"][:10] #2003-08-17T21:43:18Z
        nuevo = Path(output_dir) / f"RADARSAT_{id_lake}_{fecha}_{r.properties['sceneName']}.zip"
        shutil.move(original, nuevo)

        print(f"Descargado: {nuevo.name}")

