"""
###########################################################################
#         Velocity Model interpolation & Visualization functions          #
#--------------------------------------------------------------------------
# This script contains a series of functions designed to query a 1D or 3D #
# velocity model, perform interpolation, and visualize the results.       #
#--------------------------------------------------------------------------
# Author: Andrea C. Riaño                                                 #
# Email: andrea.riano@bsc.es                                              #  
# Date:  January 2025                                                     #
###########################################################################                                                  
"""
import cartopy.mpl.gridliner as gridliner
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib
import simplekml
import folium
import struct
import psutil
import json
import math
import glob
import csv
import os
import gc
import subprocess

from tqdm import tqdm
from netCDF4 import Dataset
from collections import defaultdict
from shapely.geometry import Polygon
from joblib import Parallel, delayed
from pyproj import Proj, Transformer, CRS
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from cartopy.io import img_tiles as cimgt
from pyproj import Geod

__version__ = "1.0.0"

def haversine(lat1, lon1, lat2, lon2):
    '''
    Calculate distance between two coordinates considering Earth's curvature
    '''
    
    # Earth's radius in km
    R = 6371   
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in kilometers
    distance = R * c
    return distance

def angle_to_east(lat1, lon1, lat2, lon2):
    """
    Calculate the angle from East between two coordinates. RWG angle format
    The angle is in degrees:
      0°   = East
     -90°  = North
     180°/-180° = West
      90°  = South
    """
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon_rad = math.radians(lon2 - lon1)

    # Compute differences
    delta_x = math.cos(lat2_rad) * math.cos(delta_lon_rad)
    delta_y = math.cos(lat2_rad) * math.sin(delta_lon_rad)

    # Angle from East in radians
    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad)

    # Normalize to [-180, 180]
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg < -180:
        angle_deg += 360

    return angle_deg

def convert_units(model_data, velocity_unit, density_unit):
    """
    Convert velocity and density units to SI (m/s for velocity, kg/m³ for density).
    
    Parameters:
        model_data (dict): Dictionary containing 'vp', 'vs' (velocities) and 'rho' (density).
    	velocity_unit (str): Unit of velocity in the input data (e.g., 'km/s', 'm/s').
    	density_unit (str): Unit of density in the input data (e.g., 'g/cm3', 'kg/m3').
    
    Returns:
    dict: Updated model_data with converted units.
    """
    if velocity_unit == 'km/s':
        model_data['vp'] = model_data['vp'] * 1000  # Convert km/s to m/s
        model_data['vs'] = model_data['vs'] * 1000  # Convert km/s to m/s
    
    if density_unit == 'g/cm3':
        model_data['rho'] = model_data['rho'] * 1000  # Convert g/cm³ to kg/m³
    
    return model_data

#AR: (2026-09-16) The original generarate_geo_corners() below (the default again since
#AR: 2026-10-09, see CORNER_METHODS) computes the corners with its own pyproj.Geod formula. I
#AR: checked it against the reference output of SCEC
#AR: (cybershake-core/PreCVM/Modelbox/reference/PAS.modelbox), and that formula lands 130 to
#AR: 167 km away from where SCEC's get_modelbox.py puts the same box, with exactly the same C1
#AR: corner, x_length, y_length and rotation. Swapping x_length and y_length (they were the other
#AR: way round from SCEC's convention, see generate_modelbox_file() below) brings it down to 16
#AR: to 23 km, which is still too much to be rounding, because the rotation formula does not
#AR: quite match SCEC's. get_modelbox.py does not compute the corners in Python at all. It calls
#AR: a compiled SCEC tool, PreCVM/Modelbox/bin/gcproj, and when I tested it directly it gives
#AR: SCEC's reference corners exactly, to the 5th decimal. So this function calls that same
#AR: binary instead of trying to copy its geodesic math by hand.
_DEFAULT_GCPROJ_BIN = "/gpfs/projects/bsc115/bsc203748/software/cybershake-core/PreCVM/Modelbox/bin/gcproj"


def generarate_geo_corners_gcproj(model_lon, model_lat, x_length, y_length, model_rot,
                                   gcproj_bin=_DEFAULT_GCPROJ_BIN):
    """
    Same 4 corners get_modelbox.py itself would compute, by calling the same real SCEC tool
    it calls (gcproj), not a from-scratch Python reimplementation.

    model_lon/model_lat here means the SAME reference point get_modelbox.py's own
    "MODEL PARAMETERS" line calls mlon/mlat, NOT the C1 corner (unlike
    generarate_geo_corners() below). On this branch, this value should come straight from
    that same get_modelbox.py run, not typed in by hand, since it is an internal reference
    point get_modelbox.py itself computes, not a point with an obvious real-world meaning.

    Returns corners in order C1, C2, C3, C4, matching gcproj's own stdin order
    ((0,0), (xlen,0), (xlen,ylen), (0,ylen)), the same order get_modelbox.py itself uses.
    """
    stdin_text = "0.0 0.0 %f 0.0 %f %f 0.0 %f\n" % (x_length, x_length, y_length, y_length)
    command = [
        gcproj_bin,
        "ref_lon=%.5f" % model_lon,
        "ref_lat=%.5f" % model_lat,
        "alpha=%f" % model_rot,
        "xlen=%f" % x_length,
        "ylen=%f" % y_length,
    ]
    process = subprocess.run(command, input=stdin_text, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError("gcproj failed with exit code %d.\nCommand: %s\nStdout: %s\nStderr: %s" % (
            process.returncode, " ".join(command), process.stdout, process.stderr))

    #AR: gcproj prints two lines per corner, "lon lat" and then the "x y" it was given (I
    #AR: checked by running it), so only the odd lines are the output.
    lines = [ln for ln in process.stdout.strip().split("\n") if ln.strip()]
    if len(lines) != 8:
        raise RuntimeError("Expected 8 output lines from gcproj (4 corners x 2 lines each), "
                            "got %d. Output: %s" % (len(lines), process.stdout))
    corners = []
    for i in range(0, 8, 2):
        lon, lat = (float(v) for v in lines[i].split())
        corners.append((lon, lat))
    return corners


def generarate_geo_corners(model_lon, model_lat, x_length, y_length, model_rot):
    """
    Corners built by chained geodesic moves so that:
      dist(C1,C2)=y_length and dist(C1,C4)=x_length (in km) on WGS84.
    Returns corners in order: C1, C2, C3, C4 (SW, NW, NE, SE).

    #AR: The standalone corner method, corner_method="geodesic". It does not reproduce SCEC's
    #AR: boxes, see generarate_geo_corners_gcproj() above for that.
    """

    geod = Geod(ellps="WGS84")

    model_rot = -1 * (model_rot + 90)
    theta = math.radians(model_rot)

    # Local unit vectors (EN) after rotation:
    # X axis in EN: (cos, sin); Y axis in EN: (-sin, cos)
    x_e, x_n = math.cos(theta), math.sin(theta)
    y_e, y_n = -math.sin(theta), math.cos(theta)

    # Convert unit EN -> azimuths (deg from North, clockwise)
    az_x = math.degrees(math.atan2(x_e, x_n))
    az_y = math.degrees(math.atan2(y_e, y_n))

    lon1, lat1 = float(model_lon), float(model_lat)  # C1

    # Step geodesically along axes
    lon2, lat2, _ = geod.fwd(lon1, lat1, az_y, y_length * 1000.0)  # C2
    lon4, lat4, _ = geod.fwd(lon1, lat1, az_x, x_length * 1000.0)  # C4
    lon3, lat3, _ = geod.fwd(lon2, lat2, az_x, x_length * 1000.0)  # C3

    return [(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4)]

def calculate_centroid(geo_corners):
    """
    Calculate the centroid of a polygon defined by its geocorners.

    Parameters:
        geo_corners: lista de coordenadas [(lon, lat), ...]

    Return:
        (lon_centroid, lat_centroid)
    """
    lon_sum = sum(lon for lon, lat in geo_corners)
    lat_sum = sum(lat for lon, lat in geo_corners)
    n = len(geo_corners)

    return (lon_sum / n, lat_sum / n)

#AR: (2026-09-16) Two fixes here, both found by testing against SCEC's reference output
#AR: (PAS.modelbox).
#AR: 1. xlen/ylen and the x/y labels of the corners were swapped with respect to SCEC's
#AR:    convention (in SCEC, C2 is x_length away from C1, and this code had it y_length away).
#AR:    The corner list in PAS.modelbox confirms it, C1 to C2 there is the x_length side.
#AR: 2. It now calls generarate_geo_corners_gcproj() (SCEC's tool) instead of
#AR:    generarate_geo_corners() (the less accurate geodesic formula of this file). The comment
#AR:    of that function gives the km errors that made me change it.
#AR: (2026-10-09) corner_method chooses between the two. "geodesic", the default, is the
#AR: standalone behaviour, unchanged. model_lon and model_lat are the C1 corner, and nothing
#AR: outside Python is needed, so it runs on any machine. "gcproj" is the SCEC convention
#AR: described above and needs the compiled gcproj tool of cybershake-core.
CORNER_METHODS = ("geodesic", "gcproj")


def _check_corner_method(corner_method):
    if corner_method not in CORNER_METHODS:
        raise ValueError("corner_method must be one of %s, got %r"
                         % (", ".join(CORNER_METHODS), corner_method))


def generate_modelbox_file(file_name, site_name, model_lon, model_lat, model_rot, x_length, y_length, output_folder,
                           corner_method="geodesic"):
    """
    Generates a model box file with the specified parameters.
    """
    _check_corner_method(corner_method)
    os.makedirs(output_folder, exist_ok=True)
    full_filename = os.path.join(output_folder, f"{file_name}.modelbox_{site_name}")

    if corner_method == "geodesic":
        geo_corners = generarate_geo_corners(model_lon, model_lat, x_length, y_length, model_rot)
        centroid_lon, centroid_lat = calculate_centroid(geo_corners)
        mlon, mlat, xlen, ylen = centroid_lon, centroid_lat, y_length, x_length
    else:
        geo_corners = generarate_geo_corners_gcproj(model_lon, model_lat, x_length, y_length, model_rot)
        centroid_lon, centroid_lat = calculate_centroid(geo_corners)
        mlon, mlat, xlen, ylen = model_lon, model_lat, x_length, y_length

    # Header section, spaced by hand to match the example
    content = f"""{site_name}
APPROXIMATE CENTROID:
 clon=   {centroid_lon:10.6f} clat=   {centroid_lat:10.6f}
MODEL PARAMETERS:
 mlon=   {mlon:10.6f} mlat=   {mlat:10.6f} mrot={model_rot:8.3f} xlen=   {xlen:9.6f} ylen=   {ylen:9.6f}
MODEL CORNERS:
"""

    x_vals = [0.0, xlen, xlen, 0.0]
    y_vals = [0.0, 0.0, ylen, ylen]

    for i in range(4):
        lon, lat = geo_corners[i]
        content += f"   {lon:9.6f}     {lat:8.6f} (x={x_vals[i]:9.3f} y={y_vals[i]:9.3f})\n"

    with open(full_filename, "w") as file:
        file.write(content)

    print(f"File '{full_filename}' has been created.")
    return geo_corners

def create_gridfile(file_name, site_name, x_length, y_length, z_length, grid_spacing, output_folder):
    """
    Creates a text file with the specified format.

    Parameters:
        file_name (str): Base name for the file.
        site_name (str): Site identifier to append to the file name.
        x_length (float): Length of x-dimension in km.
        y_length (float): Length of y-dimension in km.
        z_length (float): Length of z-dimension in km.
        grid_spacing (float): Grid spacing in km.
        output_folder (str): Path to the folder where the output file will be saved.

    Note: I leave the function here but apparently this file is not used
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Full path for the output file
    full_filename = os.path.join(output_folder,f"{file_name}.gridfile_{site_name}")

    content = f"""xlen={x_length}\n   0.0  {x_length}  {grid_spacing}\nylen={y_length}\n   0.0  {y_length}  {grid_spacing}\nzlen={z_length}\n   0.0  {z_length}  {grid_spacing}\n"""

    with open(full_filename, 'w') as file:
        file.write(content)

    print(f"File '{full_filename}' created successfully.")

def create_gridout_file(file_name, site_name, x_length, y_length, z_length,
                        grid_spacing, output_folder, nx=None, ny=None, nz=None):   
    """
    Creates a text file with grid points and spacing in the specified format:
    y first, then x, then z. Values aligned with scientific notation.
    """
    os.makedirs(output_folder, exist_ok=True)

    full_filename = os.path.join(output_folder, f"{file_name}.gridout_{site_name}")
    
    if nx is None:
        nx = round(x_length / grid_spacing)
    if ny is None:
        ny = round(y_length / grid_spacing)
    if nz is None:
        nz = round(z_length / grid_spacing)

    x_length = nx * grid_spacing
    y_length = ny * grid_spacing
    z_length = nz * grid_spacing

    def format_axis(name, length, n_points):
        lines = []
        lines.append(f"{name}len={length:.5f}")
        lines.append(f"n{name}={n_points}")
        for i in range(n_points):
            coord = i * grid_spacing
            lines.append(f"{i:5d}  {coord:11.6e}  {grid_spacing:11.6e}")
        return "\n".join(lines)

    # Correct order: Y, then X, then Z
    sections = [
        format_axis('x', y_length, ny),
        format_axis('y', x_length, nx),
        format_axis('z', z_length, nz)
    ]

    content = "\n".join(sections) + "\n"

    with open(full_filename, 'w') as file:
        file.write(content)

    print(f"File '{full_filename}' created successfully.")

def create_model_coords_file(file_name, site_name, model_lon, model_lat, model_rot,
                             x_length, y_length, grid_spacing, output_folder,
                             nx=None, ny=None, corner_method="geodesic"):
    """
    Creates the model coordinates file applying the same rotation criteria as `generar_geo_corners`.

    Output files follow the RWG format, as UCVM is not used.
    To maintain consistency with the Graves format used in CyberShake (except for AWP),
    X and Y values are swapped. CyberShake includes functions to convert between RWG and AWP formats as needed.

    Parameters:
        file_name (str): Base name for the file.
        site_name (str): Site identifier to append to the file name.
        model_lon: Longitude of the bottom-west corner of the model.
    	model_lat: Latitude of the bottom-west corner of the model.
    	model_rot: rotation of y-axis from south (counter-clockwise positive).
        x_length (float): Length of x-dimension in km.
        y_length (float): Length of y-dimension in km.
        grid_spacing (float): Grid spacing in km.
        output_folder: path to folder where outputfile is stored.

    Returns:
        None: The function writes the coordinates file to disk but does not return any value.
    """
    
    geod = Geod(ellps="WGS84")
    lon0, lat0 = float(model_lon), float(model_lat)

    if grid_spacing <= 0:
        raise ValueError("grid_spacing must be > 0")

    # Node-centered counts for the effective lengths (includes both ends)
    # --- Make modelbox match node grid with N points (0..N-1)

    if nx is None:
        nx = round(x_length / grid_spacing)
    if ny is None:
        ny = round(y_length / grid_spacing)

    x_length = nx * grid_spacing
    y_length = ny * grid_spacing

    #AR: (2026-09-16) The same swap as in generate_modelbox_file(), and it uses SCEC's gcproj
    #AR: tool instead of the geodesic formula of this file, see the comment of
    #AR: generarate_geo_corners_gcproj(). I did NOT change the X/Y loop order below (bottom and top
    #AR: through c1/c4 and c2/c3). The docstring of this file says the RWG format swaps X and Y on
    #AR: purpose with respect to AWP, and I have not checked whether that is what happens here or
    #AR: whether it is a second case of the same bug, so I left it as it was.
    _check_corner_method(corner_method)
    if corner_method == "geodesic":
        geo_corners = generarate_geo_corners(model_lon, model_lat, x_length, y_length, model_rot)
    else:
        geo_corners = generarate_geo_corners_gcproj(model_lon, model_lat, x_length, y_length, model_rot)
    c1, c2, c3, c4 = geo_corners  # SW, NW, NE, SE
    #print(c1, c2, c3, c4 )

    def geodesic_lerp(lon1, lat1, lon2, lat2, f):
        if f <= 0.0: return lon1, lat1
        if f >= 1.0: return lon2, lat2
        az12, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
        lon, lat, _ = geod.fwd(lon1, lat1, az12, dist_m * f)
        return lon, lat

    os.makedirs(output_folder, exist_ok=True)
    full_filename = os.path.join(output_folder, f"{file_name}.model_coords_{site_name}")

    # Write in cybershake orther: X outside, Y inside
    with open(full_filename, "w") as f:
        for j in range(nx):
            fx = j / (nx - 1) if nx > 1 else 0.0

            # bottom and top points at this x-fraction
            lonB, latB = geodesic_lerp(*c1, *c4, fx)
            lonT, latT = geodesic_lerp(*c2, *c3, fx)

            for i in range(ny):
                fy = i / (ny - 1) if ny > 1 else 0.0
                lon, lat = geodesic_lerp(lonB, latB, lonT, latT, fy)

                # 3rd col = i (Y), 4th col = j (X)
                f.write(f"{lon:11.6f} {lat:11.6f} {i:5d} {j:5d}\n")

    print(f"File '{full_filename}' created successfully.")

def select_projection(model_lon, model_lat, x_length, y_length):
    """
    Selects the most appropriate projection based on the model area to minimize distortions when plotting the region.

    Parameters:
    - model_lon (float): Longitude of the model center or reference point (degrees).
    - model_lat (float): Latitude of the model center or reference point (degrees).
    - x_length (float): Length of the model in the x-direction (km).
    - y_length (float): Length of the model in the y-direction (km).

    Returns:
    - cartopy.crs.Projection:
        - For small areas (≤ 250,000 km²): Returns a TransverseMercator projection centered on the appropriate UTM zone.
        - For large areas (> 250,000 km²): Returns a PlateCarree projection to avoid visual distortions.
    """
    area = x_length * y_length
    if area <= 250000:
        # Para áreas pequeñas, usa UTM centrado en la zona
        zona_utm = int((model_lon + 180) / 6) + 1
        return ccrs.TransverseMercator(central_longitude=(zona_utm - 1) * 6 - 180 + 3)
    else:
        # Para áreas más grandes, usa PlateCarree para evitar distorsiones visuales
        return ccrs.PlateCarree()

def plot_modelbox_with_cartopy(region, model_lon, model_lat, x_length, y_length, model_rot, output_folder, display=False):
    """
    Plots the rotated model domain box on a map using Cartopy and saves it as a PNG image.

    Parameters:
        region (str): Name of the region to use in the output filename.
        model_lon (float): Longitude of the model center (degrees).
        model_lat (float): Latitude of the model center (degrees).
        x_length (float): Length of the model in the x-direction (km).
        y_length (float): Length of the model in the y-direction (km).
        model_rot (float): Rotation of the domain (degrees, relative to the X-axis).
        output_folder (str): Path to the folder where the output image will be saved.
        display (bool, optional): If True, the figure will be displayed after saving (default is False).

    Returns:
        None: Saves the domain plot as '{region}_domain.png' in the specified output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{region}_domain.png")

    geo_corners = generarate_geo_corners(model_lon, model_lat, x_length, y_length, model_rot)

    min_lon = min(lon for lon, lat in geo_corners) - 0.2
    max_lon = max(lon for lon, lat in geo_corners) + 0.2
    min_lat = min(lat for lon, lat in geo_corners) - 0.2
    max_lat = max(lat for lon, lat in geo_corners) + 0.2

    # Select projection
    crs_corrected = select_projection(model_lon, model_lat, x_length, y_length)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': crs_corrected})
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.set_aspect('auto')

    imagery = cimgt.GoogleTiles(style="satellite")
    ax.add_image(imagery, 9, interpolation='nearest')

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True

    gl.xlabel_style = {'size': 8, 'color': 'black', 'rotation': 0}
    gl.ylabel_style = {'size': 8, 'color': 'black', 'rotation': 0}

    gl.xformatter = matplotlib.ticker.FuncFormatter(
        lambda x, pos: f'{x:.1f}°E' if x > 0 else f'{x:.1f}°W'
    )
    gl.yformatter = matplotlib.ticker.FuncFormatter(
        lambda y, pos: f'{y:.1f}°N' if y > 0 else f'{y:.1f}°S'
    )

    ax.set_xlabel("Longitude (°)", fontsize=8, labelpad=20)
    ax.set_ylabel("Latitude (°)", fontsize=8, labelpad=20)

    lons, lats = zip(*geo_corners + [geo_corners[0]])
    ax.plot(lons, lats, 'b-', linewidth=2, transform=ccrs.PlateCarree())

    plt.savefig(output_file, dpi=500, bbox_inches='tight')

    if display:
        plt.show()

def plot_modelbox_with_folium(region, model_lon, model_lat, x_length, y_length, model_rot, output_folder, display=False):
    """
    Plots the rotated model domain box on an interactive Folium map and saves it as HTML and KML files.

    Parameters:
        region (str): Name of the region to use in the output filenames.
        model_lon (float): Longitude of the model center (degrees).
        model_lat (float): Latitude of the model center (degrees).
        x_length (float): Length of the model in the x-direction (km).
        y_length (float): Length of the model in the y-direction (km).
        model_rot (float): Rotation angle of the model (degrees, from North to local x-axis).
        output_folder (str): Path to the folder where output files will be saved.
        display (bool, optional): If True, returns the Folium map for display (default is False).

    Returns:
        Folium.Map (optional): The interactive map object, only if display=True.

    Outputs:
        Saves a KML file named '{region}_domain.kml' in the specified output folder.
        Saves an interactive HTML map named '{region}_domain.html' in the specified output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    full_filename = os.path.join(output_folder, f"{region}_domain")
    geo_corners = generarate_geo_corners(model_lon, model_lat, x_length, y_length, model_rot)

    # Mapa base Folium
    m = folium.Map(location=[model_lat, model_lon], zoom_start=5)

    # KML
    kml = simplekml.Kml()

    folium.Polygon(
        locations=[(lat, lon) for lon, lat in geo_corners],
        color="blue",
        fill=True,
        fill_opacity=0.2,
        popup="Model Box"
    ).add_to(m)

    # En KML (lon, lat)
    kml_poly = kml.newpolygon(
        name="Model Box",
        outerboundaryis=geo_corners
    )

    kml_poly.style.linestyle.color = simplekml.Color.blue
    kml_poly.style.linestyle.width = 2
    kml_poly.style.polystyle.color = simplekml.Color.changealphaint(80, simplekml.Color.blue)

    for i, (lon, lat) in enumerate(geo_corners):
        label = f"C{i+1}"
        folium.Marker(
            tooltip=label,
            location=[lat, lon],
            popup=f"({lat:.4f}, {lon:.4f})",
            icon=folium.Icon(color="red")
        ).add_to(m)

        kml.newpoint(
            name=label,
            coords=[(lon, lat)]
        ).description = f"Corner {label}: ({lat:.4f}, {lon:.4f})"

    bounds = [(lat, lon) for lon, lat in geo_corners]
    m.fit_bounds(bounds)

    kml.save(full_filename + ".kml")
    print(f"KML file saved to {full_filename + '.kml'}")

    m.save(full_filename + ".html")
    print(f"HTML map saved to {full_filename + '.html'}")

    if display:
        return m

def plot_and_export_grid_map(output_folder, region, model_coords_file, display=False):
    """
    Plots the grid points from the model coordinates file on an interactive Folium map
    and exports the result to HTML and KML formats.

    Parameters:
        output_folder (str): Path to the folder where the output files will be saved.
        region (str): Name of the region, used in the output filenames.
        model_coords_file (str): Path to the file containing the grid coordinates
          (generated by the create_model_coords_file function).
        display (bool, optional): If True, returns the Folium map for display (default is False).

    Returns:
        Folium.Map (optional): The interactive map object, only if display=True.

    Outputs:
        Saves an interactive HTML map named '{region}_grid.html' in the specified output folder.
        Saves a KML file named '{region}_grid.kml' with all the grid points in the specified output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Full path for the output file
    full_filename = os.path.join(output_folder, f"{region}_grid")
    
    # Read grid data from the grid coordinates created with the function create_model_coords_file
    grid_data = []
    with open(model_coords_file, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue  # Skip comments
            parts = line.strip().split()
            if len(parts) == 4:
                lon, lat, x_index, y_index = map(float, parts[:4])
                grid_data.append((lon, lat, int(x_index), int(y_index)))

    # Extract longitude and latitude for plotting
    lons = [point[0] for point in grid_data]
    lats = [point[1] for point in grid_data]

    # Create a Folium map centered on the grid
    center_lat = (max(lats) + min(lats)) / 2
    center_lon = (max(lons) + min(lons)) / 2
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=1)

    # Add grid points to the map
    for lon, lat, x_idx, y_idx in grid_data:
        popup_content = f"""
        <strong>Grid Indices:</strong> ({x_idx}, {y_idx})<br>
        <strong>Coordinates:</strong> ({lat:.4f}, {lon:.4f})
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(folium_map)

    # Adjust the map view to fit the model box polygon
    min_lat = min(point[1] for point in grid_data)  # Minimum latitude
    max_lat = max(point[1] for point in grid_data)  # Maximum latitude
    min_lon = min(point[0] for point in grid_data)  # Minimum longitude
    max_lon = max(point[0] for point in grid_data)  # Maximum longitude
    folium_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    # Save the map as an HTML file
    folium_map.save(full_filename + ".html")
    print(f"HTML map saved to {full_filename + '.html'}")
    
    # Create a KML file with the grid points
    kml = simplekml.Kml()
    for lon, lat, x_idx, y_idx in grid_data:
        # Add a point with detailed description
        point = kml.newpoint(
            name=f"Grid Point ({x_idx}, {y_idx})",
            coords=[(lon, lat)]
        )
        point.description = f"""
        Grid Indices: ({x_idx}, {y_idx})<br>
        lat,lon: ({lat:.4f}, {lon:.4f})
        """
    kml.save(full_filename + ".kml")
    print(f"KML file saved to {full_filename + '.kml'}")
    
    # Show the map
    if display:
        return folium_map
    
#AR: Nueva función que interpola y escribe el binario slab a slab (capa por capa en profundidad)
#AR: para evitar materializar los arrays NX*NY*NZ completos en memoria, que causaban OOM en dominios grandes.
#AR: En cada iteración solo se tiene en RAM una slab (NX*NY puntos), no el dominio 3D completo.
#AR: (2026-10-09) fmt and h_fraction added, the two choices the CyberShake writer already had
#AR: (write_awp_media_from_scec_grid below), so the standalone program can make them too.
#AR:   fmt         "awp" (default) or "awpz", how Vp is set at the surface points whose Vs is
#AR:               raised to the minimum. awp keeps the point's own Vp/Vs. awpz uses the Vp/Vs
#AR:               of the point below, capped at MAX_SURFACE_VPVS.
#AR:   h_fraction  depth of the surface layer as a fraction of the grid spacing. The default,
#AR:               0.25, is what this function always did (grid_spacing / 4).
#AR: With the defaults the file written is the same as before, byte for byte.
def interpolate_and_save_streaming(model_coords_file, velocity_model, region, nx, ny, nz,
                                   grid_spacing, is_anelastic, output_folder,
                                   fmt="awp", h_fraction=0.25):
    """
    Interpolate the 3D velocity model and write the binary output one depth slab at a time.

    This avoids materializing the full NX*NY*NZ coordinate and result arrays in memory,
    which would require hundreds of GB for large domains. At any point only one depth
    slab (NX*NY points) is in RAM.
    """
    if fmt not in SURFACE_FORMATS:
        raise ValueError("format must be one of %s, got %r" % (", ".join(SURFACE_FORMATS), fmt))
    if not 0.0 <= float(h_fraction) < 1.0:
        raise ValueError("h_fraction must be 0 or a fraction of the grid spacing below 1, got %r"
                         % h_fraction)
    os.makedirs(output_folder, exist_ok=True)
    Dh = int(grid_spacing * 1000)
    filename = os.path.join(output_folder, f"{region}_{nx}x{ny}x{nz}_Dh{Dh}m.bin")

    # Read 2D horizontal coordinates once (only NX*NY points, no depth tiling)
    lons_2d = []
    lats_2d = []
    x_idx_arr = []
    y_idx_arr = []
    with open(model_coords_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            lon, lat, xi, yi = parts
            lons_2d.append(float(lon))
            lats_2d.append(float(lat))
            x_idx_arr.append(int(float(xi)))
            y_idx_arr.append(int(float(yi)))

    lons_2d = np.array(lons_2d, dtype=np.float32)
    lats_2d = np.array(lats_2d, dtype=np.float32)
    x_idx_arr = np.array(x_idx_arr, dtype=np.int32)
    y_idx_arr = np.array(y_idx_arr, dtype=np.int32)
    nxy = nx * ny

    if len(lons_2d) != nxy:
        raise ValueError(
            f"model_coords has {len(lons_2d)} horizontal points, but NX*NY = {nxy}."
        )

    print(f"Streaming binary output to: {filename}")
    with open(filename, "wb") as out:
        for iz in tqdm(range(nz), desc="Interpolating slab", unit="slab"):
            z = iz * grid_spacing
            z_val = np.float32(grid_spacing * h_fraction) if z == 0 else np.float32(z)
            slab_depth = np.full(nxy, z_val, dtype=np.float32)
            slab_points = np.column_stack((lats_2d, lons_2d, slab_depth))

            vpvs_low = None
            if iz == 0 and nz > 1 and fmt == "awpz":
                below = velocity_model.query(
                    np.column_stack((lats_2d, lons_2d,
                                     np.full(nxy, np.float32(grid_spacing), dtype=np.float32))),
                    verbose=False, raw=True)
                with np.errstate(divide='ignore', invalid='ignore'):
                    vpvs_low = np.where(below['vs'] > 0, below['vp'] / below['vs'],
                                        MAX_SURFACE_VPVS)
                vpvs_low = np.minimum(vpvs_low, MAX_SURFACE_VPVS)
                del below

            # verbose=False suprime el log y quality check por cada slab
            slab_vals = velocity_model.query(slab_points, verbose=False, vpvs_low=vpvs_low)

            # Reorder from model_coords (RWG) order to AWP (y,x) layout
            vp2 = np.empty((ny, nx), dtype=np.float32)
            vs2 = np.empty((ny, nx), dtype=np.float32)
            rh2 = np.empty((ny, nx), dtype=np.float32)
            vp2[x_idx_arr, y_idx_arr] = slab_vals['vp']
            vs2[x_idx_arr, y_idx_arr] = slab_vals['vs']
            rh2[x_idx_arr, y_idx_arr] = slab_vals['rho']

            if is_anelastic:
                qp2 = np.empty((ny, nx), dtype=np.float32)
                qs2 = np.empty((ny, nx), dtype=np.float32)
                qp2[x_idx_arr, y_idx_arr] = slab_vals['qp']
                qs2[x_idx_arr, y_idx_arr] = slab_vals['qs']
                layer = np.stack([vp2, vs2, rh2, qp2, qs2], axis=-1)
            else:
                layer = np.stack([vp2, vs2, rh2], axis=-1)

            layer.ravel(order="C").tofile(out)

    print(f"Binary file created: {filename}")
    return filename
#AR: Fin de interpolate_and_save_streaming


#AR: (2026-09-17) Writes the AWP velocity mesh for CyberShake from the grid files that
#AR: SCEC's gen_grid.py and gen_model_cords produce, instead of from create_model_coords_file()
#AR: and create_gridout_file() of this package.
#AR: 
#AR: Why. I checked it against SCEC on 2026-09-17, and the coordinates of this branch were wrong.
#AR: After switching the corner function to gcproj, the corners come back in SCEC's order (C2 is
#AR: the end of the X side), but create_model_coords_file() expects c1 to c4 to be the X side.
#AR: So the nx points ended up along the Y side and the ny points along the X side, and a 1 km
#AR: grid was sampled every 0.417 km one way and every 2.409 km the other, with the four corners
#AR: still in the right place. The gridout also listed Y first, and the nodes sat on the corner
#AR: where SCEC puts them half a cell inside. Instead of patching each of those by hand,
#AR: CyberShake mode now takes gridout and model_coords straight from SCEC's tools, so they are
#AR: the same as in the UCVM route by construction. This function only samples the velocity
#AR: model at those points and writes the binary.
#AR: 
#AR: The binary follows exactly what UCVM writes for CyberShake. I checked it against the
#AR: awp.TEST.media of the TEST_25 run that AWP used.
#AR:   float32, Vp Vs rho per point, in m/s and kg/m3
#AR:   layer k=0 is the surface, queried at depth 0 m, and layer k at k*spacing
#AR:   inside a layer, rows follow the X index of the modelbox and each row runs along its Y
#AR:     index, so offset = (k*NX_mb*NY_mb + ix*NY_mb + iy)*3. That is what AWP reads with its
#AR:     NX = modelbox Y and NY = modelbox X (build_IN3D_GPU takes NY from the first block of
#AR:     gridout).
#AR:   the minimums and the ratio fix come from UCVM's ucvm-single_mpi.c, which
#AR:     VelocityModel3D.query already applies in the same order (apply_corrections_vectorized).
#AR: 
#AR: (2026-10-08) Two options of SCEC's mesher, with their names and defaults, so that a mesh
#AR: built here and one built by UCVM with the same options follow the same rules.
#AR: 
#AR: fmt, "awp" or "awpz". It changes one thing, the Vp given to a SURFACE point whose Vs is
#AR: raised to the minimum. Vp is always set to the minimum Vs times a Vp/Vs ratio.
#AR:   awp    the ratio is the one of the point, as the model gives it. Where the model has a
#AR:          very low Vs at the surface that ratio is large and Vp comes out far too high.
#AR:   awpz   the ratio is the one of the point right below, one grid point down, as the model
#AR:          gives it, and never more than 4. SCEC wrote this for Study 24.8
#AR:          (ucvm-single_mpi.c, "updated Vp/Vs ratio approach to avoid too-large Vp values").
#AR: Below the surface the two are the same. In UCVM awpz also changes how the work is split,
#AR: by columns, which is what lets it see the point below. Here the layer below is read first.
#AR: 
#AR: h_fraction is the depth at which the surface layer is sampled, as a fraction of the grid
#AR: spacing. 0, the default, samples it at depth 0, as SCEC does when --h_fraction is not
#AR: given. Study 24.8 used 0.25, 20 m for its 80 m grid, the middle of the half cell the
#AR: surface point stands for. The other layers stay at k times the spacing. The original
#AR: mesher of this repository always used a quarter of the spacing (Interp_velocity_3D.py).
SURFACE_FORMATS = ("awp", "awpz")
MAX_SURFACE_VPVS = 4.0


def write_awp_media_from_scec_grid(scec_model_coords_file, scec_gridout_file, velocity_model,
                                   output_file, fmt="awp", h_fraction=0.0):
    """Samples velocity_model on SCEC's grid and writes the AWP media file.

    scec_model_coords_file: gen_model_cords output, one line per surface point,
        "lon lat ix iy", ix along the modelbox X side, iy along the Y side.
    scec_gridout_file: gen_model_cords gridout. Read the same way SCEC's single_exe.py reads it,
        first block X count, second Y count, third Z count, then the Z positions in km.
    velocity_model: a VelocityModel3D already built from the model in m/s and kg/m3.
    fmt: "awp" or "awpz", how Vp is set at surface points raised to the minimum Vs.
    h_fraction: depth of the surface layer as a fraction of the grid spacing, 0 for depth 0.
    """
    if fmt not in SURFACE_FORMATS:
        raise ValueError("format must be one of %s, got %r" % (", ".join(SURFACE_FORMATS), fmt))
    if not 0.0 <= float(h_fraction) < 1.0:
        raise ValueError("h_fraction must be 0 or a fraction of the grid spacing below 1, got %r"
                         % h_fraction)
    with open(scec_gridout_file) as f:
        g = f.readlines()
    nx_mb = int(g[1].split("=")[1])
    ny_mb = int(g[1 + nx_mb + 2].split("=")[1])
    zblock = 1 + nx_mb + 2 + ny_mb + 2
    nz = int(g[zblock].split("=")[1])
    z_km = np.array([float(g[zblock + 1 + k].split()[1]) for k in range(nz)], dtype=np.float32)

    d = np.loadtxt(scec_model_coords_file, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 4 or d.shape[0] != nx_mb * ny_mb:
        raise ValueError("%s should have %d lines of 'lon lat ix iy', found shape %s"
                         % (scec_model_coords_file, nx_mb * ny_mb, d.shape))
    lons = d[:, 0].astype(np.float32)
    lats = d[:, 1].astype(np.float32)
    ix = d[:, 2].astype(np.int64)
    iy = d[:, 3].astype(np.int64)
    if ix.max() != nx_mb - 1 or iy.max() != ny_mb - 1:
        raise ValueError("Index ranges in %s (ix up to %d, iy up to %d) do not match gridout "
                         "(%d x %d)" % (scec_model_coords_file, ix.max(), iy.max(), nx_mb, ny_mb))
    del d

    nxy = nx_mb * ny_mb
    out_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_dir, exist_ok=True)
    print("Writing AWP media %s, %d x %d x %d (modelbox X, Y, Z)" % (output_file, nx_mb, ny_mb, nz))
    with open(output_file, "wb") as out:
        for k in tqdm(range(nz), desc="Interpolating slab", unit="slab"):
            depth_k = z_km[k]
            vpvs_low = None
            if k == 0 and nz > 1:
                if h_fraction > 0.0:
                    depth_k = np.float32(h_fraction * (z_km[1] - z_km[0]))
                if fmt == "awpz":
                    below = velocity_model.query(
                        np.column_stack((lats, lons, np.full(nxy, z_km[1], dtype=np.float32))),
                        verbose=False, raw=True)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        vpvs_low = np.where(below["vs"] > 0, below["vp"] / below["vs"],
                                            MAX_SURFACE_VPVS)
                    vpvs_low = np.minimum(vpvs_low, MAX_SURFACE_VPVS)
                    del below
                print("Surface layer sampled at %.1f m, format %s" % (depth_k * 1000.0, fmt))
            pts = np.column_stack((lats, lons, np.full(nxy, depth_k, dtype=np.float32)))
            vals = velocity_model.query(pts, verbose=False, vpvs_low=vpvs_low)
            for key in ("vp", "vs", "rho"):
                v = vals[key]
                if not np.all(np.isfinite(v)) or np.any(v <= 0):
                    malos = int(np.sum(~np.isfinite(v) | (v <= 0)))
                    raise ValueError(
                        "Layer %d (depth %.3f km) has %d points with a non-positive or non-finite "
                        "%s. UCVM stops in this case too. Most likely the domain reaches cells "
                        "with no data (fill values) in the velocity model." % (k, z_km[k], malos, key))
            layer = np.empty((nx_mb, ny_mb, 3), dtype=np.float32)
            layer[ix, iy, 0] = vals["vp"]
            layer[ix, iy, 1] = vals["vs"]
            layer[ix, iy, 2] = vals["rho"]
            layer.ravel(order="C").tofile(out)
    return output_file


def check_model_has_no_fill_values(model_data, lats, lons, max_depth_km, fill_threshold=-9999.0):
    """Stops if the velocity model has no-data values anywhere the domain will sample.

    #AR: The CVM-S5 NetCDF uses -99999 where it has no data. Interpolating across those cells
    #AR: gives meaningless values that can still come out positive after the harmonic average, so
    #AR: checking the interpolated output alone is not enough. This looks at the model cells that
    #AR: surround the domain (one extra cell on each side) before anything is interpolated.
    """
    mlat = np.asarray(model_data["latitude"])
    mlon = np.asarray(model_data["longitude"])
    mdep = np.asarray(model_data["depth"])

    def rango(eje, lo, hi):
        i0 = max(0, int(np.searchsorted(eje, lo, side="right")) - 2)
        i1 = min(len(eje), int(np.searchsorted(eje, hi, side="left")) + 2)
        return slice(i0, i1)

    sl = (rango(mlat, float(np.min(lats)), float(np.max(lats))),
          rango(mlon, float(np.min(lons)), float(np.max(lons))),
          rango(mdep, 0.0, float(max_depth_km)))
    if float(np.min(lats)) < mlat.min() or float(np.max(lats)) > mlat.max() or \
       float(np.min(lons)) < mlon.min() or float(np.max(lons)) > mlon.max() or \
       float(max_depth_km) > mdep.max():
        raise ValueError(
            "The domain goes outside the velocity model's coverage (lat %.3f..%.3f, lon %.3f..%.3f, "
            "depth to %.1f km, model covers lat %.3f..%.3f, lon %.3f..%.3f, depth to %.1f km)."
            % (np.min(lats), np.max(lats), np.min(lons), np.max(lons), max_depth_km,
               mlat.min(), mlat.max(), mlon.min(), mlon.max(), mdep.max()))
    for key in ("vp", "vs", "rho"):
        sub = np.asarray(model_data[key])[sl]
        n = int(np.sum(sub <= fill_threshold))
        if n:
            raise ValueError(
                "The velocity model has %d no-data cells (%s <= %g) around the domain. Pick a "
                "domain inside the model's data coverage or fill those cells first."
                % (n, key, fill_threshold))
#AR: Fin de write_awp_media_from_scec_grid


#AR: Nueva función para el modelo 1D. A diferencia de interpolate_and_save_streaming
#AR: (3D), aquí el modelo solo depende de Z. Todos los NX*NY puntos a la misma profundidad
#AR: tienen EXACTAMENTE el mismo valor, por eso el interpolador se evalúa solo NZ veces (en
#AR: lugar de NX*NY*NZ). Para un dominio grande (por ejemplo 5000x5000x600) se pasa de 15 mil
#AR: millones de evaluaciones a 600. El binario se escribe capa a capa con np.full, sin
#AR: necesidad de model_coords.
def interpolate_1d_and_save_streaming(model_data, region, nx, ny, nz,
                                       grid_spacing, is_anelastic, output_folder):
    """
    Interpolate a 1D velocity model at NZ unique depth values and write the binary
    output one depth slab at a time.

    Since a 1D model depends only on depth, all NX*NY surface points at a given depth
    share identical values. This function evaluates interpolators only NZ times (not
    NX*NY*NZ) and broadcasts each scalar to a (NY, NX) slab via np.full.

    Parameters:
        model_data (dict): 1D model dict with 'depth', 'vp', 'vs', 'rho' (and optionally 'qp', 'qs').
        region (str): Region name for the output filename.
        nx, ny, nz (int): Grid dimensions.
        grid_spacing (float): Grid spacing in km.
        is_anelastic (bool): Whether to write Qp/Qs columns.
        output_folder (str): Folder for the output binary file.

    Returns:
        str: Path to the binary file.
    """
    os.makedirs(output_folder, exist_ok=True)
    Dh = int(grid_spacing * 1000)
    filename = os.path.join(output_folder, f"{region}_{nx}x{ny}x{nz}_Dh{Dh}m.bin")

    # NZ unique depth values, the ONLY points where the interpolator is evaluated
    depth_vals = np.arange(nz, dtype=np.float64) * grid_spacing
    depth_vals[0] = grid_spacing / 4  # evitar z=0 (problema en superficie)

    # Construir interpoladores y evaluar en NZ profundidades únicas
    depth_model = np.array(model_data['depth'], dtype=np.float64)
    vp_interp  = harmonic_interp1d(depth_model, np.array(model_data['vp'],  dtype=np.float64))
    vs_interp  = harmonic_interp1d(depth_model, np.array(model_data['vs'],  dtype=np.float64))
    rho_interp = interp1d(depth_model, np.array(model_data['rho'], dtype=np.float64),
                          kind='linear', bounds_error=True, fill_value=None, assume_sorted=True)

    vp_1d  = vp_interp(depth_vals).astype(np.float32)
    vs_1d  = vs_interp(depth_vals).astype(np.float32)
    rho_1d = rho_interp(depth_vals).astype(np.float32)

    if is_anelastic:
        qp_interp = interp1d(depth_model, np.array(model_data['qp'], dtype=np.float64),
                              kind='linear', bounds_error=True, fill_value=None, assume_sorted=True)
        qs_interp = interp1d(depth_model, np.array(model_data['qs'], dtype=np.float64),
                              kind='linear', bounds_error=True, fill_value=None, assume_sorted=True)
        qp_1d = qp_interp(depth_vals).astype(np.float32)
        qs_1d = qs_interp(depth_vals).astype(np.float32)

    print(f"Streaming binary output to: {filename}")
    with open(filename, "wb") as out:
        for iz in tqdm(range(nz), desc="Writing 1D slab", unit="slab"):
            # Todos los puntos (y,x) del slab reciben el mismo valor escalar
            vp2 = np.full((ny, nx), vp_1d[iz],  dtype=np.float32)
            vs2 = np.full((ny, nx), vs_1d[iz],  dtype=np.float32)
            rh2 = np.full((ny, nx), rho_1d[iz], dtype=np.float32)

            if is_anelastic:
                qp2 = np.full((ny, nx), qp_1d[iz], dtype=np.float32)
                qs2 = np.full((ny, nx), qs_1d[iz], dtype=np.float32)
                layer = np.stack([vp2, vs2, rh2, qp2, qs2], axis=-1)
            else:
                layer = np.stack([vp2, vs2, rh2], axis=-1)

            layer.ravel(order="C").tofile(out)

    print(f"Binary file created: {filename}")
    return filename
#AR: Fin de interpolate_1d_and_save_streaming


def save_binary_model(region, NX, NY, NZ, Dhkm, z, Vp, Vs, Rho, output_folder,
                      Qp=None, Qs=None, elastic=True):
    """
    Escribe binario AWP correcto considerando que Vp/Vs/Rho vienen en el orden del model_coords_file (RWG).

    - Lee *.model_coords_<region> desde output_folder
    - Reordena cada capa usando (x_index, y_index)
    - Escribe layout AWP: [z][y][x][var] float32 (x más rápido)
    """

    os.makedirs(output_folder, exist_ok=True)
    Dh = int(Dhkm * 1000)

    # nombre (solo etiqueta)
    filename = os.path.join(output_folder, f"{region}_{NX}x{NY}x{NZ}_Dh{Dh}m.bin")

    nvar = 3 if elastic else 5
    if not elastic and (Qp is None or Qs is None):
        raise ValueError("If elastic=False, you must provide Qp and Qs.")

    # --- encontrar model_coords file ---
    # (se genera en main como: <domain_data>.model_coords_<Region>)
    pattern = os.path.join(output_folder, "*.model_coords_*")
    matches = glob.glob(pattern)
    
    if not matches:
        # Raise an error if no model_coords file is found in the output folder.
        raise FileNotFoundError(
            f"No model_coords file found in: {output_folder}\n"
            f"(Searched using pattern: {pattern})"
        )

    if len(matches) > 1:
    # Raise an error if multiple model_coords files are found.
    # Only one should exist for the current simulation run.
        raise RuntimeError(
            f"More than one model_coords file found: {matches}. "
            f"Please remove the files that do not correspond to this run."
        )

    # Si hay varios coordfiles, tomar el primero pero esto puede generar error si no corresponde con la corrida
    # Aunque al inicio de la corrida se borra el ouput folder para evitar este error
    model_coords_file = matches[0]
    print(f"Using model_coords: {model_coords_file}")

    # --- leer x_index, y_index en el orden ORIGINAL (el mismo orden que usó GridCoords_LatLonDepth) ---
    x_idx = np.empty(NX * NY, dtype=np.int32)
    y_idx = np.empty(NX * NY, dtype=np.int32)

    p = 0
    with open(model_coords_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            # lon lat x_index y_index  (según GridCoords_LatLonDepth) 
            _, _, xi, yi = parts
            x_idx[p] = int(float(xi))
            y_idx[p] = int(float(yi))
            p += 1

    if p != NX * NY:
        raise ValueError(
            f"model_coords has {p} horizontal points, but NX*NY = {NX*NY}."
        )

    # --- asegurar float32 1D ---
    Vp = np.asarray(Vp, dtype=np.float32).ravel()
    Vs = np.asarray(Vs, dtype=np.float32).ravel()
    Rho = np.asarray(Rho, dtype=np.float32).ravel()
    if not elastic:
        Qp = np.asarray(Qp, dtype=np.float32).ravel()
        Qs = np.asarray(Qs, dtype=np.float32).ravel()

    expected = NX * NY * NZ
    if Vp.size != expected or Vs.size != expected or Rho.size != expected:
        raise ValueError(
            f"Vp/Vs/Rho array size does not match NX*NY*NZ = {expected}."
        )
    if not elastic and (Qp.size != expected or Qs.size != expected):
        raise ValueError(
            f"Qp/Qs array size does not match NX*NY*NZ = {expected}."
        )

    nxy = NX * NY

    # --- escribir por capas, reordenando con (x_idx,y_idx) ---
    with open(filename, "wb") as out:
        for iz in range(NZ):
            s = iz * nxy
            e = s + nxy

            # capa en el ORDEN ORIGINAL (archivo model_coords)
            vp0 = Vp[s:e]
            vs0 = Vs[s:e]
            rh0 = Rho[s:e]

            # reordenar a (NY,NX) en orden AWP (y,x)
            vp2 = np.empty((NY, NX), dtype=np.float32)
            vs2 = np.empty((NY, NX), dtype=np.float32)
            rh2 = np.empty((NY, NX), dtype=np.float32)

            #print("vp2 shape:", vp2.shape)
            #print("y_idx min/max:", y_idx.min(), y_idx.max())
            #print("x_idx min/max:", x_idx.min(), x_idx.max())
            #print("NY,NX:", NY, NX)
            #print("vp0 size:", vp0.size, "idx size:", y_idx.size, x_idx.size)

            vp2[x_idx, y_idx] = vp0
            vs2[x_idx, y_idx] = vs0
            rh2[x_idx, y_idx] = rh0

            if elastic:
                layer = np.stack([vp2, vs2, rh2], axis=-1)  # (NY,NX,3)
            else:
                qp0 = Qp[s:e]
                qs0 = Qs[s:e]
                qp2 = np.empty((NY, NX), dtype=np.float32)
                qs2 = np.empty((NY, NX), dtype=np.float32)
                qp2[x_idx, y_idx] = qp0
                qs2[x_idx, y_idx] = qs0
                layer = np.stack([vp2, vs2, rh2, qp2, qs2], axis=-1)  # (NY,NX,5)

            # AWP espera x más rápido => ravel(order="C") sobre (NY,NX,nvar)
            layer.ravel(order="C").tofile(out)

    print(f"Binary file created: {filename}")
    return filename

def plot_surface_XY(binary_file_path, variable_name, index, num_x, num_y, num_z, variable_names, output_folder, plane_type='XY', x_coords=None, y_coords=None, z_coords=None, display=False):
    """
    Plots a horizontal (XY) slice of a selected variable from a binary model file and exports the figure.

    Parameters:
        binary_file_path (str): Path to the binary model file.
        variable_name (str): Name of the variable to plot (must exist in variable_names).
        index (int): Index of the Z layer to extract (depth index).
        num_x (int): Number of grid points in the x-direction.
        num_y (int): Number of grid points in the y-direction.
        num_z (int): Number of grid points in the z-direction.
        variable_names (list of str): List of variable names in the binary file (e.g., ['Vp', 'Vs', 'Rho']).
        output_folder (str): Folder path where the figure will be saved.
        plane_type (str, optional): Type of plane to plot ('XY' supported). Defaults to 'XY'.
        x_coords (array-like, optional): coordinates for the x-axis. Defaults to index-based coordinates.
        y_coords (array-like, optional): coordinates for the y-axis. Defaults to index-based coordinates.
        z_coords (array-like, optional): coordinates for the z-axis (needed to label depth). Defaults to index-based coordinates.
        display (bool, optional): If True, displays the figure. Defaults to False.

    Returns:
        None

    Outputs:
        Saves a figure as '{variable_name}_{plane_type}_Depth_{depth_value}km.svg' in the specified output folder.
        Prints the path to the saved figure.
        """
    
    # Verificar si la variable existe
    if variable_name not in variable_names:
        raise ValueError(f"The variable '{variable_name}' is not in the list of variables.")

    # Determinar la posición de la variable en la lista y numero total de variables
    var_index = variable_names.index(variable_name)
    num_variables = len(variable_names)

    #AR: Reemplazado np.fromfile (cargaba todo el binario en RAM) por np.memmap +
    #AR: extracción directa del slab solicitado. Solo se carga en RAM la capa pedida,
    #AR: no el archivo completo. Esto permite trabajar con archivos de cientos de GB.
    expected_size = num_x * num_y * num_z * num_variables
    data = np.memmap(binary_file_path, dtype=np.float32, mode='r')

    # Validar el tamaño de los datos
    if data.size != expected_size:
        raise ValueError("The data size does not match the provided dimensions.")

    # Definir coordenadas físicas si no se proporcionan
    if x_coords is None:
        x_coords = np.linspace(0, num_x - 1, num_x)
    if y_coords is None:
        y_coords = np.linspace(0, num_y - 1, num_y)
    if z_coords is None:
        z_coords = np.linspace(0, num_z - 1, num_z)

    # Seleccionar y graficar el plano correspondiente
    if plane_type == 'XY':
        if not (0 <= index < num_z):
            raise IndexError("The Z index is out of the valid range.")

        # Extraer solo el slab de profundidad solicitado (no toda la variable)
        slab_start = index * num_y * num_x * num_variables
        slab_raw = np.array(data[slab_start:slab_start + num_y * num_x * num_variables])
        del data
        plane_data = slab_raw[var_index::num_variables].reshape(num_y, num_x)
        var_min = float(plane_data.min())
        var_max = float(plane_data.max())
        del slab_raw

        # Graficar
        plt.figure(figsize=(8, 6))
        im = plt.imshow(plane_data, cmap='jet', origin='lower', aspect='equal',
                        extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]], 
                        vmin=var_min, vmax=var_max) 
        plt.colorbar(
            label=(
                f'{variable_name} [m/s]' if variable_name in ['Vp', 'Vs'] else
                f'{variable_name} [$kg/m^3$]' if variable_name == 'Rho' else
                ''
            ),
            shrink=0.4, pad=0.025
        )
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        
    else:
        raise ValueError("The plane type must be 'XY'.")

    # Coordenada correspondiente 
    coord_sorted = sorted(set(z_coords))
    coord_name = coord_sorted[index]
    
    # Guardar la figura
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f'{variable_name}_{plane_type}_DepthIndex_{index}.svg')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Mostrar la figura dependiendo del flag
    if display:
        plt.show()
    
    print(f"Figure saved in: {output_path}")

def plot_surface_XZorYZ(binary_file_path, variable_name, index, num_x, num_y, num_z, variable_names, output_folder, plane_type='XZ', x_coords=None, y_coords=None, z_coords=None, display=False):
    """
    Plots a vertical (XZ or YZ) slice of a selected variable from a binary model file and exports the figure.

    Parameters:
        binary_file_path (str): Path to the binary model file.
        variable_name (str): Name of the variable to plot (must exist in variable_names).
        index (int): Index of the Y coordinate (for plane_type='YZ') or X coordinate (for plane_type='XZ') to extract the vertical slice.
        num_x (int): Number of grid points in the x-direction.
        num_y (int): Number of grid points in the y-direction.
        num_z (int): Number of grid points in the z-direction.
        variable_names (list of str): List of variable names in the binary file (e.g., ['Vp', 'Vs', 'Rho']).
        output_folder (str): Folder path where the figure will be saved.
        plane_type (str, optional): Type of vertical plane to plot ('XZ' or 'YZ'). Defaults to 'XZ'.
        x_coords (array-like, optional): coordinates for the x-axis. Defaults to index-based coordinates.
        y_coords (array-like, optional): coordinates for the y-axis. Defaults to index-based coordinates.
        z_coords (array-like, optional): coordinates for the z-axis (depth). Defaults to index-based coordinates.
        display (bool, optional): If True, displays the figure. Defaults to False.

    Returns:
        None

    Outputs:
        Saves a figure as '{variable_name}_{plane_type}_ycoord_{coord_value}km.svg' in the specified output folder.
        Prints the path to the saved figure.
    """

    # Verificar si la variable existe
    if variable_name not in variable_names:
        raise ValueError(f"The variable '{variable_name}' is not in the list of variables.")

    # Determinar la posición de la variable en la lista
    var_index = variable_names.index(variable_name)

    # Calcular el número total de variables
    num_variables = len(variable_names)

    #AR: Reemplazado np.fromfile + loops Python dobles por np.memmap con lectura
    #AR: por filas/columnas. Evita cargar el binario completo en RAM y elimina el
    #AR: cuello de botella del doble bucle Python para dominios grandes.
    expected_size = num_x * num_y * num_z * num_variables
    data = np.memmap(binary_file_path, dtype=np.float32, mode='r')

    # Validar el tamaño de los datos
    if data.size != expected_size:
        raise ValueError("The data size does not match the provided dimensions.")

    # Definir coordenadas físicas si no se proporcionan
    if y_coords is None:
        x_coords = np.linspace(0, num_x - 1, num_x)
    if x_coords is None:
        y_coords = np.linspace(0, num_y - 1, num_y)
    if z_coords is None:
        z_coords = np.linspace(0, num_z - 1, num_z)

    # Seleccionar y graficar el plano correspondiente
    if plane_type == 'YZ':
        if not (0 <= index < num_y):
            raise IndexError("The X index is out of the valid range.")

        # Extraer el plano YZ (y=index) leyendo una fila contigua por cada nivel z
        plane_data = np.empty((num_z, num_x), dtype=np.float32)
        for iz in range(num_z):
            base = iz * num_y * num_x * num_variables + index * num_x * num_variables
            row = np.array(data[base:base + num_x * num_variables])
            plane_data[iz, :] = row[var_index::num_variables]
        del data
        var_min = float(plane_data.min())
        var_max = float(plane_data.max())

        # Graficar
        plt.figure(figsize=(8, 6))
        plt.imshow(plane_data, cmap='jet', origin='lower', aspect='equal',
                   extent=[y_coords[0], y_coords[-1], z_coords[0], z_coords[-1]],
                   vmin=var_min, vmax=var_max)
        plt.colorbar(
            label=(
            f'{variable_name} [m/s]' if variable_name in ['Vp', 'Vs'] else
            f'{variable_name} [$kg/m^3$]' if variable_name == 'Rho' else
            ''
                ),
                shrink=(num_z/num_x), pad=0.015
        )
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('Depth [km]')
        plt.gca().invert_yaxis()
        
        # Coordenada correspondiente para nombre de la figura, como RWG: Y es X y X es Y
        coordSorted= sorted(set(y_coords))
        coordName=coordSorted[index]
        figname=f'{variable_name}_{plane_type}_xcoordIndex_{index}.svg'
    
    elif plane_type == 'XZ':
        if not (0 <= index < num_x):
            raise IndexError("The Y index is out of the valid range.")

        #AR: Reemplazado el doble loop Python por indexación vectorizada en el memmap.
        #AR: Para cada nivel z, se leen las posiciones (iz, iy, x=index) usando fancy
        #AR: indexing con paso num_x*num_variables, evitando cargar todo el array.
        plane_data = np.empty((num_z, num_y), dtype=np.float32)
        for iz in range(num_z):
            base_z = iz * num_y * num_x * num_variables
            positions = (base_z
                         + np.arange(num_y, dtype=np.int64) * num_x * num_variables
                         + index * num_variables
                         + var_index)
            plane_data[iz, :] = data[positions]
        del data
        var_min = float(plane_data.min())
        var_max = float(plane_data.max())

        # Graficar
        plt.figure(figsize=(8, 6))
        plt.imshow(plane_data, cmap='jet', origin='lower', aspect='equal',
                   extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                   vmin=var_min, vmax=var_max)
        plt.colorbar(
            label=(
            f'{variable_name} [m/s]' if variable_name in ['Vp', 'Vs'] else
            f'{variable_name} [$kg/m^3$]' if variable_name == 'Rho' else
            ''
                ),
                shrink=(num_z/num_y), pad=0.015
        )
        plt.xlabel('y-coordinate [km]')
        plt.ylabel('Depth [km]')
        plt.gca().invert_yaxis()

        # Coordenada correspondiente para nombre de la figura, como RWG: Y es X y X es Y
        coordSorted= sorted(set(x_coords))
        coordName=coordSorted[index]
        figname=f'{variable_name}_{plane_type}_ycoordIndex_{index}.svg'
    
    else:
        raise ValueError("The plane type must be 'XZ' or 'YZ'.")

    # Guardar la figura
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, figname)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    if display:
        plt.show()

    print(f"Figure saved in: {output_path}")


def parse_1d_model(file_path, is_anelastic=False):
    """
    Parses a 1D velocity model from a text file (CSV, TSV, or space-separated) and handles optional anelastic parameters.

    Parameters:
        file_path (str): Path to the text file containing the 1D model.
        is_anelastic (bool, optional): If True, expects the model to include anelastic parameters (Qp, Qs). Defaults to False.

    Returns:
        dict: A dictionary with the following keys:
            'depth' (np.ndarray): Depth values (typically in km).
            'vp' (np.ndarray): P-wave velocity values (m/s).
            'vs' (np.ndarray): S-wave velocity values (m/s).
            'rho' (np.ndarray): Density values (kg/mÂ³).
            'qp' (np.ndarray, optional): P-wave quality factor (only if is_anelastic=True).
            'qs' (np.ndarray, optional): S-wave quality factor (only if is_anelastic=True).

    Notes:
        The function auto-detects the delimiter (comma, tab, or space).
        The column order is read from the header row (column names like "depth",
        "vp", "vs", "rho"/"density", "qp", "qs", with or without a trailing unit in
        parentheses, e.g. "vp(km/s)"). It does NOT assume a fixed column order.
        If no recognizable header is found, falls back to the historical fixed order
        (depth, vp, vs, rho, [qp, qs]).
        Rows with invalid or incomplete data are ignored.
    """
    #AR: (2026-07-16) The column order is now read from the header row (for example
    #AR: "depth(km) rho(g/cm3) vp(km/s) vs(km/s)") instead of assuming it is always
    #AR: (depth, vp, vs, rho). An Iceland input file had rho before vp and vs in its header, and the
    #AR: old fixed order read it wrong without any warning, so vp got the values of rho, vs got the
    #AR: values of vp and rho got the values of vs. It only goes back to the old fixed order when
    #AR: the file has no header it can recognize, so input files without a header still work.
    
    
    KNOWN_COLUMNS = {
        'depth': 'depth',
        'vp': 'vp',
        'vs': 'vs',
        'rho': 'rho',
        'density': 'rho',
        'qp': 'qp',
        'qs': 'qs',
    }

    def _base_name(token):
        # #AR: Strip a trailing unit in parentheses, e.g. "rho(g/cm3)" -> "rho"
        return token.split('(')[0].strip().lower()

    data = defaultdict(list)

    with open(file_path, 'r') as file:
        raw_lines = file.readlines()

    non_comment_lines = [line for line in raw_lines if line.strip() and not line.strip().startswith('%')]
    if not non_comment_lines:
        raise ValueError(f"No data found in 1D velocity model file: {file_path}")

    first_line = non_comment_lines[0].strip()
    delimiter = ',' if ',' in first_line else '\t' if '\t' in first_line else None  # Auto-detect

    if delimiter:
        rows = list(csv.reader(non_comment_lines, delimiter=delimiter))
    else:
        rows = [line.split() for line in non_comment_lines]

    # #AR: Try to read the column order from the header row.
    header_tokens = [_base_name(tok) for tok in rows[0]]
    column_index = {}
    for i, tok in enumerate(header_tokens):
        if tok in KNOWN_COLUMNS:
            column_index[KNOWN_COLUMNS[tok]] = i

    required = {'depth', 'vp', 'vs', 'rho'}
    if is_anelastic:
        required = required | {'qp', 'qs'}

    if required.issubset(column_index.keys()):
        # #AR: Header recognized, so use its column order and skip it as a data row.
        data_rows = rows[1:]
    else:
        # #AR: No header recognized, so go back to the old fixed order
        # #AR: and treat every row (the first one too) as data.
        column_index = {'depth': 0, 'vp': 1, 'vs': 2, 'rho': 3}
        if is_anelastic:
            column_index.update({'qp': 4, 'qs': 5})
        data_rows = rows

    for row in data_rows:
        try:
            depth_val = float(row[column_index['depth']])
            vp_val = float(row[column_index['vp']])
            vs_val = float(row[column_index['vs']])
            rho_val = float(row[column_index['rho']])
        except (IndexError, ValueError):
            # Skip bad rows
            continue

        data['depth'].append(depth_val)
        data['vp'].append(vp_val)
        data['vs'].append(vs_val)
        data['rho'].append(rho_val)

        if is_anelastic:
            try:
                data['qp'].append(float(row[column_index['qp']]))
                data['qs'].append(float(row[column_index['qs']]))
            except (IndexError, ValueError, KeyError):
                pass

    # Convert lists to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    return data


def force_gc():
    """
    Forces garbage collection to manually free up unused memory.

    This function triggers Python's garbage collector to clean up
    unreferenced objects, helping to manage memory usage in large
    data processing.
    """
    gc.collect()

def harmonic_interp1d(x, y):
    """
    Performs 1D harmonic interpolation for positive values.

    This function linearly interpolates the inverse of 'y' (1/y)
    and then returns the inverse of the interpolated result.
    Harmonic interpolation is particularly useful for physical
    quantities such as velocities, where the harmonic mean
    better represents the overall behavior.

    Parameters:
        x (array-like): Independent coordinates (e.g., depths).
        y (array-like): Dependent values (e.g., velocities) corresponding to each 'x'.

    Returns:
        function: An interpolation function that accepts new 'x' values
                  and returns interpolated values using harmonic interpolation.

    Raises:
        ValueError: If a queried depth exceeds the maximum depth of the input model
                    or if any interpolation error occurs.
    """
    x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    try:
        inv_y = 1.0 / y
        interp_func = interp1d(x, inv_y, kind='linear', bounds_error=True, fill_value=None)  # No extrapolation
        return lambda xi: 1.0 / interp_func(xi)
    except ValueError as e:
        raise ValueError("Error: queried depth of one or more points is higher than the maximum depth of the input velocity model.") from e
    
def interpolate_batch(batch_z, method, depth, vp_interp, vs_interp, rho_interp, is_anelastic, qp_interp=None, qs_interp=None):
    """
    Interpolates a batch of depth values at once using provided interpolation functions.

    This function allows for efficient batch interpolation of multiple depths,
    supporting both elastic and anelastic models. It also provides the option
    to adjust depths to the nearest available value in the model using
    'previous' or 'next' methods.

    Parameters:
        batch_z (array-like): Array of depdepth values to interpolate.
        method (str): Method to handle depths not exactly matching the model.
                      Options are:
                        - 'previous': use the closest smaller or equal depth.
                        - 'next': use the closest greater or equal depth.
                      If no method is specified, depths are used as-is.
        depth (array-like): Array of depths from the input model, used for boundary checks.
        vp_interp (function): Interpolation function for P-wave velocity (Vp).
        vs_interp (function): Interpolation function for S-wave velocity (Vs).
        rho_interp (function): Interpolation function for density (Rho).
        is_anelastic (bool): Whether the model includes anelastic parameters (Qp, Qs).
        qp_interp (function, optional): Interpolation function for Qp (if anelastic).
        qs_interp (function, optional): Interpolation function for Qs (if anelastic).

    Returns:
        np.ndarray: Interpolated values as a 2D array. The columns are:
                    - Vp, Vs, Rho (if elastic).
                    - Vp, Vs, Rho, Qp, Qs (if anelastic).

    Raises:
        ValueError: If any depth is outside the domain of the model.
    """

    batch_z = np.array(batch_z, dtype=np.float32)
    if method == 'previous':
        batch_z = np.array([max((d for d in depth if d <= z), default=depth[0]) for z in batch_z], dtype=np.float32)
    elif method == 'next':
        batch_z = np.array([min((d for d in depth if d >= z), default=depth[-1]) for z in batch_z], dtype=np.float32)

    if any(z < depth[0] or z > depth[-1] for z in batch_z):
        raise ValueError(f"Some depths in batch_z are out of the interpolation range ({depth[0]} to {depth[-1]}).")

    try:
        vp_vals = vp_interp(batch_z).astype(np.float32)
        vs_vals = vs_interp(batch_z).astype(np.float32)
        rho_vals = rho_interp(batch_z).astype(np.float32)
    except ValueError as e:
        raise ValueError(f"Depths out of domain: {e}")

    if is_anelastic:
        qp_vals = qp_interp(batch_z).astype(np.float32)
        qs_vals = qs_interp(batch_z).astype(np.float32)
        return np.column_stack((vp_vals, vs_vals, rho_vals, qp_vals, qs_vals))

    return np.column_stack((vp_vals, vs_vals, rho_vals))

def quality_checks(results):
    """
    Perform quality checks efficiently on a large dataset.

    This function checks for common data issues in the interpolated results,
    including the presence of NaN or Inf values, and reports the minimum and
    maximum values for each parameter. It also computes the Vp/Vs ratio
    safely, handling potential division by zero.

    Parameters:
        results (dict): Dictionary containing arrays of model parameters.
                        Expected keys include 'vp', 'vs', 'rho', and optionally 'qp', 'qs'.

    Prints:
        A table summarizing:
            - Whether each parameter contains NaN values.
            - Whether each parameter contains Inf values.
            - Minimum and maximum values of each parameter.
            - Minimum and maximum of the Vp/Vs ratio.
        """
    print("\n--- Quality Checks of the Interpolation ---")
    print("| Parameter  | NaN?  | Inf?  | Min    | Max    |")
    print("|-----------|------|------|--------|--------|")

    for key, arr in results.items():
        arr = np.asarray(arr)  # Asegurar que sea un array NumPy
        isnan = np.isnan(arr).any()
        isinf = np.isinf(arr).any()
        arr_min = np.nanmin(arr)  # Ignora NaN en min/max
        arr_max = np.nanmax(arr)

        print(f"| {key:<10} | {isnan:<4} | {isinf:<4} | {arr_min:6.2f} | {arr_max:6.2f} |")

    # Calcular Vp/Vs de forma segura
    vp = np.asarray(results['vp'])
    vs = np.asarray(results['vs'])

    with np.errstate(divide='ignore', invalid='ignore'):  # Evitar warnings por divisiones por cero
        vp_vs_ratio = np.where(vs != 0, vp / vs, np.nan)  # Evitar divisiones por cero

    vp_vs_min = np.nanmin(vp_vs_ratio)  
    vp_vs_max = np.nanmax(vp_vs_ratio)

    print(f"| Vp/Vs Ratio | -    | -    | {vp_vs_min:6.2f} | {vp_vs_max:6.2f} |")
    print("---------------------------------------------------\n")

#AR: (2026-09-16) Picks how many CPUs to use by itself, and works the same on a laptop and
#AR: on MN5 without telling it which.
#AR: 
#AR: Why SLURM_CPUS_PER_TASK and not SLURM_NTASKS. My launch scripts (run_MN5_GPU.slurm) ask
#AR: SLURM for "--ntasks=1 --cpus-per-task=112", one task with 112 CPUs. SLURM_NTASKS counts
#AR: tasks, so it would say 1, not 112, and SLURM_CPUS_PER_TASK is the one that says 112. I keep
#AR: SLURM_NTASKS only as a backup, in case a future script asks SLURM for CPUs another way.
#AR: Outside MN5 and SLURM neither variable exists, so it uses the CPU count of the computer.
def get_n_jobs(n_jobs=None, usage_fraction=0.5, max_jobs=None):
    """
    Picks a worker count that adapts to wherever this runs, no environment flag needed.

    Parameters:
        n_jobs (int, optional): Explicit override, used as-is if given.
        usage_fraction (float): Fraction of the available CPUs to actually use.
        max_jobs (int, optional): Hard cap on the result.

    Returns:
        int: Number of workers to use, at least 1.
    """
    if n_jobs is not None:
        return int(n_jobs)

    total = int(os.environ.get(
        "SLURM_CPUS_PER_TASK",
        os.environ.get("SLURM_NTASKS", psutil.cpu_count(logical=True))
    ))

    n_jobs = max(1, math.floor(total * usage_fraction))

    if max_jobs is not None:
        n_jobs = min(n_jobs, max_jobs)

    return n_jobs


# Interpolación 1D optimizada con memoria reducida
def interpolate_1d_model(model_data, coordinates, grid_spacing=None, method='linear', is_anelastic=False, n_jobs=None):
    """
    Interpolate a 1D velocity model over a set of 3D coordinates.
    Only the depth coordinate is used for interpolation.

    Parameters:
        model_data (dict): Must contain depth, vp, vs, rho and optionally qp, qs.
        coordinates (array-like): Nx3 array with (x, y, z) or equivalent coordinates.
        grid_spacing (float, optional): Used to shift z=0 points slightly downward.
        method (str): 'linear', 'previous', or 'next'.
        is_anelastic (bool): Whether qp and qs should also be interpolated.
        n_jobs (int, optional): Number of parallel workers.

    Returns:
        dict: Interpolated values with keys vp, vs, rho and optionally qp, qs.
    """
    memory_total_gb = psutil.virtual_memory().total / (1024 ** 3)
    batch_size = min(50000, max(1000, int(memory_total_gb * 500)))

    if n_jobs is None:
        n_jobs = get_n_jobs(usage_fraction=0.5, max_jobs=100)
        print(f"Using {n_jobs} workers")

    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must be an NX3 array.")

    depth = np.array(model_data['depth'], dtype=np.float32)
    vp = np.array(model_data['vp'], dtype=np.float32)
    vs = np.array(model_data['vs'], dtype=np.float32)
    rho = np.array(model_data['rho'], dtype=np.float32)

    if grid_spacing is not None:
        coordinates[coordinates[:, 2] == 0, 2] = grid_spacing / 4

    if method == 'linear':
        vp_interp = harmonic_interp1d(depth, vp)
        vs_interp = harmonic_interp1d(depth, vs)
        rho_interp = interp1d(
            depth, rho, kind='linear',
            bounds_error=True, fill_value=None, assume_sorted=True
        )
    elif method in ['previous', 'next']:
        vp_interp = interp1d(
            depth, vp, kind='nearest',
            bounds_error=True, fill_value=None, assume_sorted=True
        )
        vs_interp = interp1d(
            depth, vs, kind='nearest',
            bounds_error=True, fill_value=None, assume_sorted=True
        )
        rho_interp = interp1d(
            depth, rho, kind='nearest',
            bounds_error=True, fill_value=None, assume_sorted=True
        )
    else:
        raise ValueError("Unknown interpolation method. Use 'linear', 'previous' or 'next'.")

    qp_interp, qs_interp = None, None
    if is_anelastic:
        qp_interp = interp1d(
            depth, np.array(model_data['qp'], dtype=np.float32),
            kind='linear', bounds_error=True, fill_value=None, assume_sorted=True
        )
        qs_interp = interp1d(
            depth, np.array(model_data['qs'], dtype=np.float32),
            kind='linear', bounds_error=True, fill_value=None, assume_sorted=True
        )

    num_points = len(coordinates)
    batch_indices = list(range(0, num_points, batch_size))

    print(f"[interpolate_1d_model] n_jobs = {n_jobs}")
    print(f"[interpolate_1d_model] batch_size = {batch_size}")
    print(f"[interpolate_1d_model] num_points = {num_points}")
    print(f"[interpolate_1d_model] num_batches = {len(batch_indices)}")

    results = {
        'vp': np.empty(num_points, dtype=np.float32),
        'vs': np.empty(num_points, dtype=np.float32),
        'rho': np.empty(num_points, dtype=np.float32)
    }
    if is_anelastic:
        results['qp'] = np.empty(num_points, dtype=np.float32)
        results['qs'] = np.empty(num_points, dtype=np.float32)

    def _run_batch(start_idx):
        end_idx = min(start_idx + batch_size, num_points)
        batch = coordinates[start_idx:end_idx]
        depth_batch = batch[:, 2].astype(np.float32, copy=False)

        batch_data = interpolate_batch(
            depth_batch,
            method,
            depth,
            vp_interp,
            vs_interp,
            rho_interp,
            is_anelastic,
            qp_interp,
            qs_interp
        )
        return start_idx, batch_data

    batch_results_with_indices = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_run_batch)(start_idx)
        for start_idx in tqdm(batch_indices, desc="Interpolating", unit="batch")
    )

    for start_idx, batch_data in sorted(batch_results_with_indices, key=lambda x: x[0]):
        end_idx = min(start_idx + batch_size, num_points)

        results['vp'][start_idx:end_idx] = batch_data[:, 0]
        results['vs'][start_idx:end_idx] = batch_data[:, 1]
        results['rho'][start_idx:end_idx] = batch_data[:, 2]

        if is_anelastic:
            results['qp'][start_idx:end_idx] = batch_data[:, 3]
            results['qs'][start_idx:end_idx] = batch_data[:, 4]

        del batch_data

    del batch_results_with_indices
    gc.collect()

    quality_checks(results)
    return results

def plot_1d_model(output_folder, model_data , is_anelastic=False, transition_type="stepwise", scatter_data=None, display=False):
    """
    Plot a 1D velocity model including Vp, Vs, Density, and optionally Qp and Qs.

    This function generates visualizations of a 1D Earth model, allowing the
    choice between stepwise or linear transitions between layers. It also supports
    overlaying scatter data (e.g., from interpolated results) for comparison.
    The plots are saved to the specified output folder, and optionally displayed.

    Parameters:
        output_folder (str): Path to the folder where the generated plots will be saved.
        model_data (dict): Dictionary containing arrays for:
                           - 'depth': Depth values [km].
                           - 'vp': P-wave velocity [m/s].
                           - 'vs': S-wave velocity [m/s].
                           - 'rho': Density [kg/m³].
                           Optionally:
                           - 'qp': P-wave quality factor.
                           - 'qs': S-wave quality factor.
        is_anelastic (bool): Whether the model includes anelastic parameters (Qp and Qs).
        transition_type (str): How to represent the transitions between layers.
                               Options are:
                               - 'stepwise': Display layers with abrupt changes.
                               - 'linear': Display layers with smooth transitions.
        scatter_data (dict, optional): Optional dictionary of scatter points to overlay.
                                       Can include:
                                       - 'Depth', 'Vp', 'Vs', 'Rho', 'Qp', 'Qs'.
        display (bool): If True, display the plots interactively.
                        If False, only save the plots to disk.

    Saves:
        - 'vp_vs_rho.png': Plot of Vp, Vs, and Density vs Depth.
        - 'qp_qs.png': Plot of Qp and Qs vs Depth (only if is_anelastic=True).

    Prints:
        Confirmation of the folder where the plots were saved.
    """

    font_size = 9
    plt.rcParams['font.size'] = font_size
    
    depth = model_data['depth']
    vp = model_data['vp']
    vs = model_data['vs']
    rho = model_data['rho']

    # Crear el output folder si no existe
    os.makedirs(output_folder, exist_ok=True)

    # ----- PRIMERA FIGURA: VP, VS, RHO -----
    plt.figure(figsize=(3, 5))  # Tamaño ajustado

    if transition_type == "stepwise":
        plt.step(vp, depth, label='Vp [m/s]', where='post', color='blue')
        plt.step(vs, depth, label='Vs [m/s]', where='post', color='green')
        plt.step(rho, depth, label='Density [kg/m³]', where='post', color='orange')
    elif transition_type == "linear":
        plt.plot(vp, depth, label='Vp [m/s]', color='blue')
        plt.plot(vs, depth, label='Vs [m/s]', color='green')
        plt.plot(rho, depth, label='Density [kg/m³]', color='orange')

    # Agregar datos dispersos si están disponibles - si no se dan grafica unicamente el modelo 1D de entrada
    if scatter_data:
        Depth = scatter_data.get('Depth', [])
        Vp = scatter_data.get('Vp', [])
        Vs = scatter_data.get('Vs', [])
        Rho = scatter_data.get('Rho', [])

        if Vp: plt.scatter(Vp, Depth, label='Interp. Vp', color='blue', marker='o')
        if Vs: plt.scatter(Vs, Depth, label='Interp. Vs', color='green', marker='^')
        if Rho: plt.scatter(Rho, Depth, label='Interp. Density', color='orange', marker='s')

    plt.legend(loc='lower left', frameon=False)
    plt.xlim(0, max(np.max(vp), np.max(vs), np.max(rho)))
    plt.ylim(np.min(depth), np.max(depth))
    plt.gca().invert_yaxis()
    plt.xlabel('Value')
    plt.ylabel('Depth [km]')
    plt.grid(False)

    # Guardar y/o mostrar la primera figura
    vp_vs_rho_path = os.path.join(output_folder, "vp_vs_rho.png")
    plt.savefig(vp_vs_rho_path, dpi=300, bbox_inches='tight')
    if display:
        plt.show()
    else:
        plt.close()

    # ----- SEGUNDA FIGURA: QP, QS (Si es anelástico) -----
    if is_anelastic and 'qp' in model_data and 'qs' in model_data:
        qp = model_data['qp']
        qs = model_data['qs']

        plt.figure(figsize=(3, 5))  # Tamaño ajustado

        if transition_type == "stepwise":
            plt.step(qp, depth, label='Qp', where='post', color='red', linestyle='--')
            plt.step(qs, depth, label='Qs', where='post', color='purple', linestyle='--')
        elif transition_type == "linear":
            plt.plot(qp, depth, label='Qp', color='red', linestyle='--')
            plt.plot(qs, depth, label='Qs', color='purple', linestyle='--')

        # Agregar datos dispersos si están disponibles
        if scatter_data:
            Qp = scatter_data.get('Qp', [])
            Qs = scatter_data.get('Qs', [])

            if Qp: plt.scatter(Qp, Depth, label='Interp. Qp', color='red', marker='x')
            if Qs: plt.scatter(Qs, Depth, label='Interp. Qs', color='purple', marker='d')

        plt.legend(loc='lower left', frameon=False)
        plt.xlim(0, max(np.max(qp), np.max(qs)))
        plt.ylim(np.min(depth), np.max(depth))
        plt.gca().invert_yaxis()
        plt.xlabel('Value')
        plt.ylabel('Depth [km]')
        plt.grid(False)

        # Guardar o mostrar la segunda figura
        qp_qs_path = os.path.join(output_folder, "qp_qs.png")
        plt.savefig(qp_qs_path, dpi=300, bbox_inches='tight')
        if display:
            plt.show()
        else:
            plt.close()
    
    print(f"Plots saved in {output_folder}")

def read_velocity_model_from_netcdf(filename):
    """
    Load a 3D velocity model from a NetCDF file.

    This function reads a velocity model stored in NetCDF format, extracting
    the grid information (latitude, longitude, depth) and the physical properties
    of the model (such as Vp, Vs, Density, Qp, and Qs, if available).

    Parameters:
        filename (str): Path to the NetCDF file containing the velocity model.

    Returns:
        dict: A dictionary containing:
            - 'latitude' (np.ndarray): Latitude coordinates of the grid.
            - 'longitude' (np.ndarray): Longitude coordinates of the grid.
            - 'depth' (np.ndarray): Depth values of the grid.
            - 'vp' (np.ndarray, optional): P-wave velocity [m/s], if present.
            - 'vs' (np.ndarray, optional): S-wave velocity [m/s], if present.
            - 'rho' (np.ndarray, optional): Density [kg/m³], if present.
            - 'qp' (np.ndarray, optional): P-wave quality factor, if present.
            - 'qs' (np.ndarray, optional): S-wave quality factor, if present.

    Notes:
        Only the properties available in the NetCDF file are included in the output dictionary.
    """
    with Dataset(filename, "r") as ncfile:
        # Read dimensions
        latitude = ncfile.variables['latitude'][:]
        longitude = ncfile.variables['longitude'][:]
        depth = ncfile.variables['depth'][:]

        # Read properties
        model_data = {
            'latitude': latitude,
            'longitude': longitude,
            'depth': depth,
        }
        for prop in ['vp', 'vs', 'rho', 'qp', 'qs']:
            if prop in ncfile.variables:
                model_data[prop] = ncfile.variables[prop][:]

        return model_data

def write_velocity_model_to_netcdf(filename, model_data):
    """
    Save a 3D velocity model to a NetCDF file.
    
    Parameters:
        filename (str): Path to the output NetCDF file.
        model_data (dict): Dictionary containing the model data.
        Required keys: 'latitude', 'longitude', 'depth', 'vp', 'vs', 'rho'
        Optional keys: 'qp', 'qs'

    Example:
    if __name__ == "__main__":
        # Example model data
        model_data = {
        'latitude': np.linspace(0, 100, 11),
        'longitude': np.linspace(0, 100, 11),
        'depth': np.linspace(0, 50, 6),
        'vp': np.random.rand(11, 11, 6),
        'vs': np.random.rand(11, 11, 6),
        'rho': np.random.rand(11, 11, 6),
        'qp': np.random.rand(11, 11, 6),
        'qs': np.random.rand(11, 11, 6),
        }
        write_velocity_model_to_netcdf("velocity_model.nc", model_data)
    """
    with Dataset(filename, "w", format="NETCDF4") as ncfile:
        # Define dimensions
        ncfile.createDimension("latitude", len(model_data['latitude']))
        ncfile.createDimension("longitude", len(model_data['longitude']))
        ncfile.createDimension("depth", len(model_data['depth']))

        # Create coordinate variables
        x_var = ncfile.createVariable("latitude", "f4", ("latitude",))
        y_var = ncfile.createVariable("lonngitude", "f4", ("longitude",))
        z_var = ncfile.createVariable("depth", "f4", ("depth",))
        
        # Assign coordinate values
        x_var[:] = model_data['latitude']
        y_var[:] = model_data['longitude']
        z_var[:] = model_data['depth']

        # Create and assign property variables
        for prop in ['vp', 'vs', 'rho', 'qp', 'qs']:
            if prop in model_data:
                var = ncfile.createVariable(prop, "f4", ("latitude", "longitude", "depth"))
                var[:, :, :] = model_data[prop]

        # Add global attributes
        ncfile.description = "3D Velocity Model"
        ncfile.history = "Created using Python and netCDF4"
        
class VelocityModelLoader:
    """
    Loader for 3D velocity models from various file formats.

    Supports loading models from JSON, NetCDF, CSV, and plain text files.
    Ensures that essential properties ('vp', 'vs', 'rho') are present in the model.

    Parameters:
        filepath (str): Path to the velocity model file.

    Attributes:
        filepath (str): Path to the file to be loaded.
        data (dict or None): Loaded velocity model data.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load(self):
        """
        Detect file format and load velocity model data accordingly.

        Supported formats:
            - JSON (.json)
            - NetCDF (.nc)
            - CSV (.csv)
            - Plain text (.txt)

        Returns:
            dict: Loaded velocity model data.
        """
        if self.filepath.endswith(".json"):
            self.data = self._load_json()
        elif self.filepath.endswith(".nc"):
            self.data = self._load_netcdf()
        elif self.filepath.endswith(".csv"):
            self.data = self._load_csv()
        elif self.filepath.endswith(".txt"):
            self.data = self._load_text()
        else:
            raise ValueError(f"Unsupported file format: {self.filepath}")
        return self.data

    def _load_json(self):
        """Load velocity model from a JSON file."""
        with open(self.filepath, "r") as f:
            json_data = json.load(f)
        
        required_keys = {'vp', 'vs', 'rho'}
        if not required_keys.issubset(json_data.keys()):
            raise ValueError("JSON file must contain at least 'vp', 'vs', and 'rho'")
        
        return {key: np.array(value) if isinstance(value, list) else value for key, value in json_data.items()}

    def _load_netcdf(self):
        """Load velocity model from a NetCDF file."""
        with Dataset(self.filepath, "r") as ncfile:
            data = {
                'latitude': ncfile.variables['latitude'][:],
                'longitude': ncfile.variables['longitude'][:],
                'depth': ncfile.variables['depth'][:],
            }
            
            for prop in ['vp', 'vs', 'rho', 'qp', 'qs']:
                if prop in ncfile.variables:
                    data[prop] = ncfile.variables[prop][:]
            
            if not {'vp', 'vs', 'rho'}.issubset(data.keys()):
                raise ValueError("NetCDF file must contain at least 'vp', 'vs', and 'rho'")
            
            return data

    def _load_csv(self):
        """Load velocity model from a CSV file."""
        data = np.genfromtxt(self.filepath, delimiter=",", names=True)
        required_keys = {'vp', 'vs', 'rho'}
        available_keys = set(data.dtype.names)
        
        if not required_keys.issubset(available_keys):
            raise ValueError("CSV file must contain at least 'vp', 'vs', and 'rho'")
        
        grid_shape = (len(np.unique(data['latitude'])), len(np.unique(data['longitude'])), len(np.unique(data['depth'])))
        model = {
            'latitude': np.unique(data['latitude']),
            'longitude': np.unique(data['longitude']),
            'depth': np.unique(data['depth']),
        }
        
        for prop in ['vp', 'vs', 'rho', 'qp', 'qs']:
            if prop in available_keys:
                model[prop] = data[prop].reshape(grid_shape)
        
        return model

    def _load_text(self):
        """Load velocity model from a plain text file."""
        with open(self.filepath, "r") as f:
            lines = f.readlines()
        
        header, *data = lines
        header = header.strip().split()
        data = np.array([list(map(float, line.strip().split())) for line in data])
        
        required_columns = {'vp', 'vs', 'rho'}
        if not required_columns.issubset(set(header)):
            raise ValueError("Text file must contain at least 'vp', 'vs', and 'rho'")
        
        grid_shape = (len(np.unique(data[:, 0])), len(np.unique(data[:, 1])), len(np.unique(data[:, 2])))
        model = {
            'latitude': np.unique(data[:, 0]),
            'longitude': np.unique(data[:, 1]),
            'depth': np.unique(data[:, 2]),
        }
        
        for prop in ['vp', 'vs', 'rho', 'qp', 'qs']:
            if prop in header:
                idx = header.index(prop)
                model[prop] = data[:, idx].reshape(grid_shape)
        
        return model

def GridCoords_LatLonDepth(region, model_coords_file, z_length, grid_spacing, nz=None):
    """
    Generate 3D grid coordinates (latitude, longitude, depth) for a simulation domain.

    Reads horizontal grid coordinates from a file and creates a full 3D grid by
    extending these coordinates through depth levels.

    Parameters:
        region (str): Name of the region (used for figure names or logging).
        model_coords_file (str): Path to the file containing horizontal grid coordinates
                                 (output from the create_model_coords_file function).
        z_length (float): Total depth of the simulation domain (in km).
        grid_spacing (float): Grid spacing (in km).

    Returns:
        tuple:
            lat_grid (np.ndarray): Flattened array of latitudes for all 3D grid nodes.
            lon_grid (np.ndarray): Flattened array of longitudes for all 3D grid nodes.
            depth_grid (np.ndarray): Flattened array of depths for all 3D grid nodes.

    Notes:
        - The depth grid is created from 0 to `z_length` with steps of `grid_spacing`.
        - The horizontal coordinates are repeated for each depth level.
        - The function ignores comment lines (starting with "#") in the coordinates file.
    """
    # Read grid data from the grid coordinates created with the function create_model_coords_file
    grid_data = []
    with open(model_coords_file, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue  # Skip comments
            parts = line.strip().split()
            if len(parts) == 4:
                lon, lat, x_index, y_index = map(float, parts[:4])
                grid_data.append((lon, lat, int(x_index), int(y_index)))

    # Extract longitude and latitude for plotting
    lons = [point[0] for point in grid_data]
    lats = [point[1] for point in grid_data]

    # Create depths vector 
    if nz is None:
        nz = round(z_length / grid_spacing)

    z_length = nz * grid_spacing
    depth = np.arange(nz, dtype=np.float32) * grid_spacing

    # Create 3D mesh nodes vectors 
    depth_grid = np.repeat(depth, len(lons)) # Create depth vector containing all the 3D mesh nodes  
    lat_grid = np.tile(lats, len(depth)) # Create depth vector containing all the 3D mesh nodes  
    lon_grid = np.tile(lons, len(depth)) # Create depth vector containing all the 3D mesh nodes  
    
    return lat_grid, lon_grid, depth_grid

#AR: (2026-10-06) The minimum velocities and density of the mesh, from one number, the minimum
#AR: Vs. This is the rule of SCEC's own mesher, UCVM/single_exe.csh in cybershake-core:
#AR:     if a minimum Vs is given
#AR:         if it is 900        Vp minimum 1800, density minimum 2000
#AR:         otherwise           Vp minimum 3.4 x Vs minimum, density minimum that same number
#AR:     if none is given        Vs 500, Vp 1700, density 1700
#AR: So 500 m/s, the default, gives the 1700 and 1700 this code always had (3.4 x 500), and
#AR: 400 m/s gives 1360 and 1360. The 1700 is not a floor of its own, what SCEC keeps is the
#AR: factor 3.4. Making the density minimum equal to the Vp minimum as a number is what their
#AR: script does, we follow it so that both meshers agree, its physical reason is not known to
#AR: us. The 900 case is a special case written in their script, copied for the same reason.
#AR: Units are m/s and kg/m3.
#AR: Lowering the minimum Vs shortens the shortest wavelength, so the grid spacing has to go
#AR: down with it to keep the points per wavelength (5 in CyberShake, 500 m/s at 0.5 Hz and
#AR: 200 m, or 400 m/s at 1 Hz and 80 m).
def scec_minimums(min_vs=500.0):
    """Returns (min_vs, min_vp, min_rho) in m/s, m/s and kg/m3 for a minimum Vs."""
    min_vs = float(min_vs)
    if min_vs <= 0.0:
        raise ValueError("The minimum Vs must be positive, got %s" % min_vs)
    if min_vs == 900.0:
        return min_vs, 1800.0, 2000.0
    min_vp = 3.4 * min_vs
    return min_vs, min_vp, min_vp


#AR: Vp/Vs is never left below this, for any minimum Vs. It is a physical limit (below it
#AR: Poisson's ratio would be negative or close to it) and SCEC's mesher has the same check.
MIN_VP_VS_RATIO = 1.45


class VelocityModel3D:
    """
    3D Velocity Model with interpolation and physical corrections.

    Provides interpolated values of seismic properties (Vp, Vs, Rho, Qp, Qs)
    at arbitrary coordinates, with physical corrections applied to ensure
    realistic values.

    Parameters:
        model_data (dict): Dictionary containing grid coordinates and property arrays.
        fill_value (float or None): Value used when extrapolating outside the grid.
        bounds_error (bool): If True, raises an error when interpolating out of bounds.

    Attributes:
        latitude (np.ndarray): Latitude grid points.
        longitude (np.ndarray): Longitude grid points.
        depth (np.ndarray): Depth grid points.
        properties (dict): Dictionary of interpolators for the model properties.
    """
    def __init__(self, model_data, fill_value=None, bounds_error=False, min_vs=500.0):
        #AR: (2026-10-06) min_vs sets the three minimums applied to every mesh point, see
        #AR: scec_minimums above. The default leaves everything as it was.
        self.min_vs, self.min_vp, self.min_rho = scec_minimums(min_vs)
        self.latitude = model_data['latitude']
        self.longitude = model_data['longitude']
        self.depth = model_data['depth']
        self.properties = {}
        self.fill_value = fill_value
        self.bounds_error = bounds_error

        # Validar dimensiones
        self._validate_dimensions(model_data)

        # Interpoladores para Vp y Vs (usando 1/Vp y 1/Vs para interpolación armónica)
        for key in ['vp', 'vs']:
            if key in model_data:
                self.properties[key] = RegularGridInterpolator(
                    (self.latitude, self.longitude, self.depth), 
                    1.0 / model_data[key],  
                    bounds_error=self.bounds_error, 
                    fill_value=(1.0 / self.fill_value if self.fill_value else None)
                )

        # Interpoladores estándar para rho, Qp y Qs
        for key in ['rho', 'qp', 'qs']:
            if key in model_data:
                self.properties[key] = RegularGridInterpolator(
                    (self.latitude, self.longitude, self.depth), 
                    model_data[key], 
                    bounds_error=self.bounds_error, 
                    fill_value=self.fill_value
                )

        # Precargar interpoladores con un punto para evitar latencias (demoras iniciando el interpolador)
        dummy_point = np.array([[self.latitude[0], self.longitude[0], self.depth[0]]])
        for interpolator in self.properties.values():
            interpolator(dummy_point)
    
    def _validate_dimensions(self, model_data):
        expected_shape = (len(self.latitude), len(self.longitude), len(self.depth))
        for key, data in model_data.items():
            if key in ['vp', 'vs', 'rho', 'qp', 'qs']:
                if data.shape != expected_shape:
                    raise ValueError(
                        f"Property '{key}' has shape {data.shape}, but expected {expected_shape}."
                    )

    def apply_corrections(self, vp, vs, rho):
        """
        Apply physical corrections to Vp, Vs, and Rho values (scalar version, kept for reference).

        Rules:
            - Minimum Vs, 500 m/s by default (adjusting Vp to preserve the ratio).
            - Minimum Vp, 3.4 times the minimum Vs (1700 m/s by default).
            - Minimum Rho, the same number as the minimum Vp (1700 kg/m³ by default).
            See scec_minimums for where the rule comes from.
            - Minimum Vp/Vs ratio of 1.45.

        Parameters:
            vp (float): P-wave velocity.
            vs (float): S-wave velocity.
            rho (float): Density.

        Returns:
            tuple: Corrected (vp, vs, rho).
        """

        if vs < self.min_vs:
            vp_vs_ratio = vp / vs if vs > 0 else 1.75
            vs = self.min_vs
            vp = self.min_vs * vp_vs_ratio

        if vp < self.min_vp:
            vp = self.min_vp

        if rho < self.min_rho:
            rho = self.min_rho

        vp_vs_ratio = vp / vs
        if vp_vs_ratio < MIN_VP_VS_RATIO:
            vs = vp / MIN_VP_VS_RATIO

        return vp, vs, rho

    #AR: Nueva versión vectorizada de apply_corrections. Aplica exactamente las mismas
    #AR: reglas físicas que apply_corrections pero sobre arrays numpy completos, sin el loop
    #AR: Python escalar de _interpolate_and_correct_batch. En el paso de correcciones es
    #AR: entre 50 y 200 veces más rápida.
    def apply_corrections_vectorized(self, vp, vs, rho, vpvs_low=None):
        """
        Vectorized physical corrections over full numpy arrays (same rules as apply_corrections).

        vpvs_low, when given, is the Vp/Vs ratio to use at the points whose Vs is raised to the
        minimum, one value per point, instead of the point's own ratio. See SURFACE_FORMATS.
        """
        vp = np.asarray(vp, dtype=np.float64)
        vs = np.asarray(vs, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)

        # Minimum Vs (500 m/s by default): preserve Vp/Vs ratio before clamping
        if vpvs_low is None:
            with np.errstate(divide='ignore', invalid='ignore'):
                vp_vs_ratio = np.where(vs > 0, vp / vs, 1.75)
        else:
            vp_vs_ratio = np.asarray(vpvs_low, dtype=np.float64)
        mask_vs_low = vs < self.min_vs
        vs = np.where(mask_vs_low, self.min_vs, vs)
        vp = np.where(mask_vs_low, self.min_vs * vp_vs_ratio, vp)

        # Minimum Vp (3.4 x minimum Vs, 1700 m/s by default)
        np.maximum(vp, self.min_vp, out=vp)

        # Minimum Rho (1700 kg/m³ by default)
        np.maximum(rho, self.min_rho, out=rho)

        # Minimum Vp/Vs ratio = 1.45
        vp_vs_ratio = vp / vs
        vs = np.where(vp_vs_ratio < MIN_VP_VS_RATIO, vp / MIN_VP_VS_RATIO, vs)

        return vp, vs, rho

    def _interpolate_and_correct_batch(self, batch_with_indices, raw=False, vpvs_low=None):
        """Interpola un batch de puntos y aplica correcciones físicas."""
        batch, indices = batch_with_indices

        # Realizar la interpolación para todo el batch de puntos
        vp_batch = 1.0 / self.properties['vp'](batch)
        vs_batch = 1.0 / self.properties['vs'](batch)
        rho_batch = self.properties['rho'](batch) if 'rho' in self.properties else np.full(vp_batch.shape, self.min_rho)

        #AR: Reemplazado el loop Python escalar (for vp, vs, rho in zip(...))
        #AR: por apply_corrections_vectorized que opera sobre el array completo.
        #AR: Elimina 250,000+ llamadas Python por batch → operaciones numpy puras.
        #AR: (2026-10-08) raw skips the corrections, used to read the Vp/Vs ratio of the layer
        #AR: below the surface as the model gives it. vpvs_low is passed on, see
        #AR: apply_corrections_vectorized.
        if not raw:
            vp_batch, vs_batch, rho_batch = self.apply_corrections_vectorized(
                vp_batch, vs_batch, rho_batch, vpvs_low=vpvs_low)

        return indices, vp_batch.astype(np.float32), vs_batch.astype(np.float32), rho_batch.astype(np.float32)

    def calculate_optimal_batch_size(self, memory_percentage=0.8):
        """
        Calculate optimal batch size based on available system memory.

        Parameters:
            memory_percentage (float): Fraction of total memory to consider usable.

        Returns:
            int: Optimal batch size.
        """
        memory_available_gb = self.get_total_memory() * memory_percentage
        if memory_available_gb > 32:
            return 250000  
        elif memory_available_gb > 16:
            return 100000  
        elif memory_available_gb > 8:
            return 50000  
        else:
            return 10000  

    def get_total_memory(self):
        """
        Get total system memory.

        Returns:
            float: Total memory in GB.
        """
        return psutil.virtual_memory().total / (1024 ** 3)

    #AR: Añadido parámetro verbose=True para poder suprimir logs y quality checks
    #AR: cuando se llama en bucle por slab desde interpolate_and_save_streaming.
    def query(self, coordinates, grid_spacing=None, n_jobs=None, output_csv=None, verbose=True,
              raw=False, vpvs_low=None):
        """
        Interpolate model properties at given 3D coordinates in parallel batches.

        Parameters:
            coordinates (array-like): Nx3 array with (latitude, longitude, depth).
            grid_spacing (float, optional): If provided, points at depth=0 are shifted
                slightly downward to avoid surface interpolation issues.
            n_jobs (int, optional): Number of parallel workers.
            output_csv (str, optional): Path to save interpolated results as CSV.
            verbose (bool): If False, suppresses per-call logs and quality checks.
                            Use False when calling in a slab loop to avoid spam.
            raw (bool): If True, the values are returned as interpolated, with no minimums.
            vpvs_low (array, optional): one Vp/Vs ratio per point, used at the points whose
                            Vs is raised to the minimum instead of their own ratio.

        Returns:
            dict: Interpolated and corrected properties.
        """

        #AR: (2026-09-16) Same auto-adapting worker count as interpolate_1d_model() above, see
        #AR: get_n_jobs() for why.
        if n_jobs is None:
            n_jobs = get_n_jobs(usage_fraction=0.5, max_jobs=100)

        batch_size = self.calculate_optimal_batch_size()

        coordinates = np.asarray(coordinates, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("Coordinates must be an NX3 array of (latitude, longitude, depth).")

        # Shift zero-depth points slightly downward
        if grid_spacing is not None:
            coordinates[coordinates[:, 2] == 0, 2] = grid_spacing / 4

        num_points = len(coordinates)
        batch_indices = list(range(0, num_points, batch_size))

        if verbose:
            print(f"[query] n_jobs = {n_jobs}")
            print(f"[query] batch_size = {batch_size}")
            print(f"[query] num_points = {num_points}")
            print(f"[query] num_batches = {len(batch_indices)}")

        results = {
            'vp': np.empty(num_points, dtype=np.float32),
            'vs': np.empty(num_points, dtype=np.float32),
            'rho': np.empty(num_points, dtype=np.float32)
        }

        has_qp = 'qp' in self.properties
        has_qs = 'qs' in self.properties
        if has_qp:
            results['qp'] = np.empty(num_points, dtype=np.float32)
        if has_qs:
            results['qs'] = np.empty(num_points, dtype=np.float32)

        def _run_batch(start_idx):
            end_idx = min(start_idx + batch_size, num_points)
            batch = coordinates[start_idx:end_idx]
            indices = np.arange(start_idx, end_idx, dtype=np.int64)
            return self._interpolate_and_correct_batch(
                (batch, indices), raw=raw,
                vpvs_low=None if vpvs_low is None else vpvs_low[start_idx:end_idx])

        batch_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_batch)(start_idx)
            for start_idx in tqdm(batch_indices, desc="Interpolating", unit="batch")
        )

        for batch_result in batch_results:
            if has_qp or has_qs:
                # Compatible if later you extend _interpolate_and_correct_batch
                if len(batch_result) == 5:
                    indices_batch, vp_batch, vs_batch, rho_batch, extra = batch_result
                    results['vp'][indices_batch] = vp_batch
                    results['vs'][indices_batch] = vs_batch
                    results['rho'][indices_batch] = rho_batch
                    if has_qp and 'qp' in extra:
                        results['qp'][indices_batch] = extra['qp']
                    if has_qs and 'qs' in extra:
                        results['qs'][indices_batch] = extra['qs']
                else:
                    indices_batch, vp_batch, vs_batch, rho_batch = batch_result
                    results['vp'][indices_batch] = vp_batch
                    results['vs'][indices_batch] = vs_batch
                    results['rho'][indices_batch] = rho_batch
            else:
                indices_batch, vp_batch, vs_batch, rho_batch = batch_result
                results['vp'][indices_batch] = vp_batch
                results['vs'][indices_batch] = vs_batch
                results['rho'][indices_batch] = rho_batch

            del batch_result

        del batch_results
        gc.collect()

        if output_csv is not None:
            rows = []
            for idx in range(num_points):
                row = [
                    idx,
                    coordinates[idx][0],
                    coordinates[idx][1],
                    coordinates[idx][2],
                    results['vp'][idx],
                    results['vs'][idx],
                    results['rho'][idx]
                ]
                if 'qp' in results:
                    row.append(results['qp'][idx])
                if 'qs' in results:
                    row.append(results['qs'][idx])
                rows.append(row)

            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ["Index", "Latitude", "Longitude", "Depth", "Vp", "Vs", "Rho"]
                if 'qp' in results:
                    header.append("Qp")
                if 'qs' in results:
                    header.append("Qs")
                writer.writerow(header)
                writer.writerows(rows)

            print(f"Results saved in {output_csv}")

        #AR: Quality checks protegidos por verbose para no inundar la consola cuando
        #AR: se llama por slab (NZ veces) desde interpolate_and_save_streaming.
        if verbose:
            check_nan = {key: np.isnan(results[key]).any() for key in results}
            check_inf = {key: np.isinf(results[key]).any() for key in results}
            check_min = {key: np.min(results[key]) for key in results}
            check_max = {key: np.max(results[key]) for key in results}

            vp_vs_ratio = results['vp'] / results['vs']
            check_vp_vs_min = np.min(vp_vs_ratio)
            check_vp_vs_max = np.max(vp_vs_ratio)

            print("\n--- Quality Checks ---")
            print("| Parameter | NaN? | Inf? | Min | Max |")
            print("|-----------|------|------|-----|-----|")
            for key in results:
                print(
                    f"| {key:<9} | {str(check_nan[key]):<4} | {str(check_inf[key]):<4} | "
                    f"{check_min[key]:.3f} | {check_max[key]:.3f} |"
                )
            print(f"| Vp/Vs     |  -   |  -   | {check_vp_vs_min:.3f} | {check_vp_vs_max:.3f} |")

        return results
