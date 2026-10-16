"""
###########################################################################
#        1D Velocity Model Interpolation & Visualization Script           #
#--------------------------------------------------------------------------
# This script calls functions from VelocityModel_utils.py to perform      #
# interpolation of a 1D velocity model and visualize the results.         #
#--------------------------------------------------------------------------
# Author: Andrea C. Riaño                                                 #
# Email: andrea.riano@bsc.es                                              #
# Date: January 2025                                                      #
###########################################################################
"""

import VelocityModel_utils
import importlib
import subprocess
import datetime
import argparse
import platform
import shutil
import time
import json
import sys
import os
import numpy as np
import pandas as pd

__version__ = "1.0.0"

# List of required libraries
required_libraries = [
    "cartopy", "folium", "joblib", "matplotlib", "netCDF4", 
    "numpy", "pandas", "psutil", "pyproj", "scipy", 
    "shapely", "simplekml", "tqdm"
]

# Check if libraries are installed
missing_libraries = []
for lib in required_libraries:
    try:
        __import__(lib)
    except ImportError:
        missing_libraries.append(lib)

# Install missing libraries
if missing_libraries:
    print(f"The following dependencies are missing: {', '.join(missing_libraries)}")
    print("Installing missing dependencies quietly...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
            stdout=subprocess.DEVNULL,  # Suppresses standard output
            stderr=subprocess.DEVNULL   # Suppresses standard error
        )
        print("Missing dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during installation: {e}")
else:
    print("All required dependencies are already installed.")

# Function to read configuration from a file
def read_inputparameters(parameters_file):
    config = {}
    with open(parameters_file, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            # Skip empty lines or comment lines (lines starting with #)
            if not line or line.startswith('#'):
                continue
            # Ensure there is exactly one '=' in the line
            if '=' in line:
                try:
                    key, value = line.split('=', 1)  # Split only on the first '='
                    config[key.strip()] = value.strip()  # Strip spaces and add to dictionary
                except ValueError:
                    print(f"Warning: Skipping malformed line: '{line}'")
            else:
                print(f"Warning: Skipping line without '=': '{line}'")
    return config

def resolve_mesh(config, x_length, y_length, z_length, grid_spacing):
    mesh_mode = config.get('mesh_mode', 'auto').strip().lower()

    nx = int(config['nx']) if 'nx' in config and config['nx'].strip() else None
    ny = int(config['ny']) if 'ny' in config and config['ny'].strip() else None
    nz = int(config['nz']) if 'nz' in config and config['nz'].strip() else None
    z_block = int(config['z_block']) if 'z_block' in config and config['z_block'].strip() else None

    if nx is not None and ny is not None and nz is not None:
        x_length = nx * grid_spacing
        y_length = ny * grid_spacing
        z_length = nz * grid_spacing
    else:
        nx = round(x_length / grid_spacing)
        ny = round(y_length / grid_spacing)
        nz = round(z_length / grid_spacing)

        x_length = nx * grid_spacing
        y_length = ny * grid_spacing
        z_length = nz * grid_spacing

    if mesh_mode == 'gpu' and z_block is not None and nz % z_block != 0:
        raise ValueError(
            f"mesh_mode='gpu' requires nz to be a multiple of z_block. "
            f"Got nz={nz}, z_block={z_block}"
        )

    return x_length, y_length, z_length, nx, ny, nz

# Main function
def main():

   # import partition utils
    global Partition_utils

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    PARTITION_UTILS_FILE = os.path.abspath(
        os.path.join(
            SCRIPT_DIR,
            "../../CyberShake_Partition/Partition_utils/Partition_utils.py"
        )
    )

    print("SCRIPT_DIR:", SCRIPT_DIR)
    print("PARTITION_UTILS_FILE:", PARTITION_UTILS_FILE)
    print("FILE EXISTS:", os.path.isfile(PARTITION_UTILS_FILE))

    if not os.path.isfile(PARTITION_UTILS_FILE):
        raise FileNotFoundError(
            f"Partition_utils.py not found at: {PARTITION_UTILS_FILE}"
        )

    spec = importlib.util.spec_from_file_location("Partition_utils", PARTITION_UTILS_FILE)
    Partition_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Partition_utils)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Velocity Model Script")
    parser.add_argument("parameters", help="Path to the configuration file")
    args = parser.parse_args()

    # Read the configuration
    config = read_inputparameters(args.parameters)

    # Convert the configuration to the correct types (e.g., float, int)
    input_folder = config['input_folder']
    file_name = config['vm_file']
    velocity_units = config['velocity_units']
    density_units = config['density_units']
    model_lon = float(config['model_lon'])
    model_lat = float(config['model_lat'])
    x_length = float(config['x_length'])
    y_length = float(config['y_length'])
    z_length = float(config['z_length'])
    model_rot = float(config['model_rot'])
    grid_spacing = float(config['grid_spacing'])
    Region = config['region']
    output_folder = config['output_folder']
    is_anelastic = config['is_anelastic'].strip().lower() in ("true", "1", "yes")
    variable_name = config['variable_name']
    plane_type = config['plane_type']
    index = int(config['index'])
    index_z = int(config['index_z'])
    Domain_dynmap = config['Domain_dynmap'].strip().lower() in ("true", "1", "yes")
    Grid_dynmap = config['Grid_dynmap'].strip().lower() in ("true", "1", "yes")
    
    mesh_mode = config.get('mesh_mode', 'auto').strip().lower()
    nx = int(config['nx']) if 'nx' in config and config['nx'].strip() else None
    ny = int(config['ny']) if 'ny' in config and config['ny'].strip() else None
    nz = int(config['nz']) if 'nz' in config and config['nz'].strip() else None
    z_block = int(config['z_block']) if 'z_block' in config and config['z_block'].strip() else None

    x_length, y_length, z_length, nx, ny, nz = resolve_mesh(
        config, x_length, y_length, z_length, grid_spacing
    )

    updates = {
        'x_length': str(x_length),
        'y_length': str(y_length),
        'z_length': str(z_length),
        'grid_spacing': str(grid_spacing),
        'nx': str(nx),
        'ny': str(ny),
        'nz': str(nz)
    }

    if mesh_mode in ('auto', 'standalone'):
	    updates['mesh_mode'] = 'standalone'
        
    if z_block is not None:
       updates['z_block'] = str(z_block)

    Partition_utils.update_parameters_file(args.parameters, updates)

    print("\n--- START OF INTERPOLATION ---\n")
    
    # Define the path of the velocity model file 
    file_path = input_folder+file_name
    
    # Define output folders 
    output_folder = output_folder+'VelModel/'
    output_folder_AWP = output_folder+'AWP_inputfiles/'
    output_folder_Fig = output_folder+'Figures/'
    
    # Remove existing output directories to prevent errors during execution.
    # The binary file generation reads only the first modelcoords file it encounters.
    if os.path.isdir(output_folder_AWP):
        print(f"Removing pre-existing output directory to avoid runtime conflicts:{output_folder_AWP}")
        shutil.rmtree(output_folder_AWP)
    
    # Define variable_names depending on the value of is_anelastic
    if is_anelastic:
        variable_names = ['Vp', 'Vs', 'Rho', 'Qp', 'Qs']
    else:
        variable_names = ['Vp', 'Vs', 'Rho']
    
    # Name of output files
    domain_data = f'{int(x_length)}km_{int(y_length)}km_{int(z_length)}km_Spacing_{int(float(grid_spacing)*1000)}m'
    
    # Read 1D velocity model
    model_data = VelocityModel_utils.parse_1d_model(file_path, is_anelastic=is_anelastic)
    
    # Convert units from input velocity model to units required for storing the model in the binary (m/s and kg/m3)
    model_data = VelocityModel_utils.convert_units(model_data,velocity_units,density_units)

    print("--> Saving AWP mesh coordinates files.")
    # Generate velocity model coordinates file
    VelocityModel_utils.create_model_coords_file(
        file_name=domain_data,
        site_name=Region,
        model_lon=model_lon,
        model_lat=model_lat,
        x_length=x_length,
        y_length=y_length,
        model_rot=model_rot,
        grid_spacing=grid_spacing,
        output_folder=output_folder_AWP,
        nx=nx,
        ny=ny
    )

    # Generate model box
    VelocityModel_utils.generate_modelbox_file(
        file_name=domain_data,
        site_name=Region,
        model_lon=model_lon,
        model_lat=model_lat,
        x_length=x_length,
        y_length=y_length,
        model_rot=model_rot,
        output_folder=output_folder_AWP)

    # Generate gridout file
    VelocityModel_utils.create_gridout_file(
        file_name=domain_data,
        site_name=Region,
        x_length=x_length,
        y_length=y_length,
        z_length=z_length,
        grid_spacing=grid_spacing,
        output_folder=output_folder_AWP,
        nx=nx,
        ny=ny,
        nz=nz
    )
    if Domain_dynmap or Grid_dynmap: 
        print("--> Saving static and dynamic maps.")
        
        # Plot domain on surface - static map 
        VelocityModel_utils.plot_modelbox_with_cartopy(
            region=Region,
            model_lon=model_lon,
            model_lat=model_lat,
            x_length=x_length,
            y_length=y_length,
            model_rot=model_rot,
            output_folder=output_folder_Fig)
            
        # Plot domain on surface - dynamic map
        if Domain_dynmap: 
            VelocityModel_utils.plot_modelbox_with_folium(
                region=Region,
                model_lon=model_lon,
                model_lat=model_lat,
                x_length=x_length,
                y_length=y_length,
                model_rot=model_rot,
                output_folder=output_folder_Fig)

        # Plot mesh points on surface - dynamic map
        if Grid_dynmap:
            model_coords_file = output_folder_AWP + domain_data + '.model_coords_' + Region
            VelocityModel_utils.plot_and_export_grid_map(
                region=Region, 
                model_coords_file=model_coords_file,
                output_folder=output_folder_Fig)
      
    #AR: Eliminados np.meshgrid, coordinates y la llamada a interpolate_1d_model.
    #AR: En un modelo 1D el interpolador solo depende de Z, y evaluarlo NX*NY*NZ veces
    #AR: (por ejemplo 15 mil millones) cuando solo hay NZ profundidades distintas
    #AR: (por ejemplo 600) no sirve de nada. Con interpolate_1d_and_save_streaming
    #AR: - el interpolador se evalúa solo NZ veces,
    #AR: - cada capa se llena con np.full (un escalar da un array 2D uniforme),
    #AR: - nunca se crean arrays de tamaño NX*NY*NZ en memoria,
    #AR: - y el binario sale idéntico al de antes.
    VelocityModel_utils.force_gc()

    # Ejes 1D de coordenadas (arrays pequeños para los plots)
    X = np.arange(nx, dtype=np.float32) * grid_spacing
    Y = np.arange(ny, dtype=np.float32) * grid_spacing
    Z = np.arange(nz, dtype=np.float32) * grid_spacing

    #AR: (2026-09-16) transition_type="stepwise" ported from a local branch. This only changes
    #AR: how the plot is drawn (straight lines between points vs. flat steps), it does not
    #AR: touch the saved binary file.
    # Plot the input 1D velocity model
    VelocityModel_utils.plot_1d_model(output_folder_Fig, model_data, is_anelastic=is_anelastic, transition_type="stepwise")

    print("--> Interpolating and saving velocity model (1D streaming, NZ evaluations only).")
    VM_binFile = VelocityModel_utils.interpolate_1d_and_save_streaming(
        model_data=model_data,
        region=Region,
        nx=nx, ny=ny, nz=nz,
        grid_spacing=grid_spacing,
        is_anelastic=is_anelastic,
        output_folder=output_folder_AWP
    )
    VelocityModel_utils.force_gc()
    #AR: Fin del bloque modificado, interpolación 1D por capas

    print("--> Plotting results.")
    # Plot XY surface
    #AR: x_coords, y_coords, z_coords son ahora arrays 1D, no el ravel de coordenadas 3D
    VelocityModel_utils.plot_surface_XY(
        binary_file_path=VM_binFile,
        variable_name=variable_name,
        index=index_z,
        num_x=nx,
        num_y=ny,
        num_z=nz,
        variable_names=variable_names,
        output_folder=output_folder_Fig,
        x_coords=X,
        y_coords=Y,
        z_coords=Z)

    # Plot XZ or YZ surface
    VelocityModel_utils.plot_surface_XZorYZ(
        binary_file_path=VM_binFile,
        variable_name=variable_name,
        index=index,
        num_x=nx,
        num_y=ny,
        num_z=nz,
        variable_names=variable_names,
        plane_type=plane_type,
        output_folder=output_folder_Fig,
        x_coords=X,
        y_coords=Y,
        z_coords=Z)

    print("\n--- END OF INTERPOLATION ---\n")
    
    #AR: (2026-09-16) This used to assume that parameters.in is always in input_folder, next
    #AR: to the velocity model file. That is true for the standalone examples, where both are in
    #AR: the same inputData/ folder. It is false in a CyberShake run, where
    #AR: step_partitionVelocityModel writes parameters.in in the working folder of the site and
    #AR: input_folder points to a separate shared folder, VelocityModelInterpolator_inputfiles/,
    #AR: that only has the velocity model file. args.parameters is where parameters.in really is.
    #AR: Before, this call crashed with FileNotFoundError at the very end of the run, after the
    #AR: interpolation had already finished.
    # Copy parameters.in and the VM input data file to the output folder
    file_path_parameters_file = args.parameters
    shutil.copy(file_path_parameters_file, os.path.join(output_folder, os.path.basename(file_path_parameters_file)))
    shutil.copy(file_path, os.path.join(output_folder, os.path.basename(file_path)))

    # Collect metadata
    metadata = {
        "script_version": __version__,
        "velocity_utilities_version": getattr(VelocityModel_utils, "__version__", "Unknown"),  # Check if utilities has a version
        "execution_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "os": platform.system() + " " + platform.release(),
        "input_parameters": file_path_parameters_file,
        "output_directory": output_folder,
    }

    # Save metadata as a JSON file
    metadata_file = os.path.join(output_folder, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {metadata_file}")
    print(f"--> Computation completed!")	
# Run the main function
if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"Error: {e}")
