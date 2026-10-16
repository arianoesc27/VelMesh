"""
###########################################################################
#                             Model partition                             #
#--------------------------------------------------------------------------
# This script calls functions from Partition_utils.py to perform          #
# partition of a 3D velocity model.                                       #
#--------------------------------------------------------------------------
# Author: Andrea C. Riaño                                                 #
# Email: andrea.riano@bsc.es                                              #
# Date: January 2025                                                      #
###########################################################################
"""
import Partition_utils
import subprocess
import datetime
import argparse
import platform
import shutil
import json
import sys
import os

__version__ = "1.0.0"

# List of required libraries
required_libraries = [
    "numpy", "sympy", "matplotlib"
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

# Main function
def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Partition Script")
    parser.add_argument("parameters", help="Path to the configuration file")
    args = parser.parse_args()

    # Read the configuration
    config = read_inputparameters(args.parameters)

    # Convert the configuration to the correct types (e.g., float, int)
    x_length = float(config['x_length'])
    y_length = float(config['y_length'])
    z_length = float(config['z_length'])
    vs_min = float(config['vs_min'])
    f_max = float(config['f_max'])
    ppwl = float(config['ppwl'])
    gpus_per_node = int(config['gpus_per_node'])
    points_per_gpu = int(config['points_per_gpu'])
    z_block = int(config['z_block']) if 'z_block' in config and config['z_block'].strip() else None
    # The old key was called prefer_power_of_2_z_block. The real rule is a multiple of 32,
    # not a power of two, so the new key is named after that. The old name is still read so
    # that existing parameters.in files keep working.
    _pref = config.get('prefer_multiple_of_32_z_block',
                       config.get('prefer_power_of_2_z_block', 'True'))
    prefer_multiple_of_32 = _pref.strip().lower() in ('true', '1', 'yes')
    z_block_max_relative_error = float(config.get('z_block_max_relative_error', '0.10').strip())
    # Optional. The block sizes AWP has a binary for, as a list such as 32,64,128,200.
    # Left out, the list built on MareNostrum 5 is used (Partition_utils.COMPILED_Z_BLOCKS).
    _avail = config.get('available_z_blocks', '').strip()
    available_z_blocks = [int(v) for v in _avail.replace(',', ' ').split()] if _avail else None
    Region = config['region']
    input_folder = config['input_folder']
    output_folder = config['output_folder']
    output_folder = output_folder + "Partition"

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Compute partition parameters
    [new_x_length, new_y_length, new_z_length, grid_spacing, nx, ny, nz, px, py, pz, n_gpus, n_nodes, z_block_used] = Partition_utils.compute_partition_GPU(
        Lx=x_length,
        Ly=y_length,
        Lz=z_length,
        Vs_min=vs_min,
        f_max=f_max,
        N_lambda=ppwl,
        gpus_per_node=gpus_per_node,
        points_per_gpu=points_per_gpu,
        z_block=z_block,
        prefer_multiple_of_32=prefer_multiple_of_32,
        z_block_max_relative_error=z_block_max_relative_error,
        available_z_blocks=available_z_blocks
    )

    # Copy parameters.in and the VM input data file to the output folder
    file_path_parameters_file = input_folder+'parameters.in'
    new_filename = "parameters_partition.in"
    shutil.copy(file_path_parameters_file, os.path.join(output_folder, new_filename))

    # Collect metadata
    metadata = {
        "partition_script_version": __version__,
        "partition_utilities_version": getattr(Partition_utils, "__version__", "Unknown"),  # Check if utilities has a version
        "execution_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "os": platform.system() + " " + platform.release(),
        "input_parameters": os.path.join(output_folder, new_filename),
        "output_directory": output_folder,
    }

    # Save metadata as a JSON file
    metadata_file = os.path.join(output_folder, "metadata_partition.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Metadata saved to {metadata_file}\n")

    # Update them in the config dictionary
    grid_spacing = grid_spacing / 1000

    updates = {
        'x_length': str(new_x_length),
        'y_length': str(new_y_length),
        'z_length': str(new_z_length),
        'grid_spacing': str(grid_spacing),
        'nx': str(nx),
        'ny': str(ny),
        'nz': str(nz),
        'mesh_mode': 'gpu',
        'z_block': str(z_block_used)
    }

    Partition_utils.update_parameters_file(args.parameters, updates)

    # Ouputfile with partition results
    results_file = os.path.join(output_folder, "partition_results.out")
    Partition_utils.write_partition_info(results_file, new_x_length, new_y_length, new_z_length, grid_spacing, nx, ny, nz, px, py, pz, n_gpus, n_nodes)

# Run the main function
if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"Error: {e}")
