#!/bin/bash
set -euo pipefail

#AR: (2026-09-16) The path comes from an environment variable instead of being written
#AR: here. This script is copied into folders of each run (for example
#AR: software/PartitionAndVelocityModel/), not only run from Running_MN5/ inside the repository,
#AR: so finding the path from $(dirname "$0") would break in those copies. Set it once in your
#AR: shell profile, the same convention as UCVM_INSTALL_PATH. If it is not set, the script uses
#AR: the location of the repository I know about.
CYBERSHAKE_VELMODEL_HOME="${CYBERSHAKE_VELMODEL_HOME:-/gpfs/projects/bsc115/bsc203748/software/cybershake_bsc_partitionandvelmodel}"
if [[ ! -d "$CYBERSHAKE_VELMODEL_HOME" ]]; then
  echo "Error: CYBERSHAKE_VELMODEL_HOME ($CYBERSHAKE_VELMODEL_HOME) does not exist. Set it to where cybershake_bsc_partitionandvelmodel actually lives."
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <parameters.in>"
  exit 1
fi

CONFIG_FILE="$1"
PYTHON_PATH=$(which python3)

if [[ -z "${PYTHON_PATH}" ]]; then
  echo "Error: python3 not found in PATH."
  exit 1
fi
# Automatically detect the Python executable
PYTHON_PATH=$(which python3)

# Check if Python was found
if [ -z "$PYTHON_PATH" ]; then
    echo "Error: Python is not installed or not found in the system PATH."
    exit 1
fi

# The path to the input parameters file
CONFIG_FILE="inputData/parameters.in"

###########################################################################
# Partition
###########################################################################
echo "[CPU] Running partitioning..."

SCRIPT_PATH="${CYBERSHAKE_VELMODEL_HOME}/CyberShake_Partition/Partition_utils/Partition_CPU.py"
$PYTHON_PATH $SCRIPT_PATH $CONFIG_FILE

# Ask only in interactive mode; continue automatically in batch mode
if [[ -t 0 ]]; then
    echo "Do you want to continue to velocity model interpolation? (y/n)"
    read -r user_input
    if [[ "$user_input" != "y" && "$user_input" != "Y" ]]; then
        echo "Stopping script as requested."
        exit 0
    fi
else
    echo "Non-interactive mode detected. Continuing automatically to velocity model interpolation."
fi

###########################################################################
# 3D Velocity model interpolation
###########################################################################
echo "[CPU] Running 3D velocity-model interpolation..."
SCRIPT_PATH="${CYBERSHAKE_VELMODEL_HOME}/CyberShake_VelocityModelInterpolator/VelocityModel_utils/Interp_velocity_3D.py"
$PYTHON_PATH $SCRIPT_PATH $CONFIG_FILE
