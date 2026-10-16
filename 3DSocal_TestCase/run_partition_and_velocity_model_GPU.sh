#!/bin/bash

###########################################################################
#       Partition and velocity model interpolation Automation             #
#--------------------------------------------------------------------------
# This script automatically detects the Python executable and runs a      #
# Python script that calculates the partition parameters for CYbershake.  #
# accepts parameters_partition.in as an input parameters file.            #
#--------------------------------------------------------------------------
# Author: Andrea C. Riaño                                                 #
# Email: andrea.riano@bsc.es                                              #
# Date: February 2025                                                     #
###########################################################################

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
# The path to the Python script
SCRIPT_PATH="../CyberShake_Partition/Partition_utils/Partition_GPU.py"
# Run the Python script
$PYTHON_PATH $SCRIPT_PATH $CONFIG_FILE

# Prompt to continue
echo "Do you want to continue to velocity model interpolation? (y/n)"
read -r user_input
if [[ "$user_input" != "y" && "$user_input" != "Y" ]]; then
    echo "Stopping script as requested."
    exit 0
fi

###########################################################################
# 3D Velocity model interpolation
###########################################################################
#The path to the Python script
SCRIPT_PATH="../CyberShake_VelocityModelInterpolator/VelocityModel_utils/Interp_velocity_3D.py"
# Run the Python script
$PYTHON_PATH $SCRIPT_PATH $CONFIG_FILE
