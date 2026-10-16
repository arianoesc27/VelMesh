#!/bin/bash

###########################################################################
#                    Velocity Model Interpolation Automation              #
#--------------------------------------------------------------------------
# This script automatically detects the Python executable and runs a      #
# Python script that performs interpolation of a 3D velocity model. It    #
# accepts an input parameters file for the model configuration.           #
#--------------------------------------------------------------------------
# Author: Andrea C. Riaño                                                 #
# Email: andrea.riano@bsc.es                                              #
# Date: January 2025                                                      #
###########################################################################

# Automatically detect the Python executable
PYTHON_PATH=$(which python3)

# Check if Python was found, if not, exit with an error message
if [ -z "$PYTHON_PATH" ]; then
    echo "Error: Python is not installed or not found in the system PATH."
    exit 1
fi

# The path to the Python script
SCRIPT_PATH="VelocityModel_utils/Interp_velocity_3D.py"

# The path to the configuration file
CONFIG_FILE="Example_3DVelocityModel/inputData/parameters.in" 

# Run the Python script with the config file as an argument
$PYTHON_PATH $SCRIPT_PATH $CONFIG_FILE
