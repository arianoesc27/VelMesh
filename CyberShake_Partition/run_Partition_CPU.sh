#!/bin/bash

###########################################################################
#                          Partition Automation                           #
#--------------------------------------------------------------------------
# This script automatically detects the Python executable and runs a      #
# Python script that calculates the partition parameters for CyberShake.  #
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

# The path to the Python script
SCRIPT_PATH="Partition_utils/Partition_CPU.py"

# The path to the input parameters file
CONFIG_FILE="inputData/parameters.in"

# Run the Python script
$PYTHON_PATH $SCRIPT_PATH $CONFIG_FILE

