#!/bin/bash

# Default Python version
PYTHON_VERSION="11"
VENV_NAME="rl-homework-venv"

# Check if argument is provided
if [ $# -eq 1 ]; then
    if [[ "$1" =~ ^(10|11|12)$ ]]; then
        PYTHON_VERSION="$1"
    else
        echo "Error: Python version must be 10, 11, or 12"
        exit 1
    fi
fi

# Create virtual environment
python3."$PYTHON_VERSION" -m venv "$VENV_NAME"

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

# Determine OS and set activation script
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    ACTIVATE_SCRIPT="${VENV_NAME}/Scripts/activate"
else
    ACTIVATE_SCRIPT="${VENV_NAME}/bin/activate"
fi

# Activate virtual environment
source "$ACTIVATE_SCRIPT"

$SHELL install_deps.sh

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi
echo "Virtual environment $VENV_NAME created and activated with Python 3.$PYTHON_VERSION"
