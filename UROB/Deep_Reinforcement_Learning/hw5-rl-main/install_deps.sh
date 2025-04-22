#!/bin/bash

# Define dependencies as an array
DEPS=(
    "dm-control==1.0.14"
    "mujoco==3.2.6"
    "gymnasium==1.0.0"
    "numpy==1.26.4"
    "jupyter==1.0.0"
    "matplotlib==3.9.0"
    "imageio==2.31.1"
    "pytest==7.4.0"
)

# Install all common dependencies
pip install "${DEPS[@]}"

# Detect whether we're on mac/windows or linux
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    pip install torch
else
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
