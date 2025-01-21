#!/bin/bash

echo "-----------------------------------------------------------------------------"
echo "|                          Installation Requirements                        |"
echo "| Author : Alexis EGEA                                                      |"
echo "-----------------------------------------------------------------------------"

echo "activation of conda..."
CONDA_PATH="$HOME/anaconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
else
    echo "Conda not found at $CONDA_PATH"
    exit 1
fi

read -p "Enter the name of your conda env: " env
conda activate $env

echo "...done"
echo "_____________________________________________________________________________"

echo "installation requirement.txt"
pip install -r requirement.txt
echo "Done, project ready to be executed"

echo 
read -p "Press any key to close the terminal window"