#!/bin/bash

echo  "-----------------------------------------------------------------------------"
echo "|              © Execution Project Human Activity Recognition               |"
echo "| Author : Alexis EGEA                                                       |"
echo "-----------------------------------------------------------------------------"
echo

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	PYTHON_CMD=python3
elif [[ "$OSTYPE" == "cygwin"* || "$OSTYPE" == "msys"* ]]; then
 	PYTHON_CMD=python
else
	echo "Unsupported OS '$OSTYPE'"
	exit 1
fi

CONDA_PATH="$HOME/anaconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
else
    echo "Conda not found at $CONDA_PATH"
    exit 1
fi
conda activate gpu

cd src/
$PYTHON_CMD -m main
cd ..

echo
read -p "Press any key to close the terminal window"