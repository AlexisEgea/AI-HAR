# Human Activity Recognition

## Description

This project aims to develop and test various neural network architectures for recognizing and classifying human activities.   
Human activity classification can have significant applications across various domains. 
For example, in medicine, studying balance through insole sensors and/or analyzing the position of a person based on the activity they are performing 
(walking, running, etc.) can help diagnose conditions such as scoliosis, providing insights into whether a patient may be affected by this disease.

The data used in this project consists of 3D coordinates from 43 sensors placed on the human body. 
The pelvis (the last sensor) serves as the reference point, representing the center of the sensor array.
The data are private and cannot be shared.

## Model Structure

- **Model Input**: `[6938, 100, 126]`
  - **6938**: Number of data samples
  - **100**: Number of timesteps per series (time frames for each activity)
  - **126**: Number of features (coordinates from 43 sensors - the pelvis = 43 x 3 (<- x, y, z)) input parameters per timestep

- **Model Output**: `[1]`   
  The model predicts the index of the activity class (e.g., 1 for "run").

## Features

- **Implemented Models**:  
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
  - Hybrid model combining LSTM and RNN (in development)
  - Transformer-based architecture

- **Dataset Processing**:
  - Flexible support for custom datasets in `.h5` format
  - Automatic segmentation of time-series data
  - Deletion of Pelvis (last 3 columns (x, y, z))
  - Centralization from Pelvis
  - Normalization to have data in [-1, 1]

- **Training and Evaluation**:  
  - Configurable hyperparameters (e.g., epochs, batch size, learning rate)
  - Number of classes returned dynamically
  - Weighted loss function for handling imbalanced datasets
  - Monitors training and validation metrics (e.g., accuracy, loss)
  - Prediction generation and saving result in `.csv` format

- **Visualization**:  
  - Frame (image) and Video data visualization
  - Performance metrics visualization (accuracy and loss)

## Directory Structure:

````commandline
AI-HAR
├── data
│   ├── dataset
│   │   ├── test_insoles.h5
│   │   ├── test_mocap.h5
│   │   ├── train_insoles.h5
│   │   ├── train_labels.h5
│   │   └── train_mocap.h5
│   ├── prediction                -> result of test prediction
│   │   └── prediction_0.csv
│   └── saved_model         
│       └── transformer_har_0
├── src
│   ├── config               
│   │   └── config.json
│   ├── model
│   │   ├── base_model.py
│   │   ├── GRU.py
│   │   ├── LSTM.py
│   │   ├── LSTM_RNN.py
│   │   └── TransformerHAR.py
│   ├── train
│   │   ├── eval.py
│   │   └── train.py
│   └── utils
│       ├── loading_data.py
│       ├── plotting.py
│       ├── processing_data.py
│       └── util_methods.py
├── main.py
├── installation-requirements.sh
├── README.md
└── requirements.txt
````

## Requirement

- Having Conda installed on your machine.

## Execution 

To get the project running on your machine, you can run the `installation-requirements.sh` script to use the `conda` environment and install require libraries.  

Once the prerequisites are installed, you can run the `HAR.sh` script to execute the project.  
Follow the project instructions and have fun :)  

### On Ubuntu:

To execute a `.sh` Script:  
   1. Open a terminal.  
   2. Run the command `chmod +x script_name.sh` to make the script executable.  
   3. Execute the script by running `./script_name.sh`.  

Commands to run to make the project work:
1. Cloning the project to your machine
```sh
git clone https://github.com/AlexisEgea/AI-HAR.git
```
2. Installing the prerequisites
```sh
chmod +x installation-requirements.sh
```
```sh
./installation-requirements.sh
```
3. Run the project
```sh
chmod +x HAR.sh
```
```sh
./HAR.sh
```

### On Windows:

Double-click on the `installation-requirements.sh` script, preferably with `Git Bash`, to install the prerequisites:
```
installation-requirements.sh
```

Double-click on the `HAR.sh` script, preferably with `Git Bash`, to run the project:
```
HAR.sh
```
---

Note that the current version of the project has been tested on both Linux and Windows (Git Bash) with cpu and gpu. If you encounter difficulties running the project, feel free to use an IDE (PyCharm, VS Code, or another), use your `conda` environment and run the `main.py` file.

## Contact Information

For inquiries or feedback, please contact me at [alexisegea@outlook.com](mailto:alexisegea@outlook.com).

## Copyright

© 2025 Alexis EGEA. All Rights Reserved.

