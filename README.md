# Safe Reinforcement Learning by Shielding for Autonomous Vehicles
This repository contains the code that was used for the thesis "Safe Reinforcement Learning by Shielding for Autonomous Vehicles", 
written by Job Zoon to obtain the Master's Degree in Computer Science at the Delft Univesity of Technology. The work for this thesis was done at the
Netherlands Organisation for Applied Scientific Research. The thesis can be found in https://repository.tudelft.nl/.

## Prerequisites
The code was run with the following software versions on Windows:
* CARLA 0.9.9.4
* Python 3.7.9
* Keras 2.2.4
* Numpy 1.18.4
* Pygame 1.9.6
* Scipy 1.4.1
* Tensorflow 1.15.4
* tqdm 4.50.2

## Get started
Follow the following steps to get started with the code.
1. Download the two Windows files from https://github.com/carla-simulator/carla/releases/tag/0.9.9
1. Extract CARLA_0.9.9.4.zip where you want to install CARLA
1. Extract AdditionalMaps_0.9.9.4.zip in the root folder of CARLA
1. Open a terminal in the root folder of CARLA and use the following commands to check if everything works
   1. CarlaUE4.exe
   1. cd PythonAPI/examples
   1. python spawn_npc.py
1. Download this repository and put it in the PythonAPI/examples folder
1. Create three directories in this folder with the following names: *logs*, *manual_logs* and *models*

## Training
To train a model, first set the desired parameters in *parameters.py*. Then run *main.py*. The resulting model will be saved in the *models* directory, TensorBoard logs can be found in the *logs* directory, and CSV logs can be found in the *manual_logs* directory.

## Evaluate
To evaluate one or multiple models, use the *run_rl.py* file and insert the models in the *MODEL_PATHS* variable.
