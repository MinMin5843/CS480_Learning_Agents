# Overview
This project implements a full machine-learning pipeline for classifying handwritten digits from the Optdigits Test Set that consists of 32x32 bitmap images, each representened as 32 lines of 0 or 1 characters followed by a digital label from 0-9. The following tasks are completed as part of this project:
    ~ Task 1: Encode the data from the link (https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits-orig.windep.Z) into input and target files for neural network training.
    ~ Task 2: Train the neural network (pattern net) based on your input/output files. Repeat with 10, 100, and 500 hidden nodes. 

## Before running, be sure to do the following:
1. Activate your virtual environment 
2. Install the dependencies:
    ```bash 
    pip install numpy torch scikit-learn matplotlib seaborn

## How to run the program
1. For Task 1, run the following command:
    ```bash 
    python src/encode.py
2. Afterwards, run the following command for Task 2:
    ```bash
    python main.py

