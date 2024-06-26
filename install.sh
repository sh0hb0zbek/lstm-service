#!/bin/bash
# download anaconda
wget https://repo.continuum.io/archive/Anaconda3-2021.11-Linux-x86_64.sh
# install anaconda
bash Anaconda3-2021.11-Linux-x86_64.sh
`source ~/.bashrc`
# create virtual environment
conda env create -f environment.yml
`source ~/anaconda3/etc/profile.d/conda.sh`
# activate virtual environment
conda activate lstm_env
