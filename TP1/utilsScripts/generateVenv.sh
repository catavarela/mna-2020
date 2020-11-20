#!/bin/bash

# get path to base of project
BASEOFPROJECT="$(dirname $BASH_SOURCE[0])/.."
# cd to base of project
cd $BASEOFPROJECT

# generate virtual environment
# if virtual env is not installed use 
#python3 -m venv test_venv
virtualenv env

#activate venv and install requirements
source env/bin/activate
pip install -r requirements.txt
