#!/bin/bash

python3 -m venv ../acmeEnv
source ../acmeEnv/bin/activate
pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt