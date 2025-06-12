#!/bin/bash

python3.12 -m venv venv
source venv/bin/activate

python3.12 -m pip install -r requirements.txt
python3.12 download_model.py --output api/src/models/v1_0
uvicorn api.src.main:app --host 127.0.0.1 --port 8880
