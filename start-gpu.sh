#!/bin/bash


# python3.12 -m venv venv
# source venv/bin/activate
# python3.12 -m pip install -r requirements.txt
# python3.12 download_model.py --output api/src/models/v1_0
# uvicorn api.src.main:app --host 127.0.0.1 --port 8880


# immediately exit if any command returns a non-zero (error) status.
set -e

if [ ! -d "venv" ]; then
  python3.12 -m venv venv
  source venv/bin/activate
  python3.12 -m pip install --upgrade pip
  python3.12 -m pip install -r requirements.txt
else
  source venv/bin/activate
fi

MODEL_DIR="api/src/models/v1_0"
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR")" ]; then
  python3.12 download_model.py --output "$MODEL_DIR"
fi

uvicorn api.src.main:app --host 127.0.0.1 --port 8880

