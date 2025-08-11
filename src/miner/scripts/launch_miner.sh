#!/bin/bash

echo "[*] Launching miner"

cd /app
source .venv/bin/activate
python ./launch_miner.py
