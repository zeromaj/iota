cp .env src/miner/.env
cd src/miner
uv sync
uv run --package miner main.py