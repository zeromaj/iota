#!/bin/bash

# Install dependencies using uv
echo "Installing dependencies..."
uv sync

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

echo "Setup complete! Pre-commit hooks are installed and ready to use."
echo "The virtual environment is activated. To activate it in the future, run: source .venv/bin/activate"
