#!/usr/bin/env bash
# Simple setup script to create a venv and install requirements
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment created in .venv. Activate with: source .venv/bin/activate"
