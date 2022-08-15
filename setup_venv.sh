#! /bin/bash
# set -euo pipefail

PYTHON_VERSION=3.10.6

# NOTE: Before installing python, make sure you have libsqlite3-dev, python3-tk, libffi-dev packages installed

# Install python
# echo "Install python $PYTHON_VERSION..."
# pyenv install -s "$PYTHON_VERSION"

# Services
service=qa-system

echo "Setting up virtual environments for services..."

cd "$service"
pyenv virtualenv -f "$PYTHON_VERSION" "$service"
VIRTUALENV_NAME="$service"
pyenv local "$PYTHON_VERSION"/envs/"$VIRTUALENV_NAME"
cd -


echo "Installing service dependencies..."
echo $PWD

cd "$service"
pip install pip-tools
pip-sync
cd -

