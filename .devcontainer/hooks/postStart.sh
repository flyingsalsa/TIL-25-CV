#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting postCreateCommand script..."

# --- Installation Section ---

# Check if poetry is installed, install if not (using pipx recommended)
if ! command -v poetry &> /dev/null
then
    echo "Poetry not found. Installing poetry via pipx..."
    # Ensure pipx is available (might need 'apt-get update && apt-get install -y pipx' earlier if not in base image)
    # Or use the official installer: curl -sSL https://install.python-poetry.org | python3 -
    pipx install poetry
    # Ensure poetry installed via pipx is in the PATH for subsequent commands in this script
    export PATH="$HOME/.local/bin:$PATH"
    echo "Poetry installed."
else
    echo "Poetry is already installed."
fi

# Navigate to the workspace directory (adjust if your workspace name is different)
cd /workspaces/til-25-cv

# Install project dependencies using poetry
echo "Installing project dependencies with Poetry..."
poetry install --no-root # '--no-root' is often used in dev containers

echo "postCreateCommand script finished."