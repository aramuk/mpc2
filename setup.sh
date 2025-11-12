#!/bin/bash

if [ ! $(which uv) ]; then
    echo "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync

if [ ! -d "mp-spdz-0.4.1" ]; then
    echo "installing MP-SPDZ"
    wget "https://github.com/data61/MP-SPDZ/releases/download/v0.4.1/mp-spdz-0.4.1.tar.xz"
    tar -xvf mp-spdz-0.4.1.tar.xz
fi

if [ ! -d "BEHAVIOR-1K" ]; then
    echo "installing Behavior-1K"
    # Clone the latest stable release (recommended)
    git clone -b v3.7.1 https://github.com/StanfordVL/BEHAVIOR-1K.git

    uv run python -m ensurepip
    uv run python -m pip install pip --upgrade
 
fi 

if [ 1 ]; then
    # cd BEHAVIOR-1K

    cd BEHAVIOR-1K
    uv run ./setup.sh --omnigibson --bddl --dataset --primitives --confirm-no-conda --accept-nvidia-eula --accept-dataset-tos
    cd ..
fi

