#!/bin/bash

pip_install_from_git_repo() {
    repo="$1"
    commit="${2:main}"
    dirname=$(basename $repo)
    [ ! -d "$dirname" ] && git clone "$repo"
    pushd "$dirname"
    git fetch origin "$commit"
    git checkout "$commit"
    python -m pip install -e .
    popd
}

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

    source .venv/bin/activate

    pushd BEHAVIOR-1K
    # Fix some install issues
    python -m pip install pymeshlab~=2022.2
    python -m pip install setuptools<=79
    pip_install_from_git_repo https://github.com/StanfordVL/curobo cbaf7d32436160956dad190a9465360fad6aba73
    pip_install_from_git_repo https://github.com/huggingface/lerobot 577cd10974b84bea1f06b6472eb9e5e74e07f77a

    pushd bddl
    python -m pip install -e .
    popd

    ./setup.sh --omnigibson --bddl --primitives --confirm-no-conda --accept-nvidia-eula --accept-dataset-tos
    popd

fi 


