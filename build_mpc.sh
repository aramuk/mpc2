#!/usr/bin/env bash

set -e

rootdir=$(dirname $(readlink -f "${BASH_SOURCE[0]}"))
initial_path="$1"
progname=$(basename "$1")
stem=$(basename "$progname" .py)
shift


if [ -z "$initial_path" ]; then
    echo "Usage: $0 <mpc_path> [args...]"
    exit 1
fi

mpspdz_dir="$rootdir/mp-spdz-0.4.1"
compile_path="$mpspdz_dir/Programs/Source/$progname"
cp "$initial_path" "$compile_path"

pushd "$mpspdz_dir"
./compile.py "$stem"

popd
