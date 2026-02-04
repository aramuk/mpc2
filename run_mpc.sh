#!/usr/bin/env bash

set -ex

rootdir=$(dirname $(readlink -f "${BASH_SOURCE[0]}"))
initial_path="$1"
progname=$(basename "$1")
stem=$(basename "$progname" .py)
shift
party="$1"
shift


if [ -z "$initial_path" ] || [ -z "$party" ]; then
    echo "Usage: $0 <mpc_path> [args...]"
    exit 1
fi

if [ "$party" != "robot" ] && [ "$party" != "server" ]; then
    echo "Party must be either robot or server"
    exit 1
fi

mpspdz_dir="$rootdir/mp-spdz-0.4.1"
compile_path="$mpspdz_dir/Programs/Source/$progname"
cp "$initial_path" "$compile_path"

source .venv/bin/activate
pushd "$mpspdz_dir"
if [ "$party" == "robot" ]; then
    ./mascot-party.x -N 2 -p 0 $stem  
else
    ./mascot-party.x -N 2 -p 1 $stem  
fi
# ./mascot-party.x -N 2 -p 0 $progname &
# ./mascot-party.x -N 2 -p 1 $progname

popd
