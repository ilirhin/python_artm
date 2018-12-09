#!/usr/bin/env bash
mkdir Release
cd Release
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(realpath ..) .. \
    -DPYTHON_EXECUTABLE=$(which python)
make
cd ..

