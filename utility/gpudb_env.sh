#!/bin/bash

export GPUDB_PATH=/home/syma/src/gpudb-read-only
export GPUDB_CUDA_PATH=$GPUDB_PATH/src/cuda
#export PRELOAD=/home/syma/src/gdaemon/ic_async.so
export PRELOAD=/home/syma/src/gdaemon/ic_stream.so
