#!/bin/bash

CDIR=`dirname $0`
source $CDIR/gpudb_env.sh

RUNPATH=$GPUDB_CUDA_PATH

EXTENDER=$CDIR/cuExtender.py;

REFDIR=$1;
CUDIR=$2;
OUTDIR=$3;

CUS="$(ls $REFDIR| grep ref| sed 's/\.ref//g')";
for cu in $CUS; do
	$EXTENDER --ref=$REFDIR/$cu.ref --src=$CUDIR/$cu.cu > $OUTDIR/$cu.cu
done
