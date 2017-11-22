#!/bin/bash

NODES=10
OSMU=10
OSSIG=5
ISMU=0
ISSIG=5
EPOCHS=50
DIMENS=60
POINTS=10000
LAYERS=2

echo "Using $LAYERS layer(s), $DIMENS dimensions, $POINTS points per sphere for $EPOCHS epochs"
echo Outer sphere: mu = $OSMU sigma=$OSSIG
echo Inner sphere: mu = $ISMU sigma=$ISSIG
while [ $NODES -lt 50 ]
do
  echo "Using $NODES nodes"
  singularity exec --bind /usr/lib64/nvidia:/host-libs /cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow-gpu:latest python3 ANN.py 60 10000 10 5 0 5 $NODES 50 > /dev/null
  let NODES=$NODES+10
done
