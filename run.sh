#!/bin/bash

NODES=50
OSMU=15
OSSIG=5
ISMU=0
ISSIG=5
EPOCHS=50
DIMENS=60
POINTS=10000

while [ $NODES -lt 201 ]
do

LAYERS=1
  while [ $LAYERS -lt 11 ]
  do
    NODES_PER_LAYER=$((NODES / LAYERS))

    OSMU=15
    while [ $OSMU -lt 26 ]
    do
      
      echo "Using $LAYERS layer(s), $NODES_PER_LAYER nodes per layer, $DIMENS dimensions, $POINTS points per sphere for $EPOCHS epochs"
      echo "Total number of nodes: $NODES"
      echo Outer sphere: mu = $OSMU sigma=$OSSIG
      echo Inner sphere: mu = $ISMU sigma=$ISSIG

      singularity exec --bind /usr/lib64/nvidia:/host-libs /cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow-gpu:latest python3 ANN.py $DIMENS $POINTS $OSMU $OSSIG $ISMU $ISSIG $NODES_PER_LAYER $EPOCHS $LAYERS > /dev/null
    OSMU=$((OSMU+5))

    done

    LAYERS=$((LAYERS+1))

  done

  NODES=$((NODES+50))

done
