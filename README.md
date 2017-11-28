# neuralnetwork
singularity exec --bind /usr/lib64/nvidia:/host-libs /cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow-gpu:latest python3 ANN.py $DIMENS $POINTS $OSMU $OSSIG $ISMU $ISSIG $NODES_PER_LAYER $EPOCHS $LAYERS
