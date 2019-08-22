#!/usr/bin/env bash
echo "NOTE THAT THE VARIABLE DATA_LOCATION NEEDS TO BE SPECIFIED BEFORE RUNNING SCRIPT"

#SET THIS VARIABLE
DATA_LOCATION=/home/user

GPU_ID=${1?Error: No gpu id specified, usage: ./run_with_docker GPU_ID SCRIPT_NAME}
SCRIPT_TO_RUN=${2? No script specified, usage: ./run_with_docker GPU_ID SCRIPT_NAME}

DIRTOADD=$(readlink -f $PWD)
THISFOLDERINREMOTE=/app

docker run --rm -it --init \
  --runtime=nvidia \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --workdir=$THISFOLDERINREMOTE \
  --volume=$DIRTOADD:/app \
  --volume=$DATA_LOCATION:/data/storage \
  -e NVIDIA_VISIBLE_DEVICES=$GPU_ID \
  maunz/for_semseg python3 $SCRIPT_TO_RUN
