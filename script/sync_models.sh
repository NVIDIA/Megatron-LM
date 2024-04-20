#!/bin/bash

if [ $# -ne 1 ]; then
    echo "1 arguments are required: iteration."
    echo "Usage: $0 1000"
    exit 1
fi

required_env_vars=("CHECKPOINT_LOCAL_PATH" "CHECKPOINT_PATH")
for var in "${required_env_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "ERROR: env $var is not set." >&2
    exit 1
  fi
done

ITER_DIR_NAME=$(printf "iter_%07d" $1)
LOCAL_DIR=$CHECKPOINT_LOCAL_PATH
DESTINATION_PATH=$CHECKPOINT_PATH

# prepare_temp_dir
if [[ $DESTINATION_PATH == gs://* ]]; then
    gsutil cp -r $LOCAL_DIR/* $DESTINATION_PATH/
    echo "rm -rf $LOCAL_DIR/*"
    rm -rf $LOCAL_DIR/*
else
    mkdir -p $DESTINATION_PATH/$ITER_DIR_NAME
    echo "rsync --remove-source-files -a $LOCAL_DIR/$ITER_DIR_NAME/* $DESTINATION_PATH/$ITER_DIR_NAME/"
    rsync --remove-source-files -a $LOCAL_DIR/$ITER_DIR_NAME/* $DESTINATION_PATH/$ITER_DIR_NAME/
    rsync -a $LOCAL_DIR/latest_checkpointed_iteration.txt $DESTINATION_PATH/
fi