#!/bin/bash
# Set up the working environment.
cd ..
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/object_detection"
SLIM_DIR="${CURRENT_DIR}/slim"
export PYTHONPATH="$PYTHONPATH:${CURRENT_DIR}"
export PYTHONPATH="$PYTHONPATH:${SLIM_DIR}"
cd "${WORK_DIR}"

echo "Starting tensorboard..."
google-chrome "127.0.0.1:6006"
tensorboard --logdir='train_log'
