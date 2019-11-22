#!/bin/bash
# Set up the working environment.
cd ..
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/object_detection"
SLIM_DIR="${CURRENT_DIR}/slim"
LOG_DIR="${WORK_DIR}/train_log"
export PYTHONPATH="$PYTHONPATH:${CURRENT_DIR}"
export PYTHONPATH="$PYTHONPATH:${SLIM_DIR}"
cd "${WORK_DIR}"

echo "Starting avaluation..."
python3 eval.py \
	--logtostderr \
	--checkpoint_dir="${LOG_DIR}" \
	--eval_dir="${LOG_DIR}/eval" \
	--pipeline_config_path="${WORK_DIR}/training/detector.config"
