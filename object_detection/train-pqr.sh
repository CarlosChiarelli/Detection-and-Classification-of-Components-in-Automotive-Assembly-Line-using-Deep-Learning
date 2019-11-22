#!/bin/bash
# Set up the working environment.
cd ..
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/object_detection"
SLIM_DIR="${CURRENT_DIR}/slim"
export PYTHONPATH="$PYTHONPATH:${CURRENT_DIR}"
export PYTHONPATH="$PYTHONPATH:${SLIM_DIR}"
cd "${WORK_DIR}"

echo "Configuring initital parameters..."
python init_config.py \
	--batch_size=8

echo "Creating csv from xml..."
python xml_to_csv.py

echo "Creating train tfrecord..."
python generate_tfrecord.py \
	--csv_input="data/train_labels.csv" \
	--output_path="data/train.record"

echo "Creating test tfrecord..."
python generate_tfrecord.py \
	--csv_input="data/test_labels.csv" \
	--output_path="data/test.record"

echo "Starting training..."
python model_main.py \
    --pipeline_config_path="training/detector.config" \
    --model_dir="train_log/" \
    --num_train_steps=40000 \
    --sample_1_of_n_eval_examples=1 \
    --alsologtostderr
