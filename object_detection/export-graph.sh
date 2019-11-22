#!/bin/bash
# Set up the working environment.
cd ..
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/object_detection"
SLIM_DIR="${CURRENT_DIR}/slim"
export PYTHONPATH="$PYTHONPATH:${CURRENT_DIR}"
export PYTHONPATH="$PYTHONPATH:${SLIM_DIR}"
cd "${WORK_DIR}"

echo "Searching last checkpoint..."
files=($(find ./train_log -iname 'model.ckpt-*.index' -type f))
max=0
for F in ${files[@]}; do
	number="${F//[^0-9]/}"
	echo "model-"$number
	if (( $number > $max )); then max=$number; fi; 
done
echo "Final" $max

echo "Deleting inference_graph folder..."
if [ -d "inference_graph" ]; then rm -Rf "inference_graph"; fi

echo "Exporting inference graph..."
python export_inference_graph.py \
	--input_type="image_tensor" \
	--pipeline_config_path="training/detector.config" \
	--trained_checkpoint_prefix="train_log/model.ckpt-"$max \
	--output_directory="inference_graph"
