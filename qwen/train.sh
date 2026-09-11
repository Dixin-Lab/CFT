#!/bin/bash
export TORCH_DISABLE_ADDR2LINE=1
export TOKENIZERS_PARALLELISM=false
export MODEL_DIR="/path/to/Qwen-Image-Edit-2509"
export OUTPUT_DIR="/path/to/qwen_cft"
export TRAIN_DATA="/path/to/trainset.jsonl"
export LOG_PATH="$OUTPUT_DIR/log"

accelerate launch --multi_gpu --mixed_precision=bf16 --num_machines 1 --num_processes 8 --main_process_port 28521 qwen/train.py \
    --dataset_root_path "/path/to/dataset" \
    --json_for_ref "/path/to/trainset_group.json" \
    --pretrained_model_name_or_path "$MODEL_DIR" \
    --tracker_project_name "cft-qwen" \
    --resolution 1024 \
    --source_column="source" \
    --target_column="target" \
    --caption_column="text" \
    --output_dir="$OUTPUT_DIR" \
    --logging_dir="$LOG_PATH" \
    --mixed_precision="bf16" \
    --train_data_dir="$TRAIN_DATA" \
    --rank=128 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --learning_rate=1e-4 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=4 \
    --max_train_steps 4000 \
    --validation_steps=100000 \
    --checkpointing_steps=200 \
    --alpha 0.1 \
    --seed 123456
