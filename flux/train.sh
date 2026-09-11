#!/bin/bash

export TORCH_DISABLE_ADDR2LINE=1
export TOKENIZERS_PARALLELISM=false

accelerate launch --multi_gpu --mixed_precision=bf16 --num_machines 1 --num_processes 8 --main_process_port 28521 flux/train.py \
    --pretrained_model_name_or_path="/path/to/FLUX.1-Kontext-dev" \
    --jsonl_for_train="/path/to/trainset.jsonl" \
    --json_for_ref="/path/to/trainset_group.json" \
    --dataset_root_path="/path/to/dataset" \
    --image_column=target \
    --mask_column=source \
    --caption_column=text \
    --output_dir="/path/to/flux_cft" \
    --tracker_project_name "cft-flux" \
    --mixed_precision="bf16" \
    --resolution=1024 \
    --dataloader_num_workers=8 \
    --learning_rate=1e-4 \
    --max_train_steps=4000 \
    --validation_steps=100000 \
    --checkpointing_steps=200 \
    --alpha 0.1 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --proportion_empty_prompts=0.0 \
    --gradient_checkpointing \
    --gaussian_init_lora \
    --rank=128 \
    --seed=123456 \
    --use_8bit_adam \
    --report_to="wandb"
