#!/bin/bash
HOME_PATH='/data/codes/lixiang/Video-LLaVA-main'
JSON_FOLDER="/data/codes/lixiang/Video-LLaVA-main/dataset/soccernet_json"
#IMAGE_FOLDER= None
#VIDEO_FOLDER="/data/lx/Video-LLaVA-main/dataset"
VIDEO_FOLDER="/data/codes/lixiang/soccernet/caption_anno_clips"
cd /data/codes/lixiang/Video-LLaVA-main

#HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed  --include localhost:0 videollava/train/train_mem.py \
#HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python videollava/train/train_mem.py \
#--data_path ${JSON_FOLDER}/videochatgpt_tune_.json ${JSON_FOLDER}/nlp_tune.json \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed  videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data/models/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d \
    --version v1 \
    --data_path ${JSON_FOLDER}/soccernet_finetune_train.json\
    --image_folder None \
    --image_tower  tower/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower tower/LanguageBind_Video_merge \
    --audio_tower tower/LanguageBind_Audio \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/Video-LLaVA-Pretrain-7B/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/videollava-7b_my_finetune_0902 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
