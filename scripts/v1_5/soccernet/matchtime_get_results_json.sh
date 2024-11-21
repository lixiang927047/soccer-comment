#!/bin/bash
HOME_PATH='/data/codes/lixiang/Video-LLaVA-main'
OUTPUT='/data/codes/lixiang/Video-LLaVA-main/dataset/soccernet_json/test_results_1115_9900'
cp -rf /data/codes/lixiang/Video-LLaVA-main/dataset/soccernet_json/test_results  ${OUTPUT}
rm ${OUTPUT}/*/*/*/*.json

cd /data/codes/lixiang/Video-LLaVA-main

# --model_path /data/codes/lixiang/Video-LLaVA-main/checkpoints/videollava-7b_my_finetune_matchtime_video_1107_5.3epoch/checkpoint-12000 \
# --model_path /data/codes/lixiang/Video-LLaVA-main/cache_dir/models--LanguageBind--Video-LLaVA-7B/snapshots/aecae02b7dee5c249e096dcb0ce546eb6f811806 \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed /data/codes/lixiang/Video-LLaVA-main/videollava/eval/my_get_resultsjson.py \
    --model_path /data/codes/lixiang/Video-LLaVA-main/checkpoints/videollava-7b_my_finetune_matchtime_video_1115/checkpoint-9900 \
    --cache_dir /data/codes/lixiang/Video-LLaVA-main/cache_dir/ \
    --video_dir /root/codes/soccernet/caption_anno_clips_matchtime_15soffset/caption_anno_clips_matchtime_15soffset \
    --gt_file /data/codes/lixiang/Video-LLaVA-main/dataset/soccernet_json/soccernet_finetune_matchtime_video_train_1107.json \
    --output_dir ${OUTPUT}