# SoccerComment⚽️
## 1. video训练
```bash
bash scripts/v1_5/soccernet/matchtime_finetune_video.sh
```
## 2. video + audio 训练
```bash
bash scripts/v1_5/soccernet/matchtime_finetune_video_audio.sh
```
## 3. evaluation
  3.1. 使用checkpoint跑结果
  ```bash
  bash scripts/v1_5/soccernet/matchtime_get_results_json.sh
  ```
  3.2. 使用官方eval程序进行测试
  (修改Predictions_path为之前跑结果的输出地址）
  ```bash
  python EvaluateDenseVideoCaption.py --SoccerNet_path /root/codes/Video-LLaVA-main/dataset/soccernet_json/test_labels/
  --Predictions_path [/root/codes/Video-LLaVA-main/dataset/soccernet_json/test_results_1115_19900/]
  ```
## 4. dataset preparation 训练数据集制作
带audio的训练集制作
```bash
python get_soccernet_dataset.py
```
纯video的训练集制作
```bash
python get_soccernet_video_dataset.py
```
