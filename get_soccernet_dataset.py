import os
import json
import uuid
import random
import numpy as np

test_urllocal_list = ['england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal', 'england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea', 'england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea', 'england_epl/2015-2016/2015-08-29 - 17-00 Liverpool 0 - 3 West Ham', 'england_epl/2015-2016/2015-09-20 - 18-00 Southampton 2 - 3 Manchester United', 'england_epl/2015-2016/2015-09-26 - 19-30 Newcastle Utd 2 - 2 Chelsea', 'england_epl/2015-2016/2015-10-03 - 19-30 Chelsea 1 - 3 Southampton', 'england_epl/2015-2016/2015-10-24 - 17-00 West Ham 2 - 1 Chelsea', 'england_epl/2015-2016/2015-11-07 - 20-30 Stoke City 1 - 0 Chelsea', 'england_epl/2015-2016/2015-11-08 - 19-00 Arsenal 1 - 1 Tottenham', 'england_epl/2015-2016/2015-12-28 - 20-30 Manchester United 0 - 0 Chelsea', 'england_epl/2015-2016/2016-02-03 - 22-45 Watford 0 - 0 Chelsea', 'england_epl/2015-2016/2016-03-01 - 22-45 Norwich 1 - 2 Chelsea', 'england_epl/2016-2017/2016-08-27 - 14-30 Tottenham 1 - 1 Liverpool', 'england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester', 'england_epl/2016-2017/2016-10-15 - 14-30 Chelsea 3 - 0 Leicester', 'england_epl/2016-2017/2017-01-21 - 15-30 Liverpool 2 - 3 Swansea', 'england_epl/2016-2017/2017-05-06 - 17-00 Leicester 3 - 0 Watford', 'europe_uefa-champions-league/2014-2015/2014-11-04 - 20-00 Zenit Petersburg 1 - 2 Bayer Leverkusen', 'europe_uefa-champions-league/2014-2015/2015-02-24 - 22-45 Manchester City 1 - 2 Barcelona', 'europe_uefa-champions-league/2014-2015/2015-03-10 - 22-45 Real Madrid 3 - 4 Schalke', 'europe_uefa-champions-league/2014-2015/2015-03-17 - 22-45 Monaco 0 - 2 Arsenal', 'europe_uefa-champions-league/2014-2015/2015-04-15 - 21-45 FC Porto 3 - 1 Bayern Munich', 'europe_uefa-champions-league/2014-2015/2015-04-22 - 21-45 Real Madrid 1 - 0 Atl. Madrid', 'europe_uefa-champions-league/2014-2015/2015-05-05 - 21-45 Juventus 2 - 1 Real Madrid', 'europe_uefa-champions-league/2015-2016/2015-09-29 - 21-45 Bayern Munich 5 - 0 D. Zagreb', 'europe_uefa-champions-league/2015-2016/2015-11-03 - 22-45 Real Madrid 1 - 0 Paris SG', 'europe_uefa-champions-league/2015-2016/2015-11-03 - 22-45 Sevilla 1 - 3 Manchester City', 'europe_uefa-champions-league/2015-2016/2015-11-03 - 22-45 Shakhtar Donetsk 4 - 0 Malmo FF', 'europe_uefa-champions-league/2015-2016/2015-11-25 - 22-45 Shakhtar Donetsk 3 - 4 Real Madrid', 'europe_uefa-champions-league/2015-2016/2016-04-05 - 21-45 Bayern Munich 1 - 0 Benfica', 'europe_uefa-champions-league/2016-2017/2016-11-01 - 20-45 Besiktas 1 - 1 Napoli', 'europe_uefa-champions-league/2016-2017/2016-11-01 - 22-45 Manchester City 3 - 1 Barcelona', 'europe_uefa-champions-league/2016-2017/2016-11-23 - 22-45 Arsenal 2 - 2 Paris SG', 'europe_uefa-champions-league/2016-2017/2017-03-08 - 22-45 Barcelona 6 - 1 Paris SG', 'europe_uefa-champions-league/2016-2017/2017-04-12 - 21-45 Bayern Munich 1 - 2 Real Madrid', 'europe_uefa-champions-league/2016-2017/2017-05-02 - 21-45 Real Madrid 3 - 0 Atl. Madrid', 'france_ligue-1/2016-2017/2016-08-28 - 21-45 Monaco 3 - 1 Paris SG', 'france_ligue-1/2016-2017/2016-11-30 - 23-00 Paris SG 2 - 0 Angers', 'germany_bundesliga/2014-2015/2015-05-09 - 16-30 Bayern Munich 0 - 1 FC Augsburg', 'germany_bundesliga/2015-2016/2015-08-29 - 19-30 Bayern Munich 3 - 0 Bayer Leverkusen', 'germany_bundesliga/2015-2016/2015-09-12 - 16-30 Bayern Munich 2 - 1 FC Augsburg', 'germany_bundesliga/2015-2016/2015-10-24 - 16-30 Bayern Munich 4 - 0 FC Koln', 'germany_bundesliga/2015-2016/2015-11-08 - 17-30 Dortmund 3 - 2 Schalke', 'germany_bundesliga/2016-2017/2016-09-10 - 19-30 RB Leipzig 1 - 0 Dortmund', 'germany_bundesliga/2016-2017/2016-10-01 - 19-30 Bayer Leverkusen 2 - 0 Dortmund', 'germany_bundesliga/2016-2017/2016-11-05 - 17-30 Hamburger SV 2 - 5 Dortmund', 'germany_bundesliga/2016-2017/2016-11-19 - 20-30 Dortmund 1 - 0 Bayern Munich', 'germany_bundesliga/2016-2017/2016-12-16 - 22-30 Hoffenheim 2 - 2 Dortmund', 'germany_bundesliga/2016-2017/2017-01-21 - 17-30 SV Werder Bremen 1 - 2 Dortmund', 'germany_bundesliga/2016-2017/2017-01-29 - 19-30 1. FSV Mainz 05 1 - 1 Dortmund', 'germany_bundesliga/2016-2017/2017-03-04 - 17-30 Dortmund 6 - 2 Bayer Leverkusen', 'germany_bundesliga/2016-2017/2017-04-29 - 16-30 Dortmund 0 - 0 FC Koln', 'italy_serie-a/2014-2015/2015-04-29 - 21-45 Juventus 3 - 2 Fiorentina', 'italy_serie-a/2015-2016/2015-08-29 - 21-45 AC Milan 2 - 1 Empoli', 'italy_serie-a/2015-2016/2015-09-20 - 16-00 Genoa 0 - 2 Juventus', 'italy_serie-a/2015-2016/2015-09-27 - 21-45 Inter 1 - 4 Fiorentina', 'italy_serie-a/2016-2017/2016-08-27 - 21-45 Napoli 4 - 2 AC Milan', 'italy_serie-a/2016-2017/2016-09-11 - 16-00 AC Milan 0 - 1 Udinese', 'italy_serie-a/2016-2017/2016-09-20 - 21-45 AC Milan 2 - 0 Lazio', 'italy_serie-a/2016-2017/2016-09-24 - 21-45 Napoli 2 - 0 Chievo', 'italy_serie-a/2016-2017/2016-09-25 - 13-30 Torino 3 - 1 AS Roma', 'italy_serie-a/2016-2017/2016-09-25 - 21-45 Fiorentina 0 - 0 AC Milan', 'italy_serie-a/2016-2017/2016-10-02 - 21-45 AS Roma 2 - 1 Inter', 'italy_serie-a/2016-2017/2016-11-20 - 17-00 Atalanta 2 - 1 AS Roma', 'italy_serie-a/2016-2017/2016-11-26 - 22-45 Empoli 1 - 4 AC Milan', 'italy_serie-a/2016-2017/2016-12-04 - 17-00 Lazio 0 - 2 AS Roma', 'italy_serie-a/2016-2017/2017-01-08 - 17-00 Genoa 0 - 1 AS Roma', 'italy_serie-a/2016-2017/2017-01-29 - 17-00 Sampdoria 3 - 2 AS Roma', 'italy_serie-a/2016-2017/2017-02-07 - 22-45 AS Roma 4 - 0 Fiorentina', 'italy_serie-a/2016-2017/2017-02-25 - 20-00 Napoli 0 - 2 Atalanta', 'italy_serie-a/2016-2017/2017-02-26 - 22-45 Inter 1 - 3 AS Roma', 'italy_serie-a/2016-2017/2017-04-01 - 21-45 AS Roma 2 - 0 Empoli', 'italy_serie-a/2016-2017/2017-04-30 - 21-45 Inter 0 - 1 Napoli', 'italy_serie-a/2016-2017/2017-05-20 - 21-45 Napoli 4 - 1 Fiorentina', 'spain_laliga/2014-2015/2015-02-14 - 20-00 Real Madrid 2 - 0 Dep. La Coruna', 'spain_laliga/2014-2015/2015-04-18 - 21-00 Real Madrid 3 - 1 Malaga', 'spain_laliga/2014-2015/2015-04-25 - 17-00 Espanyol 0 - 2 Barcelona', 'spain_laliga/2014-2015/2015-04-29 - 21-00 Real Madrid 3 - 0 Almeria', 'spain_laliga/2014-2015/2015-05-02 - 17-00 Cordoba 0 - 8 Barcelona', 'spain_laliga/2014-2015/2015-05-09 - 19-00 Barcelona 2 - 0 Real Sociedad', 'spain_laliga/2015-2016/2015-08-29 - 23-30 Real Madrid 5 - 0 Betis', 'spain_laliga/2015-2016/2015-09-19 - 17-00 Real Madrid 1 - 0 Granada CF', 'spain_laliga/2015-2016/2015-11-08 - 18-00 Barcelona 3 - 0 Villarreal', 'spain_laliga/2015-2016/2015-12-05 - 18-00 Real Madrid 4 - 1 Getafe', 'spain_laliga/2015-2016/2015-12-30 - 18-00 Real Madrid 3 - 1 Real Sociedad', 'spain_laliga/2015-2016/2016-02-27 - 18-00 Real Madrid 0 - 1 Atl. Madrid', 'spain_laliga/2015-2016/2016-03-02 - 23-00 Levante 1 - 3 Real Madrid', 'spain_laliga/2015-2016/2016-05-08 - 18-00 Real Madrid 3 - 2 Valencia', 'spain_laliga/2015-2016/2016-05-14 - 18-00 Dep. La Coruna 0 - 2 Real Madrid', 'spain_laliga/2016-2017/2016-09-10 - 21-30 Barcelona 1 - 2 Alaves', 'spain_laliga/2016-2017/2016-09-21 - 21-00 Real Madrid 1 - 1 Villarreal', 'spain_laliga/2016-2017/2016-09-24 - 21-45 Las Palmas 2 - 2 Real Madrid', 'spain_laliga/2016-2017/2016-11-26 - 18-15 Real Madrid 2 - 1 Gijon', 'spain_laliga/2016-2017/2016-12-18 - 22-45 Barcelona 4 - 1 Espanyol', 'spain_laliga/2016-2017/2017-02-11 - 22-45 Osasuna 1 - 3 Real Madrid', 'spain_laliga/2016-2017/2017-03-12 - 22-45 Real Madrid 2 - 1 Betis', 'spain_laliga/2016-2017/2017-04-02 - 17-15 Real Madrid 3 - 0 Alaves', 'spain_laliga/2016-2017/2017-04-08 - 21-45 Malaga 2 - 0 Barcelona', 'spain_laliga/2016-2017/2017-04-26 - 20-30 Barcelona 7 - 1 Osasuna']


def process_and_save_videos(input_folder, meta_filepath, output_folder, num_examples):
    # Define JSON output path
    train_json_output_path = os.path.join(output_folder, 'soccernet_finetune_matchtime_video_train_official_audio_1117.json')
    eval_json_output_path = os.path.join(output_folder, 'soccernet_finetune_matchtime_video_eval_official_audio_1117.json')

    
    # Initialize list to hold all JSON data
    train_json_data_list, eval_json_data_list = [], []

    # Get a list of all video and JSON files in the input folder
    files = os.listdir(input_folder)
    video_files = [f for f in files if f.endswith('.mp4')]
    json_files = [f for f in files if f.endswith('.json')]
    audio_files = [f for f in files if f.endswith('.wav')]

    # meta_npyfile = np.load(meta_filepath, allow_pickle=True).tolist()
    # meta_files = [k for k in meta_npyfile.keys()]
 
    # eval_rate = 0.1
    # eval_num = int(len(video_files) * eval_rate)
    # eval_id = random.sample(list(range(len(video_files))), eval_num)
    # import pdb; pdb.set_trace()
    # Process up to num_examples videos
    audio_num = 0
    for i in range(min(num_examples, len(video_files))):
        video_file = video_files[i]
        audio_file = video_file.replace('.mp4', '.wav')
        json_file = video_file.replace('.mp4', '.json')
        meta_file = video_file.replace('.mp4', '.mp3')
        

        if json_file in json_files:
            with open(os.path.join(input_folder, json_file), 'r') as jf:
                data = json.load(jf)

            # Create a unique ID for each video
            unique_id = str(uuid.uuid4())
            # Structure for LLaVA JSON
            # if audio_file in audio_files:
                # json_data = {
                #     "id": unique_id,
                #     "video": video_file, #os.path.join(input_folder, video_file),
                #     "audio": audio_file,
                #     "conversations": [
                #         {
                #             "from": "human",
                #             "value": "Here is a video clip of a soccer match: <video>.\nHere is the corresponding audio: <audio>.\nWhat happened in the video? Please give an accurate and appealing commentary."
                #         },
                #         {
                #             "from": "gpt",
                #             "value": data.get('anonymized', 'No description available')
                #         }
                #     ]
                # }
            # else:
            json_data = {
                "id": unique_id,
                "video": video_file,
                "audio": None,
                "metadata": None,
                "conversations": [
                    {
                        "from": "human",
                        "value": "Here is a video clip of a soccer match: <video>.\nWhat happened in the video? Please give an accurate and appealing commentary."
                    },
                    {
                        "from": "gpt",
                        "value": data.get('annotation').get('anonymized', 'No description available')
                    }
                ]
            }

            raw_value = json_data['conversations'][0]['value']
            if audio_file in audio_files:
                raw_value = raw_value.replace('\nWhat happened in the video?', 
                                              '\nHere is the corresponding audio: <audio>.\nWhat happened in the video?')
                json_data['audio'] = audio_file
                audio_num += 1
            # if meta_file in meta_files:
            #     metatext = meta_npyfile[meta_file]
            #     raw_value = raw_value.replace('\nWhat happened in the video?', 
            #                                   f'\nHere is the corresponding metadata: {metatext}.\nWhat happened in the video?')
            #     json_data['metadata'] = meta_file
            json_data['conversations'][0]['value'] = raw_value


            # if i in eval_id:
            #     eval_json_data_list.append(json_data)
            # else:
            #     train_json_data_list.append(json_data)
            test_flag=False
            for test_url in test_urllocal_list:
                if test_url.replace('/', '_').replace(' ', '_') in video_file:
                    test_flag=True
                    break
            if test_flag:
                eval_json_data_list.append(json_data)
            else: 
                train_json_data_list.append(json_data)


    # Save the JSON data list to a file
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(audio_num)

    with open(train_json_output_path, 'w') as json_file:
        json.dump(train_json_data_list, json_file, indent=4)
    with open(eval_json_output_path, 'w') as json_file:
        json.dump(eval_json_data_list, json_file, indent=4)

# Usage example
input_folder = '/root/codes/soccernet/caption_anno_clips_matchtime_15soffset/caption_anno_clips_matchtime_15soffset'
output_folder = 'dataset/soccernet_json'
# meta_filepath = '/data/codes/lixiang/Video-LLaVA-main/dataset/soccernet_json/transcriptions_matchtime_audio.npy'
num_examples = 1e10

process_and_save_videos(input_folder, None, output_folder, num_examples)

