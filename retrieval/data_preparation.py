import os
import json
import copy
import random
from PIL import Image
from typing import Dict, Sequence
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import transformers
from transformers import PreTrainedTokenizer, PretrainedConfig
from model import MultimodalEncoder
from argparse import Namespace
from utils import order_pick_k
from torch.utils.data import DataLoader
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']='2,3'



MAX_IMAGE_LENGTH = 128
MAX_VIDEO_LENGTH = 128
IGNORE_INDEX = -100
VIDEO_BASEPATH = '/root/codes/Video-LLaVA-main'

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class MultimodalDataset(Dataset):
    """Dataset for supervised fine-tuning with multimodal data."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: Dict):
        super(MultimodalDataset, self).__init__()
        list_data_dict = []
        data = json.load(open(data_path, "r"))
        for i in data[:500]:
            i['id'] = len(list_data_dict)
            list_data_dict.append(i)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['texts'])
            cur_len = cur_len if ('image' in sample or 'video' in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"


            if 'image' not in sources[0] and 'video' in sources[0]: #and 'audio' not in sources[0]:
                video_file = self.list_data_dict[i]['video']
                video_folder = self.data_args['video_folder']
                video_processor = self.data_args['video_processor']
                video_file = video_file if isinstance(video_file, list) else [video_file]
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                image = [video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video]
                #sources = preprocess_multimodal(copy.deepcopy([e["texts"] for e in sources]), self.data_args)
                sources = copy.deepcopy([e for e in sources])
                text = " ".join([conv['value'] for conv in sources[0]['conversations'] if conv['from'] == 'gpt'])
                
                data_dict = {
                    'input_ids': self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.tokenizer.model_max_length)['input_ids'][0]
                }
                #missing padding using rnn


            elif 'image' not in sources[0] and 'video' in sources[0] and 'audio' in sources[0]:
                video_file = self.list_data_dict[i]['video']
                video_folder = self.data_args['video_folder']
                video_processor = self.data_args['video_processor']
                video_file = video_file if isinstance(video_file, list) else [video_file]
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                video = [video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video]

                audio_file = self.list_data_dict[i]['audio']
                audio_folder = self.data_args['audio_folder']
                audio_processor = self.data_args['audio_processor']
                audio_file = audio_file if isinstance(audio_file, list) else [audio_file]
                audio_file = order_pick_k(audio_file, MAX_VIDEO_LENGTH)
                audio = [os.path.join(audio_folder, file) for file in audio_file]
                audio = [audio_processor(i, return_tensors='pt')['pixel_values'][0] for i in audio]

                image = video + audio
                #sources = preprocess_multimodal(copy.deepcopy([e["texts"] for e in sources]), self.data_args)
                #data_dict = preprocess(sources, self.tokenizer, has_image=True, has_audio=True)

            else:
                sources = copy.deepcopy([e["texts"] for e in sources])
                #data_dict = preprocess(sources, self.tokenizer, has_image=False)

            # if isinstance(i, int):
            #     data_dict = dict(input_ids=data_dict["input_ids"][0]
            #                         )
            if 'image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i]:
                data_dict['image'] = image
            elif self.data_args['is_multimodal']:
                crop_size = {'height': 224, 'width': 224}
                data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]
            return data_dict
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))
            #return None  # Return None or a default value instead of calling __getitem__ recursively
    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print(instances.keys())
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images

            batch['image'] = images
        else:
            raise ValueError(f'pretrain, {instances}')
        return batch
    

    @classmethod
    def from_config(cls, tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
        """Create dataset and collator for supervised fine-tuning."""
        dataset = cls(tokenizer=tokenizer,
                      data_path=data_args.data_path,
                      data_args=data_args)
        return dict(train_dataset=dataset,
                    eval_dataset=None,
                    data_collator=dataset.collate_fn)
    

class MultimodalConfig(PretrainedConfig):
    model_type = "multimodal"
    def __init__(self, mm_projector_type='mlp2x_gelu', mm_vision_select_layer=-2, mm_use_im_start_end=False, mm_use_im_patch_token=False, image_aspect_ratio='pad', group_by_modality_length=True, **kwargs):
        super().__init__(**kwargs)
        self.mm_projector_type = mm_projector_type
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.image_aspect_ratio = image_aspect_ratio
        self.group_by_modality_length = group_by_modality_length

class ImageTowerConfig(MultimodalConfig):
    model_type = "image_tower"
    def __init__(self, image_tower='openai/clip-vit-large-patch14-336', **kwargs):
        super().__init__(**kwargs)
        self.image_tower = image_tower

class VideoTowerConfig(MultimodalConfig):
    model_type = "video_tower"
    def __init__(self, video_tower=f'{VIDEO_BASEPATH}/tower/LanguageBind_Video_merge', **kwargs):
        super().__init__(**kwargs)
        self.video_tower = video_tower

    
if __name__ == "__main__":
    model_args = Namespace(
        model_name_or_path='/data/models/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d', # models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7,
        version="llava_llama_2"
    )

    training_args = Namespace(
        cache_dir=f'{VIDEO_BASEPATH}/cache_dir', #"/opt/cache/huggingface/hub/cache",
        model_max_length=2048,
        data_path=f'{VIDEO_BASEPATH}/dataset/soccernet_json/soccernet_finetune_matchtime_video_train_official_1115.json'
    )

    # Initialize the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )


    tokenizer.pad_token = tokenizer.unk_token

    image_tower_params = ImageTowerConfig()
    video_tower_params = VideoTowerConfig()
    
    print(video_tower_params)

    # Initialize the MultimodalEncoder
    encoder = MultimodalEncoder(video_tower_params=video_tower_params)


    dataset = MultimodalDataset(data_path=training_args.data_path 
    , tokenizer=tokenizer, data_args={
        "video_folder": "/root/codes/soccernet/caption_anno_clips_matchtime_15soffset/caption_anno_clips_matchtime_15soffset",
        "video_processor": encoder.video_tower.video_processor,
        "is_multimodal": True,
        "num_frames": encoder.video_tower.config.num_frames,
    })
    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    # Save video features and input_ids to files
    output_file = "batch_data.pt"

    batch_data = []

    for batch in tqdm(dataloader):
        video_input = batch['image']
        video_features = encoder.encode_videos(video_input)
        print(video_features.shape, batch['input_ids'].shape)
        batch_dict = {
            'video_features': video_features,
            'input_ids': batch['input_ids']
        }
        batch_data.append(batch_dict)

    # Save all batch data to a file
    torch.save(batch_data, output_file)


