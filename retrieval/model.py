# model.py

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from encoder.builder import build_image_tower, build_video_tower, build_audio_tower
from transformers import PretrainedConfig



class MultimodalEncoder(nn.Module):
    def __init__(self, image_tower_params = None, video_tower_params=None, audio_tower_params=None):
        super(MultimodalEncoder, self).__init__()
        
        # Build the encoder towers based on passed parameters
        if image_tower_params is not None:
            self.image_tower = build_image_tower(image_tower_params)
        if video_tower_params is not None:
            self.video_tower = build_video_tower(video_tower_params)
        if audio_tower_params is not None:
            self.audio_tower = build_audio_tower(audio_tower_params)
        
        #self.mm_projector =  这个要不要加，需要测试
        # Freeze the model for inference
        self.freeze()

    def encode_videos(self, videos):  # [mini_b, c, t, h, w]
        b, _, t, _, _ = videos.shape
        video_features = self.video_tower(videos)  # [mini_b, t, n, c]
        return video_features

    def freeze(self):
        """
        Freeze the model (all parameters are set to `requires_grad=False`).
        This ensures the model is used only for inference.
        """
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image=None, video=None, audio=None):
        """
        Forward pass for different modalities (image/video/audio).
        Args:
            - image (tensor): Image input tensor (optional).
            - video (tensor): Video input tensor (optional).
            - audio (tensor): Audio input tensor (optional).
        
        Returns:
            - Encoded features for the given modality.
        """
        if image is not None:
            return self.image_tower(image)  # Forward pass through the image tower
        elif video is not None:
            return self.video_tower(video)  # Forward pass through the video tower
        elif audio is not None:
            return self.audio_tower(audio)  # Forward pass through the audio tower
        else:
            raise ValueError("At least one modality (image, video, or audio) must be provided.")


class ProjectionMLP(nn.Module):
    def __init__(self, video_emb_dim, text_emb_dim, proj_dim):
        super(ProjectionMLP, self).__init__()
        self.video_mlp = nn.Sequential(
            nn.Linear(video_emb_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, proj_dim)
        )
        self.text_mlp = nn.Sequential(
            nn.Linear(text_emb_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, proj_dim)
        )
        # if audio_emb_dim:
        #     self.audio_mlp = nn.Sequential(
        #         nn.Linear(audio_emb_dim, 1024),
        #         nn.ReLU(),
        #         nn.Linear(1024, proj_dim)
        #     )
    
    def forward(self, video_emb, text_emb, audio_emb):
        video_proj = self.video_mlp(video_emb)
        text_proj = self.text_mlp(text_emb)
        #audio_proj = self.audio_mlp(audio_emb)
        return video_proj, text_proj#, audio_proj

if __name__ == "__main__":

    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="right",
    #     use_fast=False,
    # )

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
        def __init__(self, video_tower='/data/mnt/nas/smb_share/dataset/my_soccercomment/Video-LLaVA-main/tower/LanguageBind_Video_merge', **kwargs):
            super().__init__(**kwargs)
            self.video_tower = video_tower

    image_tower_params = ImageTowerConfig()
    video_tower_params = VideoTowerConfig()
    
    print(video_tower_params)



    # Initialize the MultimodalEncoder
    encoder = MultimodalEncoder(video_tower_params=video_tower_params)

    # Example input tensors
    image_input = torch.randn(1, 3, 224, 224)  # Example image tensor
    video_input = torch.randn(1, 3, 16, 224, 224)  # Example video tensor
    audio_input = torch.randn(1, 16000)  # Example audio tensor
    
    # Forward pass through the encoder
    #image_features = encoder(image=image_input)
    video_features = encoder(video=video_input)
    #audio_features = encoder(audio=audio_input)
    
    #print("Image Features:", image_features)
    print("Video Features:", video_features)
    #print("Audio Features:", audio_features)