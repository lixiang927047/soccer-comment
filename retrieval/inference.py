# inference.py

import torch
import torch.nn as nn
from model import ProjectionMLP
import numpy as np

def get_text_embedding(text_input):
    # Implement this function to get text embedding from raw text input
    # For example purposes, we'll return a random tensor
    text_emb = np.random.randn(768)
    return text_emb

def get_video_embedding(video_input):
    # Implement this function to get video embedding from raw video input
    video_emb = np.random.randn(512)
    return video_emb

def get_audio_embedding(audio_input):
    # Implement this function to get audio embedding from raw audio input
    audio_emb = np.random.randn(256)
    return audio_emb

def inference(model, query_input, modality, database_embeddings, database_labels, device):
    # Get the embedding of the query input
    if modality == 'text':
        query_emb = get_text_embedding(query_input)
        query_emb = torch.tensor(query_emb).to(device)
        query_proj = model.text_mlp(query_emb.unsqueeze(0))
    elif modality == 'video':
        query_emb = get_video_embedding(query_input)
        query_emb = torch.tensor(query_emb).to(device)
        query_proj = model.video_mlp(query_emb.unsqueeze(0))
    elif modality == 'audio':
        query_emb = get_audio_embedding(query_input)
        query_emb = torch.tensor(query_emb).to(device)
        query_proj = model.audio_mlp(query_emb.unsqueeze(0))
    else:
        raise ValueError("Invalid modality")
    
    # Normalize the embeddings
    query_proj = nn.functional.normalize(query_proj, dim=1)
    database_embeddings = nn.functional.normalize(database_embeddings, dim=1)
    
    # Compute similarities
    similarities = torch.matmul(database_embeddings, query_proj.T).squeeze(1)
    
    # Get top k results
    top_k = similarities.topk(k=5)
    top_k_indices = top_k.indices.cpu().numpy()
    top_k_scores = top_k.values.cpu().numpy()
    
    # Get corresponding labels or data
    results = [(database_labels[i], top_k_scores[idx]) for idx, i in enumerate(top_k_indices)]
    
    return results

if __name__ == "__main__":
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_emb_dim = 512
    text_emb_dim = 768
    audio_emb_dim = 256
    proj_dim = 128

    model = ProjectionMLP(video_emb_dim, text_emb_dim, audio_emb_dim, proj_dim).to(device)
    model.load_state_dict(torch.load('projection_mlp.pth', map_location=device))
    model.eval()

    # Prepare database embeddings (load from saved file)
    database_embeddings = torch.load('database_embeddings.pt').to(device)
    database_labels = np.load('database_labels.npy', allow_pickle=True)

    # Perform inference
    query_input = "Your query text or video or audio data"
    modality = 'text'  # or 'video' or 'audio'
    results = inference(model, query_input, modality, database_embeddings, database_labels, device)
    print("Top 5 retrieval results:")
    for label, score in results:
        print(f"Label: {label}, Similarity Score: {score:.4f}")
