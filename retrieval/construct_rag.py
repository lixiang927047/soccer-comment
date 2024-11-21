# construct_rag_database.py

import torch
from model import ProjectionMLP
import numpy as np

def construct_rag_database():
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_emb_dim = 512
    text_emb_dim = 768
    audio_emb_dim = 256
    proj_dim = 128

    model = ProjectionMLP(video_emb_dim, text_emb_dim, audio_emb_dim, proj_dim).to(device)
    model.load_state_dict(torch.load('projection_mlp.pth', map_location=device))
    model.eval()

    # Load embeddings
    video_embeddings = np.load('video_embeddings.npy')
    text_embeddings = np.load('text_embeddings.npy')
    audio_embeddings = np.load('audio_embeddings.npy')

    # Convert to torch tensors
    video_embeddings = torch.tensor(video_embeddings, dtype=torch.float32).to(device)
    text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32).to(device)
    audio_embeddings = torch.tensor(audio_embeddings, dtype=torch.float32).to(device)

    # Project embeddings into unified space
    with torch.no_grad():
        video_proj = model.video_mlp(video_embeddings)
        text_proj = model.text_mlp(text_embeddings)
        audio_proj = model.audio_mlp(audio_embeddings)
        # Combine all projected embeddings
        database_embeddings = torch.cat([video_proj, text_proj, audio_proj], dim=0)
        database_labels = ['video'] * len(video_embeddings) + ['text'] * len(text_embeddings) + ['audio'] * len(audio_embeddings)

    # Save the database embeddings and labels
    torch.save(database_embeddings.cpu(), 'database_embeddings.pt')
    np.save('database_labels.npy', database_labels)

if __name__ == "__main__":
    construct_rag_database()
