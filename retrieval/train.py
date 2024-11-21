# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import ProjectionMLP
import numpy as np


# Define the InfoNCE loss function
def info_nce_loss(embeddings_a, embeddings_b, temperature=0.07):
    batch_size = embeddings_a.size(0)
    embeddings_a = nn.functional.normalize(embeddings_a, dim=1)
    embeddings_b = nn.functional.normalize(embeddings_b, dim=1)
    logits = torch.matmul(embeddings_a, embeddings_b.T) / temperature  # [batch_size, batch_size]
    labels = torch.arange(batch_size).to(embeddings_a.device)
    loss_a = nn.CrossEntropyLoss()(logits, labels)
    loss_b = nn.CrossEntropyLoss()(logits.T, labels)
    loss = (loss_a + loss_b) / 2.0
    return loss

# Training function
def train_model(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            video_embs = batch['video_features'].to(device)
            text_embs = batch['input_ids'].to(device)
            #audio_embs = audio_embs.to(device)
            optimizer.zero_grad()
            video_proj, text_proj = model(video_embs, text_embs)#, audio_embs)
            loss_vt = info_nce_loss(video_proj, text_proj)
            #loss_va = info_nce_loss(video_proj, audio_proj)
            #loss_ta = info_nce_loss(text_proj, audio_proj)
            loss = loss_vt #+ loss_va + loss_ta) / 3.0
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")



class VideoFeatureDataset(Dataset):
    def __init__(self, data_file: str):
        self.data = torch.load(data_file)
        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'video_features': self.data[idx]['video_features']
        }



if __name__ == "__main__":
    video_feature_dataset = VideoFeatureDataset('batch_data.pt') #already batched
    #video_feature_dataloader = DataLoader(video_feature_dataset, batch_size=32, shuffle=True)
    print(video_feature_dataset)

    # Initialize model, optimizer, and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_emb_dim = video_feature_dataset[0]['video_features'].shape[1]
    text_emb_dim = video_feature_dataset[0]['input_ids'].shape[1]
    #audio_emb_dim = video_feature_dataset[0].shape[1]
    proj_dim = 128  # Set your projection dimension

    model = ProjectionMLP(video_emb_dim, text_emb_dim, proj_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    train_model(model, video_feature_dataset, optimizer, device, num_epochs=10)

    # Save the trained model
    torch.save(model.state_dict(), 'projection_mlp.pth')
