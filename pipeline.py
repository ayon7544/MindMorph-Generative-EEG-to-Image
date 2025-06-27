# pipeline.py

import torch
from torch.utils.data import DataLoader
from scripts.dataloader import EEGImageNetDataset
from scripts.preprocess_eeg import preprocess_eeg_batch
from scripts.encoders import NERVEncoder, VQGANEncoder
from scripts.beip_model import BEIPContrastiveModel
from torchvision import transforms
import os
from PIL import Image

# -------- Config --------
data_path = "EEG-ImageNet_1.pth"
image_folder = "./imagenet"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Transform --------
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------- Load Dataset --------
data = torch.load(data_path, map_location='cpu')
dataset = EEGImageNetDataset(data['dataset'], image_folder, image_transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

# -------- Load Models --------
eeg_encoder = NERVEncoder().to(device)
image_encoder = VQGANEncoder().to(device)
beip_model = BEIPContrastiveModel(eeg_dim=512, img_dim=512).to(device)

# -------- Optimizer --------
optimizer = torch.optim.Adam(beip_model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# -------- Training Loop --------
epochs = 10
for epoch in range(epochs):
    beip_model.train()
    total_loss = 0
    for eeg, img, label in train_loader:
        eeg, img = eeg.to(device), img.to(device)

        # Preprocess EEG signals
        eeg_features = preprocess_eeg_batch(eeg)  # Shape: [B, C, T] -> [B, feature_dim]
        eeg_embeddings = eeg_encoder(eeg_features)
        img_embeddings = image_encoder(img)

        # BEIP contrastive alignment
        loss = beip_model(eeg_embeddings, img_embeddings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# -------- Save Model --------
torch.save(beip_model.state_dict(), "beip_model.pth")
print("✅ BEIP model saved.")
