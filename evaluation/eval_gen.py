# evaluate.py

import torch
from torch.utils.data import DataLoader
from scripts.dataloader import EEGImageNetDataset
from scripts.preprocess_eeg import preprocess_eeg_batch
from scripts.encoders import NERVEncoder, VQGANEncoder
from scripts.beip_model import BEIPContrastiveModel
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import numpy as np
import os
from tqdm import tqdm

# -------- Config --------
data_path = "EEG-ImageNet_1.pth"
image_folder = "./imagenet"
batch_size = 32
model_path = "beip_model.pth"
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
_, test_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# -------- Load Models --------
eeg_encoder = NERVEncoder().to(device)
image_encoder = VQGANEncoder().to(device)
beip_model = BEIPContrastiveModel(eeg_dim=512, img_dim=512).to(device)
beip_model.load_state_dict(torch.load(model_path, map_location=device))
beip_model.eval()

# -------- Evaluation --------
cos_scores = []
top1_correct = 0
top5_correct = 0
total_samples = 0

print("🔍 Running Evaluation...")

with torch.no_grad():
    for eeg, img, label in tqdm(test_loader):
        eeg, img = eeg.to(device), img.to(device)

        eeg_features = preprocess_eeg_batch(eeg)
        eeg_embeddings = eeg_encoder(eeg_features)
        img_embeddings = image_encoder(img)

        # Compute cosine similarity between EEG and all image embeddings in batch
        eeg_np = eeg_embeddings.cpu().numpy()
        img_np = img_embeddings.cpu().numpy()
        sim_matrix = cosine_similarity(eeg_np, img_np)

        # Top-1 and Top-5 accuracy
        top_preds = np.argsort(-sim_matrix, axis=1)
        for i in range(len(label)):
            total_samples += 1
            if top_preds[i][0] == i:
                top1_correct += 1
            if i in top_preds[i][:5]:
                top5_correct += 1

        # Store diagonal similarity
        cos_scores.extend(np.diag(sim_matrix))

# -------- Report --------
mean_sim = np.mean(cos_scores)
top1_acc = top1_correct / total_samples
top5_acc = top5_correct / total_samples

print("\n📈 Evaluation Results")
print(f"Mean EEG/Image Cosine Similarity: {mean_sim:.4f}")
print(f"Top-1 Accuracy: {top1_acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")