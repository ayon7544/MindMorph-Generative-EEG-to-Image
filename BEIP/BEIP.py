import torch.nn.functional as F

# Instantiate encoders
eeg_encoder = NERVEncoder(n_channels=32, in_samples=250, embed_dim=1024).to('cuda')
# Freeze VQGAN encoder (as feature extractor)
for p in vqgan.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(eeg_encoder.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    eeg_encoder.train()
    total_loss = 0.0
    for eeg_batch, img_batch, _ in train_loader:
        # Preprocess EEG and images
        preproc_eegs = [preprocess_eeg(e.numpy()) for e in eeg_batch]  # list of tensors
        eeg_in = torch.stack(preproc_eegs).to('cuda')  # [B, D_feat]
        # Forward EEG encoder
        eeg_emb = eeg_encoder(eeg_in)  # [B, 1024]
        # Forward image encoder
        img = img_batch.to('cuda')
        img_latent = encode_image(img)  # [B, latent_dim]
        img_emb = img_latent  # treat VQGAN latent as embedding
        
        # Normalize embeddings (optional)
        eeg_emb = F.normalize(eeg_emb, dim=1)
        img_emb = F.normalize(img_emb, dim=1)
        
        # Contrastive loss (InfoNCE / CLIP-style)
        logits = torch.matmul(eeg_emb, img_emb.T)  # [B, B] similarity matrix
        labels = torch.arange(logits.size(0)).to('cuda')
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss {total_loss/len(train_loader):.4f}")
