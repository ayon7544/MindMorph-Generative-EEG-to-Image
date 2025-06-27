import torch
from torch.utils.data import Dataset, DataLoader

# Example: load the pre-saved EEG-ImageNet data
data = torch.load("EEG-ImageNet_2.pth","EEG-ImageNet_1.pth")  # user-provided file
# Assume data is a dict: {'signals': [num_trials x channels x time], 'labels': [num_trials], 'image_paths': [...]}

signals = data['eeg']      # shape: (N_trials, n_channels, n_samples)
labels = data['labels']        # shape: (N_trials,)
image_paths = data['image']  # paths to corresponding ImageNet images

class EEGImageDataset(Dataset):
    def __init__(self, signals, labels, image_paths, transform=None):
        self.signals = signals
        self.labels = labels
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        eeg = self.signals[idx]            # EEG: Tensor [channels x time]
        label = self.labels[idx]           # e.g. ImageNet class index
        img = self.load_image(self.image_paths[idx])  # load corresponding image
        if self.transform:
            img = self.transform(img)
        return eeg, img, label

    def load_image(self, path):
        from PIL import Image
        image = Image.open(path).convert('RGB')
        # Resize/crop to model input (e.g. 224x224) and normalize
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        return preprocess(image)

# Create train/test splits and dataloaders
dataset = EEGImageDataset(signals, labels, image_paths)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
