import mne
import numpy as np
from PyEMD import EMD

def preprocess_eeg(eeg_signal, sfreq=250):
    """
    eeg_signal: np.array [n_channels x n_samples]
    sfreq: sampling frequency in Hz (modify as per dataset)
    Returns processed feature vector
    """
    # 1. Bandpass filter 0.5-80 Hz
    eeg_tensor = torch.tensor(eeg_signal, dtype=torch.float32)
    eeg_np = eeg_tensor.numpy()
    # MNE filter expects shape (n_channels, n_times)
    eeg_filtered = mne.filter.filter_data(eeg_np, sfreq, l_freq=0.5, h_freq=80.0, verbose=False)

    # 2. Common Spatial Patterns (CSP)
    # For simplicity, fit CSP on-the-fly if labels available. In practice pre-fit CSP.
    # Here we just simulate applying a pre-learned CSP matrix (random for illustration).
    # Assume n_components = n_channels
    csp_matrix = np.eye(eeg_filtered.shape[0])  # Identity (placeholder)
    eeg_csp = csp_matrix.dot(eeg_filtered)

    # 3. Empirical Mode Decomposition (EMD)
    # Decompose each channel and take first few IMFs as features
    emd = EMD()
    imfs_list = []
    for ch in range(eeg_csp.shape[0]):
        imfs = emd.emd(eeg_csp[ch])
        # Stack first 3 IMFs (pad if fewer)
        n_imfs = imfs.shape[0]
        imfs = np.vstack([imfs, np.zeros((max(0, 3-n_imfs), imfs.shape[1]))])[:3]  # shape (3, N)
        imfs_list.append(imfs)
    # imfs_list: length=n_channels, each 3xN array
    imfs_array = np.stack(imfs_list)  # shape (n_channels, 3, n_samples)
    imfs_features = imfs_array.reshape(-1)  # flatten all IMFs as feature vector

    return torch.tensor(imfs_features, dtype=torch.float32)
