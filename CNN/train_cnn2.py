import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from wettbewerb import load_references, get_3montages
import mne
from scipy import signal as sig
import ruptures as rpt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score

# Daten laden und vorbereiten
training_folder = "../../test_3"
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder)

# EEG-Daten vorbereiten
def prepare_data(data, channels, sampling_frequencies, time_steps=400):
    num_records = len(data)
    formatted_data = np.empty((num_records, 3, time_steps))
    for i in range(num_records):
        _fs = sampling_frequencies[i]
        montages, montage_data, _ = get_3montages(channels[i], data[i])
        for j in range(montage_data.shape[0]):
            signal = montage_data[j, :]
            signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50., 100.]), n_jobs=2, verbose=False)
            signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
            montage_data[j, :] = signal_filter
        montage_data_resized = montage_data[:, :time_steps] if montage_data.shape[1] > time_steps else np.pad(montage_data, ((0,0), (0, time_steps - montage_data.shape[1])), 'constant')
        formatted_data[i] = montage_data_resized
    return formatted_data

formatted_data = prepare_data(data, channels, sampling_frequencies)

# Dataset-Klasse
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(np.array(labels)).long()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

eeg_dataset = EEGDataset(formatted_data, np.array([label[0] for label in eeg_labels]))

# CNN-Modell
class CNN(nn.Module):
    def __init__(self, num_classes, seq_length):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=6, kernel_size=5),
            nn.BatchNorm1d(num_features=6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(6, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )

        # Anpassung für die Berechnung der Größe des linearen Layers
        linear_input_size = self._get_conv_output(seq_length)

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, 3, shape)
            output = self.classifier(input)
            return output.numel()

    def forward(self, x):
        x = self.classifier(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


# Teilen Sie Ihre Daten in Trainings- und Validierungssets
dataset_size = len(eeg_dataset)
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(eeg_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

# Geräteeinstellungen (nutzt GPU, falls verfügbar, sonst CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Definieren Sie Ihr Modell, Verlustfunktion und Optimierer
model = CNN(num_classes=2, seq_length=400).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    predictions, targets = [], []
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        predictions.extend(predicted.view(-1).cpu().numpy())
        targets.extend(labels.view(-1).cpu().numpy())
        
    f1 = f1_score(targets, predictions, average='macro')
    return total_loss / total_samples, total_correct / total_samples, f1

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    predictions, targets = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            predictions.extend(predicted.view(-1).cpu().numpy())
            targets.extend(labels.view(-1).cpu().numpy())

    f1 = f1_score(targets, predictions, average='macro')
    return total_loss / total_samples, total_correct / total_samples, f1

# Trainingszyklus
epochs = 50
for epoch in range(epochs):
    train_loss, train_acc, train_f1 = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
    val_loss, val_acc, val_f1 = validate(model, val_dataloader, criterion, device)
    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")