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
training_folder  = "../../shared_data/training_mini"
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



import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0.0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.f1_score_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, f1_score, model):
        score = f1_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(f1_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(f1_score, model)
            self.counter = 0

    def save_checkpoint(self, f1_score, model):
        if self.verbose:
            self.trace_func(f'Validation F1 Score increased ({self.f1_score_max:.6f} --> {f1_score:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.f1_score_max = f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=2, seq_length=400).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

early_stopping = EarlyStopping(patience=100, verbose=True, delta=0.01, path='checkpoint.pt')

def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs, device):
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_predictions = []
        train_targets = []

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            _, predictions = torch.max(outputs, 1)
            train_predictions.extend(predictions.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_f1 = f1_score(train_targets, train_predictions, average='macro')

        val_losses = []
        val_predictions = []
        val_targets = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                _, predictions = torch.max(outputs, 1)
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_targets, val_predictions, average='macro')

        print(f'Epoch {epoch+1}: Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')

        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()

train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs=200, device=device)

# Nach dem Training das beste Modell laden
model.load_state_dict(torch.load('checkpoint.pt'))

