# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------
Automatic detection of onset of seizures on EEG datasets using 
non-overlapping Sliding Windows (Epochs) and Convolutional Neural Network (CNN)

Authors:
Emanuel Iwanow de Araujo
Sophia Bianchi Moyen
Michael Stivaktakis
-----------------------------------------------------------------------
"""

"""
--------------------------------------------------------
Importing libraries and functions
--------------------------------------------------------
"""

# Basic libraries
import numpy as np

# From the WKI competition
from wettbewerb import load_references, get_3montages, get_6montages

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import mne
from scipy import signal as sig
from imblearn.under_sampling import RandomUnderSampler

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
"""
---------------------------------------------------------------------------
Loading Dataset
---------------------------------------------------------------------------
"""

# Daten laden und vorbereiten
training_folder  = "../training"
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder)

"""
---------------------------------------------------------------------
Parameters
---------------------------------------------------------------------
"""
N_samples = 2000            # Number of samples per division for the sliding windows (Epochs)
scaler = StandardScaler()   # Scaler chosen for normalization of signal

# Create an array of signal for each of the 3 montages
mont1_signal = []
mont2_signal = []
mont3_signal = []
whole_mont = [mont1_signal,mont2_signal,mont3_signal]  # Array with all montages

for i,_id in enumerate(ids):
    """
    ------------------------------------------------------------------------
    ******************** Generating Epochs *********************************
    Each signal is split into smaller Epochs composed of a fixed number of samples
    ------------------------------------------------------------------------
    """
    # Getting montage data
    _montage, _montage_data, _is_missing = get_3montages(channels[i], data[i])
        
    # Getting frequency used for sampling in signal in dataset
    _fs = sampling_frequencies[i]

    for j, signal_name in enumerate(_montage):
        # Get signal from montage
        signal = _montage_data[j]
        """
        -------------------- Pre-Processing ---------------------------
        """
        # Notch-Filter to compensate net frequency of 50 Hz and its harmonic 100 Hz
        signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array(50), n_jobs=2, verbose=False)
        # Bandpassfilter between 0.5Hz and 70Hz to filter out noise
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=3, h_freq=40.0, n_jobs=2, verbose=False)
        # Defining number of divisions for signal according to user-defined N_samples per Epoch
        N_div = len(signal_filter)//N_samples
        # Normalizing data
        norm_montage_data = scaler.fit_transform(signal_filter.reshape(-1,1)).reshape(1,-1)[0]
    
        for i in range(N_div):
            # Adds Epoch to array
            montage_array = norm_montage_data[i*N_samples:(i+1)*N_samples]
            whole_mont[j].append(montage_array)


"""
-------------------------------------------------------------------------
Relabling dataset based on Epochs acquired from signal windowing
-------------------------------------------------------------------------
"""
labels = []
for i,_id in enumerate(ids):
    if eeg_labels[i][0]:
    # If signal contains seizure -> relabels each epoch according to where seizure happens
        onset = eeg_labels[i][1]                 # Getting onset
        offset = eeg_labels[i][2]                # Getting offset
        sample_freq = sampling_frequencies[i]    # Getting frequency 
        total_time = len(data[i][1])/sample_freq # Getting length of signal in seconds
        N_div = len(data[i][1])//N_samples       # Getting number of Epochs for that signal
        
        for num in range(N_div):
            """
            If Epoch starts before/at onset and seizure continues until end of Epoch 
            OR
            If Epoch starts after/at onset and ends before offset

            >> THEN >>             
            Label Epoch as having seizure
            """
            start_epoch_timestep = (total_time/N_div)*(num)
            end_epoch_timestep = (total_time/N_div)*(num+1)
            if ((start_epoch_timestep <= onset) and (end_epoch_timestep > onset)) or ((start_epoch_timestep >= onset) and (start_epoch_timestep < offset)):
                # Seizure present in Epoch
                labels.append([1])
            else:
                # No seizure present in Epoch
                labels.append([0])
                
    else:
    # If signal doesn't contain seizure -> label = False all Epochs
        N_div = len(data[i][1])//N_samples
        for num in range(N_div):
            labels.append([0])
            
# Reshape labels array
labels = np.reshape(labels, (1,-1))[0]



"""
-----------------------------------------------------------------
************ Dealing with Class Imbalance ***********************

To deal with the class imbalance (More Epochs with no seizure than
with seizure), we are doing an Undersampling

-----------------------------------------------------------------
"""
"""
# Instanziierung von RandomUnderSampler
undersample = RandomUnderSampler()

# Erstellen einer Funktion, die das Resampling durchführt
def resample_signal(signal, labels):
    # Anwenden von RandomUnderSampler
    signal_resampled, labels_resampled = undersample.fit_resample(signal, labels)
    return signal_resampled, labels_resampled

# Anwenden der Funktion auf jedes Signal
mont1_signal_resampled, labels_resampled_1 = resample_signal(np.array(mont1_signal), labels)
mont2_signal_resampled, labels_resampled_2 = resample_signal(np.array(mont2_signal), labels)
mont3_signal_resampled, labels_resampled_3 = resample_signal(np.array(mont3_signal), labels)

# Sicherstellen, dass die Labels für alle Signale gleich sind, da sie das gleiche Set von Beispielen repräsentieren sollten
assert np.array_equal(labels_resampled_1, labels_resampled_2)
assert np.array_equal(labels_resampled_1, labels_resampled_3)

labels_resampled = labels_resampled_1

whole_mont_resampled = [mont1_signal_resampled,mont2_signal_resampled,mont3_signal_resampled]


whole_mont_resampled_np = np.array(whole_mont_resampled)
"""

labels_resampled = labels
whole_mont_resampled_np = np.array(whole_mont)

"""
---------------------------------------------------------------------
PyTorch Dataset class for the EEG signals and labels
---------------------------------------------------------------------
"""
# Dataset-Klasse
class EEGDataset(Dataset):
    def __init__(self, whole_mont_resampled_np, labels_resampled):
        # Hier wird angenommen, dass `data` bereits ein Tensor ist. Wenn nicht, sollten Sie `data` in einen Tensor umwandeln.
        self.data = torch.from_numpy(whole_mont_resampled_np).float()
        self.labels = torch.from_numpy(np.array(labels_resampled)).long()

    def __len__(self):
        return self.data.shape[1]  # Anzahl der Beispiele entspricht nun dem zweiten Dimension

    def __getitem__(self, idx):
        # Für jedes Beispiel: Holt das idx-te Beispiel über alle Kanäle
        # Keine Notwendigkeit, permute oder unsqueeze zu verwenden
        sample = self.data[:, idx, :]  # Behält die Form [Kanäle, Länge] bei
        label = self.labels[idx]
        return sample, label

# Initialisierung des EEGDataset mit vorbereiteten Daten und Labels
eeg_dataset = EEGDataset(whole_mont_resampled_np, labels_resampled)

"""
---------------------------------------------------------------------
CNN architecture
The model includes two convolutional layers with batch normalization
and ReLU activation, each followed by max pooling to reduce dimensionality.
The extracted features are then passed through three fully connected layers
with batch normalization and ReLU activation, culminating in a classification layer.
Convolutional Layers: The model begins with two convolutional layers, where the first layer takes 3 input channels and applies filters of size 5 to capture temporal patterns, increasing to 6 output channels. The subsequent convolutional layer further deepens feature extraction, increasing to 16 output channels.
These layers are crucial for identifying spatial and temporal dynamics inherent in EEG signals.
---------------------------------------------------------------------
"""
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

"""
---------------------------------------------------------------------
Divide the dataset into Training and Validation
---------------------------------------------------------------------
"""
dataset_size = len(eeg_dataset)
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(eeg_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)


"""
---------------------------------------------------------------------
Define an Early Stopping class that stops the training 
if the validation F1 is not more improving
---------------------------------------------------------------------
"""
class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0.0, path='checkpoint2.pt', trace_func=print):
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
model = CNN(num_classes=2, seq_length=2000).to(device)

# Gewichte definieren:
weights = torch.tensor([1.0, 1.0], device=device) 
criterion = nn.CrossEntropyLoss(weight=weights)
#criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

early_stopping = EarlyStopping(patience=50, verbose=True, delta=0.01, path='checkpoint2.pt')

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
        train_precision = precision_score(train_targets, train_predictions, average='macro')
        train_recall = recall_score(train_targets, train_predictions, average='macro')

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
        val_precision = precision_score(val_targets, val_predictions, average='macro')
        val_recall = recall_score(val_targets, val_predictions, average='macro')

        print(f'Epoch {epoch+1}: Train F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}')
        print(f'Epoch {epoch+1}: Val F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')

        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()

train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs=300, device=device)

# Nach dem Training das beste Modell laden
model.load_state_dict(torch.load('checkpoint2.pt'))

