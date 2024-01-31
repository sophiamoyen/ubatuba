import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from wettbewerb import load_references, get_3montages

# EEGNet Modelldefinition
class EEGNet(nn.Module):
    def __init__(self, num_channels=3, time_steps=400):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        reduced_size = time_steps // 4
        self.fc1 = nn.Linear(64 * reduced_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# EEGDataset Klasse
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.float32)

# Daten laden
training_folder = "../../shared_data/training_mini"
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder)

# Daten vorbereiten
def prepare_data(data, channels, time_steps=400):
    num_records = len(data)
    formatted_data = np.empty((num_records, 3, time_steps)) # [Anzahl der Beispiele, 3 Kanäle, Zeitpunkte]

    for i in range(num_records):
        montages, montage_data, _ = get_3montages(channels[i], data[i])
        montage_data_resized = montage_data[:, :time_steps] if montage_data.shape[1] > time_steps else np.pad(montage_data, ((0,0), (0, time_steps - montage_data.shape[1])), 'constant')
        formatted_data[i] = montage_data_resized

    return formatted_data

formatted_data = prepare_data(data, channels)
binary_labels = [label[0] for label in eeg_labels]  # Nehmen Sie nur den ersten Wert (Anfall ja/nein)

# Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(formatted_data, binary_labels, test_size=0.2, random_state=42)

train_dataset = EEGDataset(X_train, y_train)
test_dataset = EEGDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modellinstanz, Verlustfunktion und Optimierer
model = EEGNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trainingsprozess
best_f1 = 0.0
best_model = None

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validierung und Berechnung des F1-Scores
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = outputs.squeeze().round()
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    epoch_f1 = f1_score(all_labels, all_predictions)
    if epoch_f1 > best_f1:
        best_f1 = epoch_f1
        best_model = model.state_dict()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, F1-Score: {epoch_f1:.4f}')

# Speichern des besten Modells
if best_model is not None:
    torch.save(best_model, 'best_model.pth')
