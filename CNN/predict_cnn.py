import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from wettbewerb import get_3montages

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

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels, data, fs, reference_system, model_name='model.json'):
    '''
    ... [Dokumentation und Parameter]
    '''

    # Initialisiere Return (Ergebnisse)
    seizure_present = True # gibt an ob ein Anfall vorliegt
    seizure_confidence = 0.5 # gibt die Unsicherheit des Modells an (optional)
    onset = 4.2   # gibt den Beginn des Anfalls an (in Sekunden)
    onset_confidence = 0.99 # gibt die Unsicherheit bezüglich des Beginns an (optional)
    offset = 999999  # gibt das Ende des Anfalls an (optional)
    offset_confidence = 0   # gibt die Unsicherheit bezüglich des Endes an (optional)
    
    # Laden des trainierten Modells
    model = EEGNet(num_channels=3, time_steps=400)  # Anpassen der Parameter entsprechend
    model.load_state_dict(torch.load(model_name))
    model.eval()

    # Konvertieren der Eingabedaten in das richtige Format
    montages, montage_data, _ = get_3montages(channels, data)
    inputs = montage_data[:, :400] if montage_data.shape[1] > 400 else np.pad(montage_data, ((0,0), (0, 400 - montage_data.shape[1])), 'constant')
    inputs = torch.from_numpy(inputs).float().unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs)
        seizure_present = outputs.item() > 0.5  # Schwellenwert für binäre Klassifikation
        seizure_confidence = outputs.item()

    # Zurückgeben der Vorhersagen im gleichen Format wie zuvor
    prediction = {
        "seizure_present": seizure_present,
        "seizure_confidence": seizure_confidence,
        "onset": 5.0,  # Dummy-Werte, da das Modell Onset/Offset nicht vorhersagt
        "onset_confidence": 0.5,
        "offset": 9999,
        "offset_confidence": 0.5
    }

    return prediction
