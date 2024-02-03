# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author:  Maurice Rohr, Dirk Schweickard
"""

from model_cnn import CNN
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
from wettbewerb import get_3montages

# Pakete aus dem Vorlesungsbeispiel
import mne
from scipy import signal as sig
import ruptures as rpt

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from wettbewerb import load_references, get_3montages, get_6montages
import mne
from scipy import signal as sig
from imblearn.under_sampling import RandomUnderSampler

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str, model_name : str='checkpoint.pt') -> Dict[str,Any]:
    '''
    Parameters
    ----------
    channels : List[str]
        Namen der übergebenen Kanäle
    data : ndarray
        EEG-Signale der angegebenen Kanäle
    fs : float
        Sampling-Frequenz der Signale.
    reference_system :  str
        Welches Referenzsystem wurde benutzt, "Bezugselektrode", nicht garantiert korrekt!
    model_name : str
        Name eures Models,das ihr beispielsweise bei Abgabe genannt habt. 
        Kann verwendet werden um korrektes Model aus Ordner zu laden
    Returns
    -------
    prediction : Dict[str,Any]
        enthält Vorhersage, ob Anfall vorhanden und wenn ja wo (Onset+Offset)
    '''

    #------------------------------------------------------------------------------
    # Euer Code ab hier  

    # Initialisiere Return (Ergebnisse)
    seizure_present = [0] # gibt an ob ein Anfall vorliegt
    seizure_confidence = 0.5 # gibt die Unsicherheit des Modells an (optional)
    onset = 4.2   # gibt den Beginn des Anfalls an (in Sekunden)
    onset_confidence = 0.99 # gibt die Unsicherheit bezüglich des Beginns an (optional)
    offset = 999999  # gibt das Ende des Anfalls an (optional)
    offset_confidence = 0   # gibt die Unsicherheit bezüglich des Endes an (optional)

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
    print("Antes de definir o CNN")
    cnn_classifier = CNN(num_classes=2, seq_length=2000)
    print("A")
    checkpoint = torch.load(model_name)
    print("B")
    cnn_classifier.load_state_dict(checkpoint)
    print("C")
    cnn_classifier.eval()

    
    print("Cortando sinal")
    number_montages = 3
    N_samples = 2000 # Number of samples per division
    # Decompose the wave
    wavelet = 'db4'
    scaler = StandardScaler()
    new_signal = []

    mont1_signal = []
    mont2_signal = []
    mont3_signal = []
    whole_mont = [mont1_signal,mont2_signal,mont3_signal]
    for i,_id in enumerate(ids):
    
        if number_montages == 6:
            _montage, _montage_data, _is_missing = get_6montages(channels[i], data[i])
        else:
            _montage, _montage_data, _is_missing = get_3montages(channels[i], data[i])
        
        _fs = sampling_frequencies[i]
        features_per_id = []

        for j, signal_name in enumerate(_montage):
            signal = _montage_data[j]
            # Notch-Filter to compensate net frequency of 50 Hz
            signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
            # Bandpassfilter between 0.5Hz and 70Hz to filter out noise
            signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
            # Defining number of divisions for signal
            N_div = len(signal_filter)//N_samples
            # Normalizing data
            norm_montage_data = scaler.fit_transform(signal_filter.reshape(-1,1)).reshape(1,-1)[0]
    
            for i in range(N_div):
                montage_array = norm_montage_data[i*N_samples:(i+1)*N_samples]
                whole_mont[j].append(montage_array)




    whole_mont_resampled_np = np.array(whole_mont)
    print("Convertendo em torch")
    # Konvertieren der Daten in PyTorch Tensoren und Vorbereiten für das Modell
    data_tensor = torch.tensor(whole_mont_resampled_np, dtype=torch.float).permute(1, 0, 2)  # Permutieren zu [Anzahl der Beispiele, Kanäle, Länge]

    print("Device CUDA")
    # Stellen Sie sicher, dass Ihr Modell und Daten auf dem gleichen Gerät sind
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_classifier = cnn_classifier.to(device)
    data_tensor = data_tensor.to(device)

    # Bereiten Sie die Struktur für die Erfassung der Vorhersagen vor
    seizure_present_array = np.zeros(data_tensor.shape[0], dtype=bool)

    # Vorhersagen durchführen und überprüfen, ob ein Anfall vorhanden ist
    print("antes",seizure_present)
    with torch.no_grad():
        predictions = cnn_classifier(data_tensor)  # Vorhersagen für das gesamte Batch
        predicted_classes = torch.argmax(predictions, dim=1)
        seizure_indices = torch.where(predicted_classes == 1)[0]  # Angenommen, Klasse 1 steht für "seizure"

        if len(seizure_indices) > 0:
            seizure_present = [1]
            print("depois",seizure_present)
            # Erster Anfallindex kann als Anfallsbeginn betrachtet werden
            first_seizure_index = seizure_indices[0].item()
            onset = N_samples * first_seizure_index / fs  # Berechnung des Anfallsbeginns in Sekunden
            offset = N_samples * seizure_indices[-1].item() / fs  # Berechnung des Anfallsendes
            # Annahme: Confidence-Werte sind statisch, könnten dynamisch angepasst werden
            seizure_confidence = 0.8  # Beispielwert, anpassen basierend auf Ihrer Modellausgabe oder Heuristik
            offset_confidence = 0.8  # Beispielwert, anpassen basierend auf Analyse


    
 

     
     
    
#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
