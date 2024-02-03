# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------
Script to test automatic detection of onset of seizures on EEG datasets using 
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

import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
from wettbewerb import get_3montages

# Pakete aus dem Vorlesungsbeispiel
import mne
from scipy import signal as sig
import ruptures as rpt


import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from wettbewerb import load_references, get_3montages, get_6montages
import mne
from scipy import signal as sig
from imblearn.under_sampling import RandomUnderSampler
from model_cnn import CNN



###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str, model_name : str='model_cnn.pt') -> Dict[str,Any]:
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
    seizure_present = [0]       # gibt an ob ein Anfall vorliegt
    seizure_confidence = 0.5    # gibt die Unsicherheit des Modells an (optional)
    onset = 4.2                 # gibt den Beginn des Anfalls an (in Sekunden)
    onset_confidence = 0.99     # gibt die Unsicherheit bezüglich des Beginns an (optional)
    offset = 999999             # gibt das Ende des Anfalls an (optional)
    offset_confidence = 0       # gibt die Unsicherheit bezüglich des Endes an (optional)

    
    # Initialize and load the model
    cnn_classifier = CNN(num_classes=2, seq_length=2000)
    A = torch.load(model_name)
    cnn_classifier.load_state_dict(A)
    cnn_classifier.eval()

    """
    ---------------------------------------------------------------------
    Parameters
    ---------------------------------------------------------------------
    """
    N_samples = 2000          # Number of samples per division for the sliding windows (Epochs)
    scaler = StandardScaler() # Scaler chosen for normalization of signal

    # Create an array of signal for each of the 3 montages
    mont1_signal = []
    mont2_signal = []
    mont3_signal = []
    whole_mont = [mont1_signal,mont2_signal,mont3_signal] # Array with all montages
    
    
    # Getting the montages
    _montage, _montage_data, _is_missing = get_3montages(channels, data)
    _fs = fs
    
    for j, signal_name in enumerate(_montage):
            # Get signal from montage
            signal = _montage_data[j]
            
            """
            -------------------- Pre-Processing ---------------------------
            """
            # Notch-Filter to compensate net frequency of 50 Hz
            signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
            # Bandpassfilter between 0.5Hz and 70Hz to filter out noise
            signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
            # Defining number of divisions for signal
            N_div = len(signal_filter)//N_samples
            # Normalizing data
            norm_montage_data = scaler.fit_transform(signal_filter.reshape(-1,1)).reshape(1,-1)[0]
    
            for i in range(N_div):
                # Add Epochs per array
                montage_array = norm_montage_data[i*N_samples:(i+1)*N_samples]
                whole_mont[j].append(montage_array)
        
    
    # Renaming it
    whole_mont_resampled_np = np.array(whole_mont)

    # Konvertieren der Daten in PyTorch Tensoren und Vorbereiten für das Modell
    data_tensor = torch.tensor(whole_mont_resampled_np, dtype=torch.float).permute(1, 0, 2)  # Permutieren zu [Anzahl der Beispiele, Kanäle, Länge]

    # Stellen Sie sicher, dass Ihr Modell und Daten auf dem gleichen Gerät sind
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_classifier = cnn_classifier.to(device)
    data_tensor = data_tensor.to(device)

    # Bereiten Sie die Struktur für die Erfassung der Vorhersagen vor
    seizure_present_array = np.zeros(data_tensor.shape[0], dtype=bool)

    # Vorhersagen durchführen und überprüfen, ob ein Anfall vorhanden ist
    with torch.no_grad():
        predictions_classifier = cnn_classifier(data_tensor)  # Vorhersagen für das gesamte Batch
        predicted_classes = torch.argmax(predictions_classifier, dim=1)
        
        seizure_indices = torch.where(predicted_classes == 1)[0]  # Angenommen, Klasse 1 steht für "seizure"

        # Aktualisieren des seizure_present_array basierend auf den gefundenen Anfall-Indizes
        seizure_present_array[seizure_indices] = True

        # Überprüfen, ob mindestens ein Anfall im Batch vorhergesagt wurde
        if seizure_present_array.any():
            seizure_present = [1]
            # Der erste und der letzte Index in seizure_indices repräsentieren den Beginn und das Ende des Anfalls
            first_seizure_index = seizure_indices[0].item()
            last_seizure_index = seizure_indices[-1].item()

            # Berechnung des Anfallsbeginns in Sekunden
            onset = N_samples * first_seizure_index / fs
            # Berechnung des Anfallsendes in Sekunden
            offset = N_samples * last_seizure_index / fs

            # Annahme: Confidence-Werte sind statisch, könnten dynamisch angepasst werden
            seizure_confidence = 0.8  # Beispielwert, anpassen basierend auf Ihrer Modellausgabe oder Heuristik
            offset_confidence = 0.8  # Beispielwert, anpassen basierend auf Analyse
        else:
            # Falls kein Anfall vorhergesagt wurde, setzen Sie Standardwerte
            seizure_present = [0]
            onset = 0  # Kein Anfall, daher kein Beginn
            offset = 0  # Kein Anfall, daher kein Ende
            seizure_confidence = 0  # Kein Anfall, daher keine Confidence
            offset_confidence = 0  # Kein Anfall, daher keine Confidence
    

     
    
#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
