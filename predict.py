# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author:  Maurice Rohr, Dirk Schweickard
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
import pywt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler



###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str, model_name : str='model.joblib') -> Dict[str,Any]:
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
    onset = 4.5   # gibt den Beginn des Anfalls an (in Sekunden)
    onset_confidence = 0.99 # gibt die Unsicherheit bezüglich des Beginns an (optional)
    offset = 999999  # gibt das Ende des Anfalls an (optional)
    offset_confidence = 0   # gibt die Unsicherheit bezüglich des Endes an (optional)
    '''
    # Hier könnt ihr euer vortrainiertes Modell laden (Kann auch aus verschiedenen Dateien bestehen)
    with open(model_name, 'rb') as f:  
        parameters = json.load(f)         # Lade simples Model (1 Parameter)
    '''
    rf_classifier = joblib.load(model_name)
    
    N_samples = 2000 # Numeber of samples per division
    # Decompose the wave
    wavelet = 'db4'
    scaler = StandardScaler()
    normalization = True
    features = []
    montage, montage_data, is_missing = get_3montages(channels, data)
    N_div = len(montage_data[0])//N_samples
    _fs = fs
    
    """
    # Notch Filter
    montage_data[0] = mne.filter.notch_filter(x=montage_data[0], Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
    montage_data[1] = mne.filter.notch_filter(x=montage_data[1], Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
    montage_data[2] = mne.filter.notch_filter(x=montage_data[2], Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
    # Noise Filter
    montage_data[0] = mne.filter.filter_data(data=montage_data[0], sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
    montage_data[1] = mne.filter.filter_data(data=montage_data[1], sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
    montage_data[2] = mne.filter.filter_data(data=montage_data[0], sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
    """
    # Normalizing data
    if normalization:
        norm_montage0_data = scaler.fit_transform(montage_data[0].reshape(-1,1)).reshape(1,-1)[0]
        norm_montage1_data = scaler.fit_transform(montage_data[1].reshape(-1,1)).reshape(1,-1)[0]
        norm_montage2_data = scaler.fit_transform(montage_data[2].reshape(-1,1)).reshape(1,-1)[0]
    else:
        norm_montage0_data = montage_data[0]
        norm_montage1_data = montage_data[1]
        norm_montage2_data = montage_data[2]
    
    for i in range(N_div):
        features_per_div = np.zeros((15))
        montage0_array = norm_montage0_data[i*N_samples:(i+1)*N_samples]
        montage1_array = norm_montage1_data[i*N_samples:(i+1)*N_samples]
        montage2_array = norm_montage2_data[i*N_samples:(i+1)*N_samples]
        ca4, cd4, cd3, cd2, cd1 = pywt.wavedec(montage0_array, wavelet, level=4)
        montage0_dwt = [ca4, cd4, cd3, cd2, cd1]
        ca4, cd4, cd3, cd2, cd1 = pywt.wavedec(montage1_array, wavelet, level=4)
        montage1_dwt = [ca4, cd4, cd3, cd2, cd1]
        ca4, cd4, cd3, cd2, cd1 = pywt.wavedec(montage2_array, wavelet, level=4)
        montage2_dwt = [ca4, cd4, cd3, cd2, cd1]
        for w in range(len(montage0_dwt)):
            features_per_div[w] = np.sum(np.abs(np.diff(montage0_dwt[w])))/len(montage0_dwt[w]) 
            features_per_div[5+w] = np.sum(np.abs(np.diff(montage1_dwt[w])))/len(montage1_dwt[w])
            features_per_div[10+w] = np.sum(np.abs(np.diff(montage2_dwt[w])))/len(montage2_dwt[w])
        features.append(features_per_div)
        
        
    seizure_present_array = np.zeros(len(features),dtype=bool)
    for f in range(len(features)):
        seizure_present_array[f] = rf_classifier.predict(features[f].reshape(1, -1))
    
    for s in range(len(seizure_present_array)):
        if seizure_present_array[s] == 1:
            #print("Epilepsia detectada")
            if seizure_present == [0]:
                onset = N_samples*s/fs
            seizure_present = [1]
    

#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
