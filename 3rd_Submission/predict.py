# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author:  Maurice Rohr, Dirk Schweickard
"""


import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
from wettbewerb import get_3montages, get_6montages

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
    features = []
    _montage, _montage_data, _is_missing = get_6montages(channels, data)
    _fs = fs
    features_per_id = []
    
    for j, signal_name in enumerate(_montage):
        features_per_mont = []
        signal = _montage_data[j]
        # Notch-Filter to compensate net frequency of 50 Hz
        signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array(50), n_jobs=2, verbose=False)
        # Bandpassfilter between 0.5Hz and 70Hz to filter out noise
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=40.0, n_jobs=2, verbose=False)
        # Defining number of divisions for signal
        N_div = len(signal_filter)//N_samples
        # Normalizing data
        norm_montage_data = scaler.fit_transform(signal_filter.reshape(-1,1)).reshape(1,-1)[0]
    
        for i in range(N_div):
            features_per_div = np.zeros(5)
            montage_array = norm_montage_data[i*N_samples:(i+1)*N_samples]
            ca4, cd4, cd3, cd2, cd1 = pywt.wavedec(montage_array, wavelet, level=4)
            montage_dwt = [ca4, cd4, cd3, cd2, cd1]
            
            for w in range(len(montage_dwt)):
                features_per_div[w] = np.sum(np.abs(np.diff(montage_dwt[w])))/len(montage_dwt[w]) 
                
            features_per_mont.append(features_per_div)
            
        features_per_id.append(features_per_mont)
        
    for n_piece in range(len(features_per_mont)):
        list_montages = []
        for n_montage in range(len(features_per_id)):
            list_montages += features_per_id[n_montage][n_piece].flatten().tolist()
        features.append(np.array(list_montages))
        
        
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
                               
                               
        
