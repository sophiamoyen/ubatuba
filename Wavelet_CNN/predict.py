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

from scipy.stats import kurtosis, skew
import antropy as ant
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Model libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC



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
    
    N_samples = 4000 # Numeber of samples per division
    # Decompose the wave
    wavelet = 'db4'
    scaler = StandardScaler()
    normalization = True
    features = []
    montage, montage_data, is_missing = get_3montages(channels, data)
    N_div = len(montage_data[0])//N_samples
    _fs = fs
    # Notch Filter
    montage_data[0] = mne.filter.notch_filter(x=montage_data[0], Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
    montage_data[1] = mne.filter.notch_filter(x=montage_data[1], Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
    montage_data[2] = mne.filter.notch_filter(x=montage_data[2], Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
    # Noise Filter
    montage_data[0] = mne.filter.filter_data(data=montage_data[0], sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
    montage_data[1] = mne.filter.filter_data(data=montage_data[1], sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
    montage_data[2] = mne.filter.filter_data(data=montage_data[0], sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
    
    # Normalizing data
    if normalization:
        norm_montage0_data = scaler.fit_transform(montage_data[0].reshape(-1,1)).reshape(1,-1)[0]
        norm_montage1_data = scaler.fit_transform(montage_data[1].reshape(-1,1)).reshape(1,-1)[0]
        norm_montage2_data = scaler.fit_transform(montage_data[2].reshape(-1,1)).reshape(1,-1)[0]
    else:
        norm_montage0_data = montage_data[0]
        norm_montage1_data = montage_data[1]
        norm_montage2_data = montage_data[2]
    
    new_montages = [norm_montage0_data, norm_montage1_data, norm_montage1_data]
    for i in range(N_div):
        features_per_div = np.zeros((27))
        m = 0
        for signal_filter in new_montages:
            sig_min = np.min(signal_filter) # Min
            sig_max = np.max(signal_filter) # Max
            sig_mean = np.mean(signal_filter) # Mean
            sig_ll = np.sum(np.abs(np.diff(signal_filter))) # Line Length
            sig_std = np.std(signal_filter) #Std Dev
            sig_kurtosis = kurtosis(signal_filter.tolist()) #Kurtosis
            sig_skew = skew(signal_filter.tolist()) #Skewness
            sig_en = np.mean(signal_filter**2) #Energy
            sig_entspec = ant.spectral_entropy(signal_filter,_fs,method='fft') # Entropy Spectral
            
            features_per_div[m:(m+9)] = np.array([sig_min, sig_max, sig_mean, sig_ll, sig_std, sig_kurtosis, sig_skew, sig_en,sig_entspec])
            m += 9
        features.append(features_per_div)
    
    seizure_present_array = np.zeros(len(features),dtype=bool)
    for f in range(len(features)):
        seizure_present_array[f] = rf_classifier.predict(features[f].reshape(1,-1))
    
    for s in range(len(seizure_present_array)):
        if seizure_present_array[s] == 1:
            #print("Epilepsia detectada")
            if seizure_present == [0]:
                onset = N_samples*s/fs
            seizure_present = [1]
    
    print("a")
#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
