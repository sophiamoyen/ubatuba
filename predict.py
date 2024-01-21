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
from sklearn.metrics import accuracy_score
import joblib

from train import selector
from scipy.stats import kurtosis, skew
import antropy as ant
from sklearn.ensemble import GradientBoostingClassifier


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
    seizure_present = True # gibt an ob ein Anfall vorliegt
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
    clf = joblib.load(model_name)
    
     # Berechne Montage
    _montage, _montage_data, _is_missing = get_3montages(channels, data)
    id_feature = np.zeros(27)
        
    m = 0
    for j, signal_name in enumerate(_montage):
        # Ziehe erste Montage des EEG
        signal = _montage_data[j]
        # Notch-Filter to compensate net frequency of 50 Hz
        signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
        # Bandpassfilter between 0.5Hz and 70Hz to filter out noise
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
        
        """Feature calculation"""
        sig_min = np.min(signal_filter) # Min
        sig_max = np.max(signal_filter) # Max
        sig_mean = np.mean(signal_filter) # Mean
        sig_ll = np.sum(np.abs(np.diff(signal_filter))) # Line Length
        sig_std = np.std(signal_filter) #Std Dev
        sig_kurtosis = kurtosis(signal_filter.tolist()) #Kurtosis
        sig_skew = skew(signal_filter.tolist()) #Skewness
        sig_en = np.mean(signal_filter**2) #Energy 
        sig_entspec = ant.spectral_entropy(signal_filter,fs,method='fft') # Entropy Spectral
        
        id_feature[m:(m+9)] = np.array([sig_min, sig_max, sig_mean, sig_ll, sig_std, sig_kurtosis, sig_skew, sig_en, sig_entspec])
        m += 9
    
    id_feature = id_feature.reshape(1,-1)
    X_selected = selector.transform(id_feature)
    
    seizure_present = clf.predict(X_selected)

#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!