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
    rf_classifier = joblib.load(model_name)
    
    wavelet = 'db4'
    montage, montage_data, is_missing = get_3montages(channels, data)
    montage_line_length_array = np.zeros((15))
    for j, signal_name in enumerate(montage):
        ca4, cd4, cd3, cd2, cd1 = pywt.wavedec(montage_data[j], wavelet, level=4)
        montage_line_length_array[(5*j)] = np.sum(np.abs(np.diff(ca4)))/len(ca4)
        montage_line_length_array[(5*j)+1] = np.sum(np.abs(np.diff(cd4)))/len(cd4)
        montage_line_length_array[(5*j)+2] = np.sum(np.abs(np.diff(cd3)))/len(cd3)
        montage_line_length_array[(5*j)+3] = np.sum(np.abs(np.diff(cd2)))/len(cd2)
        montage_line_length_array[(5*j)+4] = np.sum(np.abs(np.diff(cd1)))/len(cd1)
    dataset_montage_line_length_array = np.array([montage_line_length_array])
    
    seizure_present = rf_classifier.predict(dataset_montage_line_length_array)
    seizure_present = seizure_present[0]

#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
