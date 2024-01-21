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
    seizure_present = False # gibt an ob ein Anfall vorliegt
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
    
    N_div = 100 # Numeber of subdivisions
    # Decompose the wave
    wavelet = 'db4'
    dataset_montage_line_length_array = np.zeros((N_div,15))
    montage, montage_data, is_missing = get_3montages(channels, data)
    montage_line_length_array = np.zeros((15))
    for y in range(N_div):
        for j, signal_name in enumerate(montage):
            montage_divided = np.array_split(montage_data[j],N_div)
            ca4, cd4, cd3, cd2, cd1 = pywt.wavedec(montage_divided[y], wavelet, level=4)
            dwt_array = [ca4, cd4, cd3, cd2, cd1]
            for w in range(len(dwt_array)):
                montage_line_length_array[(5*j)+w] = np.sum(np.abs(np.diff(dwt_array[w])))/len(dwt_array[w])                
        dataset_montage_line_length_array[y*N_div:y*N_div+15] = montage_line_length_array

    features = dataset_montage_line_length_array
    seizure_present_array = np.zeros(len(features))
    for f in range(len(features)):
        seizure_present_array[f] = rf_classifier.predict(features[f])
    
    for s in range(len(seizure_present_array)):
        if seizure_present_array[s] == 1:
            seizure_present == True
    

#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
