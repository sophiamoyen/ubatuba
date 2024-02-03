# -*- coding: utf-8 -*-
"""

Script to test de trained model

Authors:
Emanuel Iwanow de Araujo
Sophia Bianchi Moyen
Michael
"""
"""
--------------------------------------------------------
Importing libraries and functions
--------------------------------------------------------
"""
from typing import List, Tuple, Dict, Any

# Pakete aus dem Vorlesungsbeispiel
# Basic libraries
import numpy as np
import pandas as pd

# From the WKI competition
from wettbewerb import load_references, get_3montages, get_6montages

# Signal/Dataset processing
import mne
from scipy import signal as sig
from scipy.stats import kurtosis, skew
import antropy as an
import pywt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Feature Analysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Classification Model
from sklearn.ensemble import RandomForestClassifier

# Libraries to save/open model
import os
import joblib
import json
import csv


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
    
    _montage, _montage_data, _is_missing = get_6montages(channels, data)
    N_samples = 2000           # Number of samples per division
    wavelet = 'db4'            # Wavelet tyoe used for DWT
    num_coef_dwt = 5           # Number of output coefficients for DWT
    num_features = 1           # Number of statistical features extracted
    scaler = StandardScaler()  # Scaler chosen for normalization of signal
    _fs = fs                   # Getting sampling frequency
    
    # Empty list to store features for signal
    features_per_id = []
    
    """
    ------------- Generating Statistical Features from Epochs ---------------

    """
    for j, signal_name in enumerate(_montage):
        features_per_mont = []
        signal = _montage_data[j]

        """
        -------------------- Pre-Processing ---------------------------
        """
        # Notch-Filter to compensate net frequency of 50 Hz its harmonic 100 Hz
        signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array(50), n_jobs=2, verbose=False)
        # Bandpassfilter between 0.5Hz and 70Hz to filter out noise
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=40.0, n_jobs=2, verbose=False)
        # Defining number of divisions for signal according to user-defined N_samples per Epoch
        N_div = len(signal_filter)//N_samples
        # Normalizing data
        norm_montage_data = scaler.fit_transform(signal_filter.reshape(-1,1)).reshape(1,-1)[0]
    
        for i in range(N_div):
            """
            ----------------- Discrete Wavelet Decomposition ----------
            """
            # Creates empty array to store features for this Epoch
            features_per_div = np.zeros(num_features*num_coef_dwt)
            
            # Create Epoch for pre-processing
            montage_array = norm_montage_data[i*N_samples:(i+1)*N_samples]
            
            # Extracting coefficients from Discrete Wavelet Decomposition (DWT)
            ca4, cd4, cd3, cd2, cd1 = pywt.wavedec(montage_array, wavelet, level=4)
            montage_dwt = [ca4, cd4, cd3, cd2, cd1]
            
            m = 0 # Support variable for inserting features in designated location in array
            for signal_coefficient in montage_dwt:
                """
                --------------------- Feature calculation -------------------------
                For each coefficient signal (CA4, CD4, CD3, CD2, CD1) generated by the DWT,
                calculates the features Line Length
                --------------------------------------------------------------------
                """
                # Calculates the LINE LENGTH feature of the decomposed signal
                line_length = np.sum(np.abs(np.diff(signal_coefficient)))/len(signal_coefficient) 
                features_per_div[m] = line_length
                m += 1
                
            # Adds features colected for each Epoch to each Montage array
            features_per_mont.append(features_per_div)
    
        # Adds features from each montage to list of features overall
        features_per_id.append(features_per_mont)
        
    # Formatting lists of features collected into one list "features"
    features = []
    for n_piece in range(len(features_per_mont)):
        list_montages = []
        for n_montage in range(len(features_per_id)):
            list_montages += features_per_id[n_montage][n_piece].flatten().tolist()
        features.append(np.array(list_montages))
        
    """
    ------------------------------------------------------------------------------
    Prediction of seizure onset
    ------------------------------------------------------------------------------
    """
    seizure_present_array = np.zeros(len(features),dtype=bool)
    for f in range(len(features)):
    # Predicts if there is indication of seizure in each Epoch of signal
        seizure_present_array[f] = rf_classifier.predict(features[f].reshape(1, -1))
        
    
    for s in range(len(seizure_present_array)):
        if seizure_present_array[s] == 1:
            if seizure_present == [0]:
                onset = N_samples*s/fs
            seizure_present = [1]

    

#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
