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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History 
history = History()
 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str, model_name : str='model1.joblib') -> Dict[str,Any]:
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
    model = joblib.load(model_name)
    print("Model loaded")
    number_montages = 3
    N_samples = 3000 # Number of samples per division
    # Decompose the wave
    wavelet = 'db4'
    scaler = StandardScaler()
    new_signal = []

    mont1_signal = []
    mont2_signal = []
    mont3_signal = []
    whole_mont = [mont1_signal,mont2_signal,mont3_signal]
    
    if number_montages == 6:
        _montage, _montage_data, _is_missing = get_6montages(channels, data)
    else:
        _montage, _montage_data, _is_missing = get_3montages(channels, data)
        
    _fs = fs
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

    scales = range(1,128)
    waveletname = 'morl'
    test_size = len(whole_mont[0])

    test_data_cwt = np.ndarray(shape=(test_size, 127, 127, 3))

    print("Gerando CWT...")
    for i in range(0,test_size):
        for j in range(len(whole_mont)):
            signal = whole_mont[j][i]
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            coeff_ = coeff[:,:127]
            test_data_cwt[i, :, :, j] = coeff_
            
    print("CWT gerada")
    
    img_x = 127
    img_y = 127
    img_z = 3
    input_shape = (img_x, img_y, img_z)

    batch_size = 16
    epochs = 10
    num_classes = 2

    x_test = test_data_cwt.astype('float32')
    y_test = model.predict(x_test, batch_size=batch_size)
    print("y_test",y_test)
    
    seizure_present_array = np.zeros(len(y_test),dtype=bool)
    y = []
    for m in range(len(y_test)):
        if y_test[m,1] > 0.5:
            seizure_present_array[m] = True
            y.appeend(1)
        else:
            seizure_present_array[m] = False
            y.append(0)
            
    print(y)
    for s in range(len(seizure_present_array)):
        if seizure_present_array[s] == True:
            #print("Epilepsia detectada")
            seizure_present = True
            if seizure_present == True:
                onset = N_samples*s/fs

    
    print("predição",seizure_present)

    

#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
