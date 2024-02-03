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



###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str, model_name : str='model.json') -> Dict[str,Any]:
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
    onset = 4.2   # gibt den Beginn des Anfalls an (in Sekunden)
    onset_confidence = 0.99 # gibt die Unsicherheit bezüglich des Beginns an (optional)
    offset = 999999  # gibt das Ende des Anfalls an (optional)
    offset_confidence = 0   # gibt die Unsicherheit bezüglich des Endes an (optional)

    # Hier könnt ihr euer vortrainiertes Modell laden (Kann auch aus verschiedenen Dateien bestehen)
    with open(model_name, 'rb') as f:  
        parameters = json.load(f)         # Lade simples Model (1 Parameter)
        th_opt = parameters['std_thresh']

    
    # Wende Beispielcode aus Vorlesung an 
    
    _montage, _montage_data, _is_missing = get_3montages(channels, data)
    signal_std = np.zeros(len(_montage))
    for j, signal_name in enumerate(_montage):
        # Ziehe erste Montage des EEG
        signal = _montage_data[j]
        # Wende Notch-Filter an um Netzfrequenz zu dämpfen
        signal_notch = mne.filter.notch_filter(x=signal, Fs=fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
        # Wende Bandpassfilter zwischen 0.5Hz und 70Hz um Rauschen aus dem Signal zu filtern
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
        
        # Berechne short time fourier transformation des Signal: signal_filtered = filtered signal of channel, fs = sampling frequency, nperseg = length of each segment
        # Output f= array of sample frequencies, t = array of segment times, Zxx = STFT of signal
        f, t, Zxx = sig.stft(signal_filter, fs, nperseg=fs * 3)
        # Berechne Schrittweite der Frequenz
        df = f[1] - f[0]
        # Berechne Engergie (Betrag) basierend auf Real- und Imaginärteil der STFT
        E_Zxx = np.sum(Zxx.real ** 2 + Zxx.imag ** 2, axis=0) * df
        
        signal_std[j] = np.std(signal_filter)



        # Erstelle neues Array in der ersten Iteration pro Patient
        if j == 0:
            # Initilisiere Array mit Energiesignal des ersten Kanals
            E_array = np.array(E_Zxx)
        else:
            # Füge neues Energiesignal zu vorhandenen Kanälen hinzu (stack it)
            E_array = np.vstack((E_array, np.array(E_Zxx)))
            
    # Berechne Feature zur Seizure Detektion
    signal_std_max = signal_std.max()
    # Klassifiziere Signal
    seizure_present = signal_std_max>th_opt
    
    # Berechne Gesamtenergie aller Kanäle für jeden Zeitppunkt
    E_total = np.sum(E_array, axis=0)
    # Berechne Stelle der maximalen Energie
    max_index = E_total.argmax()

    # Berechne "changepoints" der Gesamtenergie
    # Falls Maximum am Anfang des Signals ist muss der Onset ebenfalls am Anfang sein und wir können keinen "changepoint" berechnen
    if max_index == 0:
        onset = 0.0
        onset_confidence = 0.2
        
    else:
        # Berechne "changepoint" mit dem ruptures package
        # Setup für  "linearly penalized segmentation method" zur Detektion von changepoints im Signal mi rbf cost function
        algo = rpt.Pelt(model="rbf").fit(E_total)
        # Berechne sortierte Liste der changepoints, pen = penalty value
        result = algo.predict(pen=10)
        #Indices sind ums 1 geshiftet
        result1 = np.asarray(result) - 1
        # Selektiere changepoints vor Maximum
        result_red = result1[result1 < max_index]
        # Falls es mindestens einen changepoint gibt nehmen wir den nächsten zum Maximum
        if len(result_red)<1:
            # Falls keine changepoint gefunden wurde raten wir, dass er "nahe" am Maximum ist
            print('No changepoint, taking maximum')
            onset_index = max_index
        else:
            # Der changepoint entspricht gerade dem Onset 
            onset_index = result_red[-1]
        # Gebe Onset zurück
        onset = t[onset_index]      
     
     
    
#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
