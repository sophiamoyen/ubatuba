# -*- coding: utf-8 -*-



import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from wettbewerb import load_references, get_3montages
import mne
from scipy import signal as sig
import ruptures as rpt
import json
import pywt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

training_folder  = "../training"

print('Loading Dataset')
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder) # Importiere EEG-Dateien, zugehörige Kanalbenennung, Sampling-Frequenz (Hz) und Name (meist fs=256 Hz), sowie Referenzsystem
print('Dataset loaded')

# Decompose the wave
wavelet = 'db4'
dataset_montage_line_length_array = np.zeros((len(ids),15))
for i,_id in enumerate(ids):
    montage, montage_data, is_missing = get_3montages(channels[i], data[i])
    montage_line_length_array = np.zeros((15))
    for j, signal_name in enumerate(montage):
        ca4, cd4, cd3, cd2, cd1 = pywt.wavedec(montage_data[j], wavelet, level=4)
        montage_line_length_array[(5*j)] = np.sum(np.abs(np.diff(ca4)))/len(ca4)
        montage_line_length_array[(5*j)+1] = np.sum(np.abs(np.diff(cd4)))/len(cd4)
        montage_line_length_array[(5*j)+2] = np.sum(np.abs(np.diff(cd3)))/len(cd3)
        montage_line_length_array[(5*j)+3] = np.sum(np.abs(np.diff(cd2)))/len(cd2)
        montage_line_length_array[(5*j)+4] = np.sum(np.abs(np.diff(cd1)))/len(cd1)
    dataset_montage_line_length_array[i] = montage_line_length_array

features = dataset_montage_line_length_array
labels = np.array(eeg_labels, dtype=int)[:,0]

rf_classifier = RandomForestClassifier()
rf_classifier.fit(features, labels)

# Speichere Modell
joblib.dump(rf_classifier, 'model.joblib') 
'''
with open('model.json', 'w', encoding='utf-8') as f:
    json.dump(rf_classifier.get_params(), f, ensure_ascii=False, indent=4)
    print('Seizure Detektionsmodell wurde gespeichert!')
'''
