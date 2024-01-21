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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import pandas as pd


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

training_folder  = "../../shared_data/training_mini"

print('Loading Dataset')
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder) # Importiere EEG-Dateien, zugehörige Kanalbenennung, Sampling-Frequenz (Hz) und Name (meist fs=256 Hz), sowie Referenzsystem
print('Dataset loaded')

N_div = 100 # Numeber of subdivisions
# Decompose the wave
wavelet = 'db4'
dataset_montage_line_length_array = np.zeros((len(ids)*N_div,15))
for i,_id in enumerate(ids):
    montage, montage_data, is_missing = get_3montages(channels[i], data[i])
    montage_line_length_array = np.zeros((15))
    for y in range(N_div):
        for j, signal_name in enumerate(montage):
            montage_divided = np.array_split(montage_data[j],N_div)
            ca4, cd4, cd3, cd2, cd1 = pywt.wavedec(montage_divided[y], wavelet, level=4)
            dwt_array = [ca4, cd4, cd3, cd2, cd1]
            for w in range(len(dwt_array)):
                montage_line_length_array[(5*j)+w] = np.sum(np.abs(np.diff(dwt_array[w])))/len(dwt_array[w])                
        dataset_montage_line_length_array[i*N_div:i*N_div+15] = montage_line_length_array

features = dataset_montage_line_length_array

labels = np.zeros((len(eeg_labels)*N_div))
for i,_id in enumerate(ids):
    if eeg_labels[i][0]:
        onset = eeg_labels[i][1]
        offset = eeg_labels[i][2]
        sample_freq = sampling_frequencies[i]
        total_time = len(data[i][1])/sample_freq
        for num in range(N_div):
            if (total_time/N_div)*(num+1) >= onset and (total_time/N_div)*(num+1) <= offset:
                labels[i*N_div+num] = 1

rf_classifier = rf = RandomForestClassifier(
    n_estimators=500,  # Number of trees in the forest
    max_features="sqrt",  # Number of features to consider at each split
    max_depth=8,  # Maximum depth of each tree
    min_samples_leaf=4,  # Minimum number of samples required to be at a leaf node
)


rf_classifier.fit(features, labels)

# Speichere Modell
joblib.dump(rf_classifier, 'model.joblib') 

