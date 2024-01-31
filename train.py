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
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

training_folder  = "../test_3"

print('Loading Dataset')
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder) # Importiere EEG-Dateien, zugehörige Kanalbenennung, Sampling-Frequenz (Hz) und Name (meist fs=256 Hz), sowie Referenzsystem
print('Dataset loaded')

N_samples = 2000 # Numeber of samples per division
# Decompose the wave
wavelet = 'db4'
scaler = StandardScaler()
normalization = True
features = []
for i,_id in enumerate(ids):
    montage, montage_data, is_missing = get_3montages(channels[i], data[i])
    _fs = sampling_frequencies[i]
    N_div = len(montage_data[0])//N_samples
    
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
    
    for i in range(N_div):
        features_per_div = np.zeros((15))
        montage0_array = norm_montage0_data[i*N_samples:(i+1)*N_samples]
        montage1_array = norm_montage1_data[i*N_samples:(i+1)*N_samples]
        montage2_array = norm_montage2_data[i*N_samples:(i+1)*N_samples]
        ca04, cd04, cd03, cd02, cd01 = pywt.wavedec(montage0_array, wavelet, level=4)
        montage0_dwt = [ca04, cd04, cd03, cd02, cd01]
        ca14, cd14, cd13, cd12, cd11 = pywt.wavedec(montage1_array, wavelet, level=4)
        montage1_dwt = [ca14, cd14, cd13, cd12, cd11]
        ca24, cd24, cd23, cd22, cd21 = pywt.wavedec(montage2_array, wavelet, level=4)
        montage2_dwt = [ca24, cd24, cd23, cd22, cd21]
        for w in range(len(montage0_dwt)):
            features_per_div[w] = np.sum(np.abs(np.diff(montage0_dwt[w])))/len(montage0_dwt[w]) 
            features_per_div[5+w] = np.sum(np.abs(np.diff(montage1_dwt[w])))/len(montage1_dwt[w])
            features_per_div[10+w] = np.sum(np.abs(np.diff(montage2_dwt[w])))/len(montage2_dwt[w])
        features.append(features_per_div)


labels = []
for i,_id in enumerate(ids):
    if eeg_labels[i][0]:
        onset = eeg_labels[i][1]
        offset = eeg_labels[i][2]
        sample_freq = sampling_frequencies[i]
        total_time = len(data[i][1])/sample_freq
        N_div = len(data[i][1])//N_samples
        for num in range(N_div):
            if (((total_time/N_div)*(num) <= onset) and ((total_time/N_div)*(num+1) > onset)) or (((total_time/N_div)*(num) >= onset) and ((total_time/N_div)*(num) < offset)):
                labels.append([1])
            else:
                labels.append([0])
    else:
        N_div = len(data[i][1])//N_samples
        for num in range(N_div):
            labels.append([0])
labels = np.reshape(labels, (1,-1))[0]

rf_classifier = RandomForestClassifier(
    n_estimators=300,  # Number of trees in the forest
    max_features="sqrt",  # Number of features to consider at each split
    max_depth=10,  # Maximum depth of each tree
    min_samples_leaf=1,  # Minimum number of samples required to be at a leaf node
)

oversample = SMOTE()
features_smote, labels_smote = oversample.fit_resample(features,labels)
rf_classifier.fit(features_smote, labels_smote)


# Speichere Modell
joblib.dump(rf_classifier, 'model.joblib') 

