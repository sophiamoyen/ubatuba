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
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier

from scipy.stats import kurtosis, skew
import antropy as ant


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

training_folder  = "../../shared_data/training_mini/"

print('Loading Dataset')
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder) # Importiere EEG-Dateien, zugehörige Kanalbenennung, Sampling-Frequenz (Hz) und Name (meist fs=256 Hz), sowie Referenzsystem
print('Dataset loaded')

N_samples = 4000 # Numeber of samples per division
# Decompose the wave
wavelet = 'db4'
scaler = StandardScaler()
normalization = True
features = []

for i,_id in enumerate(ids):
    montage, montage_data, is_missing = get_3montages(channels[i], data[i])
    _fs = sampling_frequencies[i]
    N_div = len(montage_data[0])//N_samples
    print(N_div)
    
    # Notch Filter
    montage_data[0] = mne.filter.notch_filter(x=montage_data[0], Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
    montage_data[1] = mne.filter.notch_filter(x=montage_data[1], Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
    montage_data[2] = mne.filter.notch_filter(x=montage_data[2], Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
    # Noise Filter
    montage_data[0] = mne.filter.filter_data(data=montage_data[0], sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
    montage_data[1] = mne.filter.filter_data(data=montage_data[1], sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
    montage_data[2] = mne.filter.filter_data(data=montage_data[0], sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
    
    if normalization:
    # Normalizing data
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

# select top 10 features using mutual_info_classif

features_names = ['a_min','a_max','a_mean','a_ll','a_std','a_kurt','a_skew','a_energy','a_ent',
                 'b_min','b_max','b_mean','b_ll','b_std','b_kurt','b_skew','b_energy','b_ent',
                 'c_min','c_max','c_mean','c_ll','c_std','c_kurt','c_skew','c_energy','c_ent']

mutual_info = mutual_info_classif(features, labels)
mutual_info = pd.Series(mutual_info)
mutual_info.index = features_names
print(mutual_info.sort_values(ascending=False))
"""
selector = SelectKBest(mutual_info_classif, k=5)
features = selector.fit_transform(features, labels)
"""


# fix class imbalance issue with SMOTE
smote = SMOTE()
features, labels = smote.fit_resample(features, labels)

rf_classifier = rf = RandomForestClassifier(
                    n_estimators=300,  # Number of trees in the forest
                    max_features="sqrt",  # Number of features to consider at each split
                    max_depth=10,  # Maximum depth of each tree
                    min_samples_leaf=1,  # Minimum number of samples required to be at a leaf node
                    )

rf_classifier.fit(features, labels)

# Speichere Modell
joblib.dump(rf_classifier, 'model.joblib') 

