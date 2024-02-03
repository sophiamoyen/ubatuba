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
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

training_folder  = "../shared_data/training_mini"

print('Loading Dataset')
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder) # Importiere EEG-Dateien, zugehörige Kanalbenennung, Sampling-Frequenz (Hz) und Name (meist fs=256 Hz), sowie Referenzsystem
print('Dataset loaded')

import numpy as np
from wettbewerb import load_references, get_3montages, get_6montages
import pywt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

number_montages = 6
N_samples = 2000 # Number of samples per division
# Decompose the wave
wavelet = 'db4'
scaler = StandardScaler()
features = []
for i,_id in enumerate(ids):
    
    if number_montages == 6:
        _montage, _montage_data, _is_missing = get_6montages(channels[i], data[i])
    else:
        _montage, _montage_data, _is_missing = get_3montages(channels[i], data[i])
        
    _fs = sampling_frequencies[i]
    features_per_id = []

    for j, signal_name in enumerate(_montage):
        features_per_mont = []
        signal = _montage_data[j]
        # Notch-Filter to compensate net frequency of 50 Hz
        signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array(50), n_jobs=2, verbose=False)
        # Bandpassfilter between 0.5Hz and 70Hz to filter out noise
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=3, h_freq=40.0, n_jobs=2, verbose=False)
        # Defining number of divisions for signal
        N_div = len(signal_filter)//N_samples
        # Normalizing data
        norm_montage_data = scaler.fit_transform(signal_filter.reshape(-1,1)).reshape(1,-1)[0]
    
        for i in range(N_div):
            features_per_div = np.zeros(5)
            montage_array = norm_montage_data[i*N_samples:(i+1)*N_samples]
            ca4, cd4, cd3, cd2, cd1 = pywt.wavedec(montage_array, wavelet, level=4)
            montage_dwt = [ca4, cd4, cd3, cd2, cd1]
            
            for w in range(len(montage_dwt)):
                # Calculates the line length feature of the decomposed signal
                features_per_div[w] = np.sum(np.abs(np.diff(montage_dwt[w])))/len(montage_dwt[w]) 

                
            features_per_mont.append(features_per_div)
    
        features_per_id.append(features_per_mont)
        
    for n_piece in range(len(features_per_mont)):
        list_montages = []
        for n_montage in range(len(features_per_id)):
            list_montages += features_per_id[n_montage][n_piece].flatten().tolist()
        features.append(np.array(list_montages))


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

if number_montages == 6:
    column_values = ['mont1_ca4','mont1_cd4','mont1_cd3','mont1_cd2','mont1_cd1',
                     'mont2_ca4','mont2_cd4','mont2_cd3','mont2_cd2','mont2_cd1',
                     'mont3_ca4','mont3_cd4','mont3_cd3','mont3_cd2','mont3_cd1',
                     'mont4_ca4','mont4_cd4','mont4_cd3','mont4_cd2','mont4_cd1',
                     'mont5_ca4','mont5_cd4','mont5_cd3','mont5_cd2','mont5_cd1',
                     'mont6_ca4','mont6_cd4','mont6_cd3','mont6_cd2','mont6_cd1']
else:
    column_values = ['mont1_ca4','mont1_cd4','mont1_cd3','mont1_cd2','mont1_cd1',
                     'mont2_ca4','mont2_cd4','mont2_cd3','mont2_cd2','mont2_cd1',
                     'mont3_ca4','mont3_cd4','mont3_cd3','mont3_cd2','mont3_cd1']
    
df_features = pd.DataFrame(data = features,
                          columns = column_values)


"""
# select top 10 features using mutual_info_classif
selector = SelectKBest(mutual_info_classif, k=5)
X_selected = selector.fit_transform(features, labels)
"""
mutual_info = mutual_info_classif(features, labels)
mutual_info = pd.Series(mutual_info)
mutual_info.index = df_features.columns
print(mutual_info.sort_values(ascending=False))

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

