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
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

training_folder  = "../../shared_data/training_mini"

print('Loading Dataset')
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder) # Importiere EEG-Dateien, zugehörige Kanalbenennung, Sampling-Frequenz (Hz) und Name (meist fs=256 Hz), sowie Referenzsystem
print('Dataset loaded')

# Código da Sophia (Features)
import numpy as np
from scipy.stats import kurtosis, skew
import antropy as ant

feature = np.zeros((len(ids),(27))) # Empty array for all 9 features for every montage of every id
for i,_id in enumerate(ids):
    # Berechne Montage
    _montage, _montage_data, _is_missing = get_3montages(channels[i], data[i])
    _fs = sampling_frequencies[i]
    id_feature = np.zeros(27)
        
    m = 0
    for j, signal_name in enumerate(_montage):
        # Ziehe erste Montage des EEG
        signal = _montage_data[j]
        # Notch-Filter to compensate net frequency of 50 Hz
        signal_notch = mne.filter.notch_filter(x=signal, Fs=_fs, freqs=np.array([50.,100.]), n_jobs=2, verbose=False)
        # Bandpassfilter between 0.5Hz and 70Hz to filter out noise
        signal_filter = mne.filter.filter_data(data=signal_notch, sfreq=_fs, l_freq=0.5, h_freq=70.0, n_jobs=2, verbose=False)
        
        """Feature calculation"""
        sig_min = np.min(signal_filter) # Min
        sig_max = np.max(signal_filter) # Max
        sig_mean = np.mean(signal_filter) # Mean
        sig_ll = np.sum(np.abs(np.diff(signal_filter))) # Line Length
        sig_std = np.std(signal_filter) #Std Dev
        sig_kurtosis = kurtosis(signal_filter.tolist()) #Kurtosis
        sig_skew = skew(signal_filter.tolist()) #Skewness
        sig_en = np.mean(signal_filter**2) #Energy 
        sig_entspec = ant.spectral_entropy(signal_filter,_fs,method='fft') # Entropy Spectral
        
        id_feature[m:(m+9)] = np.array([sig_min, sig_max, sig_mean, sig_ll, sig_std, sig_kurtosis, sig_skew, sig_en, sig_entspec])
        m += 9
    
    feature[i,:] = id_feature
    
labels = np.array(eeg_labels, dtype=int)[:,0]

column_values = ['a_min','a_max','a_mean','a_ll','a_std','a_kurt','a_skew','a_energy','a_ent',
                 'b_min','b_max','b_mean','b_ll','b_std','b_kurt','b_skew','b_energy','b_ent',
                 'c_min','c_max','c_mean','c_ll','c_std','c_kurt','c_skew','c_energy','c_ent']
df_features = pd.DataFrame(data = feature,
                  index = ids,
                  columns = column_values)


# select top 10 features using mutual_info_classif
selector = SelectKBest(mutual_info_classif, k=5)
X_selected = selector.fit_transform(feature, labels)

mutual_info = mutual_info_classif(feature, labels)
mutual_info = pd.Series(mutual_info)
mutual_info.index = df_features.columns
print(mutual_info.sort_values(ascending=False))

"""
clf = rf = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    max_features="sqrt",  # Number of features to consider at each split
    max_depth=12,  # Maximum depth of each tree
    min_samples_leaf=4,  # Minimum number of samples required to be at a leaf node
)


clf.fit(X_selected, labels)
"""
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf.fit(X_selected, labels)

# Speichere Modell
joblib.dump(clf, 'model.joblib') 

'''
with open('model.json', 'w', encoding='utf-8') as f:
    json.dump(rf_classifier.get_params(), f, ensure_ascii=False, indent=4)
    print('Seizure Detektionsmodell wurde gespeichert!')
'''
