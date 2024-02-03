# -*- coding: utf-8 -*-


import csv
import matplotlib.pyplot as plt
import numpy as np
import os
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
from wettbewerb import load_references, get_3montages, get_6montages
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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


### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

training_folder  = "../../test_2"

print('Loading Dataset')
ids, channels, data, sampling_frequencies, reference_systems, eeg_labels = load_references(training_folder) # Importiere EEG-Dateien, zugehörige Kanalbenennung, Sampling-Frequenz (Hz) und Name (meist fs=256 Hz), sowie Referenzsystem
print('Dataset loaded')

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
for i,_id in enumerate(ids):
    
    if number_montages == 6:
        _montage, _montage_data, _is_missing = get_6montages(channels[i], data[i])
    else:
        _montage, _montage_data, _is_missing = get_3montages(channels[i], data[i])
        
    _fs = sampling_frequencies[i]
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
"""
print("Sinais divididos")
oversample = SMOTE()
undersample = RandomUnderSampler()

mont1_signal, labels1 = undersample.fit_resample(whole_mont[0],labels)
mont2_signal, labels2 = undersample.fit_resample(whole_mont[1],labels)
mont3_signal, labels3 = undersample.fit_resample(whole_mont[2],labels)

whole_mont_sampled = [mont1_signal,mont2_signal,mont3_signal]
"""
whole_mont_sampled = whole_mont

print("Sinais resampled")

scales = range(1,128)
waveletname = 'morl'
train_size = len(labels)

train_data_cwt = np.ndarray(shape=(train_size, 127, 127, 3))

print("Gerando CWT...")
for i in range(0,train_size):
    for j in range(len(whole_mont_sampled)):
        signal = whole_mont_sampled[j][i]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:127]
        train_data_cwt[i, :, :, j] = coeff_
        
size_train = round(0.9*len(labels))
x_train = train_data_cwt[:train_size,:,:,:]
x_test = train_data_cwt[train_size:,:,:,:]
y_train = labels[:train_size]
y_test = labels[train_size:]

print("CWT gerados...")

img_x = 127
img_y = 127
img_z = 3
input_shape = (img_x, img_y, img_z)
 
batch_size = 16
epochs = 10
num_classes = 2
 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
 
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
print("Adicionado ao modelo...")


model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

print("Compilando modelo...")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
 
print("Dando fit...")
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

# Speichere Modell
joblib.dump(model, 'model.joblib') 

print("Calculando resultados...")
train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))



