# Team Ubatuba

This repository contains a compilation of code developed for automatic detection of onset of seizures from EEG signals.
Several techniques were tested and stored in each folder. The best one until now was using sliding windows (Epochs) and a CNN for classification using `pytorch`, which is the one corresponding to the `train.py`and `predict.py`in the main folder.

Other methods used were:

- `9_Featires`: Prediction of seizure using 9 statistical features were extracted from the raw signal from 3 or 6 montages (Mean, Minimum, Maximum, Skewness, Kurtosis, Standard Deviation, Spectral Entropy and Line Length), based on (Siddiqui et. al., 2019 [1]). Then the best features are selected using Mutual Information Gain and fed to a XGBoost Classifier.

- `Wavelet_CNN`: Prediction of seizure using sliding windows (Epochs), based on (Tzimourta et. al., 2019 [2]) and (Bairagi et. al., 2021 [6]), Continous Wavelet Transform (CWT) with the wavelet type Daubechies of order 4, then the scaleogram of the signals are fed into a CNN for prediction of seizure, based on (Mao et. al., 2020 [3]) and [5], for each Epoch and reanalysed as a whole signal for onset detection.

- `DWT_ll`: Prediction of seizure using sliding windows (Epochs), Discrete Wavelet Transform (DWT) with the wavelet type Daubechies of order 4, based on (Guo et. al., 2010 [4]), then the output coefficients are fed into a Random Forest Classifier for prediction of seizure for each Epoch and reanalysed as a whole signal for onset detection.

Usage:

```
python train.py
python predict_pretrained.py --test_dir ../test/
python score.py --test_dir ../test/
```

References

[1] Siddiqui, M.K., Islam, M.Z. & Kabir, M.A. A novel quick seizure detection and localization through brain data mining on ECoG dataset. Neural Comput & Applic 31, 5595–5608 (2019). https://doi.org/10.1007/s00521-018-3381-9
[2] Tzimourta, Katerina D., et al. "A robust methodology for classification of epileptic seizures in EEG signals." Health and Technology 9 (2019): 135-142.
[3] Mao, wei-lung & Fathurrahman, Haris Imam Karim & Lee, Y & Chang, Teng-Wen. (2020). EEG dataset classification using CNN method. Journal of Physics: Conference Series. 1456. 012017. 10.1088/1742-6596/1456/1/012017. 
[4] Guo, Ling, et al. "Automatic epileptic seizure detection in EEGs based on line length feature and artificial neural networks." Journal of neuroscience methods 191.1 (2010): 101-109.
[5] https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
[6] Ramendra Nath Bairagi, Md Maniruzzaman, Suriya Pervin, Alok Sarker, "Epileptic seizure identification in EEG signals using DWT, ANN and sequential window algorithm", Soft Computing Letters,Volume 3, 2021, 100026, ISSN 2666-2221 (https://www.sciencedirect.com/science/article/pii/S2666222121000150)