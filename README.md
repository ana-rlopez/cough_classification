# Dry/wet cough classifier

The current classifier was originally based on the features and model employed in a set of research articles and PhD thesis. [1,2,3]. However, later on through our experiments the model and features diverged from the original concept.
## Usage
Our latest experiment can be seen by running the Jupyter notebook 'cough_classification.ipynb'.

## Data 
The data set consisted of 36 audio wav files of cough sounds, that we scrapped from Youtube videos. Each recording was manually edited to include only 3 consecutive coughs (since in future experiments we expect to use only cough recordings in this format). In addition, a doctor annotated the recordings with the labels wet/dry.

## Pre-processing
Prior to extracting features, the signal was: 1) downsampled to 16kHz, 2) the level of the signal was normalized (for futures features that may be affected on level, such as wavelets), 3) segmented in cough segments, by removing the silences in the signal 4) high filtered (pre-emphasis filter).
The cough segments were then divided in non-overlapping Hamming-windowed frames of 25ms.
## Features
For each windowed-frame the following features were extracted:
* Mel frequency cepstral coefficients (MFCCs) (and its \delta and \delta\delta features) 
(with z-score normalization)
* Kurtosis
* Log-energy
* Zero-crossing rate (ZCR)
* Skewness
* Entropy
* Formants (the first 4)
* Fundamental frequency (F0)
To count for the time-series aspect of the signal, we grouped the frames in chunks of 10 consecutive frames, and the mean and standard deviation for all the features in each chunk were computed. The mean and standard deviation values of all features in 1 chunk constituted then one observation vector, to use in training the model (and evaluation).
## Classification model 
Originally, we employed as model for classification:

a) a logistic regressor with stochastic gradient descent (SGD) algorithm, for classification. 

However, we later on have implemented:

b) A feed-forward neural network. --> For this model only the MFCC features were employed.

c) a GRU model (sequential model)
We must note also that prior to training the model, the training set of observations was normalized by fitting a scaler function. The same fit scaler function was applied to the validation set prior to using for predicting it.

## Evaluation
The evaluation of the classifier was performed using cross-validation (specifically, using one-leave-out method). While we have in this case 1 observation
In this case, we leave out 1 of the recordings as validation set, and the rest as training.

## Results

---
| Item      |    Classifier    | Accuracy |     Output  |
| ------------- |:-------------:| -----:| --------:|
| 1       |     Logistic regression Classifier       | 75%| Dry VS Wet Cough 

b) Feed-forward network


c) GRU


## Settings
The settings of the features extracted can be found in the script 'config.py'

## Note on visual diagnostics
We persued also research on using computer vision methods to include visual diagnostics to our solution (folder 'visual_diagnostics'). However, for the moment this direction has been discontinued.

## Technologies
Project is created with:
* Python 3
* pandas == 1.0.1
* numpy == 1.18.1
* Keras == 1.4
* keras_self_attention == 0.42.0
* scipy == 1.3.2
* sklearn == 0.21.3
* librosa == 0.7.2 
* pydub
* python_speech_features
* pysptk == 0.1.18

## Setup

To run this project locally, install all requirements as listed above.

### License
[MIT](https://choosealicense.com/licenses/mit/)

### References
[1] Swarnkar, V., Abeyratne, U.R., Chang, A.B. et al. Automatic Identification of Wet and Dry Cough in Pediatric Patients with Respiratory Diseases. Ann Biomed Eng 41, 1016–1028 (2013).

[2] Kosasih, Keegan, et al. "Wavelet augmented cough analysis for rapid childhood pneumonia diagnosis." IEEE Transactions on Biomedical Engineering 62.4 (2014): 1185-1194.

[3] Amrulloh, Yusuf Aziz. "Automated methods for cough assessments and applications in screening paediatric respiratory diseases." (2014).
