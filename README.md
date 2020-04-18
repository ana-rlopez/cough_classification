# Dry/wet cough classifier

The current classifier was based on the features and model employed in a set of research articles and PhD thesis. [1,2,3].

## Usage

* The original experiment can be found in default branch, 'original_experiment'.
* Other experiments were done by other collaborators of this repository, which involved using a different model for classification, are included in the other branches. These experiments are, though, evaluated in other manner as the original experiment, or they are not yet completed.
* Also, research on visual diagnostics using computer vision methods was done by other collaborator, and can be found in folder 'visual diagnostics', in 'master' branch.

## Data 
The data set consisted of 36 audio wav files of cough sounds, that were scrapped from Youtube videos. Each recording was manually edited to include only 3 consecutive coughs (since in future experiments we expect to use only cough recordings in this format). In addition, a doctor annotated the recordings with the labels wet/dry.

## Pre-processing
Prior to extracting features, the signal was: 1) downsampled to 16kHz, 2) the level of the signal was normalized (for futures features that may be affected on level, such as wavelets), 3) segmented in cough segments, by removing the silences in the signal 4) high filtered (pre-emphasis filter).
The cough segments were then divided in non-overlapping Hamming-windowed frames of 25ms.
## Features
For each windowed-frame the following features were extracted:
* Mel frequency cepstral coefficients (MFCCs) (and its \delta and \delta\delta features)
* Kurtosis
* Log-energy
* Zero-crossing rate (ZCR)
* Skewness
* Entropy
* Formants (the first 4)
* Fundamental frequency (F0)
To count for the time-series aspect of the signal, the frames were grouped in chunks of 10 consecutive frames, and the mean and standard deviation for all the features in each chunk were computed. The mean and standard deviation values of all features in 1 chunk constituted then one observation vector, to use in training the model (and evaluation).

## Classification model 
The original model used for classification is a logistic regressor with stochastic gradient descent (SGD) algorithm, for classification. 

Prior to training the model, the training set of observations was normalized by fitting a scaler function. The same fit scaler function was applied to the validation set prior to using it for predictions.

## Evaluation
The evaluation of the classifier was performed using cross-validation (specifically, using one-leave-out method). In this case, all the chunk observations belonging to 1 recording are used as validation set, and the rest are used as training.

The measure for evaluating the classification performance was accuracy.

## Results

The accuracy results were obtained from averaging 20 rounds of: fitting the model to training data + predicting on validation data.

---
| Item      |    Classifier    | Accuracy (Avg) |     Output  |
| ------------- |:-------------:| -----:| --------:|
| 1       |     Logistic regression Classifier       | 66.3%| Dry VS Wet Cough 


## Settings
The settings of the features extracted can be found in the script 'config.py'

## Note on visual diagnostics


## Technologies
Project is created with:
* Python 3
* pandas == 1.0.1
* numpy == 1.18.1
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
