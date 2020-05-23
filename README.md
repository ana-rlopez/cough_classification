# Dry/wet cough classifier

The current classifier was based on the features and model employed in a set of research articles and PhD thesis [1,2,3].

## Installation

This repository requires Python 3.6 or greater. In addition, **jupyter-notebook** and **ffmpeg** packages are required (apt-get install <package_name>).
The required Python dependencies can be installed from the provided file **requirements.txt** via terminal as:
```bash
$ pip install -r requirements.txt
```
### Docker container

As alternative, a simple Dockerfile is provided to create a container from which the repository can be run. This option requires having first Docker installed ([instructions here](https://docs.docker.com/engine/install/)). More information about Docker in: https://docs.docker.com

For building and running the Docker container of this repository, run the following commands in your terminal:
```bash
$ make docker-build
$ make docker-run
```
Note: Depending on your Docker installation, you may need to add **sudo** to the 2 aforementioned commands.

After these commands are run, the container runs in the terminal and shows a page address for the Jupyter notebook. Open the page in a browser and you are ready.

## Usage

* The original experiment can be found in the default branch, 'original_experiment'.
The experiment can be run via the Jupyter notebook **cough_classification.ipynb**. 
<br > Please note that this repository is meant to be used with the user's own data. No data is provided in this repository.
<br> Also, the installation instructions provided cover only the usage of this branch.

* Other experiments were done by other collaborators of this repository, which involved using a different model for classification, and are included in the other branches of this repository. These experiments are, though, evaluated in other manner as the original experiment, or they are not yet completed.
* Also, preliminary research on visual diagnostics using computer vision methods was done by another collaborator, and can be found in folder 'visual diagnostics', in 'master' branch

### Data format

The data folder path can be modified in config.py. Wav recordings are expected in the data folder, with the format:
- The filename string (without the .wav extension) consists of substrings delimited by dashes (that is '-' ).
- The last substring of the filename must be the label of the recording. Thus, this substring must be 'Dry' or 'Wet'.
- The second-to-last substring of the filename is the ID of the recording.

Example of filename string following this format:
```bash
'Morning_cough-RFDRFs205-Wet'
```
For this example the ID is 'RFDRFs205' and label is 'Wet'.

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

### Metrics 
The metrics obtained tor evaluate the classification performance were:
* Accuracy
* Recall   
* F1
* Precision


### Data 

This experiment was evaluated with the following data:
The data set consisted of 36 audio wav files of cough sounds, that were scrapped from Youtube videos. The list of videos came from the free supplementary materials provided for the following research article: 'Continuous Sound Collection Using Smartphones and Machine Learning to Measure Cough' (Kvapilova et al., 2019) [4].

For this experiment, each recording was manually edited to include only 3 consecutive coughs (since in future experiments we expect to use only cough recordings in this format). In addition, a doctor annotated the recordings with the labels 'wet'/'dry'.

## Results
The accuracy results were obtained from averaging 20 rounds of: fitting the model to training data + predicting on validation data.

| Item 	| Classifier          	| Accuracy (Avg) 	| Recall (Avg) 	| F1 (Avg) 	| Precision (Avg) 	|
|------	|---------------------	|:--------------:	|:------------:	|:--------:	|:---------------:	|
| 1    	| Logistic regression 	|      0.646     	|     0.629    	|   0.630  	|      0.634      	|


## Settings
The settings of the features extracted can be found in the script 'config.py'

---

## License
[MIT](https://choosealicense.com/licenses/mit/)

## References
[1] Swarnkar, V., Abeyratne, U.R., Chang, A.B. et al. Automatic Identification of Wet and Dry Cough in Pediatric Patients with Respiratory Diseases. Ann Biomed Eng 41, 1016–1028 (2013).

[2] Kosasih, Keegan, et al. "Wavelet augmented cough analysis for rapid childhood pneumonia diagnosis." IEEE Transactions on Biomedical Engineering 62.4 (2014): 1185-1194.

[3] Amrulloh, Yusuf Aziz. "Automated methods for cough assessments and applications in screening paediatric respiratory diseases." (2014).

[4] Kvapilova L, Boza V, Dubec P, Majernik M, Bogar J, Jamison J, Goldsack J, C, Kimmel D, J, Karlin D, R: Continuous Sound Collection Using Smartphones and Machine Learning to Measure Cough. Digit Biomark 2019;3:166-175. doi: 10.1159/000504666
