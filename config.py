## RECORDINGS DATA SET

import pandas as pd
import sys
import numpy as np

#CHANGE TO YOUR OWN TRAINING SET FOLDER
#folder of training data set
FOLDER_PATH = 'data/YT_set/edited_wavs/'

#RECORDINGS PRE-PROCESSING

fs_targ = 16000 # set all audios to this sampling frequency
n_channels_targ = 1

HPF_skip = False #skip applying pre-emphasis (high-pass) filtering
norm_skip = False #skip normalization step
dB_targ = -28.0 #target level for normalization

##FEATUTRE EXTRACTION

featExtr_skip = False #skip wavs reading + feature extraction steps (if feats pickle file already available)

#initialize data frame of features:
feats = pd.DataFrame([])

#framing

frame_len_s=0.025 #12 segments seemed adequeate in paper, since segments are no longer than 400ms (400ms/12=33.3ms)
frame_step_s=frame_len_s #according to paper: non-overlapping frames

frame_len = int(round(frame_len_s*fs_targ)) #in samples
frame_step = int(round(frame_step_s*fs_targ)) #in samples
win_func =np.hamming #at least for mfcc, as in paper

#mfcc
cep_num= 13 #number of coefficients as in paper (https://link.springer.com/article/10.1007/s10439-013-0741-6)

#lp
lp_ord = int(round(2 + fs_targ/1000)) #standard rule of thumb for LP oder

#formants
nr_formants = 4 #as in paper, first 4 formants
