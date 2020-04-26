"""
This module includes functions to compute features for the given signal, and other feature-related functions.

Most of the features included in this module (and their settings) are based on the following references:
[1] Swarnkar, V., Abeyratne, U.R., Chang, A.B. et al. Automatic Identification of Wet and Dry Cough in Pediatric Patients with Respiratory Diseases. Ann Biomed Eng 41, 1016–1028 (2013).
[2] Kosasih, Keegan, et al. "Wavelet augmented cough analysis for rapid childhood pneumonia diagnosis." IEEE Transactions on Biomedical Engineering 62.4 (2014): 1185-1194.
[3] Amrulloh, Yusuf Aziz. "Automated methods for cough assessments and applications in screening paediatric respiratory diseases." (2014).
"""

import math
import configparser
import sys
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import lfilter

import python_speech_features
import librosa
import pydub
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import mediainfo
from pydub.playback import play

#TODO: re-check! seems not to work?
def get_RMS(s):
    """
    Compute RMS value of input signal.
    
    Args:
        s (numpy array): input signal.
    
    Returns:
        rms_s (float): RMS value of the input signal.
    """
    rms_s = np.sqrt(np.mean(np.power(s,2)))
    
    #TODO: add a function to compute dB, or add it here as option
    #convert to dB scale
    #s_db = 20*np.log10(s_rms/1.0)
    return rms_s

#TODO: re-check! seems not to work either?
def RMS_normalization(s, dB_targ):
    """
    Perform RMS-based normalization of a signal, based on a target level.
    
    Args:
        s (numpy array): input signal to be normalized.
        dB_targ (float/int): target level (in dB).
    
    Returns:
        scaled_s (numpy array): normalized signal.
    """
    #desired level is converted to linear scale
    rms_targ = 10**(dB_targ/20.0)
    
    #compute scaling factor
    scale = rms_targ/get_RMS(s)
    
    #scale amplitude of input signal
    scaled_s = scale*s
    
    return scaled_s

def match_target_amplitude(audioSegment_sound, target_dBFS):
    """
    Match the amplitude of an input signal to a target level.
    
    Args:
        audioSegment_sound (Audiosegment  (pydub)): input signal.
        target_dBFS (float/int): target level (in dBFS).
    
    Returns:
       matched_audio (Audiosegment  (pydub)): amplitude-matched signal.
    """
    dBFS_diff = target_dBFS - audioSegment_sound.dBFS
    matched_audio = audioSegment_sound.apply_gain(dBFS_diff)
    return matched_audio

def apply_preEmph(x):
    """
    Apply pre-emphasis (high-pass) filter.
    
    Args:
        x (numpy array): input signal.
    
    Returns:
        x_filt: high-pass-filtered signal.
    
    """
    x_filt = lfilter([1., -0.97], 1, x)
    return x_filt
        
def autocorr(x):
    """
    Compute autocorrelation of the input signal.
    
    Args:
        x (numpy array): input signal.
        
    Returns:
        autocorr_x (float): autocorrelation of the signal.
    """
    result = np.correlate(x, x, mode='full')
    autocorr_x = result[int((result.size+1)/2):] #Note: other people use re.size/2:, but this does not work for me 
                                   # TODO: check consistency in other computers   
    return autocorr_x

def get_zcr(x):
    """
    Compute zero-crossing rate of the input signal.
    
    Args:
        x (numpy array): input signal.
        
    Returns:
        zcr (float): zero-crossing rate of the signal.
    """
    zcr = (((x[:-1] * x[1:]) < 0).sum())/(len(x)-1)
    return zcr

def get_logEnergy(x):
    """
    #Compute log-energy of the input signal.
    
    Args:
        x (numpy array): input signal.
        
    Returns:
        logEnergy (float): log-energy of the signal.
    """
    logEnergy = np.log10( ( (np.power(x,2)).sum()/len(x) ) + sys.float_info.epsilon)  
    return logEnergy

def get_F0(x,fs):
    """
    Estimate fundamental frequency (F0) of the input signal using the autocorrelation-based method.
    
    Args:
        x (numpy array): input signal.
        fs (int): sampling frequency of x.
        
    Returns:
        F0 (float): estimated fundamental frequency of the signal.
    """
    xcorr_arr = autocorr(x)
    
    #looking for F0 in the frequency interval 50-500Hz, but we search in time domain
    min_ms = round(fs/500)
    max_ms = round(fs/50)
    
    xcorr_slot = xcorr_arr[max_ms+1:2*max_ms+1]
    xcorr_slot = xcorr_slot[min_ms:max_ms]
    t0 = np.argmax(xcorr_slot)
    F0 = fs/(min_ms+t0-1)
    return F0

#Estimate formants
def get_formants(x, lp_order, nr_formants,fs):
    """
    Estimate formants of the input signal using Levinson-Durbin method.
    
    Args:
        x (numpy array): input signal.
        lp_order (int): order of LPC used in the estimation.
        nr_formants (int): number of formants to return (starting from first one).
        fs (float): sampling frequency of x.
        
    Returns:
        first_formants (list): first estimated nr_formants formants.
    """
    #compute lp coefficients
    a = librosa.lpc(x, lp_ord)
    
    #get roots from lp coefficients
    rts = np.roots(a)
    rts = [r for r in rts if np.imag(r) >= 0]

    #get angles
    angz = np.arctan2(np.imag(rts), np.real(rts))

    #get formant frequencies
    formants = sorted(angz * (fs / (2 * math.pi)))
    
    first_formants = formants[0:nr_formants]
    return first_formants

def get_entropy(x, type='shannon'):
    """
    Compute entropy of input signal.
    
    Args:
        x (numpy array): input signal.
        type of entropy: 'shannon' (default), 'natural' or 'hartley'.
        
    Returns:
        entropy_x (list of floats): first estimated nr_formants formants.
    """
    #Note: default type of entropy is Shannon entropy, since this is the one used by [3].
    base = {'shannon' : 2., 'natural' : math.exp(1), 'hartley' : 10.}
    N = len(x)

    if N <= 1:
        return 0

    value,counts = np.unique(x, return_counts=True)
    probs = counts / N
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    #initialization
    entropy_x = 0. 

    # compute entropy
    for i in probs:
        entropy_x -= i * math.log(i+sys.float_info.epsilon, base[type])
    
    return entropy_x

def feature_extraction(x,feats_df,ID,label,config):
    """
    Extract features from signal x (identified as ID), and concatenate them to dataframe feats_df.
    
    Args:
        x (numpy array): input signal.       
        feats_df (dataframe): dataframe to store features.
        ID (string): identification of x.
        label (string): class 'Dry' or 'Wet', as which x has been labeled.
        config (ConfigParser): configuration file
     
     Returns:
        feats_df (dataframe): dataframe to store features, and to which has been appended the features of input x.
    """
    #segment x in frames (to compute features in a frame-basis)
    
    
    fs = config.getint('preprocess','fs_targ')
    if config.get('featExtraction','win_type') == 'hamming':
        win_func= np.hamming
    frame_len_s = config.getfloat('featExtraction','frame_len_s')
    frame_len = int(round(frame_len_s*fs)) #frame length in samples
    frame_step_s = config.getfloat('featExtraction','frame_step_s')
    frame_step = int(round(frame_step_s*fs)) #frame step in samples    
    
    x_frames = python_speech_features.sigproc.framesig(x,frame_len,frame_step,win_func)
    nr_frames = x_frames.shape[0]
    #print(nr_frames)
        
    #0)Wavelets #TODO
    
    #DOUBT: if log-energy feature is included, should I also include the first mfcc coefficient (c0) ?
    #1)mfcc
    mfcc_feat = python_speech_features.mfcc(x,fs, winlen=frame_len_s,winstep=frame_step_s, numcep=config.getint('featExtraction','cep_num'),winfunc=win_func)
    
    #deltas to capture time-varying aspect of signal
    mfcc_delta_feat = python_speech_features.delta(mfcc_feat,1) #mfcc_delta_feat = np.subtract(mfcc_feat[:-1], mfcc_feat[1:]) #same
    mfcc_deltadelta_feat = python_speech_features.delta(mfcc_delta_feat,1)          
    
    #2)zero-crossing rate
    zcr_feat = np.apply_along_axis(get_zcr, 1, x_frames)
    
    #3)Formant frequencies
    
    #Note: for the moment, it seems some frames are ill-conditioned for LP computing.
    #current solution - we skip those and fill with NaN values.
    formants_feat= np.empty((nr_frames,4))
    formants_feat[:] = np.nan
    
    for i_frame in range(0,nr_frames):
        try: 
            formants_feat[i_frame] = get_formants(x_frames[i_frame], config.getint('featExtraction','lp_ord'), config.getint('featExtraction','nr_formants'),fs)
        except:
            pass
    
    #4)Log-energy
    logEnergy_feat =  np.apply_along_axis(get_logEnergy, 1, x_frames)
    
    #5)Pitch (F0)
    F0_feat =  np.apply_along_axis(get_F0, 1, x_frames,fs)
    
    #TODO: compute also F0 with pysptk (a python wrapper for SPTK library), it probably gives better results
    #https://github.com/r9y9/pysptk/blob/master
    #F0_feat = pysptk.rapt(x.astype(np.float32), fs=fs, hopsize=frame_step, min=50, max=500, ,voice_bias=0.0 ,otype=\"f0\")
    #right frame size???
    
    #6)Kurtosis
    kurt_feat =  np.apply_along_axis(kurtosis, 1, x_frames)
    
    #7)Bispectrum Score (BGS)
    #TODO: see [3] for more info on this feature
    
    #8)Non-Gaussianity Score (NGS)
    #TODO: see [3] for more info on this feature
   
    #9) Adding skewness as measure of non-gaussianity (not in paper)
    skew_feat =  np.apply_along_axis(skew, 1, x_frames)
    
    #10) (Shannon) Entropy
    entropy_feat = np.apply_along_axis(get_entropy, 1, x_frames)
    
    
    mfcc_cols = ['mfcc_%s' % s for s in range(0,config.getint('featExtraction','cep_num'))]
    mfcc_delta_cols = ['mfcc_d%s' % s for s in range(0,config.getint('featExtraction','cep_num'))]
    mfcc_deltadelta_cols = ['mfcc_dd%s' % s for s in range(0,config.getint('featExtraction','cep_num'))]
    formants_cols = ['F%s' % s for s in range(1,config.getint('featExtraction','nr_formants')+1)]
          
    feats_segment = pd.concat([pd.DataFrame({'Id': ID, 'kurt': kurt_feat, 'logEnergy': logEnergy_feat,
                                                 'zcr': zcr_feat, 'F0': F0_feat,
                                                 'skewness': skew_feat, 'label': label, 'entropy':entropy_feat}),
                               pd.DataFrame(mfcc_feat,columns=mfcc_cols), 
                            pd.DataFrame(formants_feat,columns=formants_cols)],axis=1)
    
    #print(nr_frames)
    feats_df = feats_df.append(feats_segment,ignore_index=True, sort=False)
    
    return feats_df

def feature_extraction_Step(all_s,all_id,all_label,config):
    """
    Extract features for all signals included in all_s
    Args:
        all_s (list of numpy arrays): list of signals from which features are extracted.
        all_id (list of strings): corresponding list of IDs for signals in all_s.
        all_label (list of strings): corresponding list of labels ('Dry' or 'Wet') for signals in all_s.
        config (ConfigParser): configuration file
          
     Returns:
        feats_df (dataframe): dataframe that store features of all signals in all_s.
    """   
    #initialize data frame of features:
    feats_df = pd.DataFrame([])

    for s, ID, label in zip(all_s,all_id,all_label):

            #Pre-processing of the signals:

            ## 0 ) Resampling to target sampling frequency:
            s = s.set_frame_rate(config.getint('preprocess','fs_targ'))
            
            ## 1) Match amplitude to target level
            if config.getboolean('preprocess', 'norm_skip') is False:
                s=match_target_amplitude(s, config.getfloat('preprocess','dB_targ'))
                #print(s.rms)

            ## 2) Segmentation of cough streams (silence-based)
            #min_silence_len in ms, silence_thresh in dB
            s_segments = split_on_silence (s, min_silence_len = 600, silence_thresh =s.dBFS-10)

            #checks that segmentation and removal of silence is OK
            #print(len(s_segments))
            #play(s)
            #input("Press Enter to continue...")               
            #for i in range(len(s_segments)):
            #    play(s_segments[i])
            #    input("Press Enter to continue...")               

            ## 3) Convert s_segments to numpy array format
            AudioSegment2numpy_arr = lambda x: np.asarray(x.get_array_of_samples())
            s_segments_np = list(map(AudioSegment2numpy_arr, s_segments))

            ## 4) Pre-emphasis filtering on each segment
            if config.getboolean('preprocess','HPF_skip') is False:
                print('High-pass filtering...')       
                preEmph_filtering = lambda x: apply_preEmph(x)
                s_segments_filt = list(map(preEmph_filtering, s_segments_np))
            else:
                s_segments_filt = s_segments_np

            #Feature extraction for each segment
            print('Computing features...')
            for idx, seg_i in enumerate(s_segments_filt):
                #print('\tSegment %d' % idx)
                feats_df = feature_extraction(seg_i,feats_df,ID,label,config)
                
    return feats_df

def processingNaNvalues(feats_df):
    """
    Process rows of input dataframe containing NaN values. (At the moment, these rows are removed).
    
    Args:
        feats_df (dataframe): dataframe that store features.
        
     Returns:
        feats2_df(dataframe): same dataframe as input one, but the NaN rows have been processed.
    """
    #TODO: decide best way to deal with this. Do interpolation, or do dropping rows?  
    ##Check which columns have NaNs values
    #feats_df2 = feats_df.copy()
    #sum(feats_df.isna().any())
    #feats_df.columns[feats_df.isna().any()].tolist() --> We get just the ones we have inserted in formants
    #feats2_df = feats_df.interpolate(method ='cubic')
    feats2_df = feats_df.dropna(axis=1).copy()
    #feats2_df.dropna(axis=0, how="any", thresh=None, subset=None, inplace=True)
    #feats2_df.columns[feats2_df.isna().any()].tolist()
    #feats2_df.describe()
    #sum(feats2_df.isna().any())
    return feats2_df

def createLabelDict_addLabel2df(feats_df):
    """
    Make dictionary associating IDs of signals to the signal's label.
    
    Args: 
        feats_df (dataframe): dataframe that store features.
        
    Returns:
        label_dict: dictionary with labels associated with IDs of signals.
    """
    feats_df_unique = feats_df.drop_duplicates(subset=['Id'])
    label_dict = dict(zip(feats_df_unique.Id, feats_df_unique.label))
    return label_dict

def frame_mean_std_chunk_modeling (feats_df, label_dict, nr_frames):
    """
    Group the frames from a same recording (Id) into chunks of several consecutive frames. \ Then compute mean and standard deviation of these chunks.
    
    Args:
        feats_df (dataframe): dataframe that store features (at frame-level, i.e. one row of features corresponds to a frame).
        label_dict (dictionary): dictionary of label vs Id of a recording.
        nr_frames (int): length of the segmented chunks.
    
    Returns:
        mean_std_feats_df (dataframe): dataframe that stores mean and standard deviation of features at chunk-level.
    """
    feats_df['cum_IDidx'] = feats_df.groupby('Id').cumcount()

    def get_subidx(cum_Idx,batch_size):
        #batch needs to be an integer (or float like 3.0)
        return int(1.0*cum_Idx/batch_size)

    feats_df['subIdx'] = feats_df.apply(lambda x: get_subidx(x['cum_IDidx'], nr_frames), axis=1)
    feats_df = feats_df.drop(['cum_IDidx'],axis=1)

    mean_feats = feats_df.groupby(['Id','subIdx']).aggregate('mean').reset_index()
    std_feats = feats_df.groupby(['Id','subIdx']).agg(lambda x: x.std(ddof=0)).reset_index() #ddof=0 to compute population std (rather than sample std)
    keep_same = {'Id', 'subIdx'}
    mean_feats.columns = ['{}{}'.format(c, '' if c in keep_same else '_m') for c in mean_feats.columns]
    std_feats.columns = ['{}{}'.format(c, '' if c in keep_same else '_std') for c in std_feats.columns]

    mean_std_feats_df = pd.merge(mean_feats, std_feats, on=['Id','subIdx'], how='outer')

    mean_std_feats_df['label'] = mean_std_feats_df["Id"].map(label_dict)
    #mean_std_feats_df[['Id','label']].head(50)

    return mean_std_feats_df
