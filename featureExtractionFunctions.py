import pandas as pd
import config
import numpy as np
import python_speech_features as spe_feats
from scipy.stats import kurtosis, skew
from scipy.signal import lfilter
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import mediainfo
from pydub.playback import play
import pysptk
import math

#compute RMS value of a signal and return it (in dB scale)
#seems not to work?
def get_RMS(s):
    s_rms = np.sqrt(np.mean(np.power(s,2)))
    #convert to dB scale
    #s_db = 20*np.log10(s_rms/1.0)
    return s_rms

#seems not to work either
#RMS-based normalization of a signal, based on a target level (in dB)
def RMS_normalization(s, dB_targ):
    
    #desired level is converted to linear scale
    rms_targ = 10**(dB_targ/20.0)
    
    #compute scaling factor
    scale = rms_targ/get_RMS(s)
    
    #scale amplitude of input signal
    scaled_s = scale*s
    
    return scaled_s

def match_target_amplitude(audioSegment_sound, target_dBFS):
    dBFS_diff = target_dBFS - audioSegment_sound.dBFS
    return audioSegment_sound.apply_gain(dBFS_diff)

#Apply pre-emphasis (high-pass) filter
def apply_preEmph(x):
    x_filt = lfilter([1., -0.97], 1, x)
    return x_filt
        
#Obtain autocorrelation
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int((result.size+1)/2):] #Note: other people use re.size/2:, but this does not work for me 
                                   # TODO: check consistency in other computers

#Compute zero-crossing rate
def get_zcr(x):
    zcr = (((x[:-1] * x[1:]) < 0).sum())/(len(x)-1)
    return zcr

#Compute log-energy
def get_logEnergy(x):
    logEnergy = np.log10( ( (np.power(x,2)).sum()/len(x) ) + config.eps)  
    return logEnergy

#Estimate fundamental frequency (F0)
def get_F0(x,fs):
    #autocorrelation-based method to extract F0
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
def get_formants(x, lp_order, nr_formants):
    
    #compute lp coefficients
    a = librosa.lpc(x, lp_ord)
    

    #get roots from lp coefficients
    rts = np.roots(a)
    rts = [r for r in rts if np.imag(r) >= 0]

    #get angles
    angz = np.arctan2(np.imag(rts), np.real(rts))

    #get formant frequencies
    formants = sorted(angz * (fs_targ / (2 * math.pi)))
    
    return formants[0:nr_formants]

def get_entropy(x, type='shannon'):
    #default shannon entropy since this is the one used by the phd thesis
    
    base = {'shannon' : 2., 'natural' : math.exp(1), 'hartley' : 10.}
    N = len(x)

    if N <= 1:
        return 0

    value,counts = np.unique(x, return_counts=True)
    probs = counts / N
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0. #initialization

    # compute entropy
    for i in probs:
        ent -= i * math.log(i+config.eps, base[type])
    
    return ent

#Extract frequencies
def feature_extraction(x,fs,feats_df,lp_ord,ID,label):
#Extract features from signal x (identified as ID), and concatenate them to dataframe feats_df
#Features' reference: (see Appendix)
#[1]https://link.springer.com/article/10.1007/s10439-013-0741-6
#[2]https://espace.library.uq.edu.au/data/UQ_344963/s41943203_phd_submission.pdf?dsi_version=c5434db897ab74b192ca295a9eeca041&Expires=1585086202&Key-Pair-Id=APKAJKNBJ4MJBJNC6NLQ&Signature=c8k8DmG~KIxg0ToTO8rebm2MzHneCzJGkjSFRB7BYTEQ-MHXEr0ocHmISrldP3hFf9qmeiL11ezyefcNeRVeKIQ9PVjOl9pn7rXWcjA1o2voPn1VnDd8n7G2cT31apdj0LNMclhlXRPnCsGD66qDRqa3d-xaqqXhEqU73aw3ZgBgroO213MfJOqFhJxxXo2QEia0bSlDRTeX9KhSczFK-IFTPC6GwFL2L04por8pQRI3HF7E3f26O9zp9OhkwxSU9qfJah20WxZLA4PxREdv7JGoVBinR6T0mTcIaQi~B4IzYjSPSsTTADMNk5znVYIvSqgtMT~DY~qwlfq4SRdFjQ__
  
    
    #do features in a frame-basis
    x_frames = spe_feats.sigproc.framesig(x,config.frame_len,config.frame_step,config.win_func) #DOUBT: should I use window or not?
                                                                        #at least for formant estimation i should

    nr_frames = x_frames.shape[0]
    #print(nr_frames)
        
    #0)Wavelets #TODO
    
    #DOUBT: if log-energy feature is included, should I also include the first mfcc coefficient (c0) ?
    #1)mfcc
    mfcc_feat = spe_feats.mfcc(x,fs, winlen=config.frame_len_s,winstep=config.frame_step_s, numcep=config.cep_num,winfunc=config.win_func)
    
    #deltas to capture 
    mfcc_delta_feat = spe_feats.delta(mfcc_feat,1) #mfcc_delta_feat = np.subtract(mfcc_feat[:-1], mfcc_feat[1:]) #same
    mfcc_deltadelta_feat = spe_feats.delta(mfcc_delta_feat,1)          
    
    #2)zero-crossing rate
    zcr_feat = np.apply_along_axis(get_zcr, 1, x_frames)
    
    #3)Formant frequencies
    #using LP-coeffcs-based method
    #formant_feat = np.apply_along_axis(get_formants, 1, x_frames, lp_ord, nr_formants)
    
    #Note: for the moment, it seems some frames are ill-conditioned for lp computing,
    #current solution - we skip those and fill with NaN values
    formants_feat= np.empty((nr_frames,4))
    formants_feat[:] = np.nan
    
    for i_frame in range(0,nr_frames):
        try: 
            formants_feat[i_frame] = get_formants(x_frames[i_frame], config.lp_ord, config.nr_formants)
        except:
            pass
    
    #4)Log-energy
    logEnergy_feat =  np.apply_along_axis(get_logEnergy, 1, x_frames)
    
    #5)Pitch (F0)
    F0_feat =  np.apply_along_axis(get_F0, 1, x_frames,fs)
    
    #TODO: compute also F0 with pysptk (a python wrapper for SPTK library), it probably gives better results
    #https://github.com/r9y9/pysptk/blob/master
        	#F0_feat = pysptk.rapt(x.astype(np.float32), fs=fs, hopsize=frame_step, min=50, max=500, ,voice_bias=0.0 ,otype=\"f0\")
    
    #Compare the values between swipe and rapt 
    #F0_feat = pysptk.swipe(x.astype(np.float64), fs=fs,hopsize = config.frame_step, min=50, max=500, otype="f0")
    
    F0_feat = pysptk.rapt(x.astype(np.float32), fs=fs,hopsize = config.frame_step, min=50, max=500, otype="f0")

    
    
    #right frame size???
#Change the window size from 450 to 40 to 100
# Keep swipe , change min to 50 and max - 500
#EXample pysptk.swipe(x.astype(np.float64), fs=fs, hopsize=80, min=60, max=200, otype="f0")

    
    #6)Kurtosis
    kurt_feat =  np.apply_along_axis(kurtosis, 1, x_frames)
    
    #7)Bispectrum Score (BGS)
    #TODO: see PhD thesis for more info on this feature
    
    #8)Non-Gaussianity Score (NGS)
    #TODO: see PhD thesis for more info on this feature
   
    #9) Adding skewness as measure of non-gaussianity (not in paper)
    skew_feat =  np.apply_along_axis(skew, 1, x_frames)
    
    #DOUBT: 10) Shannon entropy GETTING -inf in all cases, WHY??? Don't include until fixed
    entropy_feat = np.apply_along_axis(get_entropy, 1, x_frames)

    
    #TODO: add small value in all entries, this may fix the problem
    
    mfcc_cols = ['mfcc_%s' % s for s in range(0,config.cep_num)]
    mfcc_delta_cols = ['mfcc_d%s' % s for s in range(0,config.cep_num)]
    mfcc_deltadelta_cols = ['mfcc_dd%s' % s for s in range(0,config.cep_num)]
    formants_cols = ['F%s' % s for s in range(1,config.nr_formants+1)]
          
    feats_segment = pd.concat([pd.DataFrame({'Id': ID, 'kurt': kurt_feat, 'logEnergy': logEnergy_feat,
                                                 'zcr': zcr_feat, 'F0': F0_feat,
                                                 'skewness': skew_feat, 'label': label, 'entropy':entropy_feat}),
                               pd.DataFrame(mfcc_feat,columns=mfcc_cols), 
                            pd.DataFrame(formants_feat,columns=formants_cols)],axis=1)
    
    #print(nr_frames)
    feats_df = feats_df.append(feats_segment,ignore_index=True, sort=False)
    
    return feats_df


def feature_extraction_Step(all_s,all_id,all_label):
    
    import config
    import pydub
    
    #initialize data frame of features:
    feats = pd.DataFrame([])

    for s, ID, label in zip(all_s,all_id,all_label):

            #Pre-processing of the signals:

            ## 0 ) Resampling to target sampling frequency:
            s = s.set_frame_rate(config.fs_targ)
            fs= config.fs_targ

            ## 1)
            if config.norm_skip is False:
                s=match_target_amplitude(s, config.dB_targ)
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
            if config.HPF_skip is False:
                print('High-pass filtering...')       
                preEmph_filtering = lambda x: apply_preEmph(x)
                s_segments_filt = list(map(preEmph_filtering, s_segments_np))
            else:
                s_segments_filt = s_segments_np

            print('Computing features...')
            #Feature extraction for each segment

            #(lambda function doesn't work )
            #feat_extr_step = lambda x, fs, feats_df, lp_ord, ID: feature_extraction(x,fs,feats_df,lp_ord,ID)
            #feats = feat_extr_step(s_segments_filt,fs,feats,lp_ord,ID)
            for idx, seg_i in enumerate(s_segments_filt):
                #print('\tSegment %d' % idx)
                feats = feature_extraction(seg_i,fs,feats,config.lp_ord,ID,label)
                
    return feats

def processingNaNvalues(feats):

    #TODO: decide best way to deal with this. Do interpolation, or do dropping rows?
    #1.Check which columns have NaNs values

    #feats2 = feats.copy()

    #sum(feats.isna().any())
    #feats.columns[feats.isna().any()].tolist() --> We get just the ones we have inserted in formants
    feats2 = feats.interpolate(method ='pad')
    
    
    
    #commenting the line below at the moment that drops the rows with NAN values
    #feats2 = feats.dropna(axis=1).copy()
    
    
    #feats2.dropna(axis=0, how="any", thresh=None, subset=None, inplace=True)

    #feats2.columns[feats2.isna().any()].tolist()
    #feats2.describe()
    #sum(feats2.isna().any())
    return feats2

#Make dictionary and add label column using it
def createLabelDict_addLabel2df(feats):
    feats_unique = feats.drop_duplicates(subset=['Id'])
    label_dict = dict(zip(feats_unique.Id, feats_unique.label))
    return label_dict


def frame_mean_std_chunk_modeling (feats2, label_dict):

    #Grouping the frames from a same recording (Id) into chunks with the same number of frames.
    #The training of the classifier will be based on these chunks mean and standard deviation.

    feats2['cum_IDidx'] = feats2.groupby('Id').cumcount()

    def get_subidx(cum_Idx,batch_size):
        #batch needs to be an integer (or float like 3.0)
        return int(1.0*cum_Idx/batch_size)

    feats2['subIdx'] = feats2.apply(lambda x: get_subidx(x['cum_IDidx'], 10), axis=1)
    feats2 = feats2.drop(['cum_IDidx'],axis=1)

    mean_feats = feats2.groupby(['Id','subIdx']).aggregate('mean').reset_index()
    std_feats = feats2.groupby(['Id','subIdx']).agg(lambda x: x.std(ddof=0)).reset_index() #ddof=0 to compute population std (rather than sample std)
    keep_same = {'Id', 'subIdx'}
    mean_feats.columns = ['{}{}'.format(c, '' if c in keep_same else '_m') for c in mean_feats.columns]
    std_feats.columns = ['{}{}'.format(c, '' if c in keep_same else '_std') for c in std_feats.columns]

    mean_std_feats = pd.merge(mean_feats, std_feats, on=['Id','subIdx'], how='outer')

    mean_std_feats['label'] = mean_std_feats["Id"].map(label_dict)
    #mean_std_feats[['Id','label']].head(50)

    return mean_std_feats


#TODO: modeling of chunks using sequence models too




