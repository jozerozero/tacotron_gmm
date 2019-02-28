import librosa
from nnmnkwii import metrics
from datasets import audio
import numpy as np
import sys

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_wave(filename, sr):
    wav=librosa.core.load(filename, sr=sr)[0]
    wav = wav / np.abs(wav).max() * 0.999
    hparams={
            'trim_fft_size' : 512,
            'trim_hop_size' : 128,
            'trim_top_db' : 60
            }
    hparams=dotdict(hparams)
    wav = audio.trim_silence(wav, hparams)
    return wav

def mcd(filename1, filename2, sr=22050):
    wav1=get_wave(filename1, sr)
    wav2=get_wave(filename2, sr)
    mfcc1=librosa.feature.mfcc(y=wav1, sr=sr)
    mfcc2=librosa.feature.mfcc(y=wav2, sr=sr)
    D, wp = librosa.core.dtw(mfcc1,mfcc2)
    #print(wp)
    print(mfcc1.shape)
    print(mfcc2.shape)
    #print(D.shape)
    mfcc1=np.array([mfcc1.T[i[0]] for i in wp])
    mfcc2=np.array([mfcc2.T[i[1]] for i in wp])
    return metrics.melcd(mfcc1, mfcc2)
    #return D[-1,-1]

if len(sys.argv)==4:
    print(mcd(sys.argv[1], sys.argv[2], sys.argv[3]))
else:
    print(mcd(sys.argv[1], sys.argv[2]))
