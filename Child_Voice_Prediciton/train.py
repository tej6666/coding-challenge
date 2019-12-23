
from __future__ import division

import numpy as np
from mel_coefficients import mfcc
import matplotlib.pyplot as plt
from LBG import lbg
from scipy.io.wavfile import read
import os
from LPC import lpc
import wave
import struct


import sounddevice as sd

def wav_to_floats(wave_file):
    w = wave.open(wave_file)
    astr = w.readframes(w.getnframes())
    # convert binary chunks to short
    a = struct.unpack("%ih" % (w.getnframes()* w.getnchannels()), astr)
    a = [float(val) / pow(2, 15) for val in a]
    return a




def training(nfiltbank, orderLPC):
    nSpeaker = 2
    nCentroid = 16
    codebooks_mfcc = np.empty((nSpeaker,nfiltbank,nCentroid))
    codebooks_lpc = np.empty((nSpeaker, orderLPC, nCentroid))
    directory = os.getcwd() + '/train';
    fname = str()

    for i in range(nSpeaker):
        fname = '/speaker' + str(i+1) + '.wav'
        print('Now speaker ', str(i+1), 'features are being trained' )
        (fs,s) = read(directory + fname)
        # read the wav file specified as first command line arg
        s = wav_to_floats(directory + fname)
        s = s[:48000]
        sd.play(s, fs)
        mel_coeff = mfcc(s, fs, nfiltbank)
        lpc_coeff = lpc(s, fs, orderLPC)
        codebooks_mfcc[i,:,:] = lbg(mel_coeff, nCentroid)
        codebooks_lpc[i,:,:] = lbg(lpc_coeff, nCentroid)
    

    codebooks = np.empty((2, nfiltbank, nCentroid))
    mel_coeff = np.empty((2, nfiltbank, 68))
   
    for i in range(2):
        fname = '/speaker' + str(i+1) + '.wav'
        (fs,s) = read(directory + fname)
        s = s[:48000]
        mel_coeff[i,:,:] = mfcc(s, fs, nfiltbank)[:,0:68]
        codebooks[i,:,:] = lbg(mel_coeff[i,:,:], nCentroid)
        
    
    plt.figure(nSpeaker + 1)
    s1 = plt.scatter(mel_coeff[0,6,:], mel_coeff[0,4,:],s = 100,  color = 'r', marker = 'o')
    c1 = plt.scatter(codebooks[0,6,:], codebooks[0,4,:], s = 100, color = 'r', marker = '+')
    s2 = plt.scatter(mel_coeff[1,6,:], mel_coeff[1,4,:],s = 100,  color = 'b', marker = 'o')
    c2 = plt.scatter(codebooks[1,6,:], codebooks[1,4,:], s = 100, color = 'b', marker = '+')
    plt.grid()
    plt.legend((s1, s2, c1, c2), ('Child','Parent','Child centroids', 'Parent centroids'), scatterpoints = 1, loc = 'upper left')
    plt.show()
   
    
    return (codebooks_mfcc, codebooks_lpc)
    
    
