
from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import EUDistance
from mel_coefficients import mfcc
from LPC import lpc
from train import training
import os
import webrtcvad
import warnings

warnings.filterwarnings("ignore")
nSpeaker = 2
nfiltbank = 24
orderLPC = 8
(codebooks_mfcc, codebooks_lpc) = training(nfiltbank, orderLPC)
directory = os.getcwd() + '/test';
fname = str()

vad = webrtcvad.Vad()

# set aggressiveness from 0 to 3
vad.set_mode(3)
