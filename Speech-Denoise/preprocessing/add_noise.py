#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:46:56 2020

@author: sleek_eagle
"""

import math   
import numpy as np 
import librosa
import matplotlib.pyplot as plt

import argparse
import os
from scipy.io.wavfile import write
import pandas as pd

'''
Signal to noise ratio (SNR) can be defined as 
SNR = 20*log(RMS_signal/RMS_noise)
where RMS_signal is the RMS value of signal and RMS_noise is that of noise.
      log is the logarithm of 10

*****additive white gausian noise (AWGN)****
 - This kind of noise can be added (arithmatic element-wise addition) to the signal
 - mean value is zero (randomly sampled from a gausian distribution with mean value of zero. standard daviation can varry)
 - contains all the frequency components in an equal manner (hence "white" noise) 
'''

#SNR in dB
#given a signal and desired SNR, this gives the required AWGN what should be added to the signal to get the desired SNR
def get_white_noise(signal,SNR) :
    #RMS value of signal
    RMS_s=math.sqrt(np.mean(signal**2))
    #RMS values of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n=RMS_n
    noise=np.random.normal(0, STD_n, signal.shape[0])
    return noise

#given a signal, noise (audio) and desired SNR, this gives the noise (scaled version of noise input) that gives the desired SNR
def get_noise_from_sound(signal,noise,SNR):
    RMS_s=math.sqrt(np.mean(signal**2))
    #required RMS of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    
    #current RMS of noise
    print('Noise:', noise)
    print("RMS_n_current:", np.mean(noise**2))
    RMS_n_current=math.sqrt(np.mean(noise**2))
    
    noise=noise*(RMS_n/RMS_n_current)
    
    return noise

#***convert complex np array to polar arrays (2 apprays; abs and angle)
def to_polar(complex_ar):
    return np.abs(complex_ar),np.angle(complex_ar)



parser = argparse.ArgumentParser(description="Mix noise to original audio ")

parser.add_argument("--csv_file", "-a", type=str, help="directory of audio")
parser.add_argument("--noise_path", "-n", type=str, help="directory of noise audio")
# parser.add_argument("--save_dir", "-s", type=str, help="Directory to save file")

args = parser.parse_args()


# noise_path=[]
# for file in os.listdir(args.noise_dir):
#     if file.endswith('.wav'):
#         filepath = os.path.join(args.noise_dir, file)
#         print("file_path:",filepath)
#         noise_path.append(filepath)
sample_rate=16000

noise, sr = librosa.load(args.noise_path, sr=sample_rate)
index = 0
noise_len = noise.shape[0]
SNR = 8
csv_df = pd.read_csv(args.csv_file, delimiter=';')

for i, row in csv_df.iterrows():
    file_path = row['audio_path']
    signal, sr = librosa.load(file_path, sr=sample_rate)
    
    signal_len = signal.shape[0]
    if index < noise_len:
        if signal_len < noise_len - index:
            noise_add = noise[index:index + signal_len]
            noise_add = get_noise_from_sound(signal, noise_add, SNR=SNR)
            signal = signal + noise_add
            index = index + signal_len
        else:
            noise_add = noise[index:]
            noise_add = get_noise_from_sound(signal[0:noise_len-index], noise_add, SNR=SNR)
            signal = signal + noise_add
            index  = noise_len
        print("SNR = " + str(20*np.log10(math.sqrt(np.mean(signal**2))/math.sqrt(np.mean(noise_add**2)))))    
        # librosa.output.write_wav(os.path.join(args.save_dir,file), result, sample_rate)
        write(file_path, sr, signal)

    else:
        print(row['audio_path'])
        break
        



print("End of this process real!")
