import librosa
import librosa.display
import numpy as np
from tools import *

import torch 
import networks
from pydub import AudioSegment
from python_speech_features import logfbank, mfcc
from tools import *

import timeit

DATA_REQUIRED_SR = 8000
FRAME_SIZE = 0.01 # giay

# filepath = '/home/huydd/NLP/ASR/SentenceSplit/Sp-Denoise/result/csty1.wav'
# filepath = '/home3/huydd/cut_audio_by_silence/Speech-Denoise/result/csty1.wav'
filepath = '/home3/huydd/huydd/data_with_noise/val_data_backup/result1140_1910.wav'
snd, sr = librosa.load(filepath,sr=8000)
print(snd.shape)
audio = audio_normalize(snd)
len_audio = audio.shape[0] # samplerate
num_frame = int((len_audio/DATA_REQUIRED_SR)/FRAME_SIZE)
print(num_frame)

# snd = librosa.to_mono(snd)
mixed_sig_stft = librosa.feature.mfcc(audio, sr=16000, n_mfcc=60, n_fft=510, hop_length=160, win_length=400)
# mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
print(mixed_sig_stft.shape)


start = timeit.default_timer()
filter_bank = filter_bank_with_audio(audio, 16000)
stop = timeit.default_timer()

# filter_bank = torch.tensor(filter_bank.transpose((1, 0)), dtype=torch.float32)
# filter_bank = filter_bank.unsqueeze(0)
print(filter_bank.shape)
print('Time: ', (stop - start))

start = timeit.default_timer()
logfbank_signal = logfbank(audio, samplerate=16000, nfilt=60, nfft=512)
stop = timeit.default_timer()

print(logfbank_signal.shape)
print('Time: ', (stop - start))
# s = mixed_sig_stft.unsqueeze(0)
# print(s.shape)
# net = networks.AudioVisualNet()
# out = net(s)
# print(out.shape)
# print(out)
# mfcc_feat = mfcc(snd)
# print(mfcc_feat.shape)