import librosa
import librosa.display
import numpy as np
from tools import *

import torch 
import networks
from pydub import AudioSegment

DATA_REQUIRED_SR = 8000
FRAME_SIZE = 0.01 # giay

# filepath = '/home/huydd/NLP/ASR/SentenceSplit/Sp-Denoise/result/csty1.wav'
# filepath = '/home3/huydd/cut_audio_by_silence/Speech-Denoise/result/csty1.wav'
filepath = '/home3/huydd/huydd/data_with_noise/val_data_backup/result1140_1910.wav'
snd, sr = librosa.load(filepath,sr=16000)
print(snd.shape)
audio = audio_normalize(snd)
len_audio = audio.shape[0] # samplerate
num_frame = int((len_audio/DATA_REQUIRED_SR)/FRAME_SIZE)
print(num_frame)

# snd = librosa.to_mono(snd)
mixed_sig_stft = librosa.feature.mfcc(audio, sr=16000, n_mfcc=60, n_fft=510, hop_length=160, win_length=400)
# mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
print(mixed_sig_stft.shape)
# s = mixed_sig_stft.unsqueeze(0)
# print(s.shape)
# net = networks.AudioVisualNet()
# out = net(s)
# print(out.shape)
# print(out)