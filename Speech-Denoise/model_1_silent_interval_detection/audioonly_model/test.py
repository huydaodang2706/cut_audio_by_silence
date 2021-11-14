import librosa
import librosa.display
import numpy as np
import transform
from tools import *

import torch 
import networks
from pydub import AudioSegment

DATA_REQUIRED_SR = 8000
FRAME_SIZE = 0.01 # giay

# filepath = '/home/huydd/NLP/ASR/SentenceSplit/Sp-Denoise/result/csty1.wav'
# filepath = '/home3/huydd/cut_audio_by_silence/Speech-Denoise/result/csty1.wav'
filepath = '/home3/huydd/audio_vietcetera/vietcetera_da_cat/vietcetera_10_result/vietcetera_10_interval_152.wav'
snd, sr = librosa.load(filepath,sr=8000)
print(snd.shape)
audio = audio_normalize(snd)
len_audio = audio.shape[0] # samplerate
num_frame = int((len_audio/DATA_REQUIRED_SR)/FRAME_SIZE)
print(num_frame)

# snd = librosa.to_mono(snd)
mixed_sig_stft = transform.fast_stft(snd, n_fft=510, hop_length=80, win_length=400)
mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
print(mixed_sig_stft.shape)
# s = mixed_sig_stft.unsqueeze(0)
# print(s.shape)
# net = networks.AudioVisualNet()
# out = net(s)
# print(out.shape)
# print(out)