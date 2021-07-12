import librosa
import librosa.display
import numpy as np
import transform
import tools
import torch 
import networks
from pydub import AudioSegment
# filepath = '/home/huydd/NLP/ASR/SentenceSplit/Sp-Denoise/result/csty1.wav'
filepath = '/home/huydd/NLP/ASR/SentenceSplit/py-webrtcvad/data/Silence/tets/silence1min.wav'
snd, sr = librosa.load(filepath,sr=16000)
# snd = librosa.to_mono(snd)
mixed_sig_stft = transform.fast_stft(snd, n_fft=510, hop_length=160, win_length=400)
mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
print(mixed_sig_stft.shape)
# s = mixed_sig_stft.unsqueeze(0)
# print(s.shape)
# net = networks.AudioVisualNet()
# out = net(s)
# print(out.shape)
# print(out)