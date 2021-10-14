import torch
from torch import Tensor
from torch.utils.mobile_optimizer import optimize_for_mobile
import librosa
import numpy as np 
import torch 
from networks import get_network
from tools import *
from utils import ensure_dir, get_parent_dir
from transform import *
import torch.nn as nn
from dataset import DATA_REQUIRED_SR
import torch.quantization.quantize_fx as quantize_fx
import copy
import timeit

SIGMOID_THRESHOLD = 0.5

net = get_network()
load_path = '/home3/huydd/huydd/model_output/huydd/model/ckpt_epoch15.pth'
net.load_state_dict(torch.load(load_path,map_location=torch.device('cpu'))['model_state_dict'])

net.eval()


# Apply quantization / script / optimize for motbile
quantized_model = torch.quantization.quantize_dynamic(
    net, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
scripted_model = torch.jit.script(quantized_model)
optimized_model = optimize_for_mobile(scripted_model)

file_path = '/home3/huydd/cut_audio_by_silence/Speech-Denoise/result/csty1.wav'
sound, sr = librosa.load(file_path, sr=DATA_REQUIRED_SR)
print("Length of the audio:", len(sound))
    
audio = audio_normalize(sound)

mixed_sig_stft = fast_stft(audio, n_fft=510, hop_length=160, win_length=400)
mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2,0,1)),dtype=torch.float32)
audio_input = mixed_sig_stft.unsqueeze(0)

start = timeit.default_timer()
output=optimized_model(audio_input)
pred_labels = (torch.sigmoid(output).detach().cpu().numpy() >= SIGMOID_THRESHOLD).astype(np.float32)

frames = pred_labels[0].tolist()
frames = list(map(int, frames[100:400]))

stop = timeit.default_timer()
print('Time 1: ', stop - start)

print("Silence frames from quantized net:",frames)

start_1 = timeit.default_timer()

output=net(audio_input)
pred_labels = (torch.sigmoid(output).detach().cpu().numpy() >= SIGMOID_THRESHOLD).astype(np.float32)

frames = pred_labels[0].tolist()
frames = list(map(int, frames[100:400]))

stop_1 = timeit.default_timer()
print('Time 2: ', stop_1 - start_1)


print("Silence frames from original network:",frames)

optimized_model.save("/home3/huydd/quantize/optimized.pth")
# torch.save(net, "/home3/huydd/quantize/silence_detection_origin.pth")

# # torch.jit.save(torch.jit.script(quantized_model), "/home3/huydd/quantize/silence_detection.pth")
# # quantized_model.save("/home3/huydd/quantize/silence_detection.pt")