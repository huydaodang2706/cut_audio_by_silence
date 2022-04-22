import argparse
import librosa
import timeit
import numpy as np 
import torch 
from networks import get_network
from tools import *

DATA_REQUIRED_SR = 8000
SIGMOID_THRESHOLD = 0.5

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def audio_normalize(snd):
    """Normalize librosa audio array"""
    max_abs = max(abs(snd))
    if max_abs > 1:
        mult_var = 1.0 / max_abs
        return snd * mult_var
    else:
        return snd

def test():
    load_path = "/home3/huydd/cut_audio_by_silence/light_weight_vad_2.0/model_ckpt/22_4_2022_aicc_6h_voice_6h_unvoice/huydd/model/best_acc.pth"
    # print('Load saved model: {}'.format(load_path))x
    gpu_flag = False
    net = get_network()
    if gpu_flag:
        net.load_state_dict(torch.load(load_path)['model_state_dict'])
    else:
        net.load_state_dict(torch.load(load_path,map_location=torch.device('cpu'))['model_state_dict'])
        
    if torch.cuda.device_count() >= 1 and gpu_flag:
        print('For single-GPU')
        net = net.cuda()    # For single-GPU
    else:
        net = net

    net.eval()

    # while True:
    # Load and preprocess input data
    file_path = "/home3/cuongld/ASR_team/data_ASR/aicc_prod/2021-12-05/utter-8kHz-1498051814-ceada8f2-853d-4c1b-9fc8-2516c91e29ef.wav"
    # file_path = "test_silience.wav"
    start = timeit.default_timer()
    sound, sr = librosa.load(file_path, sr=DATA_REQUIRED_SR)
    print("Length of the audio:", len(sound))
        
    audio = audio_normalize(sound)
    
    input_filter_bank = filter_bank_with_audio(audio, nfilt=30)
    input_filter_bank = torch.tensor(input_filter_bank.transpose((1, 0)), dtype=torch.float32)
    input_filter_bank = input_filter_bank.unsqueeze(0)
    audio_input = input_filter_bank.unsqueeze(0)
    print("Input audio shape {}".format(audio_input.shape))

    if torch.cuda.device_count() >= 1 and gpu_flag:
        audio_input = audio_input.cuda()
    
    # Prediction
    output=net(audio_input)
    pred_labels = (torch.sigmoid(output).detach().cpu().numpy() >= SIGMOID_THRESHOLD).astype(np.float32)
    
    frames = pred_labels[0].tolist()
    frames = list(map(int, frames))

    print("Silence frames:",frames)
    print(len(frames))
    stop = timeit.default_timer()
    print('Time: ', (stop - start))


if __name__ == '__main__':
    test()
            
            


