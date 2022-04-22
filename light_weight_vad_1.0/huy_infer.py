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
    load_path = "/home3/huydd/cut_audio_by_silence/light_weight_vad/model_ckpt/ckpt_27_3_2022_aicc_vad_6h_only/huydd/model/best_acc.pth"
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

    while True:
        # Load and preprocess input data
        file_path = "/home/thanhld/asr/asr-volumes/vad/vad_bytes_random-session-id3c9dd414-baea-4d1e-b270-0d7442031a94_ebb5ad96-5fed-425d-bf0f-1e9b5ffd651b_5e38197b-6775-43b9-b5ac-cc7584be501d.wav"
        # file_path = "test_silience.wav"
        start = timeit.default_timer()
        sound, sr = librosa.load(file_path, sr=DATA_REQUIRED_SR)
        print("Length of the audio:", len(sound))
            
        audio = audio_normalize(sound)
        
        input_filter_bank = filter_bank_with_audio(audio)
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
            
            


