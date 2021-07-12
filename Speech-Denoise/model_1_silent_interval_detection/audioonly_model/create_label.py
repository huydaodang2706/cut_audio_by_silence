import argparse
import json
import os
import librosa
import collections
from scipy.io.wavfile import write

import numpy as np 
import torch 
from networks import get_network
from tools import *
from utils import ensure_dir, get_parent_dir
from transform import *
from common import (EXPERIMENT_DIR, PHASE_PREDICTION, PHASE_TESTING,
                    PHASE_TRAINING, get_config)

from dataset import (DATA_REQUIRED_SR, DATA_MAX_AUDIO_SAMPLES)

SIGMOID_THRESHOLD = 0.5
EXPERIMENT_PREDICTION_OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, 'outputs')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir','-a',type=str,default='',required=True, help="Audio directory to process")
    parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
    parser.add_argument('-o', '--outputs', default=EXPERIMENT_PREDICTION_OUTPUT_DIR, type=str, help="outputs dir to write results")
    parser.add_argument('--save_dir', type=str, help="outputs dir to write results")
    parser.add_argument("--gpu", type=str2bool, nargs='?',
                        const=True, help="GPU for prediction")

    args = parser.parse_args()

    net = get_network()

    if args.ckpt == 'latest' or args.ckpt == 'best_acc':
        name = args.ckpt    
    else:
        name = "ckpt_epoch{}".format(args.ckpt)

    load_path = os.path.join(EXPERIMENT_DIR,'model',"{}.pth".format(name))

    print('Load saved model: {}'.format(load_path))
    if args.gpu:
        net.load_state_dict(torch.load(load_path)['model_state_dict'])
    else:
        net.load_state_dict(torch.load(load_path,map_location=torch.device('cpu'))['model_state_dict'])
        
    # if torch.cuda.device_count() > 1:
    #     print('For multi-GPU')
    #     net = nn.DataParallel(net.cuda())   # For multi-GPU
    if torch.cuda.device_count() >= 1 and args.gpu:
        print('For single-GPU')
        net = net.cuda()    # For single-GPU
    else:
        net = net
    # net = net
    # # Set model to evaluation mode
    net.eval()


 

    for file in os.listdir(args.audio_dir):
        if file.endswith('.wav'):
            filepath = os.path.join(args.audio_dir, file)
            print("file_path:",filepath)
            sound, sr = librosa.load(filepath, sr=DATA_REQUIRED_SR)
            
            audio = audio_normalize(sound)
            print('Sample rate:',sr)
            
            mixed_sig_stft = fast_stft(audio, n_fft=510, hop_length=160, win_length=400)
            mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2,0,1)),dtype=torch.float32)
            audio_input = mixed_sig_stft.unsqueeze(0)
            print("Audio_input:",audio_input.shape)
            
            # output = net(audio_input.cuda())
            if torch.cuda.device_count() >= 1 and args.gpu:
                audio_input = audio_input.cuda()
            
            output=net(audio_input)
            pred_labels = (torch.sigmoid(output).detach().cpu().numpy() >= SIGMOID_THRESHOLD).astype(np.float32)
            
            #print("Output:",pred_labels)
            frames = pred_labels[0].tolist()
            frames = list(map(int, frames))
            # print(frames)
            zeros_frames=[]
            for x in frames:
                zeros_frames.append(0)
            print(zeros_frames)
            # print("Output:",pred_labels[0,0:400])
            # print("Non-zero:", np.count_nonzero(pred_labels[0]) )
            f = open(os.path.join(args.save_dir,file.replace('.wav','')) + '.txt','w')
            # f.write(str(frames))
            f.write(str(zeros_frames))
            f.close()

if __name__ == '__main__':
    main()
            
            


