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
import torch.nn as nn

from dataset import (DATA_REQUIRED_SR, DATA_MAX_AUDIO_SAMPLES)
from pydub import AudioSegment

SIGMOID_THRESHOLD = 0.5
EXPERIMENT_PREDICTION_OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, 'outputs')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def vad_collector(frame_duration_s, window_size, frames,sr):
    sample_per_frame = int(sr * frame_duration_s) + 1

    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=window_size)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    # triggered = False
    triggered = True
    
    speech_timestamp = []
    start_time = 0
    for i, frame in enumerate(frames):
        is_speech = frame 
        if triggered:
            ring_buffer.append((i, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])

            if num_voiced >= 0.9 * ring_buffer.maxlen:
                triggered = False
                if ring_buffer[0][0] != 0:
                    start_time = (ring_buffer[0][0]-1) * sample_per_frame
                else:
                    start_time = ring_buffer[0][0] * sample_per_frame
                ring_buffer.clear()
        elif i == len(frames) - 1:
            end_time = (len(frames) -1) * sample_per_frame
            speech_timestamp.append({
                    'start': start_time,
                    'end': end_time
                })
            ring_buffer.clear()
        else:
            ring_buffer.append((i, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            
            if num_unvoiced == ring_buffer.maxlen:
                # end_time = i * sample_per_frame
                end_time = (ring_buffer[0][0] + 1) * sample_per_frame
                triggered = True
                speech_timestamp.append({
                    'start': start_time,
                    'end': end_time
                })
                ring_buffer.clear()
                # start_time = end_time + 1
            
    return speech_timestamp

def split_audio(audio, sr, speech_timestamps, save_dir, file):
    # os.mkdir(os.path.join(save_dir,file.replace('.wav','') + "_re"))
    # path = os.path.join(save_dir,file.replace('.wav','') + "_re")
    path = save_dir
    i = 0
    # audio, sr = librosa.load(file_path,sr=16000)
    # write(os.path.join(path ,'data_orig.wav'),sr,audio)
    if len(speech_timestamps) == 0:
        return

    if len(audio) <= sr*7:
        print('short audio: ', len(audio))
        write(os.path.join(path, file.replace('.wav','') + "_segment" + str(i) + ".wav"),sr,audio)
    else:
        for segment in speech_timestamps:
            seg = audio[segment['start']:segment['end']]
            write(os.path.join(path, file.replace('.wav','') + "_segment" + str(i) + ".wav"),sr,seg)
            i+=1



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file','-a',type=str,default='',required=True, help="Audio directory to process")
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

    # load_path = os.path.join('/home3/huydd/cut_audio_by_silence/model_output/huydd/','model',"{}.pth".format(name))
    # load_path = os.path.join('/home3/huydd/huydd/model_output/huydd/','model',"{}.pth".format(name))
    load_path = args.ckpt
    print('Load saved model: {}'.format(load_path))
    if args.gpu:
        net.load_state_dict(torch.load(load_path)['model_state_dict'])
    else:
        net.load_state_dict(torch.load(load_path,map_location=torch.device('cpu'))['model_state_dict'])
        
    # if torch.cuda.device_count() > 1 and args.gpu:
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


 
    # f = open(os.path.join(args.save_dir,'result.csv'),'w')
    # f.writelines('file_name;time\n')
    # f1 = open(os.path.join(args.save_dir,'predict.csv'),'w')
    # f1.writelines('file_name;label\n')
    window_size = 600
    stride = 500
    frame_rate = 100

    file_path = args.audio_file
    sound, sr = librosa.load(file_path, sr=DATA_REQUIRED_SR)
    print("Length of the audio:", len(sound))
    sr_window_size = int(window_size*sr/frame_rate)
    sr_stride = int(stride*sr/frame_rate)

    frame_overlap_no = (len(sound) - sr_window_size)/sr_stride + 1
    
    print("frame_overlap_no: ",frame_overlap_no)

    frame_predictions = []

    for x in range(int(frame_overlap_no)+1):
        if x*sr_stride + sr_window_size > len(sound):
            audio = sound[x*sr_stride:]
        else:
            audio = sound[x*sr_stride:x*sr_stride + sr_window_size]

        audio = audio_normalize(audio)
        # print('Sample rate:',sr)
        
        mixed_sig_stft = fast_stft(audio, n_fft=510, hop_length=160, win_length=400)
        mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2,0,1)),dtype=torch.float32)
        audio_input = mixed_sig_stft.unsqueeze(0)
        # print("Audio_input:",audio_input.shape)

        if torch.cuda.device_count() >= 1 and args.gpu:
            audio_input = audio_input.cuda()
        
        output=net(audio_input)
        pred_labels = (torch.sigmoid(output).detach().cpu().numpy() >= SIGMOID_THRESHOLD).astype(np.float32)
        
        # Get length of audio file
        # sound_check_duration = AudioSegment.from_wav(filepath).duration_seconds * 100
        # print("sound length of audio: {}".format(sound_check_duration))
        #print("Output:",pred_labels)
        frames = pred_labels[0].tolist()
        frames = list(map(int, frames))
        # print(frames)

        if len(frame_predictions) == 0:
            frame_predictions = frames
        else:
            skip_frame_no = window_size - stride
            append_frame = frames[skip_frame_no:]
            frame_predictions.pop()
            frame_predictions = frame_predictions + append_frame
    
    frame_predictions.pop()

    print("Length of total frame:",len(frame_predictions))
    speech_timestamps = vad_collector(0.01,30,frame_predictions,sr )
    
    print(speech_timestamps)

    print(len(speech_timestamps))

if __name__ == '__main__':
    main()
            
            


