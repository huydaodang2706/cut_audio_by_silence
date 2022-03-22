import pandas as pd
from pydub import AudioSegment
import argparse
import os
import shutil

# MIN va MAX SILEMCE LENGTH nen la min = 0.01s tro xuong, max = 0.3s tro xuong
MIN_SILENCE_INTERVAL = 0.01
MAX_SILENCE_INTERVAL = 0.02
# MIN va MAX AUDIO LENGTH nen chenh nhau 5-10s
MIN_AUDIO_LEN = 10 
MAX_AUDIO_LEN = 15 
MAX_ACCEPT_SILENCE_INTERVAL = 2
def export_audio(original_audio, start, end, save_path):
    # Start va end (giay)
    start = start * 1000
    end = end * 1000
    new_audio = original_audio[start:end]
    new_audio.export(save_path, format='wav')

# Predict dir must contain temp and result file
def process_csv(audio_path, pr_path, save_path, base_name, 
        min_silence=MIN_SILENCE_INTERVAL, 
        max_silence=MAX_SILENCE_INTERVAL, 
        min_audio_len=MIN_AUDIO_LEN,
        max_audio_len=MAX_AUDIO_LEN,
        max_accept_silence=MAX_ACCEPT_SILENCE_INTERVAL):
    '''
        Input:
            audio_path: path of audio
            pr_path: path of csv prediction (output vad, time is second)
        Output: 
            None
    '''
    if max_accept_silence/2 > 1:
        padding = 1
    else:
        padding = int(max_accept_silence/2)
    
    ori_audio = AudioSegment.from_wav(audio_path)
    pr_result = pd.read_csv(pr_path)
    
    # Get number of chunk in csv file
    row_index = pr_result.index
    number_of_rows = len(row_index)

    # set previous cut point and current cut point = 0
    pre_cut_p = 0
    cur_cut_p = 0

    # Flag max accept silence duration 
    flag_max = False
    # Flag skip 
    flag_skip = False
    # Index start of the block for scanning interval to cut
    index_start = 0
    next_blog = False
    file_index = 0

    while index_start < number_of_rows:
        index = index_start
        next_blog = False
        print('-----------------------------------------')
        print('Start of the block to scan: ' , index)
        
        file_index += 1
        file_save_path = os.path.join(save_path, base_name + '_' + str(file_index) + '.wav')
        # Khi bat dau 1 block thi se dat lai 1 so tham so
        silence_len = max_silence
        
        while index < number_of_rows and not next_blog:
            row = pr_result.iloc[index]
        
            if index < number_of_rows-1:
                # B1: Cap nhat previous cut point
                if index == 0:
                    pre_cut_p = row['start']
                elif flag_skip:
                    print('Skip previous cut point')
                elif flag_max:
                    pre_cut_p = row['start'] - padding
                else:
                    pre_cut_p = cur_cut_p
                    
                
                # B2: Xu ly silence interval
                silence_duration = pr_result.iloc[index + 1]['start'] - row['end']

                if silence_duration < silence_len:
                    flag_skip = True
                    # skip_silence_intervals.append(row)
                else:
                    flag_skip = False

                    # silence duration > MAX_SILENCE_INTERVAL
                    if silence_duration > max_accept_silence:
                        flag_max = True
                        cur_cut_p = row['end'] + padding
                        
                        print('Cut audio')
                        export_audio(ori_audio, pre_cut_p, cur_cut_p, file_save_path)
                        
                        index_start = index + 1
                        next_blog = True
                    else:
                        flag_max = False

                        cur_cut_p = row['end'] + int(silence_duration/2)
                    
                        audio_len = cur_cut_p - pre_cut_p

                        if audio_len > min_audio_len:
                            if audio_len < max_audio_len:
                                print('Cut audio')
                                export_audio(ori_audio, pre_cut_p, cur_cut_p, file_save_path)

                                index_start = index + 1
                                next_blog = True
                            else:
                                # Truong hop audio len vuot qua max_audio_len thi phai duyet lai tu index_start de va su dung min_silence_len
                                if silence_len == max_silence:
                                    silence_len = min_silence
                                    index = index_start - 1
                                    flag_skip = True
                                    print('Need search again from: ', index_start)
                                elif silence_len == min_silence:
                                    silence_len = 0.04
                                    # Quay tro lai index_start de scan nhung tru 1 de cong 1 ben duoi la vua
                                    index = index_start - 1
                                    flag_skip = True
                                    print('Need search again from: ', index_start)
                                else:
                                    if index_start != index:
                                        silence_duration = pr_result.iloc[index]['start'] - pr_result.iloc[index - 1]['end']
                                        cur_cut_p = pr_result.iloc[index - 1]['end'] + int(silence_duration/2)
                                        index_start = index
                                    else:
                                        index_start = index + 1
                                    
                                    export_audio(ori_audio,pre_cut_p, cur_cut_p, file_save_path)
                                    flag_skip = False
                                    next_blog = True
                                    
                        else:
                            flag_skip = True
                
            else:
                # index chay den cuoi file roi thi khong can phai xu ly gi nua ma cat luon
                pre_cut_p = cur_cut_p
                cur_cut_p  = row['end'] 
                # index += 1
                print('Cut audio')
                export_audio(ori_audio, pre_cut_p, cur_cut_p, file_save_path)
                
                index_start = index + 1
                next_blog = True

            index += 1




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir','-a',type=str,default='',required=True, help="Audio directory to process")
    parser.add_argument('--predict_dir','-p',type=str,default='',required=True, help="Audio directory to process")
    parser.add_argument('--save_dir', type=str, help="outputs dir to write results")
    # audio_dir:  thu muc chua audio o dang ban dau
    # predict_dir: thu muc chua ket qua da cat 
    # save_dir: thu muc ket qua de luu
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    for pr_dir in os.listdir(args.predict_dir):
        # 
        if pr_dir.endswith('_re'):
            i = 0
            try:
                audio_id = pr_dir.replace('_re','')
                pr_file_path = os.path.join(args.predict_dir, pr_dir, audio_id + '.csv')
                audio_path = os.path.join(args.audio_dir, audio_id, audio_id + '.wav')

                # ori_audio = AudioSegment.from_wav(audio_path)
                
                save_path = os.path.join(args.save_dir, audio_id)
                os.mkdir(save_path)

                process_csv(audio_path, pr_file_path, save_path, audio_id)    
                pr_result = pd.read_csv(pr_file_path)
            except: 
                if os.path.isdir(save_path):
                    shutil.rmtree(save_path, ignore_errors=True)
                print('File not exists')
