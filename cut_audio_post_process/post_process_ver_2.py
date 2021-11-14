import pandas as pd
from pydub import AudioSegment
import argparse
import os
# Cat audio chi dua vao max_silence, min_silence, max_accept_silence

# MIN_SILENCE_INTERVAL = 0.01
MAX_SILENCE_INTERVAL = 0.01
MIN_AUDIO_LEN = 3
# MAX_AUDIO_LEN = 15
MAX_ACCEPT_SILENCE_INTERVAL = 2
def export_audio(original_audio, start, end, save_path):
    # Start va end (giay)
    start = start * 1000
    end = end * 1000
    new_audio = original_audio[start:end]
    new_audio.export(save_path, format='wav')

# Predict dir must contain temp and result file
def  process_csv(audio_path, pr_path, save_path, base_name, 
        max_silence=MAX_SILENCE_INTERVAL, 
        min_audio_len=MIN_AUDIO_LEN,
        max_accept_silence=MAX_ACCEPT_SILENCE_INTERVAL):
    # Read audio file
    ori_audio = AudioSegment.from_wav(audio_path)
    # Read predicted csv file
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
    # index_start = 0
    file_index = 0
    index = 0
    # print('Start of the block to scan: ' , index)
   
    # file_save_path = os.path.join(save_path, base_name + '_' + str(file_index) + '.wav')
    while index < number_of_rows:
        row = pr_result.iloc[index]
        file_save_path = os.path.join(save_path, base_name + '_' + str(file_index) + '.wav')

        if index < number_of_rows-1:
            # B1: Cap nhat previous cut point
            if index == 0:
                pre_cut_p = row['start']
            elif flag_skip:
                print('Skip previous cut point')
            elif flag_max:
                pre_cut_p = row['start'] - 1
            else:
                pre_cut_p = cur_cut_p
                
            
            # B2: Xu ly silence interval
            silence_duration = pr_result.iloc[index + 1]['start'] - row['end']

            if silence_duration < max_silence:
                # Skip do silence duration nho hon muc cho phep
                flag_skip = True
                # skip_silence_intervals.append(row)
            else:
                # TH silence duration lon hon max_silence cho phep
                flag_skip = False

                # silence duration > MAX_SILENCE_INTERVAL
                if silence_duration > max_accept_silence:
                    # TH silence duration lon hon muc cho phep (thuong la 2s)
                    flag_max = True
                    cur_cut_p = row['end'] + 1
                    # index = index + 1

                    print('Cut audio')
                    export_audio(ori_audio, pre_cut_p, cur_cut_p, file_save_path)
                    file_index +=  1

                else:
                    # TH van nho hon muc cho phep (muc cho phep thuong la 2s)
                    flag_max = False

                    cur_cut_p = row['end'] + int(silence_duration/2)
                
                    audio_len = cur_cut_p - pre_cut_p

                    if audio_len > min_audio_len:
                        # cut audio
                        export_audio(ori_audio, pre_cut_p, cur_cut_p, file_save_path)
                        file_index +=  1

                    else:
                        flag_skip = True                            
            index += 1
        else:
            cur_cut_p  = row['end'] 
            print('Cut audio')
            export_audio(ori_audio, pre_cut_p, cur_cut_p, file_save_path)
            file_index +=  1

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

    cannot_do = []
    for pr_dir in os.listdir(args.predict_dir):
        # 
        i = 0
        audio_id = pr_dir.replace('_re','')
        pr_file_path = os.path.join(args.predict_dir, pr_dir, audio_id + '.csv')
        audio_path = os.path.join(args.audio_dir, audio_id, audio_id + '.wav')

        # ori_audio = AudioSegment.from_wav(audio_path)
        
        save_path = os.path.join(args.save_dir, audio_id)
        os.mkdir(save_path)
        try:
            process_csv(audio_path, pr_file_path, save_path, audio_id)    
            # print(0)
        except:
            cannot_do.append(audio_path)
            os.rmdir(save_path)
        # pr_result = pd.read_csv(pr_file_path)
    print(cannot_do)
    f = open(os.path.join(args.save_dir, 'cannot_do.txt'), 'w')
    for x in cannot_do:
        f.writelines(str(x) + '\n')