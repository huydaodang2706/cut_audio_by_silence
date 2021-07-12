import json
import os
from tqdm import tqdm
import shutil
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from util import *
from tools import *

def process_audio(csv_path, output_dir, audio_id, output_json, ext=AUDIO_EXT, print_progress=True):
    """Process an audio file, or all the audio files with the same audio_id
    
    Arguments:
        csv_path {str} -- Path to the csv file describing the dataset
        output_dir {str} -- Path to the output directory containing the processed audio file(s)
        audio_id {str} -- Audio id of the to-be-processed audio file
        output_json {str} -- Path to the json file containing detailed information about every audio file in the output directory
    
    Keyword Arguments:
        ext {str} -- Audio extension (default: {AUDIO_EXT})
        print_progress {bool} -- Print progress when True (default: {True})
    
    Returns:
        int -- Return the number of failed audio files
    """

    print('Processing "{}"...'.format(audio_id))
    ensure_dir(output_dir)
    df = pd.read_csv(csv_path, header=None)
    # print(df)
    df_todownload = df.loc[df.iloc[:, 0] == audio_id.split('/')[0]]
    # print(df_todownload)
    # num_videos = df_todownload.shape[0]
    # if num_videos == 0:
    #     return -1
    # hierarchy = OrderedDict([
    #     ('dataset_path', output_dir),
    #     ('num_videos', num_videos)
    # ])
    # file_info_list = []

    # # download the whole video first
    # whole_video_path = os.path.join(get_parent_dir(csv_path), audio_id + ext)
    # # whole_video_success = youtube_dl_full(audio_id, whole_video_path, FRAMERATE)
    # whole_video_success = os.path.exists(whole_video_path)
    # # print(whole_video_path, whole_video_success)
    # if whole_video_success:
    #     count = 0
    #     success_count = 0
    #     field_toprint = [x for x in FIELDS if x not in (
    #         'bit_stream', 'frames_path')]
    #     for index, row in df_todownload.iterrows():
    #         file_info = OrderedDict()
    #         if print_progress:
    #             print('To process: ', row)
    #         path = os.path.join(
    #             output_dir, '{}_{:07}{}'.format(row[0], count+1, ext))
    #         # print("abcd")
    #         # print(whole_video_path)
    #         duration = get_duration2_audio(whole_video_path)
    #         shutil.move(whole_video_path, path)
    #         file_info[FIELDS[0]] = path
    #         file_info[FIELDS[12]] = 0
    #         file_info[FIELDS[13]] = duration
    #         file_info[FIELDS[10]] = 0
    #         file_info[FIELDS[11]] = 0
    #         file_info[FIELDS[1]] = FRAMERATE
    #         (file_info[FIELDS[2]], file_info[FIELDS[3]]) = \
    #             get_samplerate_numaudiosamples(path, sr=AUDIO_SAMPLE_RATE)
    #         change_audiosamplerate(path, AUDIO_SAMPLE_RATE)
    #         file_info[FIELDS[4]] = duration
    #         file_info[FIELDS[5]] = math.ceil(duration*FRAMERATE)
    #         file_info[FIELDS[6]] = '1' * file_info[FIELDS[5]]
    #         file_info[FIELDS[7]] = 0
    #         file_info[FIELDS[8]] = 0
    #         file_info[FIELDS[9]] = None
    #         file_info[FIELDS[15]] = None
    #         file_info[FIELDS[14]] = path

    #         file_info_list.append(file_info)
    #         count += 1
    #         success_count += 1
    #         # print('11')
    #         if print_progress:
    #             print_dictionary(file_info, key_list=field_toprint)
    #             print(
    #                 '========== {} / {} Complete ==========\n'.format(count, num_videos))

    #     hierarchy['files'] = file_info_list
    #     json_str = json.dumps(hierarchy, **JSON_DUMP_PARAMS)
    #     # print(json_str)

    #     # write file
    #     if print_progress:
    #         print('Writing to "{}"...'.format(output_json))
    #     with open(output_json, 'wb') as f:
    #         f.write(json_str.encode())
    #     if print_progress:
    #         print('Done\n')

    #     # remove whole video
    #     if os.path.exists(whole_video_path):
    #         os.remove(whole_video_path)

    #     return num_videos-success_count

    # else:
    #     return num_videos




if __name__ == "__main__":

    DIR = '/home/huydd/NLP/ASR/SentenceSplit/Speech-Denoise/dataset'
    CSV = '/home/huydd/NLP/ASR/SentenceSplit/Speech-Denoise/dataset/sounds_of_silence.csv'
    JSON = '/home/huydd/NLP/ASR/SentenceSplit/Speech-Denoise/dataset/result/result.json'
    process_audio(CSV,DIR,1,JSON)
    