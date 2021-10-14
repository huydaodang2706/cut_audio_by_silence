# 1st param: audio_path to format audio
# 2st param: file_path to cut with py-webrtcvad
# 3st param: save_path for audio output of py-webrtcvad
# 4st param: output dir after speech denoise repo

# Format audio for acceptable input of py-we
audio_path=$1
file_path=$2
save_dir=$3
output_dir=$4
sample_rate=16000

format_code=/home3/huydd/cut_audio_by_silence/py-webrtcvad/format_audio.py
python $format_code --audio_dir=$audio_path --save_dir=$audio_path 
# Cut audio by py-webrtc vad 
py_path=/home3/huydd/cut_audio_by_silence/py-webrtcvad/example1.py

python $py_path 3 $file_path $save_dir 

# Cut audio output by silero vad by speech denoise 
silence_detection_path=/home3/huydd/cut_audio_by_silence/Speech-Denoise/model_1_silent_interval_detection/audioonly_model/inference_1.py
ckpt=15

use_gpu=True
python $silence_detection_path --audio_dir=$save_dir --ckpt=$ckpt --save_dir=$output_dir --gpu=$use_gpu
# echo $audio_path
# echo $file_path
# echo $save_dir
# echo $output_dir