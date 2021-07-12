# 1st param: folder name to create to save audio and transcript
# 2st param: playlist or link of youtube audio
# 3st param: id of youtube audio to get transcript
folder_path=$1
play_list=$2
vid_id=$3
mkdir $folder_path
python /home3/huydd/huydd/script_sp/youtube/crawlYoutube.py   \
    --playlist=$play_list --save_dir=$folder_path 

python /home3/huydd/huydd/script_sp/youtube/crawlTranscript.py  \
    --vid_id=$vid_id --save_dir=$folder_path 

