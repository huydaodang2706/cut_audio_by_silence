from pydub import AudioSegment
from pydub.playback import play
import argparse
import os

parser = argparse.ArgumentParser(description="Format audio to acceptable format ")

parser.add_argument("--audio_dir", "-a", type=str, help="directory of audio to convert type")
parser.add_argument("--save_dir", "-s", type=str, help="Directory to save file")

args = parser.parse_args()

startMin = 1
startSec = 00

endMin = 1
endSec = 20

# Time to miliseconds
startTime = startMin*60*1000+startSec*1000
endTime = endMin*60*1000+endSec*1000

for file in os.listdir(args.audio_dir):
    if file.endswith('.wav'):
        filepath = os.path.join(args.audio_dir, file)
        # print("file_path:",filepath)
        sound = AudioSegment.from_wav(filepath)
        # Change to mono-channel
        sound = sound.set_channels(1)
        # Change sample rate to 16k
        sound = sound.set_frame_rate(16000)
        # Change sample width
        sound = sound.set_sample_width(2)
        # print(sound[10:20])
        # play(sound[10:60])
        # Export new audio to directory
        sound[startTime:endTime].export(args.save_dir + "/" + file, format="wav")