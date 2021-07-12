from __future__ import unicode_literals
import os
import argparse
from youtube_transcript_api import YouTubeTranscriptApi

parser = argparse.ArgumentParser(description="crawlYoutube transcript flags")

parser.add_argument("--vid_id", "-id", type=str, help="Id of video to crawl")
parser.add_argument("--save_dir", "-s", type=str,
                    help="Directory to save crawled files")
parser.add_argument("--sublang", type=str, help="Subtitle language", default="vi")

args = parser.parse_args()

if not (args.vid_id and args.save_dir and args.sublang):
    raise ValueError("must specify --vid_id and --save_dir")

    
srt = YouTubeTranscriptApi.get_transcript(args.vid_id,languages=(args.sublang,))
with open( os.path.join(args.save_dir, 'srt_file.txt'),'w') as f:
    for i in srt:
        f.write(str(i))
