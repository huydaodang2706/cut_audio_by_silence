import argparse
import os
import json

parser = argparse.ArgumentParser(description="Format audio transcript to acceptable format ")

parser.add_argument("--script_file", "-a", type=str, help="directory of audio to convert type")
parser.add_argument("--save_dir", "-s", type=str, help="Directory to save file")

args = parser.parse_args()

f = open(args.script_file,'r')
# print(file.read())
delimiter = '}'
script = [(e + delimiter).replace('\'','\"') for e in f.read().split('}') if e] 

print(type(script))
print(script)

script_string = ""
for x in script:
    x = json.loads(x)
    # print(x)
    if x['text'] != '[âm nhạc]':
        script_string = script_string + ' ' + x['text']

print(script_string)

fw = open(args.save_dir, 'w')

fw.write(script_string)

f.close()
fw.close()
