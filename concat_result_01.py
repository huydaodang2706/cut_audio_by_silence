import os
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--py_result',type=str,default='',required=True, help="py_webrtc result")
parser.add_argument('--sp_result', type=str, help="speech-denoise result")
parser.add_argument('--save_dir', type=str, help="concat  result")

args = parser.parse_args()

py_result = pd.read_csv(args.py_result,sep=',',index_col=0)
sp_result = pd.read_csv(args.sp_result,sep=';',index_col=0)
print(py_result)
print(sp_result)
# print(crawl_link['link'])
f = open(os.path.join(args.save_dir,'predict_label.txt'),'w')
i = 0
predict_label = []

def gen_silence_label(duration):
    x = int(duration/10)
    label = []
    for i in range(x):
        label.append(0)
    return label


for index, row in py_result.iterrows():

    print(index)
    print(row)
    segment_label = eval(str(sp_result.loc[index]['label']))

    if index == 'chunk-0.wav' :
        if row['start'] != 0:
            x1 = gen_silence_label(row['start'])
            predict_label = predict_label + x1
            # print(type(segment_label))
            x2 = segment_label
            predict_label = predict_label + x2
        else:
            x2 = segment_label
            predict_label = predict_label + x2
    else:
        if row['start'] == previous_row['end']:
            x2 = segment_label
            predict_label = predict_label + x2
        else:
            x1 = gen_silence_label(row['start'] - previous_row['end'])
            predict_label = predict_label + x1
            x2 = segment_label
            predict_label = predict_label + x2

    previous_row = row
print(predict_label)
print(len(predict_label))
f.writelines(str(predict_label))