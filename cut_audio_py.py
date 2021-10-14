import argparse
import json
import os
import pandas as pd

def gen_silence_label(duration):
    x = int(duration/10)
    label = []
    for i in range(x):
        label.append(0)
    return label


parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir','-a',type=str,default='',required=True, help="Audio directory to process")
parser.add_argument('--save_dir', type=str, help="outputs dir to write results")
args = parser.parse_args()
if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

# path = os.walk(args.audio_dir)

cpu_cannot_do = []
for sub2 in os.listdir(args.audio_dir):
    sub2_path = os.path.join(args.audio_dir, sub2)
    print(sub2_path)
    # f.writelines(sub2_path + '\n')
    
    for sub3 in os.listdir(sub2_path):
        if sub3.endswith('.wav'):
            sub3_path = os.path.join(sub2_path,sub3)
            print(sub3_path)
            
            audio_path=sub2_path
            file_path=sub3_path
            
            new_dir = os.path.join(args.save_dir,sub2) + '_temp'
            print("New dir: ",new_dir)
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
            save_dir=new_dir

            new_output_dir = os.path.join(args.save_dir,sub2) + '_result'
            if not os.path.isdir(new_output_dir):
                os.mkdir(new_output_dir)
            output_dir=new_output_dir
            print(audio_path)
            print(file_path)
            print(save_dir)
            print(output_dir)
            command = 'sh cut_audio_py_1.sh ' + audio_path + ' ' + file_path + ' ' + save_dir + ' ' + output_dir
            print(command)
            os.system(command)
            # save_dir is temp dir
            # output_dir is result_dir
            try:
                # Generate start end label
                py_result = pd.read_csv(os.path.join(save_dir,'result.csv'),sep=',',index_col=0)
                sp_result = pd.read_csv(os.path.join(output_dir,'result.csv'),sep=';',index_col=0)
                
                # print(crawl_link['link'])
                f = open(os.path.join(output_dir,sub3.replace('.wav','')  + '.csv'),'w')
                f.writelines('name,start,end,duration\n')
                i = 0
                for index, row in py_result.iterrows():
                    print(index)
                    print(row)
                    # print(sp_result.iloc[index]['time'])
                    speech_segment = eval(str(sp_result.loc[index]['time']))
                    # print(speech_segment)
                    for segment in speech_segment:
                        time_start = round(row['start'] + segment['start']/16,4)
                        time_end = round(row['start'] + segment['end']/16,4)
                        duration = round(time_end - time_start,4)
                        print(time_start)
                        print(time_end)
                        f.writelines('interval_' + str(i) + ',' + str(time_start) + ',' + str(time_end) + ',' + str(duration) + '\n')
                        i += 1
                    # break
                f.close()
                # print(py_result.iloc[1])

                # generate 01 label
                py_result_1 = pd.read_csv(os.path.join(save_dir,'result.csv'),sep=',',index_col=0)
                sp_result_1 = pd.read_csv(os.path.join(output_dir,'predict.csv'),sep=';',index_col=0)
                g = open(os.path.join(output_dir,'predict_label.txt'),'w')
                i = 0
                predict_label = []

                for index, row in py_result_1.iterrows():

                    print(index)
                    print(row)
                    segment_label = eval(str(sp_result_1.loc[index]['label']))

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
                g.writelines(str(predict_label))
                g.close()   
                if os.path.isfile(os.path.join(output_dir,'predict_label.txt')):
                    os.system('rm -rf ' + audio_path )
            except:
                cpu_cannot_do.append(audio_path)
                print(output_dir + ' cannot be processed by this cpu!')
                os.system('rm -rf ' + output_dir)
                os.system('rm -rf ' + save_dir)
            for temp_path in os.listdir(save_dir):
                if temp_path.endswith('wav'):
                    os.remove(os.path.join(save_dir, temp_path))

print('CPU cannot do:')
for path in cpu_cannot_do:
    print(path)
    

