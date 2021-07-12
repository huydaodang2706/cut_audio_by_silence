import argparse
import os

parser = argparse.ArgumentParser(description="Format audio to acceptable format ")

parser.add_argument("--source1", "-a", type=str, help="directory of audio to convert type")
parser.add_argument("--source2", "-s", type=str, help="Directory to save file")

args = parser.parse_args()
f1 = open('msi_tran.txt','w')
f2 = open('msi_tran_with_silence.txt','w')
for file in os.listdir(args.source1):
    if file.endswith('.txt'):
        f_source1 = open(os.path.join(args.source1,file),'r').readlines()
        f_source2 = open(os.path.join(args.source2,file),'r').readlines()
        # f1.writelines('{},{},{}\n'.format(file,f_source1,f_source2))
        # f1.writelines('{} '.format(f_source1))
        f1.writelines(f_source1)
        f1.writelines(' ')
        # f2.writelines('{} '.format(f_source2))
        f2.writelines(f_source2)
        f2.writelines(' ')
f1.close()