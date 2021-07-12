from jiwer import wer
import argparse
# ground_truth = "hello world this is the new world"
# hypothesis = "hello world this new world"
# ground_truth = "hello world"
# hypothesis = "hello world this new world"
parser = argparse.ArgumentParser(description="API upload file wav to google API")

parser.add_argument("--gr", "-a", type=str, help="directory of audio to convert type")
parser.add_argument("--cr", "-s", type=str, help="Directory to save file")
args = parser.parse_args()

f = open(args.gr,'r')
g = open(args.cr,'r')
ground_truth = f.readlines()
hypothesis = g.readlines()

# print(ground_truth)
# print(hypothesis)
error = wer(ground_truth, hypothesis)
print(error)