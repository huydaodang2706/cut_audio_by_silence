import os
import pandas as pd

crawl_link = pd.read_csv('link_validate_silence_detection.csv',sep=',')

folder_path = './Val_crawl'
# print(crawl_link['link'])
for index, row in crawl_link.iterrows():
    command = 'sh crawl.sh ' + folder_path + '/' + row['file_name'] + ' ' + row['link'] + ' ' + row['video_id']
    print(command)
    os.system(command)
