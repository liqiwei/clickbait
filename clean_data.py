#get rid of all spaces in data
import pandas as pd

def rid_space():
    no_space =open('clickbait.txt','w')
    with open('clickbait_data.txt','r') as with_space:
        for line in with_space.readlines():
            if not line.isspace():
                line = line+","
                print(line)
                no_space.write(line)

def combine_csv():
    cb = open('clickbait.csv','r')
    non_cb = open('non_clickbait.csv','r')
    with open('all_data.csv','w') as all_data_file:
        for line in cb.readlines():
            all_data_file.write(line)
        for line in non_cb.readlines():
            all_data_file.write(line)


combine_csv()



