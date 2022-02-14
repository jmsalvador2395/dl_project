import os
import pickle
import numpy as np

def import_data():
    print("----- importing data -----")
    data_dir='./data/'
    files=os.listdir(data_dir)
    data=[]
    for f in files:
        if 'demonstrator' in f:
            print('reading {}'.format(f))
            new_data=pickle.load(open(data_dir+f, 'rb'))
            data+=new_data
    if len(data) == 0:
        print('** no data available **')
        return data
    print("----- finished importing data -----")
    return data

def main():
    data=import_data()
    print(len(data))
    
    


if __name__ == '__main__':
    main()
