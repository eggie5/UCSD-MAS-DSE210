import pprint 
import os
import numpy as np
import sys

def read_features(path):
    with open(path, 'r') as f:
        features = []
        for line in f:
            features.append(map(int, line.split()))
    return np.array(features)


def read_label(path, file_name):
    if os.path.isfile(file_name+'.npy') and os.path.isfile('pi.npy'):
        return np.load(file_name + '.npy'), np.load('pi.npy')
    with open(os.path.join(path, file_name), 'r') as f:
        label_list = []
        for line in f:
            label_list.append(int(line))
        label_array = np.array(label_list)

    return label_array
    
def read_vocab(path):
    with open(path, 'r') as f:
        vocab = []
        i=0
        for line in f:
            i+=1
            vocab.append(i)
    return np.array(vocab)