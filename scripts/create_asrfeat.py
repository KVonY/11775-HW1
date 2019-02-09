#!/bin/python
import numpy as np
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} vocab_file, file_list".format(sys.argv[0])
        print "vocab_file -- path to the vocabulary file"
        print "file_list -- the list of videos"
        exit(1)
    vocab_file = sys.argv[1]
    file_list = sys.argv[2]
    vocab = list(np.genfromtxt(vocab_file, dtype=str))
    files = open(file_list)
    output_path = "asrfeat/feature.vec"
    output = open(output_path, 'w')
    for f in files:
        video_name = f.strip()
        asr_path = "11775_asr/{}.ctm".format(video_name)
        if os.path.exists(asr_path) != True:
            print "No ASR features\n"
            z = np.zeros(len(vocab))
            z.fill(1.0 / len(vocab))
            zeros = map(str, z)
            zeros_feature = ';'.join(zeros)
            output.write(video_name + ' ' + zeros_feature + '\n')
            continue
        bow = np.zeros(len(vocab))
        asr_file = open(asr_path, 'r')
        for i in asr_file:
            word = i.split()[4]
            if word in vocab:
                idx = vocab.index(word)
                bow[idx] += 1
        # normalize
        if np.sum(bow) == 0:
            bow_vec = np.zeros(len(vocab))
            bow_vec.fill(1.0 / len(vocab))
            print "WRONG bag-of-word vector representation\n"
        else:
            bow_vec = bow/float(np.sum(bow))
        # output
        feature = ';'.join([str(i) for i in bow_vec])
        output.write(video_name + ' ' + feature + '\n')
    output.close()
    files.close()

    print "ASR features generated successfully!"
