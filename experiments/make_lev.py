# -*- coding: utf-8 -*-
import sys
#from subprocess import check_output
#from scipy.spatial.distance import cosine
from imaginet.task import load
#import csv
import simmetrics
import cPickle as pickle
import numpy as np
from scipy.stats import spearmanr

def wordcount(word, sentindex):
    if word in sentindex:
        count = len(sentindex[word])
    else:
        count = 0
    return count
    
modelname = sys.argv[1]
dataset = sys.argv[2]

# WHY DOES THIS NOT WORK WITH THE OLD MODEL #it does just was
# importing clipped)elu in defn/visual.py and that doesnt exits anymore
model = load("../reimaginet/emnlp-2016/{}.zip".format(modelname))    
# model = load("../reimaginet/vis/model.10.zip")
data = {}

with open("../boundaries/sentidlextrain.p", "rb") as s:
    sentindex = pickle.load(s)

f = open(dataset, 'r')
for line in f:
    fields = line.split("\t")
    #reader = csv.reader(d, delimiter = '\t')
    #with open('simdata_{}_{}'.format(modelname, dataset), 'wb') as s:
        #writer = csv.writer(s, delimiter='\t')
    #for row in reader:
    pair = (fields[0], fields[1])
    pairdata = {}
    pairdata['hr'] = float(fields[2])
    pairdata['word1'] = fields[0]
    pairdata['word2'] = fields[1]
    metrics = simmetrics.get_metrics(fields[0], fields[1], model)
    pairdata['cos_1'] = metrics[0]
    pairdata['cos_2'] = metrics[1]
    pairdata['cos_3'] = metrics[2]
    pairdata['lev'] = metrics[3]
    pairdata['count1'] = wordcount(fields[0], sentindex)
    pairdata['count2'] = wordcount(fields[1], sentindex)
    data[pair] = pairdata

f.close()

# pickle data for future use
with open('simdata_{}_{}.p'.format(dataset, modelname), 'wb') as f:
    pickle.dump(data, f)
"""
# load data from pickle
with open('simdata_{}.p'.format(dataset), 'rb') as f:
    data = pickle.load(f)
"""

# calculate spearmans rho to see if it makes any sense
hr = []
cos_1 = []
cos_2 = []
cos_3 = []
lev = []

for pair in data:
    hr.append(data[pair]['hr'])
    cos_1.append(data[pair]['cos_1'])
    cos_2.append(data[pair]['cos_2'])
    cos_3.append(data[pair]['cos_3'])
    lev.append(data[pair]['lev'])

print str(spearmanr(np.asarray(hr), np.asarray(cos_1)))
print str(spearmanr(np.asarray(hr), np.asarray(cos_2)))
print str(spearmanr(np.asarray(hr), np.asarray(cos_3)))
