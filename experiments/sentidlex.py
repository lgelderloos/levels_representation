from imaginet.simple_data import words
from imaginet.data_provider import getDataProvider
import cPickle as pickle
from collections import Counter
import sys

split = sys.argv[1]
prov = getDataProvider('coco', root = '../reimaginet')
sents = list(prov.iterSentences(split=split))
indexlexicon = {}

for s in sents:
    for word in words(s):
        if word in indexlexicon:
            indexlexicon[word].append(s['sentid'])
        else:
            indexlexicon[word] = [s['sentid']]
        
with open("sentidlex{}.p".format(split), "wb") as l:
    pickle.dump(indexlexicon, l)
