# -*- Coding: utf-8 -*-

from imaginet.data_provider import getDataProvider
from imaginet.simple_data import phonemes
from imaginet.task import pile, load
import sys
import cPickle as pickle
import imaginet.defn
import numpy as np
#reload(imaginet.defn.lm)

##############################################################################

# this looks for boundary after word
def get_wordends(caption):
    boundaries = []
    pos = -1
    for p in caption:
        if p == "*":
            if not pos == -1:
                boundaries[pos] = 1
        else:
            pos += 1
            boundaries.append(0)
    boundaries[pos]=1
    return boundaries            

# looks for onsets
def get_onsets(caption):
    boundaries = []
    pos = 0
    for p in caption:
        if p == "*":
            boundaries.append(1)
        else:
            if len(boundaries) == 0:
                boundaries.append(1)
            elif len(boundaries) == pos:
                boundaries.append(0)
            pos += 1
    return boundaries

##############################################################################

split = sys.argv[1]
folder = sys.argv[2]
modelpath = sys.argv[3]

prov = getDataProvider('coco', root = '../reimaginet/')
sents = list(prov.iterSentences(split=split)) 
#print "nr of images: " + str(len(set([sent['imgid'] for sent in sents])))
#print "nr of sentences: " + str(len(sents))

ipa = []
wordends = []
onsets = []
sentids = []
ipasentslist = []
ipa_sents = {}
sentences = {}
tokens = {}

print "making input data..."
for sent in sents:
    sentipa = phonemes(sent)
    sentid = sent['sentid']
    tokens[sentid] = sent['tokens']
    wordends.extend(get_wordends(sent['ipa']))
    onsets.extend(get_onsets(sent['ipa']))
    ipasentslist.append(sentipa)
    ipa_sents[sentid] = sentipa    
    sentences[sentid] = sent['raw']
    ipa.extend(sentipa)
    for i in sentipa:
        sentids.append(sentid)
#print "ipa length: " + str(len(ipa))
"""
with open('../data/{}/{}/tokens.p'.format(folder, split), 'w') as f:
    pickle.dump(tokens, f)
        
with open('../data/{}/{}/ipa.p'.format(folder, split), 'w') as f:
    pickle.dump(ipa, f)

with open('../data/{}/{}/wordends.p'.format(folder, split), 'w') as f:
    pickle.dump(wordends, f)

with open('../data/{}/{}/onsets.p'.format(folder, split), 'w') as f:
    pickle.dump(onsets, f)
        
with open('../data/{}/{}/sentids.p'.format(folder, split), 'w') as f:
    pickle.dump(sentids, f)

with open('../data/{}/{}/ipa_sents.p'.format(folder, split),'w') as f:
    pickle.dump(ipa_sents, f)

with open('../data/{}/{}/sents.p'.format(folder, split), 'w') as f:
    pickle.dump(sentences, f)

#print "timesteps: "+ str(len(wordends))
"""
print "start getting activations..."
model = load(path=modelpath)
# initialize empty numpy arrays
activations1 = np.empty([len(wordends), 1024])
activations2 = np.empty([len(wordends), 1024])
activations3 = np.empty([len(wordends), 1024])
pos = 0
for sent in ipasentslist:
    stack = pile(model, [sent])
    activations1[pos:(pos + len(sent)), :] = stack[0][:-1, 0, :]
    activations2[pos:(pos + len(sent)), :] = stack[0][:-1, 1, :]
    activations3[pos:(pos + len(sent)), :] = stack[0][:-1, 2, :]
    pos += len(sent)

print "writing activations to files..."
with open('../data/{}/{}/layer1.npy'.format(folder, split), 'wb') as f:
    np.save(f, activations1)
with open('../data/{}/{}/layer2.npy'.format(folder, split), 'wb') as f:
    np.save(f, activations2)
with open('../data/{}/{}/layer3.npy'.format(folder, split), 'wb') as f:
    np.save(f, activations3)
