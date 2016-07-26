# -*- Coding: utf-8 -*-
"""
@author: Lieke
"""

from imaginet.data_provider import getDataProvider
from imaginet.simple_data import phonemes
from imaginet.task import pile, load
import sys
import cPickle as pickle
import imaginet.defn
import numpy as np
#reload(imaginet.defn.lm)

##############################################################################

# this looks for boudnary after word
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
#folder = sys.argv[2]
#modelpath = sys.argv[3]

prov = getDataProvider('coco', root = '../reimaginet/')
sents = list(prov.iterSentences(split=split)) 
print len(set([sent['imgid'] for sent in sents]))

ipa = []
wordends = []
onsets = []
sentids = []
ipasentslist = []
ipa_sents = {}
sentences = {}
tokens = {}
 
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

with open('data/{}/{}/tokens.p'.format(folder, split), 'w') as f:
    pickle.dump(tokens, f)
        
with open('data/{}/{}/ipa.p'.format(folder, split), 'w') as f:
    pickle.dump(ipa, f)

with open('data/{}/{}/wordends.p'.format(folder, split), 'w') as f:
    pickle.dump(wordends, f)

with open('data/{}/{}/onsets.p'.format(folder, split), 'w') as f:
    pickle.dump(onsets, f)
        
with open('data/{}/{}/sentids.p'.format(folder, split), 'w') as f:
    pickle.dump(sentids, f)

with open('data/{}/{}/ipa_sents.p'.format(folder, split),'w') as f:
    pickle.dump(ipa_sents, f)

with open('data/{}/{}/sents.p'.format(folder, split), 'w') as f:
    pickle.dump(sentences, f)

# activations        
model = load(path=modelpath)
stack = pile(model, ipasentslist)
layers = [1, 2, 3]
for layer in layers:
    layer_states = np.concatenate([pile[:-1,(layer-1),:] for pile in stack])
    print layer_states.shape
    np.save('data/{}/{}/layer{}.npy'.format(folder, split, layer), layer_states)
