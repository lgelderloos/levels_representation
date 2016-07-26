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
import random

##############################################################################

def pick_other_phoneme(phon, phonemes):
    # pick random thing from phonemes
    if new_phon == phon:
        pick_other_phoneme(phon, phoneme)
    else:
        return new_phon

##############################################################################

split = sys.argv[1]
folder = sys.argv[2]
modelpath = sys.argv[3]
model = load(path=modelpath)
differences = {}

with open('phonemelist.p', 'rb') as f:
    phonemes = 

prov = getDataProvider('coco', root = '../reimaginet/')
sents = list(prov.iterSentences(split=split)) 
#print "nr of images: " + str(len(set([sent['imgid'] for sent in sents])))
#print "nr of sentences: " + str(len(sents))
for sent in sents[:100]:
    sentipa = phonemes(sent)
    new_phon = pick_other_phoneme(sentipa[0])
    modipa = sentipa
    modipa[0] = new_phon
    stack = pile(model, [sentipa, modipa])
    activations1[pos:(pos + len(sent)), :] = stack[0][:-1, 0, :]
    for pos in range(len(diffs)):
        if pos in differences:
            differences[pos].append(diffs[pos])
        else:
            differences[pos] = [diffs[pos]]
