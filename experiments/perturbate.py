# -*- coding: utf-8 -*-

import numpy
import cPickle as pickle
import random
import sys
from imaginet.task import representation, pile, load
from imaginet.simple_data import words, phonemes
from scipy.spatial.distance import cosine

###############################################################################
    
def perturbate(sentence):
    random_char = random.choice(phonemes)
    while random_char == sentence[0]:
        random_char = random.choice(phonemes)
    alt_sent = list(sentence)
    #print alt_sent[:5]
    alt_sent[0] = random_char
    #print alt_sent[:5]
    return alt_sent

def get_dists(sentence):
    global differences
    
    alt_sent = perturbate(sentence)
    #print sentence[:5]
    #print alt_sent[:5]
    acts = pile(model, [sentence, alt_sent])
    for layer in [1, 2, 3]:
        for t in range(len(alt_sent)+1):
            if not t in differences[layer]:
                differences[layer][t] = []
            differences[layer][t].append(cosine(acts[0][t,(layer-1),:], acts[1][t,(layer-1),:]))

#############################################################################

split = sys.argv[1]
outputarray = sys.argv[2]

with open("../boundaries/data/phongru/{}/ipa_sents.p".format(split), "rb") as i:
    ipasents = pickle.load(i)
with open("phonemes.p", "rb") as p:
    phonemes = pickle.load(p)

# load model
model = load('../reimaginet/emnlp-2016/phon-gru.8.zip')

differences = { 1: {}, 2: {}, 3: {} }

#c = 0
for key in ipasents:
    get_dists(ipasents[key])
    #c += 1
    #if c > 200:
    #    break
max_timesteps = len(differences[1].keys())
avgs = numpy.empty([max_timesteps, 3])
for t in range(0, max_timesteps):
    one = numpy.mean(differences[1][t])
    two = numpy.mean(differences[2][t])
    three = numpy.mean(differences[3][t])
    avgs[t, 0] = one
    avgs[t, 1] = two
    avgs[t, 2] = three
    print "t =", str(t),  str(one), str(two), str(three)

with open(outputarray, "wb") as f:
    numpy.save(f, avgs)
