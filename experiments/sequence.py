
from difflib import *
import cPickle as pickle
import random
import numpy
from scipy.spatial.distance import cdist
from imaginet.task import load, pile

def matching_positions(ref, comp):
    match = SequenceMatcher(a=ref, b=comp, autojunk=False)
    for block in match.get_matching_blocks():
        for i in range(block.a, block.a+block.size):
            yield i

def mean_length(ref, comp):
    match = SequenceMatcher(a=ref, b=comp, autojunk=False)
    lengths = []
    # exclude last block (sentence length)
    for block in match.get_matching_blocks()[:-1]:
        lengths.append(block.size)
    return numpy.mean(lengths)

def mean_matching_pos(ref, comp):
    a = list(reversed(ref))
    b = list(reversed(comp))
    return numpy.mean(list(matching_positions(a, b)))

def mmpos_for_neighbours(sentids, distance):
    mmpos = []
    mlen = []
    pairs = []
    
    for i in range(len(sentids)):
        # [1] because [0] returns index of sent
        print i
        sort = numpy.argsort(distance[i,:])
        nn_pos = sort[1]
        mlen.append(mean_length(ipas[sents[i]], ipas[sents[nn_pos]]))
        mmpos.append(mean_matching_pos(ipas[sents[i]], ipas[sents[nn_pos]])) 
        pairs.append( (sents[i], sents[nn_pos]) )
        
    return mmpos, mlen, pairs

######################################
# fix filepath
with open("../data/phongru/val/ipa_sents.p", "rb") as i:
    ipas = pickle.load(i)

#sents = ipas.keys()
#print len(sents)
layers = [1, 2, 3]

#model = load("../reimaginet/emnlp-2016/phon-gru.8.zip")
#reps = [pile(model, [ipas[sent]])[0][-1,:,:] for sent in sents]

#with open("reps_test.p", "wb") as r:
#    pickle.dump(reps, r)
#with open("sents_test.p", "wb") as r:
#    pickle.dump(sents, r)

with open("reps.p", "rb") as r:
    reps = pickle.load(r)
with open("sents.p", "rb") as s:
    sents = pickle.load(s)

data = {}
    
for layer in layers:
    acts = [numpy.asarray(rep[(layer-1),:]) for rep in reps] 
    distance = cdist(acts, acts, metric='cosine')
    #print distance.shape
    mmpos, mlen, ids_pairs = mmpos_for_neighbours(sents, distance)
    data[layer] = {'mm_positions': mmpos, 'mm_length': mlen, 'ids': ids_pairs}
    
with open("position_data.p", "wb") as p:
    pickle.dump(data, p)

print "Layer 1\tpositions: " + str(numpy.mean(data[1]['mm_positions'])) + " length: " + str(numpy.mean(data[1]['mm_length']))
print "Layer 2\tpositions: " + str(numpy.mean(data[2]['mm_positions'])) + " length: " + str(numpy.mean(data[2]['mm_length']))
print "Layer 3\tpositions: " + str(numpy.mean(data[3]['mm_positions'])) + " length: " + str(numpy.mean(data[3]['mm_length']))
