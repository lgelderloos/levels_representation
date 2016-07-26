# for inspecting word boundary predictions

import cPickle as pickle
import syss

layer = sys.argv[1]
start = int(sys.argv[2])
timesteps = int(sys.argv[3])

with open('predictions{}.p'.format(layer), 'rb') as f:
    predictions = pickle.load(f)
with open('wordends.p', 'rb') as f:
    wordends = pickle.load(f)
with open('ipa.p', 'rb') as f:
    ipa = pickle.load(f)
with open('sents.p', 'rb') as f:
    sents = pickle.load(f)
with open('sentids.p', 'rb') as f:
    sids = pickle.load(f)

chunks = []
chunk = []
current = 0
for i in range(start,timesteps):
    if sids[i] != current:
        print
        print sents[sids[i]]
        current = sids[i]
    print "".join([ipa[i]]),
        
    if predictions[i] == 1:
        print "*",
    if wordends[i] == 1:
        print "|",
