#from imaginet.simple_data import words
from imaginet.data_provider import getDataProvider
import cPickle as pickle
from collections import Counter

def ipawords(ipalist):
    word = "" 
    words = []
    for pho in ipalist:
        if pho == "*":
            # check if word is empty, otherwise add to word list
            if not word == "":
                words.append(word)
            word = ""
        else:
            word = word + pho
    return words

prov = getDataProvider('coco', root = '../reimaginet')
sents = list(prov.iterSentences(split="train"))

hugewordlist = []
for s in sents:
    hugewordlist.extend(ipawords(s['ipa']))
            
trainlexicon = Counter(hugewordlist)
with open("phongru_trainlexiconipa.p", "wb") as l:
    pickle.dump(trainlexicon, l)
