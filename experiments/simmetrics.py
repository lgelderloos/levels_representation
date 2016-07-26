# -*- coding: utf-8 -*-
import sys
from subprocess import check_output
from scipy.spatial import distance
from imaginet.task import pile, representation
import numpy as np


def espeak(words):
    return phon(check_output(["espeak", "-q", "--ipa=3", '-v', 'en', words]).decode('utf-8'))

def clean_phonemes(espeakoutput):
    '''
    Takes espeak output as input
    returns list of phonemes without word boundaries
    '''
    no_word_stress = espeakoutput.replace("ˈ", "")
    no_stress = no_word_stress.replace("ˌ", "")
    boundaries = no_stress.replace(" ","_")
    no_ii = boundaries.replace("iːː", "iː")
    phonemes = no_ii.split("_")

    #remove 'empty' phonemes
    while "" in phonemes:
        phonemes.remove('')

    return phonemes

def texttophonemes(text):
    '''
    Takes orthograpic sentence as input
    returns list of phonemes and boundaries
    boundaries are indicated by an asterisk 
    '''
    # remove punctuation and add quotes for espeak
    chars = []
    for c in text:
        if c.isspace() or c.isalnum():
            chars.append(c)
    espeaktext = '"' + ''.join(chars) + '"'

    espeakout = check_output(['espeak', '-q', '--ipa=3', '-v', 'en', espeaktext])
    # strip of newline characters etc
    espeakout = espeakout.strip()
    phonemelist = clean_phonemes(espeakout)
    return [pho.decode('utf-8') for pho in phonemelist]

def cosine_similarity(array1, array2):
    return 1.0-(distance.cosine(array1, array2))

# copied implementation from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)
        # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)
        # We call tuple() to force strings to be used as sequences
        # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1
        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(current_row[1:], np.add(previous_row[:-1], target != s))
        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(current_row[1:], current_row[0:-1] + 1)
        previous_row = current_row

    return previous_row[-1]

# returns levenshtein distance divided by length of longest item
def norm_levenshtein(item1, item2):
    lev = float(levenshtein(item1, item2))
    maxlev = float(max(len(item1), len(item2)))
    return lev/maxlev

def get_metrics(word1, word2, model, alphabet = "orthographic"):
    # only do ipa conversion if necessary
    if alphabet == "ipa":
        ipa1 = word1
        ipa2 = word2 
    elif alphabet == "orthographic":
        ipa1 = texttophonemes(word1)
        ipa2 = texttophonemes(word2)

    lev = norm_levenshtein(ipa1, ipa2)
    #act1, act2 = representation(model, [ipa1, ipa2])
    pile1 = pile(model, [ipa1])
    pile2 = pile(model, [ipa2])
    #print stack[0].shape, stack[1].shape
    cosine_sim_layer_1 = cosine_similarity(pile1[0][-1,0,:], pile2[0][-1,0,:])
    cosine_sim_layer_2 = cosine_similarity(pile1[0][-1,1,:], pile2[0][-1,1,:])
    cosine_sim_last_layer = cosine_similarity(pile1[0][-1,-1,:], pile2[0][-1,-1,:])
    return [ cosine_sim_layer_1, cosine_sim_layer_2, cosine_sim_last_layer, lev]

def get_metrics_wordmodels(word1, word2, model):
    lev = norm_levenshtein(word1, word2)
    #act1, act2 = representation(model, [ipa1, ipa2])
    rep1 = representation(model, [[word1]])
    rep2 = representation(model, [[word2]])
    #print stack[0].shape, stack[1].shape
    cosine_sim = cosine_similarity(rep1, rep2)
    return [ cosine_sim, lev ]
