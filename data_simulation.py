#! /usr/bin/env python
# alpha is a known k dimensional vector
# beta is fixed for all topics. v dimensional vector.

#random_seed : the generator used for initial topics z1.....zN, N is total number of words

# doc_term is a sparse matrix. 

import sys
import numpy as np
from scipy import sparse
import lda
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gensim
import pandas as pd
import scipy.special as sp
import math
import random

#########################################################  Simulate a corpus
k = 3  # number of topics
V = 30 # Vocabulary size V
D = 5  # Document size D
alpha = [0.01]*k # Dirichlet prior for topic proportions
xi = 10 # Poisson prior for 'number of words per document'
eta = [0.01]*V # Dirichlet prior for topic-word distribution
n = 50


psi = []    # topic-word probability
theta = []  # document-topic probability
Nd = []  # number of words in each document
Z = []  # pairs of (topic,word) Generated in D documents


doc_term = sparse.lil_matrix((D,V))


def G_P(k,D,alpha,xi,eta):

    for i in range(k):
        rows = np.random.dirichlet(eta,size = 1)
        psi.append(rows)

    for d in range(D):
        # topic proportions
        s=[1.0*random.gammavariate(a,1) for a in alpha]
        if max(s)==0:
            sample_theta=alpha
        else:
            sample_theta=[1.0*v/sum(s) for v in s]
        # number of words in a document
        sample_N = np.random.poisson(xi,size = 1)[0]

        t=[]
        W=[]
        for n in range(sample_N):
            
            # sample a topic z
            z = np.random.multinomial(1, pvals = sample_theta,size=1)
            # sample a word given topic z
            topic_index = np.nonzero(z)[1][0]
            word = np.random.multinomial(1,pvals = psi[topic_index][0],size=1)
            word_index = np.nonzero(word)[1][0]

            W.append(word)
            t.append((topic_index,word_index))

        Z.append(t)
        doc_term[d,:] = sum(W)[0]
        theta.append(sample_theta)
        Nd.append(sample_N)

G_P(k,D,alpha,xi,eta)
doc_term=doc_term.astype(np.int32)
