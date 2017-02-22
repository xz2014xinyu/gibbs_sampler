#! /usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import math
import random
sys.path.append('/Users/zhangxinyu/Desktop/gibbs_sampler')
import lda_sampler
from lda_sampler import*


num_topics = 3  # number of topics
V = 10 # Vocabulary size V
D = 5  # Document size D
alpha = [0.01]*num_topics # Dirichlet prior for topic proportions
xi = 30 # Poisson prior for 'number of words per document'
eta = [0.01]*V # Dirichlet prior for topic-word distribution
n=50



doc_term=G_P(num_topics,V,D,alpha,xi,eta)
doc_term=doc_term.astype(np.int32)

L=[]
for s in range(0,10000,100):
	np.random.seed(s)
	lda_sampling(num_topics,alpha,eta,n,doc_term)
	cl=lda_sampling.log_likelihood
	L.append(cl)

a=np.arange(0,50,1)
for i in range(100):
	plt.plot(a.tolist(),[y for y in L][i], label='id %s'%i)

plt.show()

