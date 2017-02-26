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


#doc_term=G_P(num_topics,V,D,alpha,xi,eta)
#doc_term=doc_term.astype(np.int32)

#np.save('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset2/doc_term',doc_term)


doc_term=np.load('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset2/doc_term.npy')[()]
corpus=sparse_to_arrays(doc_term)

num_topics=3
D=10
V=20 
alpha=[0.001]*num_topics #true prio 0.1....
xi=100
eta = [0.01]*V # Dirichlet prior for topic-word distribution
n=2000


L=[]
sr=range(0,10000,200) # 50 different seeds
for s in sr:
	lda_sampling(D,V,num_topics,alpha,eta,n,corpus,s)
	cl=lda_sampling.log_likelihood
	L.append(cl)

df=pd.DataFrame(L)
df.to_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset2/loglikelihoods_over50seeds.txt',header=None)