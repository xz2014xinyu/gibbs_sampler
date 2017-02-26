#! /usr/bin/env python
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


np.random.seed(1)
lda_sampling(k,alpha,eta,n,doc_term)
cl1=lda_sampling.complete_loglikelihood
topics1=lda_sampling.topics


np.random.seed(1000)
lda_sampling(k,alpha,eta,n,doc_term)
cl2=lda_sampling.complete_loglikelihood
topics2=lda_sampling.topics

pn=np.arange(0,n,1)
plt.style.use("bmh")
plt.plot(pn,cl1,pn,cl2)
plt.xlabel("Iteration");plt.ylabel("complete Log Likelihoods")
plt.show()

plt.plot(pn[0:10],cl1[0:10],pn[0:10],cl2[0:10])
plt.show()







df=pd.DataFrame(cl1,cl2)
df.to_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/two_seeds_data.txt',header=None)



## initial topics:
for i in range(D):
	print('(seed1) topics at 1st iteration for document {}:{}'.format(i,topics1[1][i]))

for i in range(D):
	print('(seed2) topics at 1st iteration for document {}:{}'.format(i,topics2[1][i]))

# seed1
Nd=np.sum(doc_term,axis=1) # number of words in each document d
corpus=[] 
z0=[] #initial topics
ii, jj = np.nonzero(doc_term)
ss = np.array(tuple(doc_term[i, j] for i, j in zip(ii, jj)))

np.random.seed(1)
for d in range(D):
	index=[i for i,x in enumerate(ii) if x==d]
	corpusd=np.repeat(jj[index],ss[index]) 
	corpus.append(corpusd)
	t=np.random.randint(0,k,np.int(Nd[d])) 
	z0.append(t)

n_zw1=np.zeros((k,V))
for j in range(k):
	for i in range(V):
		cc=[]
		for d in range(D):
			index=[a for a,x in enumerate(topics1[1][d]) if x==j]
			cnt=len([b for b in corpus[d][index] if b==i])
			cc.append(cnt)
		n_zw1[j,i]=sum(cc)

n_dz1=np.zeros((D,k))
for i in range(D):
	for j in range(k):
		cc=len([a for a,x in enumerate(topics1[1][i]) if x==j])
		n_dz1[i,j]=cc

n_z1=np.zeros(k)
for j in range(k):
	cc=[]
	for d in range(D):
		c=len([a for a,x in enumerate(topics1[1][d]) if x==j])
		cc.append(c)
	n_z1[j]=sum(cc)

n_zw2=np.zeros((k,V))
for j in range(k):
	for i in range(V):
		cc=[]
		for d in range(D):
			index=[a for a,x in enumerate(topics2[1][d]) if x==j]
			cnt=len([b for b in corpus[d][index] if b==i])
			cc.append(cnt)
		n_zw2[j,i]=sum(cc)

n_dz2=np.zeros((D,k))
for i in range(D):
	for j in range(k):
		cc=len([a for a,x in enumerate(topics2[1][i]) if x==j])
		n_dz2[i,j]=cc

n_z2=np.zeros(k)
for j in range(k):
	cc=[]
	for d in range(D):
		c=len([a for a,x in enumerate(topics2[1][d]) if x==j])
		cc.append(c)
	n_z2[j]=sum(cc)

print '(seed1) topic-word; document-topic; topics at 1st iteration:';  n_zw1; n_dz1; n_z1
print '(seed2) topic-word; document-topic; topics at 1st iteration:';  n_zw2; n_dz2; n_z2








