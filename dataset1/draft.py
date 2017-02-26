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

#np.save('/Users/zhangxinyu/Desktop/gibbs_sampler/doc_term',doc_term)
doc_term=np.load('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset1/doc_term.npy')[()]
corpus=sparse_to_arrays(doc_term)

num_topics = 3  # number of topics
V = 10 # Vocabulary size V
D = 5  # Document size D
alpha = [0.001]*num_topics # Dirichlet prior for topic proportions
#xi = 30 # Poisson prior for 'number of words per document'
eta = [0.01]*V # Dirichlet prior for topic-word distribution
n=2000


L=[]
sr=range(0,10000,200) # 50 different seeds
#for s in sr:
#	lda_sampling(D,V,num_topics,alpha,eta,n,corpus,s)
#	cl=lda_sampling.log_likelihood
#	L.append(cl)


#df=pd.DataFrame(L)
#df.to_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/loglikelihoods_over50seeds.txt',header=None)


df=pd.DataFrame.from_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset1/loglikelihoods_over50seeds.txt',header=None)
for i in range(len(sr)):
	a=df.loc[i]
	L.append(a)

logL=[]
for i in range(len(L)):
	w=np.array(L[i]).tolist()
	logL.append(w)

L=logL

## loglikelihoods plots
a=np.arange(1,n,1).tolist()
for i in range(len(sr)):
	plt.plot(a,[y for y in L][i])

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods over 50 different seeds")
plt.show()

##hist
p=[]
for i in range(len(sr)):
	o=L[i][n-2]
	p.append(o)

plt.hist(p)
plt.title("histogram of final loglikelihoods over 50 different seeds")
plt.show()


### two limits -> two sets
index1=[e for e,x in enumerate(p) if -20<=x]
index2=[e for e,x in enumerate(p) if x<=(-50)]
seedset1=[sr[i] for i in index1]
seedset2=[sr[j] for j in index2]


## set1 subplot
a=np.arange(1,n,1).tolist()
Lset1=[L[i] for i in index1]
for i in range(len(index1)):
	plt.plot(a[200:600],[y for y in Lset1][i][200:600])
plt.xlabel("Iteration");plt.ylabel("Log Likelihoods set1")
plt.show()

#first one
k=259
L1=[x for x in L if x[k-1]<=(-90)][0]
seed1=sr[L.index(L1)]
plt.plot(a[k-3:k+3],L1[k-3:k+3],color='green')
plt.xlabel("Iteration");plt.ylabel("Log Likelihoods set1 (seed200)")
plt.show()




##second
k=337
L2=[x for x in L if x[k-1]<=(-78)][0]
seed2=sr[L.index(L2)]
plt.plot(a[k-4:k+3],L2[k-4:k+3],color='blue')
plt.xlabel("Iteration");plt.ylabel("Log Likelihoods set1 (seed0)")
plt.show()


QQ.append(z_update[])
PP.append(probz)

## third
k=376
L3=[x for x in L if x[k-1]<=(-79)][0]
seed3=sr[L.index(L3)]
plt.plot(a[k-3:k+3],L3[k-3:k+3],color='blue')
plt.xlabel("Iteration");plt.ylabel("Log Likelihoods set1 (seed1400)")
plt.show()




## set2 subplot seed2=2200
a=np.arange(1,3000,1).tolist()
lda_sampling(num_topics,alpha,eta,3000,corpus,seedset2[0])
L4=lda_sampling.log_likelihood
plt.plot(a,L4)
plt.show()


k=2211
seed3=seedset2[0]
plt.plot(a[k-3:k+3],L4[k-3:k+3],color='blue')
plt.xlabel("Iteration");plt.ylabel("Log Likelihoods set2 (seed2200)")
plt.show()



z0=lda_sampling.initial_topic  # initial topics


























