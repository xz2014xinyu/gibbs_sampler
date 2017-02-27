#! /usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import math
import random
import lda
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
n=10000


L=[]
sr=range(0,20000,200) # 100 different seeds
#for s in sr:
#	np.random.seed(seed=s)
#	m1=lda.LDA(n_topics=num_topics,alpha=0.001,eta=0.01,n_iter=10000,refresh=10)
#	m1.fit(doc_term)
#	L.append(m1.loglikelihoods_)



#df=pd.DataFrame(L)
#df.to_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset2/loglikelihoods_over100seeds.txt',header=None)

df=pd.DataFrame.from_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset2/loglikelihoods_over100seeds.txt',header=None)
for i in range(len(sr)):
	a=df.loc[i]
	L.append(a)


logL=[]
for i in range(len(L)):
	w=np.array(L[i]).tolist()
	logL.append(w)

L=logL


a=np.arange(0,n,10).tolist()
for i in range(len(sr)):
	plt.plot(a,[y for y in L][i],linewidth=1.0)

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods over 100 different seeds")
plt.show()



p=[]
for i in range(len(sr)):
	o=L[i][999]
	p.append(o)

plt.hist(p)
plt.title("histogram of final loglikelihoods over 100 different seeds")
plt.show()


#### 3 sets
index1=[e for e,x in enumerate(p) if -550<=x]
index2=[e for e,x in enumerate(p) if x<=(-550) and (-580)<=x]
index3=[e for e,x in enumerate(p) if x<=(-600)]
seedset1=[sr[i] for i in index1]
seedset2=[sr[j] for j in index2]
seedset3=[sr[j] for j in index3]

## set1 subplot
a=np.arange(0,n,10).tolist()
Lset1=[L[i] for i in index1]

for i in range(len(index1)):
	plt.plot(a,[y for y in Lset1][i])

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods set1")
plt.show()



### 1st
L1=[x for x in Lset1 if x[900]<=(-550)][0]
seed1=seedset1[Lset1.index(L1)]

plt.plot(a,L1,color='green')
plt.xlabel("Iteration");plt.ylabel("Log Likelihoods set2 (seed8200)")
plt.show()



lda_sampling(D,V,num_topics,alpha,eta,2000,corpus,seed1)
l1=lda_sampling.log_likelihood
p1=lda_sampling.probs
plt.plot(l1)



## 2nd
L2=[x for x in L if x[168]<=(-650)][0]

seed2=sr[L.index(L2)]

plt.plot(a,L2,color='blue')
plt.xlabel("Iteration");plt.ylabel("Log Likelihoods set2 (seed8400)")
plt.show()


lda_sampling(D,V,num_topics,alpha,eta,3500,corpus,8500)
l2=lda_sampling.log_likelihood
p2=lda_sampling.probs



## set2 subplot
a=np.arange(0,n,10).tolist()
Lset2=[L[i] for i in index2]

for i in range(len(index2)):
	plt.plot(a,[y for y in Lset2][i])

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods set2")
plt.show()


lda_sampling(D,V,num_topics,alpha,eta,10000,corpus,17800)
l3=lda_sampling.log_likelihood
p3=lda_sampling.probs









#####################################
##################################




np.random.seed(seed=6200)
m1=lda.LDA(n_topics=num_topics,alpha=0.001,eta=0.01,n_iter=10000,refresh=10)
m1.fit(doc_term)
l1=m1.loglikelihoods_
plt.plot(l1)
plt.show()








