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
alpha=[0.001]*num_topics #true prior 0.1....
xi=100
eta = [0.01]*V # Dirichlet prior for topic-word distribution
n=10000


L=[]
sr=range(0,20000,200) # 100 different seeds
#for s in sr:
#	lda_sampling(D,V,num_topics,alpha,eta,1000,corpus,s)
#	cl=lda_sampling.log_likelihood
#	L.append(cl)


df=pd.DataFrame.from_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset2/100seeds_1000iter.txt',header=None)
for i in range(len(sr)):
	a=df.loc[i]
	L.append(a)

logL=[]
for i in range(len(L)):
	w=np.array(L[i]).tolist()
	logL.append(w)

L=logL

##plot

a=np.arange(1,1000,1).tolist()
for i in range(len(sr)):
	plt.plot(a,[y for y in L][i],linewidth=1.0)

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods over 100 different seeds")
plt.show()

a=np.arange(1,1000,1).tolist()
for i in range(len(sr)):
	plt.plot(a[800:1000],[y for y in L][i][800:1000],linewidth=1.0)

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods over 100 different seeds")
plt.show()


p=[]
for i in range(len(sr)):
	o=L[i][998]
	p.append(o)

plt.hist(p)
plt.title("histogram of final loglikelihoods over 100 different seeds")
plt.show()

##2 sets

index1=[e for e,x in enumerate(p) if (-520)<x]
index2=[e for e,x in enumerate(p) if x<=(-520)]
seedset1=[sr[i] for i in index1]
seedset2=[sr[j] for j in index2]

### for seedset2: [0, 200, 400, 2600, 2800, 3800, 4600, 5800, 7800, 12200, 12800, 13000, 15800]
#for s in seedset2:
#	lda_sampling(D,V,num_topics,alpha,eta,2000,corpus,s)
#	cl=lda_sampling.log_likelihood
#	L.append(cl)
	
#df=pd.DataFrame(L)
#df.to_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset2/seedset2_2000iter.txt',header=None)
L=[]
df=pd.DataFrame.from_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset2/seedset2_2000iter.txt',header=None)
for i in range(len(seedset2)):
	a=df.loc[i]
	L.append(a)


logL=[]
for i in range(len(L)):
	w=np.array(L[i]).tolist()
	logL.append(w)

L=logL

##plot
a=np.arange(1,2000,1).tolist()
for i in range(len(seedset2)):
	plt.plot(a,[y for y in L][i],linewidth=1.0)

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods seedset2")
plt.show()

a=np.arange(1,2000,1).tolist()
for i in range(len(seedset2)):
	plt.plot(a[1500:2000],[y for y in L][i][1500:2000],linewidth=1.0)

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods seedset2")
plt.show()


p=[]
for i in range(len(seedset2)):
	o=L[i][1998]
	p.append(o)

plt.hist(p)
plt.title("histogram of final loglikelihoods seedset2")
plt.show()


### 2 sets again

index1=[e for e,x in enumerate(p) if (-560)<x]
index2=[e for e,x in enumerate(p) if x<=(-560)]
seedset1=[seedset2[i] for i in index1]
seedset2=[seedset2[j] for j in index2]




a=np.arange(1,2000,1).tolist()
Lset1=[L[i] for i in index1]

for i in range(len(index1)):
	plt.plot(a,[y for y in Lset1][i])

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods set1")
plt.show()


### 1st     NOOOOOO
L1=[x for x in Lset1 if x[360]<=(-650)][0]
seed1=seedset1[Lset1.index(L1)]

plt.plot(a,L1,color='red')
plt.xlabel("Iteration");plt.ylabel("Log Likelihoods set2 (seed2600)")
plt.show()



lda_sampling(D,V,num_topics,alpha,eta,1600,corpus,seed1)
l1=lda_sampling.log_likelihood
p1=lda_sampling.probs
plt.plot(l1)



## 2nd
L2=[x for x in Lset1 if x[320]<=(-570) and (-630)<=x[320]][0]
seed2=seedset1[Lset1.index(L2)]

plt.plot(a,L2,color='green')
plt.xlabel("Iteration");plt.ylabel("Log Likelihoods set2 (seed200)")
plt.show()

lda_sampling(D,V,num_topics,alpha,eta,1340,corpus,seed2)
l2=lda_sampling.log_likelihood
p2=lda_sampling.probs
plt.plot(l1)

#### set2

L=[]
### for seedset2: [400, 2800, 3800, 5800, 12200, 12800]
#for s in seedset2:
#	lda_sampling(D,V,num_topics,alpha,eta,3000,corpus,s)
#	cl=lda_sampling.log_likelihood
#	L.append(cl)
	
#df=pd.DataFrame(L)
#df.to_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset2/seedset2_2000iter.txt',header=None)


L=[]
df=pd.DataFrame.from_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset2/seedset2_3000iter.txt',header=None)
for i in range(len(seedset2)):
	a=df.loc[i]
	L.append(a)


logL=[]
for i in range(len(L)):
	w=np.array(L[i]).tolist()
	logL.append(w)

L=logL

##plot
a=np.arange(1,3000,1).tolist()
for i in range(len(seedset2)):
	plt.plot(a,[y for y in L][i],linewidth=1.0)

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods seedset2")
plt.show()


for i in range(len(seedset2)):
	plt.plot(a[2500:3000],[y for y in L][i][2500:3000],linewidth=1.0)

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods seedset2")
plt.show()

p=[]
for i in range(len(seedset2)):
	o=L[i][2998]
	p.append(o)

plt.hist(p)
plt.title("histogram of final loglikelihoods seedset2")
plt.show()

## 2 sets
index1=[e for e,x in enumerate(p) if (-540)<x]
index2=[e for e,x in enumerate(p) if x<=(-540)]
seedset1=[seedset2[i] for i in index1]
seedset2=[seedset2[j] for j in index2]

L=[]
### for seedset2: [2800, 3800, 5800, 12200, 12800]
#for s in seedset2:
#	lda_sampling(D,V,num_topics,alpha,eta,5000,corpus,s)
#	cl=lda_sampling.log_likelihood
#	L.append(cl)
	
#df=pd.DataFrame(L)
#df.to_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset2/seedset2_5000iter.txt',header=None)

L=[]
df=pd.DataFrame.from_csv('/Users/zhangxinyu/Desktop/gibbs_sampler/dataset2/seedset2_5000iter.txt',header=None)
for i in range(len(seedset2)):
	a=df.loc[i]
	L.append(a)


logL=[]
for i in range(len(L)):
	w=np.array(L[i]).tolist()
	logL.append(w)

L=logL

a=np.arange(1,5000,1).tolist()
for i in range(len(seedset2)):
	plt.plot(a,[y for y in L][i],linewidth=1.0)

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods seedset2")
plt.show()


for i in range(len(seedset2)):
	plt.plot(a[4500:5000],[y for y in L][i][4500:5000],linewidth=1.0)

plt.xlabel("Iteration");plt.ylabel("Log Likelihoods seedset2")
plt.show()







