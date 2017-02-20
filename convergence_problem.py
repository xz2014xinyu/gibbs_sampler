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
V = 10 # Vocabulary size V
D = 5  # Document size D
alpha = [0.001]*k # Dirichlet prior for topic proportions
xi = 30 # Poisson prior for 'number of words per document'
eta = [0.01]*V # Dirichlet prior for topic-word distribution
n = 500


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
		s=[1.0*random.gammavariate(a,1) for a in alpha]
		if max(s)==0:
			sample_theta=alpha
		else:
			sample_theta=[1.0*v/sum(s) for v in s]
		sample_N = np.random.poisson(xi,size = 1)[0]
		t=[]
		W=[]
		for n in range(sample_N):
			z = np.random.multinomial(1, pvals = sample_theta,size=1)
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
############ gibbs
def lda_sampling(num_topics,alpha,eta,num_iterations,doc_term):
	D=doc_term.shape[0] #number of documents
	V=doc_term.shape[1] #vocabulary size
	Nd=np.sum(doc_term,axis=1) # number of words in each document d
	N=doc_term.sum() # total number of words
#convert sparse matrix to arrays of words 	
	corpus=[] 
	z0=[] #initial topics
	ii, jj = np.nonzero(doc_term)
	ss = np.array(tuple(doc_term[i, j] for i, j in zip(ii, jj)))
	for d in range(D):
		index=[i for i,x in enumerate(ii) if x==d]
		corpusd=np.repeat(jj[index],ss[index]) 
		corpus.append(corpusd)
		topics=np.random.randint(0,num_topics,np.int(Nd[d])) 
		z0.append(topics)
	z_update=[]
	z_update.append(z0)
# word count of each topic z and vocabulary w
	n_zw=np.zeros((num_topics,V))
	for j in range(num_topics):
		for i in range(V):
			cc=[]
			for d in range(D):
				index=[a for a,x in enumerate(z_update[0][d]) if x==j]
				cnt=len([b for b in corpus[d][index] if b==i])
				cc.append(cnt)
			n_zw[j,i]=sum(cc)
# word count in each document d and topic z
		n_dz=np.zeros((D,num_topics))
		for i in range(D):
			for j in range(num_topics):
				cc=len([a for a,x in enumerate(z_update[0][i]) if x==j])
				n_dz[i,j]=cc
# word count of topic z in all documents
		n_z=np.zeros(num_topics)
		for j in range(num_topics):
			cc=[]
			for d in range(D):
				c=len([a for a,x in enumerate(z_update[0][d]) if x==j])
				cc.append(c)
			n_z[j]=sum(cc)
## Sampling from full conditional posterior P(z| Z\z, w)
	log_likelihood=[]
	complete_loglikelihood=[]
	for n in range(1,num_iterations+1):
		sampled_topics1={}
		sampled_topics=[]
		for d in range(D):
			newz=np.zeros(np.int(Nd[d]))
			for i in range(Nd[d]):
				# discount for the topic z^d_i  we will sample
				n_zw[z_update[n-1][d][i],corpus[d][i]] -= 1
				n_dz[d,z_update[n-1][d][i]] -= 1
				n_z[z_update[n-1][d][i]] -=1
				# sample a new topic from multinomial and store it as 'newtopic'
				probz=np.zeros(num_topics)
				for j in range(num_topics):
						probz[j]=(n_zw[j,corpus[d][i]]+eta[corpus[d][i]])*(n_dz[d,j]+alpha[j])/(n_z[j]+sum(eta))
				probz=probz/sum(probz)
				newtopic=np.random.choice(np.arange(0,num_topics),p=probz)
				newz[i]=newtopic
				# add count for the new topic we have sampled
				n_zw[newtopic,corpus[d][i]] += 1
				n_dz[d,newtopic] += 1
				n_z[newtopic] += 1
			sampled_topics.append(newz)
		z_update.append(sampled_topics)
## calculate loglikelihood log p(w|z) and complete loglikelihood log P(w,z)
		L=[]
		cL=[]
		for i in range(num_topics):
			ll=[]
			for j in range(V):
				l=sp.gammaln(n_zw[i,j]+eta[j])-sp.gammaln(eta[j])
				ll.append(l)
			lll=sum(ll)-sp.gammaln(sum(n_zw[i,:])+sum(eta))+sp.gammaln(sum(eta))
			L.append(lll)
		L=sum(L)
		log_likelihood.append(L)
		for i in range(D):
			ll=[]
			for j in range(num_topics):
				l=sp.gammaln(alpha[j]+n_dz[i,j])-sp.gammaln(alpha[j])
				ll.append(l)
			lll=sum(ll)-sp.gammaln(sum(alpha)+sum(n_dz[i,:]))+sp.gammaln(sum(alpha))
			cL.append(lll)
		cL=sum(cL)+L
		complete_loglikelihood.append(cL)
## estimate topic proportions theta and word distributions psi
	theta_hat=np.zeros((D,num_topics))
	psi_hat=np.zeros((num_topics,V))
	for i in range(D):
		for j in range(num_topics):
			theta_hat[i,j]=(n_dz[i,j]+alpha[j])/(Nd[i]-1+sum(alpha))
	for i in range(num_topics):
		for j in range(V):
			psi_hat[i,j]=(n_zw[i,j]+eta[j])/(n_z[i]+sum(eta))
############## results
	lda_sampling.alpha=alpha
	lda_sampling.eta=eta
	lda_sampling.topics=z_update
	lda_sampling.theta_hat=theta_hat
	lda_sampling.psi_hat=psi_hat
	lda_sampling.log_likelihood=log_likelihood
	lda_sampling.complete_loglikelihood=complete_loglikelihood
	return(cL)

####### two random seeds

np.random.seed(1)
lda_sampling(k,alpha,eta,n,doc_term)
cl1=lda_sampling.complete_loglikelihood
topics1=lda_sampling.topics


np.random.seed(1000)
lda_sampling(k,alpha,eta,n,doc_term)
cl2=lda_sampling.complete_loglikelihood
topics2=lda_sampling.topics

pn=np.arange(0,500,1)
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
	print('seed1 document {}:{}'.format(i,topics1[0][i]))
for i in range(D):
	print('seed2 document {}:{}'.format(i,topics2[0][i]))

# seed1
D=doc_term.shape[0] #number of documents
V=doc_term.shape[1] #vocabulary size
Nd=np.sum(doc_term,axis=1) # number of words in each document d
N=doc_term.sum() # total number of words
corpus=[] 
z0=[] #initial topics
ii, jj = np.nonzero(doc_term)
ss = np.array(tuple(doc_term[i, j] for i, j in zip(ii, jj)))

np.random.seed(1)

for d in range(D):
	index=[i for i,x in enumerate(ii) if x==d]
	corpusd=np.repeat(jj[index],ss[index]) 
	corpus.append(corpusd)
	topics=np.random.randint(0,k,np.int(Nd[d])) 
	z0.append(topics)

z_update=[]
z_update.append(z0)

n_zw=np.zeros((k,V))

for j in range(k):
	for i in range(V):
		cc=[]
		for d in range(D):
			index=[a for a,x in enumerate(z_update[0][d]) if x==j]
			cnt=len([b for b in corpus[d][index] if b==i])
			cc.append(cnt)
		n_zw[j,i]=sum(cc)

n_dz=np.zeros((D,k))
for i in range(D):
	for j in range(k):
		cc=len([a for a,x in enumerate(z_update[0][i]) if x==j])
		n_dz[i,j]=cc

n_z=np.zeros(k)
for j in range(k):
	cc=[]
	for d in range(D):
		c=len([a for a,x in enumerate(z_update[0][d]) if x==j])
		cc.append(c)
	n_z[j]=sum(cc)







