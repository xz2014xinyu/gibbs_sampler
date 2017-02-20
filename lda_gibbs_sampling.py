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

#### #######################################  Gibbs sampler
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




#return('Topic assignments at iteration {}:{}'.format(n,sampled_topics))

















