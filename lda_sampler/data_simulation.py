#! /usr/bin/env python
import numpy as np
from scipy import sparse
import random
def G_P(k,V,D,alpha,xi,eta):
	psi = []    
	theta = []  
	Nd = [] 
	doc_term = sparse.lil_matrix((D,V))
	for i in range(k):
		rows = np.random.dirichlet(eta,size = 1)
		psi.append(rows)
	for d in range(D):
		s=[1.0*random.gammavariate(a,1) for a in alpha]
		if max(s)==0:
			sample_theta=alpha
		else:
			sample_theta=[1.0*v/sum(s) for v in s]
		np.random.seed(0)
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
		doc_term[d,:] = sum(W)[0]
		theta.append(sample_theta)
		Nd.append(sample_N)
	return(doc_term)







