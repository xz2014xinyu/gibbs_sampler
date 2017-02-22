#! /usr/bin/env python
import scipy.special as sp
def loglikelihood_(n_zw,n_dz,alpha,eta,D,num_topics):
	l1=[]
	l2=[]
	for i in range(num_topics):
		L=sum(sp.gammaln(n_zw[i,:]+eta))-sp.gammaln(sum(n_zw[i,:])+sum(eta))
		l1.append(L)
	for j in range(D):
		L=sum(sp.gammaln(n_dz[j,:]+alpha))-sp.gammaln(sum(n_dz[j,:])+sum(alpha))
		l2.append(L)
### complete loglikelihood log P(w,z)= log P(w|z) + log P(z)
	L=sum(l1)+sum(l2)-num_topics*(sum(sp.gammaln(eta))-sp.gammaln(sum(eta)))-D*(sum(sp.gammaln(alpha))-sp.gammaln(sum(alpha)))
	return(L)
