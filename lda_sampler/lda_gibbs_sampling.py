#! /usr/bin/env python
# alpha is a k dimensional vector
# eta is fixed for all topics. v dimensional vector.
# corpus is a list of arrays, each array represent a document.

#### #######################################  Gibbs sampler
import numpy as np
from log_likelihood import loglikelihood_

def lda_sampling(D,V,num_topics,alpha,eta,num_iterations,corpus,randomstate):
	Nd=[len([x for x in corpus][i]) for i in range(D)] # number of words in each document d
	z0=[]
	for d in range(D):
		np.random.seed(d)
		t=np.random.randint(0,num_topics,np.int(Nd[d])) 
		z0.append(t)
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
	probz=np.zeros(num_topics)
	np.random.seed(randomstate)
	for n in range(1,num_iterations):
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
		L=loglikelihood_(n_zw,n_dz,alpha,eta,D,num_topics)
		log_likelihood.append(L)
		print('The Log-likelihood at iteration {}. is {}. random seed{}'.format(n,L,randomstate))
###paramter estimation
	#theta_hat=np.zeros((D,num_topics))
	#psi_hat=np.zeros((num_topics,V))
	#for i in range(D):
	#	for j in range(num_topics):
	#		theta_hat[i,j]=(n_dz[i,j]+alpha[j])/(Nd[i]-1+sum(alpha))
	#for i in range(num_topics):
	#	for j in range(V):
	#		psi_hat[i,j]=(n_zw[i,j]+eta[j])/(n_z[i]+sum(eta))
############## results
	lda_sampling.topics=z_update
	#lda_sampling.theta_hat=theta_hat
	#lda_sampling.psi_hat=psi_hat
	lda_sampling.log_likelihood=log_likelihood
	lda_sampling.initial_topic=z0

#return('Topic assignments at iteration {}:{}'.format(n,sampled_topics))










