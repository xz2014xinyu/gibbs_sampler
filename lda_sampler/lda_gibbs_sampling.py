#! /usr/bin/env python
# alpha is a known k dimensional vector
# beta is fixed for all topics. v dimensional vector.

#random_seed : the generator used for initial topics z1.....zN, N is total number of words

# doc_term is a sparse matrix. 

#### #######################################  Gibbs sampler
import numpy as np

def lda_sampling(num_topics,alpha,eta,num_iterations,doc_term):
	D=doc_term.shape[0] #number of documents
	V=doc_term.shape[1] #vocabulary size
	Nd=np.sum(doc_term,axis=1) # number of words in each document d
#convert sparse matrix to arrays of words 	
	corpus=[]
	z0=[]
	ii, jj = np.nonzero(doc_term)
	ss = np.array(tuple(doc_term[i, j] for i, j in zip(ii, jj)))
	for d in range(D):
		index=[i for i,x in enumerate(ii) if x==d]
		corpusd=np.repeat(jj[index],ss[index]) 
		corpus.append(corpusd)
		t=np.repeat(0,Nd[d,0])
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
		print('The Log-likelihood at iteration {}.{}'.format(n,L))
## estimate topic proportions theta and word distributions psi
	param_estimate_(n_dz,n_zw,n_z,num_topics,D)
############## results
	lda_sampling.topics=z_update
	lda_sampling.theta_hat=param_estimate_.theta
	lda_sampling.psi_hat=param_estimate_.psi
	lda_sampling.log_likelihood=log_likelihood

#return('Topic assignments at iteration {}:{}'.format(n,sampled_topics))










