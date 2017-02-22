#! /usr/bin/env python
import numpy as np
def param_estimate_(n_dz,n_zw,n_z,num_topics,D):
	theta_hat=np.zeros((D,num_topics))
	psi_hat=np.zeros((num_topics,V))
	for i in range(D):
		for j in range(num_topics):
			theta_hat[i,j]=(n_dz[i,j]+alpha[j])/(Nd[i]-1+sum(alpha))
	for i in range(num_topics):
		for j in range(V):
			psi_hat[i,j]=(n_zw[i,j]+eta[j])/(n_z[i]+sum(eta))
	param_estimate_.theta=theta_hat
	param_estimate_.psi=psi_hat
