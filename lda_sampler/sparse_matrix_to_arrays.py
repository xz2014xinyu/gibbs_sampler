#! /usr/bin/env python
# this function convert sparse matrix to a list of arrays.
import numpy as np
def sparse_to_arrays(doc_term):
	corpus=[]
	D=doc_term.shape[0]
	ii, jj = np.nonzero(doc_term)
	ss = np.array(tuple(doc_term[i, j] for i, j in zip(ii, jj)))
	for d in range(D):
		index=[i for i,x in enumerate(ii) if x==d]
		corpusd=np.repeat(jj[index],ss[index])
		corpus.append(corpusd)
	return(corpus)