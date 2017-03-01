def common_elements_prob(list1,list2):
	list=[]
	for x in list1:
		for y in list2:
			if x[0]==y[0] and x[1]==y[1] and x[2]==y[2]:
				list.append(x)
			elif x[0]==y[1] and x[1]==y[0] and x[2]==y[2]:
				list.append(x)
			elif x[0]==y[2] and x[1]==y[0] and x[2]==y[1]:
				list.append(x)
			elif x[0]==y[1] and x[1]==y[0] and x[2]==y[2]:
				list.append(x)
			elif x[0]==y[0] and x[1]==y[2] and x[2]==y[1]:
				list.append(x)
			elif x[0]==y[2] and x[1]==y[1] and x[2]==y[0]:
				list.append(x)
	if list==[]:
		return 0
	else:
		return(np.vstack({tuple(row) for row in list}))

def common_elements_topics(list1,list2):
	list=[]
	for x in list1:
		if x in list2:
			list.append(x)
	return(list)