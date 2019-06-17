import numpy as np
import pdb
import os

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def count_map(test,data,test_lab,data_lab):
	qlen = len(test)
	dlen = len(data)

	dist = np.zeros(dlen)
	res = np.zeros(qlen)

	for i in range(qlen):
		#print i
		for  j in range(dlen):
			#pdb.set_trace()
			dist[j] = sum(test[i]^data[j])
		idx = np.argsort(dist)
		ton = 0
		for k in range(dlen):
			if sum(data_lab[idx[k]]^test_lab[i])==0:
				ton = ton+1
				res[i] += ton/(k+1.0)
		res[i] = res[i]/ton

	return np.mean(res)

def MAP_ARGV(sess, discriminator, test_feature, database_feature, test_label, database_label, dim):

	test_feature = np.asarray(test_feature)
	database_feature = np.asarray(database_feature)
	#pdb.set_trace()
	image_hash_test = sess.run(discriminator.image_hash,
			 feed_dict={discriminator.I_data: np.asarray(test_feature[0]),
						discriminator.T_data: np.asarray(test_feature[1]),
						discriminator.A_data: np.asarray(test_feature[2]),
						discriminator.V_data: np.asarray(test_feature[3]),
						discriminator.D_data: np.asarray(test_feature[4])},)

	image_hash_dataset = sess.run(discriminator.image_hash,
			 feed_dict={discriminator.I_data: np.asarray(database_feature[0]),
						discriminator.T_data: np.asarray(database_feature[1]),
						discriminator.A_data: np.asarray(database_feature[2]),
						discriminator.V_data: np.asarray(database_feature[3]),
						discriminator.D_data: np.asarray(database_feature[4])},)

	data = np.concatenate((image_hash_dataset['I'],image_hash_dataset['T'],
		image_hash_dataset['A'],image_hash_dataset['V'],image_hash_dataset['D'],))
	data_lab = np.concatenate((database_label[0],database_label[1],database_label[2],database_label[3],database_label[4])).astype(int)

#	test = np.concatenate((image_hash_test['I'],image_hash_test['T'],
#		image_hash_test['A'],image_hash_test['V'],image_hash_test['D'],))
#	test_lab = np.concatenate((test_label[0],test_label[1],test_label[2],test_label[3],test_label[4])).astype(int)

	apI = count_map(np.asarray(image_hash_test['I']),data,np.asarray(test_label[0]).astype(int),data_lab)
	apT = count_map(np.asarray(image_hash_test['T']),data,np.asarray(test_label[1]).astype(int),data_lab)
	apA = count_map(np.asarray(image_hash_test['A']),data,np.asarray(test_label[2]).astype(int),data_lab)
	apV = count_map(np.asarray(image_hash_test['V']),data,np.asarray(test_label[3]).astype(int),data_lab)
	apD = count_map(np.asarray(image_hash_test['D']),data,np.asarray(test_label[4]).astype(int),data_lab)

	filename = 'result/test_map_' + str(dim) + '.txt'
	fp = open(filename,"a")
	fp.write(str([apI,apT,apA,apV,apD])+"\n")
	fp.close()


	return np.mean([apI,apT,apA,apV,apD])
