import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import pdb

dataset = 'xmedia'
K = 10
train_size = 4000
label_dim = 20

def get_feature(feature_string_list, train_size):
	feature_list = []
	for i in range(train_size):
		feature_string = feature_string_list[i].split()
		feature_float_list = []
		for j in range(len(feature_string)):
			feature_float_list.append(float(feature_string[j]))
		feature_list.append(feature_float_list)
		
	return np.asarray(feature_list)
	
def get_distance(vec1, vec2):
	return np.sqrt(np.sum(np.square(vec1 - vec2)))
	
def is_same_cate(strA, strB, label_dim):
	labelA = strA.split()
	labelB = strB.split()	
	for i in range(label_dim):
		if labelA[i] == '1' and labelA[i] == labelB[i]:
			return True
	return False
	
def get_AP(k_nearest, label, query_index, k, label_dim):
	score = 0.0
	for i in range(k):
		if is_same_cate(label[query_index], label[int(k_nearest[i])], label_dim):
			score += 1.0
	return score / k
	
	

dataset_dir = '../' + dataset + '/'
list_dir = dataset_dir + 'list/'
feature_dir = dataset_dir + 'feature/'
result_dir = './' + dataset + '/'

train_img_string_list = open(feature_dir + 'train_img.txt', 'r').read().split('\n')
train_txt_string_list = open(feature_dir + 'train_txt.txt', 'r').read().split('\n')
train_vid_string_list = open(feature_dir + 'train_vid.txt', 'r').read().split('\n')
train_3d_string_list = open(feature_dir + 'train_3d.txt', 'r').read().split('\n')
train_aud_string_list = open(feature_dir + 'train_aud.txt', 'r').read().split('\n')
train_label = open(list_dir + 'train_label.txt', 'r').read().split('\r\n')

train_img = get_feature(train_img_string_list, train_size)
train_txt = get_feature(train_txt_string_list, train_size)
train_3d = get_feature(train_3d_string_list, train_size)
train_vid = get_feature(train_vid_string_list, train_size)
train_aud = get_feature(train_aud_string_list, train_size)

distance_img = pdist(train_img, 'euclidean')
distance_txt = pdist(train_txt, 'euclidean')		
distance_3d = pdist(train_3d, 'euclidean')
distance_vid = pdist(train_vid, 'euclidean')
distance_aud = pdist(train_aud, 'euclidean')

np.save(result_dir + 'distance_img.npy', distance_img)
np.save(result_dir + 'distance_txt.npy', distance_txt)
np.save(result_dir + 'distance_3d.npy', distance_3d)
np.save(result_dir + 'distance_aud.npy', distance_aud)
np.save(result_dir + 'distance_vid.npy', distance_vid)
 
 
# distance_img = np.load(result_dir + 'distance_img.npy')
# distance_txt = np.load(result_dir + 'distance_txt.npy')
# distance_3d = np.load(result_dir + 'distance_3d.npy')
# distance_aud = np.load(result_dir + 'distance_aud.npy')
# distance_vid = np.load(result_dir + 'distance_vid.npy')

distance_img = squareform(distance_img)
distance_txt = squareform(distance_txt)
distance_3d = squareform(distance_3d)
distance_aud = squareform(distance_aud)
distance_vid = squareform(distance_vid)


KNN_img = np.zeros((train_size, K))
KNN_txt = np.zeros((train_size, K))
KNN_aud = np.zeros((train_size, K))
KNN_3d = np.zeros((train_size, K))
KNN_vid = np.zeros((train_size, K))

KNN_cross = np.zeros((train_size, K*5))
accuracy_sum_img = 0
accuracy_sum_txt = 0
accuracy_sum_cross = 0
	
for i in range(train_size):
	k_nearest_img = np.argsort(distance_img[i])[0:K]
	k_nearest_txt = np.argsort(distance_txt[i])[0:K]
	k_nearest_3d  = np.argsort(distance_3d[i])[0:K]
	k_nearest_aud = np.argsort(distance_aud[i])[0:K]
	k_nearest_vid = np.argsort(distance_vid[i])[0:K]
	
	k_nearest_cross = np.zeros((K*5))
	for j in range(K):
		k_nearest_cross[j+K*0] = k_nearest_img[j]
		k_nearest_cross[j+K*1] = k_nearest_txt[j]
		k_nearest_cross[j+K*2] = k_nearest_3d[j]
		k_nearest_cross[j+K*3] = k_nearest_aud[j]
		k_nearest_cross[j+K*4] = k_nearest_vid[j]
		
#	k_nearest_cross = np.unique(k_nearest_cross)
	
#	accuracy_sum_img += get_AP(k_nearest_img, train_label, i, K, label_dim)
#	accuracy_sum_txt += get_AP(k_nearest_txt, train_label, i, K, label_dim)
#	accuracy_sum_cross += get_AP(k_nearest_cross, train_label, i, K*2, label_dim)
	
	KNN_img[i] = k_nearest_img
	KNN_txt[i] = k_nearest_txt
	KNN_3d[i] =  k_nearest_3d
	KNN_aud[i] = k_nearest_aud
	KNN_vid[i] = k_nearest_vid
	
	KNN_cross[i] = k_nearest_cross
	
		
print(accuracy_sum_img / train_size)
print(accuracy_sum_txt / train_size)
print(accuracy_sum_cross / train_size)

np.save(result_dir + 'KNN_img.npy', KNN_img)
np.save(result_dir + 'KNN_txt.npy', KNN_txt)
np.save(result_dir + 'KNN_3d.npy', KNN_3d)
np.save(result_dir + 'KNN_aud.npy', KNN_aud)
np.save(result_dir + 'KNN_vid.npy', KNN_vid)
np.save(result_dir + 'KNN_cross.npy', KNN_cross)

