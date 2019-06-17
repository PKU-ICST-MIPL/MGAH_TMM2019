import os, random, pdb, time
import tensorflow as tf
import numpy as np
import utils as ut
from sys import argv
from map_argv import *
from dis_model_nn import DIS
from gen_model_nn import GEN
from tensorflow.python.client import device_lib

script, hashdim, gpuid = argv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpuid


print(device_lib.list_local_devices())

OUTPUT_DIM = int(hashdim)
Kx = 10
SELECTNUM = 1
SAMPLERATIO = 10

WHOLE_EPOCH = 30
D_EPOCH = 1000
G_EPOCH = 2
GS_EPOCH = 30
D_DISPLAY = 100
G_DISPLAY = 10

I_DIM = 4096
T_DIM = 3000
A_DIM = 78
V_DIM = 4096
D_DIM = 4700

HIDDEN_DIM = 4096
CLASS_DIM = 20
MODAL_NUM = 5
TRAIN_NUM = 4000
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0005
D_LEARNING_RATE = 0.01
LAMBDA = 0
BETA = 1.0
GAMMA = 0.1

WORKDIR = '../xmedia/'
DIS_MODEL_BEST_FILE = './model/dis_best_nn_' + str(OUTPUT_DIM) + '.model'
DIS_MODEL_NEWEST_FILE = './model/dis_newest_nn_' + str(OUTPUT_DIM) + '.model'

#test_feature,database_feature,test_label,database_label = ut.load_all_query_url(WORKDIR + 'feature/',WORKDIR + 'list/', CLASS_DIM)
test_feature,database_feature,test_label,database_label = ut.load_all_query_url(WORKDIR + 'feature_znorm/',WORKDIR + 'list/', CLASS_DIM)

# pdb.set_trace()

train_feature = ut.load_train_feature(WORKDIR + 'feature_znorm/')
# label_dict = ut.load_all_label(WORKDIR + 'list/')
knn_idx = ut.load_knn(WORKDIR + 'python/xmedia/')

#
record_file_name = 'record_' + str(OUTPUT_DIM) + '.txt',
#record_file = open('record_' + str(OUTPUT_DIM) + '.txt', 'w')
#record_file.close()



def generate_samples(fix):
	data = []
	#pdb.set_trace()
	for i in range(TRAIN_NUM):
		item = []
		for j in range(MODAL_NUM):
			if j==fix:
				#print j,i
				item.append(train_feature[j][i])
			else:
				t_idx = random.randint(0,Kx-1)
				item.append(train_feature[j][knn_idx[j][i][t_idx]])

		for j in range(MODAL_NUM):
			if j==fix:
				item.append(train_feature[j][i])
			else:
				k = random.randint(0, TRAIN_NUM-1)
				while k in knn_idx[j][i]:
					k = random.randint(0,TRAIN_NUM-1)
				item.append(train_feature[j][k])
		data.append(item)

	random.shuffle(data)
	return data

def train_discriminator(sess, discriminator, dis_train_list, tag = 'I'):
	train_size = len(dis_train_list)
	index = 1
	#print train_size
	while index < train_size:
		input_data = []
		if index + BATCH_SIZE <= train_size:
			for qx in range(10):
				qua = []
				for i in range(index, index + BATCH_SIZE):
				#	print i,qx
					qua.append(dis_train_list[i][qx])
				input_data.append(np.asarray(qua))
		else:
			for i in range(index, train_size):
				for qx in range(10):
					qua = []
					for i in range(index, train_size):
					#	print i,qx
						qua.append(dis_train_list[i][qx])
					input_data.append(np.asarray(qua))

		index += BATCH_SIZE

#		input_data = np.asarray(input_data)

		# pdb.set_trace()
		dict = {discriminator.I_data:input_data[0],
		       discriminator.T_data: input_data[1],
		       discriminator.A_data: input_data[2],
		       discriminator.V_data: input_data[3],
		       discriminator.D_data: input_data[4],
		       discriminator.I_neg_data: input_data[5],
		       discriminator.T_neg_data: input_data[6],
		       discriminator.A_neg_data: input_data[7],
		       discriminator.V_neg_data: input_data[8],
		       discriminator.D_neg_data: input_data[9]}

		if tag=='I':
			_d, d_loss = sess.run([discriminator.updates_I, discriminator.loss_I],feed_dict=dict)
		if tag=='T':
			_d, d_loss = sess.run([discriminator.updates_T, discriminator.loss_T],feed_dict=dict)
		if tag=='A':
			_d, d_loss = sess.run([discriminator.updates_A, discriminator.loss_A],feed_dict=dict)
		if tag=='V':
			_d, d_loss = sess.run([discriminator.updates_V, discriminator.loss_V],feed_dict=dict)
		if tag=='D':
			_d, d_loss = sess.run([discriminator.updates_D, discriminator.loss_D],feed_dict=dict)

	print('D_Loss_%s: %.4f' % (tag,d_loss))
	return discriminator

def main():
	discriminator = DIS(I_DIM, T_DIM, A_DIM, V_DIM, D_DIM, HIDDEN_DIM, OUTPUT_DIM, WEIGHT_DECAY, D_LEARNING_RATE, BETA, GAMMA)
	#generator = GEN(I_DIM, T_DIM, A_DIM, V_DIM, D_DIM, HIDDEN_DIM, OUTPUT_DIM, WEIGHT_DECAY, D_LEARNING_RATE, BETA, GAMMA)

	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	sess.run(tf.initialize_all_variables())

	# pdb.set_trace()

	saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables()])
	# saver.restore(sess, DIS_MODEL_NEWEST_FILE)

	print('start adversarial training')
	#map_best_val_gen = 0.0
	map_best_val_dis = 0.0

	for epoch in range(WHOLE_EPOCH):
		print('Training D ...')
		for d_epoch in range(D_EPOCH):
			print('d_epoch: ' + str(d_epoch))
			if d_epoch  == 0:
				print('negative sampling for d using g ...')
				dis_train_list_I = generate_samples(0)
				dis_train_list_T = generate_samples(1)
				dis_train_list_D = generate_samples(2)
				dis_train_list_V = generate_samples(3)
				dis_train_list_A = generate_samples(4)

			discriminator = train_discriminator(sess, discriminator, dis_train_list_I, 'I')
			discriminator = train_discriminator(sess, discriminator, dis_train_list_T, 'T')
			discriminator = train_discriminator(sess, discriminator, dis_train_list_D, 'D')
			discriminator = train_discriminator(sess, discriminator, dis_train_list_V, 'V')
			discriminator = train_discriminator(sess, discriminator, dis_train_list_A, 'A')

			if (d_epoch + 1) % (D_DISPLAY) == 0:
				test_map = MAP_ARGV(sess, discriminator, test_feature, database_feature, test_label, database_label, OUTPUT_DIM)
				print('Test_MAP: %.4f' % test_map)
				if test_map > map_best_val_dis:
					map_best_val_dis = test_map
					saver.save(sess, DIS_MODEL_BEST_FILE)
			saver.save(sess, DIS_MODEL_NEWEST_FILE)


	sess.close()

if __name__ == '__main__':
	main()
