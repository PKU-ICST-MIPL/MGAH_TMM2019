import tensorflow as tf
import pdb

class DIS():
	def __init__(self, I_dim, T_dim, A_dim, V_dim, D_dim, hidden_dim, output_dim, weight_decay, learning_rate, beta, gamma):

		self.I_data = tf.placeholder(tf.float32, shape=[None, I_dim], name="I_data")
		self.T_data = tf.placeholder(tf.float32, shape=[None, T_dim], name="T_data")
		self.A_data = tf.placeholder(tf.float32, shape=[None, A_dim], name="A_data")
		self.V_data = tf.placeholder(tf.float32, shape=[None, V_dim], name="V_data")
		self.D_data = tf.placeholder(tf.float32, shape=[None, D_dim], name="D_data")
		
		self.I_neg_data = tf.placeholder(tf.float32, shape=[None, I_dim], name="I_neg_data")
		self.T_neg_data = tf.placeholder(tf.float32, shape=[None, T_dim], name="T_neg_data")
		self.A_neg_data = tf.placeholder(tf.float32, shape=[None, A_dim], name="A_neg_data")
		self.V_neg_data = tf.placeholder(tf.float32, shape=[None, V_dim], name="V_neg_data")
		self.D_neg_data = tf.placeholder(tf.float32, shape=[None, D_dim], name="D_neg_data")
		

		with tf.variable_scope('discriminator'):
			hash_sig = {}
			I_sig, self.I_code, I_param = self.build_net(self.I_data, 'I', I_dim, hidden_dim, output_dim, None)
			hash_sig['I'] = I_sig
			T_sig, self.T_code, T_param = self.build_net(self.T_data, 'T', T_dim, hidden_dim, output_dim, None)
			hash_sig['T'] = T_sig
			A_sig, self.A_code, A_param = self.build_net(self.A_data, 'A', A_dim, hidden_dim, output_dim, None)
			hash_sig['A'] = A_sig
			V_sig, self.V_code, V_param = self.build_net(self.V_data, 'V', V_dim, hidden_dim, output_dim, None)
			hash_sig['V'] = V_sig
			D_sig, self.D_code, D_param = self.build_net(self.D_data, 'D', D_dim, hidden_dim, output_dim, None)
			hash_sig['D'] = D_sig
			self.image_hash = {}
			self.image_hash['I'] = tf.cast(I_sig + 0.5, tf.int32)
			self.image_hash['T'] = tf.cast(T_sig + 0.5, tf.int32)
			self.image_hash['A'] = tf.cast(A_sig + 0.5, tf.int32)
			self.image_hash['V'] = tf.cast(V_sig + 0.5, tf.int32)
			self.image_hash['D'] = tf.cast(D_sig + 0.5, tf.int32)
			
			
				
			hash_sig_neg = {}
			I_neg_sig, I_neg_code = self.build_net(self.I_neg_data, 'I_neg', I_dim, hidden_dim, output_dim, I_param)
			hash_sig_neg['I_neg'] = I_neg_sig
			T_neg_sig, T_neg_code = self.build_net(self.T_neg_data, 'T_neg', T_dim, hidden_dim, output_dim, T_param)
			hash_sig_neg['T_neg'] = T_neg_sig
			A_neg_sig, A_neg_code = self.build_net(self.A_neg_data, 'A_neg', A_dim, hidden_dim, output_dim, A_param)
			hash_sig_neg['A_neg'] = A_neg_sig
			V_neg_sig, V_neg_code = self.build_net(self.V_neg_data, 'V_neg', V_dim, hidden_dim, output_dim, V_param)
			hash_sig_neg['V_neg'] = V_neg_sig
			D_neg_sig, D_neg_code = self.build_net(self.D_neg_data, 'D_neg', D_dim, hidden_dim, output_dim, D_param)
			hash_sig_neg['D_neg'] = D_neg_sig
		

		pos_distance_I = self.compute_distance(hash_sig['I'],hash_sig)
		neg_distance_I = self.compute_distance(hash_sig['I'],hash_sig_neg)
		pos_distance_T = self.compute_distance(hash_sig['T'],hash_sig)
		neg_distance_T = self.compute_distance(hash_sig['T'],hash_sig_neg)
		pos_distance_D = self.compute_distance(hash_sig['D'],hash_sig)
		neg_distance_D = self.compute_distance(hash_sig['D'],hash_sig_neg)
		pos_distance_V = self.compute_distance(hash_sig['V'],hash_sig)
		neg_distance_V = self.compute_distance(hash_sig['V'],hash_sig_neg)
		pos_distance_A = self.compute_distance(hash_sig['A'],hash_sig)
		neg_distance_A = self.compute_distance(hash_sig['A'],hash_sig_neg)
		
		
		self.reward = {'I':0,'T':0,'D':0,'V':0,'A':0}
		with tf.name_scope('svm_loss'):
			self.loss_I = tf.reduce_mean(tf.maximum(0.0, beta + pos_distance_I - neg_distance_I)) + self.compute_regulation(weight_decay)
			self.loss_T = tf.reduce_mean(tf.maximum(0.0, beta + pos_distance_T - neg_distance_T)) + self.compute_regulation(weight_decay)
			self.loss_D = tf.reduce_mean(tf.maximum(0.0, beta + pos_distance_D - neg_distance_D)) + self.compute_regulation(weight_decay)
			self.loss_V = tf.reduce_mean(tf.maximum(0.0, beta + pos_distance_V - neg_distance_V)) + self.compute_regulation(weight_decay)
			self.loss_A = tf.reduce_mean(tf.maximum(0.0, beta + pos_distance_A - neg_distance_A)) + self.compute_regulation(weight_decay)
	
			self.reward['I'] = tf.sigmoid(tf.maximum(0.0, beta + pos_distance_I - neg_distance_I))
			self.reward['T'] = tf.sigmoid(tf.maximum(0.0, beta + pos_distance_T - neg_distance_T))
			self.reward['D'] = tf.sigmoid(tf.maximum(0.0, beta + pos_distance_D - neg_distance_D))
			self.reward['V'] = tf.sigmoid(tf.maximum(0.0, beta + pos_distance_V - neg_distance_V))
			self.reward['A'] = tf.sigmoid(tf.maximum(0.0, beta + pos_distance_A - neg_distance_A))
		
		self.global_step = tf.Variable(0, trainable=False)
		self.lr_step = tf.train.exponential_decay(learning_rate, self.global_step, 500, 0.9, staircase=True)
		self.optimizer = tf.train.GradientDescentOptimizer(self.lr_step)
		self.updates_I = self.optimizer.minimize(self.loss_I, var_list=[var for var in tf.trainable_variables()])
		self.updates_T = self.optimizer.minimize(self.loss_T, var_list=[var for var in tf.trainable_variables()])
		self.updates_D = self.optimizer.minimize(self.loss_D, var_list=[var for var in tf.trainable_variables()])
		self.updates_V = self.optimizer.minimize(self.loss_V, var_list=[var for var in tf.trainable_variables()])
		self.updates_A = self.optimizer.minimize(self.loss_A, var_list=[var for var in tf.trainable_variables()])
		

		
	def build_layer(self, input, name, shape, l_param, activ):
		W_init_args = {}
		b_init_args = {}
		if l_param == None:
			l_param = []
			W_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
			b_init = tf.constant_initializer(value=0.0)
			l_param.append(tf.get_variable(name=name+'_W', shape=shape, initializer=W_init, **W_init_args))
			l_param.append(tf.get_variable(name=name+'_b', shape=(shape[-1]), initializer=b_init, **b_init_args))
			if activ == 'tanh':
				return tf.nn.tanh(tf.nn.xw_plus_b(input, l_param[0], l_param[1])), l_param
			elif activ == 'sigmoid':
				return tf.sigmoid(tf.nn.xw_plus_b(input, l_param[0], l_param[1])), l_param
		else:
			if activ == 'tanh':
				return tf.nn.tanh(tf.nn.xw_plus_b(input, l_param[0], l_param[1]))
			elif activ == 'sigmoid':
				return tf.sigmoid(tf.nn.xw_plus_b(input, l_param[0], l_param[1]))
			
			
	def build_net(self, input, name, input_dim, hidden_dim, output_dim, n_param):
		if n_param == None:
			n_param = []
			l1, l1_param = self.build_layer(input, name+'_l1', (input_dim, hidden_dim), None, 'tanh')
			l2, l2_param = self.build_layer(l1, name+'_l2', (hidden_dim, output_dim), None, 'sigmoid')
			n_param.append(l1_param)
			n_param.append(l2_param)
			hash_code = tf.cast(l2 + 0.5, tf.int32)
			return l2, hash_code, n_param
		else:
			l1 = self.build_layer(input, name+'_l1', (input_dim, hidden_dim), n_param[0], 'tanh')
			l2 = self.build_layer(l1, name+'_l2', (hidden_dim, output_dim), n_param[1], 'sigmoid')
			hash_code = tf.cast(l2 + 0.5, tf.int32)
			return l2, hash_code
			
	
	def compute_distance(self, ot, sig_dict):
		result = 0.0
		for i in sig_dict:
			result += tf.reduce_sum(tf.square(ot - sig_dict[i]), 1)
		return result
		
		
	def compute_neg_distance(self, pos_sig_dict, neg_sig_dict):
		result = 0.0
		for i in pos_sig_dict:
			for j in neg_sig_dict:
				if i != j:
					result += tf.reduce_sum(tf.square(pos_sig_dict[i] - neg_sig_dict[j]), 1)
		return result
			
			
	def compute_regulation(self, weight_decay):
		result = 0.0
		var_list = [var for var in tf.trainable_variables()]
		for item in var_list:
			result += tf.nn.l2_loss(item)
		return weight_decay * result
