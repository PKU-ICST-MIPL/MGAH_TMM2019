import tensorflow as tf
import pdb


class GEN:
	def __init__(self, I_dim, T_dim, A_dim, V_dim, D_dim, hidden_dim, output_dim, weight_decay, learning_rate, beta, gamma):
		md = ['I','T','A','V','D']
		self.tag = tf.placeholder(tf.uint8,name='reward_I')
		self.weight_decay = weight_decay
		
		self.reward_I = tf.placeholder(tf.float32, shape=[None], name='reward_I')
		self.reward_T = tf.placeholder(tf.float32, shape=[None], name='reward_T')
		self.reward_A = tf.placeholder(tf.float32, shape=[None], name='reward_A')
		self.reward_V = tf.placeholder(tf.float32, shape=[None], name='reward_V')
		self.reward_D = tf.placeholder(tf.float32, shape=[None], name='reward_D')
		
		self.I_data = tf.placeholder(tf.float32, shape=[None, I_dim], name="I_data")
		self.T_data = tf.placeholder(tf.float32, shape=[None, T_dim], name="T_data")
		self.A_data = tf.placeholder(tf.float32, shape=[None, A_dim], name="A_data")
		self.V_data = tf.placeholder(tf.float32, shape=[None, V_dim], name="V_data")
		self.D_data = tf.placeholder(tf.float32, shape=[None, D_dim], name="D_data")
		

		with tf.variable_scope('generator'):
			self.hash_sig = {}
			I_sig, self.I_code, I_param = self.build_net(self.I_data, 'I', I_dim, hidden_dim, output_dim, None)
			self.hash_sig['I'] = I_sig
			T_sig, self.T_code, T_param = self.build_net(self.T_data, 'T', T_dim, hidden_dim, output_dim, None)
			self.hash_sig['T'] = T_sig
			A_sig, self.A_code, A_param = self.build_net(self.A_data, 'A', A_dim, hidden_dim, output_dim, None)
			self.hash_sig['A'] = A_sig
			V_sig, self.V_code, V_param = self.build_net(self.V_data, 'V', V_dim, hidden_dim, output_dim, None)
			self.hash_sig['V'] = V_sig
			D_sig, self.D_code, D_param = self.build_net(self.D_data, 'D', D_dim, hidden_dim, output_dim, None)
			self.hash_sig['D'] = D_sig
			self.image_hash = {}
			self.image_hash['I'] = tf.cast(I_sig + 0.5, tf.int32)
			self.image_hash['T'] = tf.cast(T_sig + 0.5, tf.int32)
			self.image_hash['A'] = tf.cast(A_sig + 0.5, tf.int32)
			self.image_hash['V'] = tf.cast(V_sig + 0.5, tf.int32)
			self.image_hash['D'] = tf.cast(D_sig + 0.5, tf.int32)

		#pdb.set_trace()
		
		self.pred_score_I = self.count_pred(self.hash_sig,'I')
		self.pred_score_T = self.count_pred(self.hash_sig,'T')
		self.pred_score_A = self.count_pred(self.hash_sig,'A')
		self.pred_score_V = self.count_pred(self.hash_sig,'V')
		self.pred_score_D = self.count_pred(self.hash_sig,'D')		
		
		self.gen_prob_I = self.count_gen_prob(self.pred_score_I)
		self.gen_prob_T = self.count_gen_prob(self.pred_score_T)
		self.gen_prob_A = self.count_gen_prob(self.pred_score_A)
		self.gen_prob_V = self.count_gen_prob(self.pred_score_V)
		self.gen_prob_D = self.count_gen_prob(self.pred_score_D)
		
		self.gen_loss_I = self.count_loss(0,self.gen_prob_I)
		self.gen_loss_T = self.count_loss(1,self.gen_prob_T)
		self.gen_loss_A = self.count_loss(2,self.gen_prob_A)
		self.gen_loss_V = self.count_loss(3,self.gen_prob_V)
		self.gen_loss_D = self.count_loss(4,self.gen_prob_D)
		
		
		global_step = tf.Variable(0, trainable=False)
		lr_step = tf.train.exponential_decay(learning_rate, global_step, 20000, 0.9, staircase=True)
		self.optimizer = tf.train.GradientDescentOptimizer(lr_step)
		
		self.gen_updates_I = self.optimizer.minimize(self.gen_loss_I, var_list=[var for var in tf.trainable_variables()])
		self.gen_updates_T = self.optimizer.minimize(self.gen_loss_T, var_list=[var for var in tf.trainable_variables()])
		self.gen_updates_A = self.optimizer.minimize(self.gen_loss_A, var_list=[var for var in tf.trainable_variables()])
		self.gen_updates_V = self.optimizer.minimize(self.gen_loss_V, var_list=[var for var in tf.trainable_variables()])
		self.gen_updates_D = self.optimizer.minimize(self.gen_loss_D, var_list=[var for var in tf.trainable_variables()])
		
		
		
		
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
				
	def count_pred(self,hash_sig,tag):
		pred_score = {}
		pred_score['I'] = -tf.reduce_sum(tf.square(self.hash_sig[tag] - self.hash_sig['I']), 1)
		pred_score['T'] = -tf.reduce_sum(tf.square(self.hash_sig[tag] - self.hash_sig['T']), 1)
		pred_score['A'] = -tf.reduce_sum(tf.square(self.hash_sig[tag] - self.hash_sig['A']), 1)
		pred_score['V'] = -tf.reduce_sum(tf.square(self.hash_sig[tag] - self.hash_sig['V']), 1)
		pred_score['D'] = -tf.reduce_sum(tf.square(self.hash_sig[tag] - self.hash_sig['D']), 1)
		return pred_score
		
	def count_gen_prob(self,pred_score):
		gen_prob = {}
		gen_prob['I'] = tf.reshape(tf.nn.softmax(tf.reshape(pred_score['I'], [1, -1])), [-1]) + 0.00000000000000000001
		gen_prob['T'] = tf.reshape(tf.nn.softmax(tf.reshape(pred_score['T'], [1, -1])), [-1]) + 0.00000000000000000001
		gen_prob['A'] = tf.reshape(tf.nn.softmax(tf.reshape(pred_score['A'], [1, -1])), [-1]) + 0.00000000000000000001
		gen_prob['V'] = tf.reshape(tf.nn.softmax(tf.reshape(pred_score['V'], [1, -1])), [-1]) + 0.00000000000000000001
		gen_prob['D'] = tf.reshape(tf.nn.softmax(tf.reshape(pred_score['D'], [1, -1])), [-1]) + 0.00000000000000000001
		return gen_prob
		
	def count_loss(self,tag,gen_prob):
		gen_loss = self.compute_regulation(self.weight_decay)
		if tag!=0:
			print ("add_I")
			gen_loss += -tf.reduce_mean(tf.log(gen_prob['I']) * self.reward_I)
		if tag!=1:
			print ("add_T")
			gen_loss += -tf.reduce_mean(tf.log(gen_prob['T']) * self.reward_T)
		if tag!=2:
			print ("add_A")
			gen_loss += -tf.reduce_mean(tf.log(gen_prob['A']) * self.reward_A)
		if tag!=3:
			print ("add_V")
			gen_loss += -tf.reduce_mean(tf.log(gen_prob['V']) * self.reward_V)
		if tag!=4:
			print ("add_D")
			gen_loss += -tf.reduce_mean(tf.log(gen_prob['D']) * self.reward_D)	
		return gen_loss
			
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
			
	def compute_regulation(self, weight_decay):
		result = 0.0
		var_list = [var for var in tf.trainable_variables()]
		for item in var_list:
			result += tf.nn.l2_loss(item)
		return weight_decay * result			
	
	def save_model(self, sess, filename):
		param = sess.run(self.params)
		cPickle.dump(param, open(filename, 'w'))