import random
import numpy as np
import tensorflow as tf
import mdn
from matplotlib import pyplot as plt
import time

HIDDEN_UNITS = 256
NUMBER_MIXTURES = 5
NUMBER_DIM = 2

def data_preprocessor(filename):
	stock_data = []
	with open(filename, 'r') as f:
		idx = 0
		for line in f:
			if idx == 0:
				idx += 1
				continue
			stock_data.append(float(line.split(',')[5]))
		f.close()
	sequence = []
	for i in range(len(stock_data)-1):
		curr = stock_data[i]
		next = stock_data[i+1]
		sequence.append([np.sign(next-curr), np.abs(next-curr)])
	return stock_data, sequence

class ML_model:
	def __init__(self, data, lags, Z):
		self.data = data
		self.Z = Z
		self.lags = lags
		self.hits = {}
		print("Num CPUs:", len(tf.config.list_physical_devices('CPU') )) 
		print("Num GPUs:", len(tf.config.list_physical_devices('GPU') )) 

	def train_rnn(self, epochs, batch_size, model_name):
		# one dimension for up/down, one dimension for diff
		X_sequence = []
		
		idx = 0
		while idx < len(self.data)-1:
			curr = self.data[idx]
			next = self.data[idx+1]
			X_sequence.append([np.sign(next-curr), np.abs(next-curr)])
			idx += 1

		# prepare training dataset
		X_train = []
		Y_train = []
		for i in range(len(self.data)-self.lags-1):
			example = X_sequence[i : i + self.lags]
			X_train.append(example[:-1])
			Y_train.append(example[-1])
		print(self.lags-1, len(X_train[-1]), len(Y_train[-1]))
		X_train = np.array(X_train).reshape(len(Y_train), self.lags-1, NUMBER_DIM)
		Y_train = np.array(Y_train).reshape(len(Y_train), NUMBER_DIM)
		print(X_train.shape, Y_train.shape)
		# print(np.isnan(np.sum(X_train)))
		# print(np.isnan(np.sum(Y_train)))

		print('batch_size:{}, epoch:{}'.format(batch_size, epochs))
		# Sequential model
		model = tf.compat.v1.keras.Sequential()

		# Add two LSTM layers
		model.add(tf.compat.v1.keras.layers.CuDNNLSTM(HIDDEN_UNITS, batch_input_shape=(None,self.lags-1,NUMBER_DIM), return_sequences=True))
		model.add(tf.compat.v1.keras.layers.CuDNNLSTM(HIDDEN_UNITS))

		# Here's the MDN layer
		model.add(mdn.MDN(NUMBER_DIM, NUMBER_MIXTURES))

		# Now we compile the MDN RNN - need to use a special loss function with the right number of dimensions and mixtures.
		model.compile(loss=mdn.get_mixture_loss_func(NUMBER_DIM, NUMBER_MIXTURES), optimizer=tf.compat.v1.keras.optimizers.Adam())

		# Let's see what we have:
		model.summary()

		history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=[tf.compat.v1.keras.callbacks.TerminateOnNaN()])

		model.save('model/{}_rnn_mdn_{}_{}_{}_{}_gpu.h5'.format(model_name, NUMBER_DIM, NUMBER_MIXTURES, batch_size, epochs))
		
		# plt.figure()
		# plt.plot(history.history['loss'])
		# plt.title('loss_{}_{}_{}_{}'.format(NUMBER_DIM, NUMBER_MIXTURES, batch_size, epochs))
		# plt.show()
		print(history.history['loss'])

	def load_model(self, model_name):
		# with tf.device('/gpu:0'):
		# 	print('force using gpu...')
		decoder = tf.compat.v1.keras.Sequential()
		decoder.add(tf.compat.v1.keras.layers.CuDNNLSTM(HIDDEN_UNITS, batch_input_shape=(1,1,NUMBER_DIM), return_sequences=True, stateful=True))
		decoder.add(tf.compat.v1.keras.layers.CuDNNLSTM(HIDDEN_UNITS, stateful=True))
		decoder.add(mdn.MDN(NUMBER_DIM, NUMBER_MIXTURES))
		decoder.compile(loss=mdn.get_mixture_loss_func(NUMBER_DIM,NUMBER_MIXTURES), optimizer=tf.compat.v1.keras.optimizers.Adam())
		decoder.summary()
		decoder.load_weights(model_name) # load weights independently from file
		print('Model Loaded!')
		return decoder

	def simulate(self, T, V, state, base, decoder, plot=False):
		predicted_sequence = []

		for t in range(T):
			#start = time.time()
			input_state = np.array(state).reshape(1, 1, NUMBER_DIM)
			params = decoder.predict(input_state)
			sample_value = mdn.sample_from_output(params[0], NUMBER_DIM, NUMBER_MIXTURES)
			sample_value.reshape(NUMBER_DIM,)
			# covert to actual value changes
			diff = np.sign(sample_value[0][0]) * sample_value[0][1]
			base += diff
			predicted_sequence.append(base)
			state = sample_value
			#end = time.time()
			#print('prediction:', end-start)
			if base >= V:
				break
			
			
		if plot:
			print(predicted_sequence)
			plt.plot(range(0, len(self.data)), self.data, color='b')
			plt.plot(range(len(self.data), len(self.data) + len(predicted_sequence)), \
				predicted_sequence, color='black', label='ML-model')
			#plt.hlines(V, 0, T, color='r')
			plt.legend()
			plt.show()

		return predicted_sequence, state

	def SRS(self, T, V, state, base, rnn_model, ci_threshold, verbose=False):
		total_cost = 0
		estimate = 0
		ci = 1
		samples = []

		history = []
		while ci > ci_threshold or estimate == 0 or estimate == 1 or ci == 0:
			for i in range(10):
				# print(state, base)
				s, _ = self.simulate(T, V, state, base, rnn_model)
				t = len(s)
				total_cost += t
				# print(len(s), s[-1])
				# if t > 0 and s[-1] >= V:
				if t >= T:
					samples.append(1)
				else:
					samples.append(0)

			estimate = np.mean(samples)
			ci = self.Z * np.sqrt((estimate*(1-estimate))/len(samples))
			
			history.append((estimate, ci, total_cost))

			if verbose:
				print(history[-1])
				if len(samples) > 100:
					break
		return history

	def initialize(self, t_boundaries, v_boundaries):		
		self.hits.clear()
		for i in range(len(v_boundaries)):
			self.hits[i] = []

	def MLSS_hybird_dfs(self, V, state, base, rnn_model, idx, splits, t_boundaries, v_boundaries):
		s, t = state
		cost = 0
		hits = 0

		for i in range(splits):
			if idx == len(t_boundaries)-1:
				# print('---start: ', idx, base, state)
				q, last_state = self.simulate(np.sum(t_boundaries)-t, v_boundaries[idx], s, base, rnn_model)
				# print('===end: ', idx, q[-1], last_state)
				cost += len(q)
				# if len(q) >= t_boundaries[idx]:
				if t + len(q) >= np.sum(t_boundaries):
					hits += 1
			else:
				# print('---start: ', idx, base, state)
				q, last_state = self.simulate(np.sum(t_boundaries[:idx+1])-t, v_boundaries[idx], s, base, rnn_model)
				# print('===end: ', idx, q[-1], last_state)
				cost += len(q)
				if q[-1] >= V:
					continue
				# if len(q) >= t_boundaries[idx]:
				c,h = self.MLSS_hybird_dfs(V, (last_state, t+len(q)), q[-1], idx+1, \
					splits, t_boundaries, v_boundaries)
				cost += c
				hits += h
		return cost, hits

	def MLSS_hybrid_root_path(self, T, V, state, base, rnn_model, splits, t_boundaries, v_boundaries):
		total_cost = 0
		total_hits = 0
		
		idx = 0
		# print('---start: ', idx, base, state)
		s, last_state = self.simulate(t_boundaries[idx], v_boundaries[idx], state, base, rnn_model)
		# print('===end: ', idx, s[-1], last_state)
		t = len(s)
		total_cost += t

		if s[-1] >= V:
			return total_cost, total_hits
		
		# if t >= t_boundaries[idx]:
		# v_idx = min(len(v_boundaries)-1, np.searchsorted(v_boundaries, s[-1], side='right'))
		c, h = self.MLSS_hybird_dfs(V, (last_state,t), s[-1], rnn_model, idx+1, splits, t_boundaries, v_boundaries)
		total_cost += c
		# self.hits[v_idx].append(h)
		total_hits += h
		# else:
		# 	self.hits[len(v_boundaries)-1].append(0)

		return total_cost, total_hits

	def MLSS_hybrid(self, T, V, state, base, rnn_model, splits, t_boundaries, v_boundaries, ci_threshold, verbose=False):
		total_cost = 0
		estimate = 1
		ci = 1

		m = len(t_boundaries)

		root_paths = 0
		root_path_hits = []

		# self.initialize(t_boundaries, v_boundaries)

		history = []

		while ci > ci_threshold or estimate == 0 or estimate == 1 or ci == 0:
			for i in range(5):
				root_paths += 1
				# print(state, base)
				c, h = self.MLSS_hybrid_root_path(T, V, state, base, rnn_model, splits, t_boundaries, v_boundaries)
				total_cost += c
				root_path_hits.append(h)

			# estimation
			estimate = 0
			estimated_var = 0

			estimate = np.sum(root_path_hits) / (len(root_path_hits) * splits ** (m-1))
			estimated_var = np.var(root_path_hits) / (len(root_path_hits) * splits ** (2*(m-1)))

			# # merge value partition with the same splitting ratio
			# split_dict = defaultdict(list)
			# for k in self.hits:
			# 	split_dict[splits[k]].extend(self.hits[k])

			# assert root_paths == np.sum([len(split_dict[v]) for v in split_dict])

			# for v in split_dict:
			# 	estimate += np.sum(split_dict[v]) / (root_paths * v ** (m-1))

			# if len(split_dict) == 1:
			# 	v = list(split_dict.keys())[0]
			# 	estimated_var = np.var(split_dict[v]) / (root_paths * v ** (2*(m-1)))
			# else:
			# 	for v in split_dict:
			# 		# estimated_var += len(self.hits[k]) * np.var(self.hits[k]) / (root_paths * splits[k] ** (m-1)) ** 2
			# 		p_k = len(split_dict[v]) / root_paths
			# 		avg_hits_k = np.mean(split_dict[v])
			# 		var_hits_k = np.var(split_dict[v])

			# 		estimated_var += (var_hits_k * p_k) / (root_paths * v ** (2*(m-1)))
			# 		estimated_var += (avg_hits_k**2 * p_k * (1-p_k)) / (root_paths * v ** (2*(m-1)))

			# # for k in self.hits:
			# # 	estimate += np.sum(self.hits[k]) / (root_paths * splits[k] ** (m-1))
			# # 	estimated_var += len(self.hits[k]) * np.var(self.hits[k]) / (root_paths * splits[k] ** (m-1)) ** 2


			ci = self.Z * np.sqrt(estimated_var)
			history.append((estimate, ci, total_cost))

			if verbose == True:
				print(history[-1])
				if len(root_path_hits) > 100:
					break

		return history

def ML_model_train():
	raw_data, transformed_data = data_preprocessor('amazon_stock.csv')
	model = ML_model(raw_data, 50, 1.96)
	model.train_rnn(100, 32, 'amazon')

def ML_model_test():
	raw_data, transformed_data = data_preprocessor('amazon_stock.csv')
	model = ML_model(raw_data, 50, 1.96)
	
	rnn_mdn = model.load_model('model/amazon_rnn_mdn_2_5_32_100_gpu.h5')

	predicated_traces = []
	for i in range(1):
		s, _ = model.simulate(200, 3000, transformed_data[-1], raw_data[-1], rnn_mdn)
		predicated_traces.append(s)
		print(s)
	# 	plt.plot(s)
	# plt.show()

def ML_model_SRS_test(datafile, modelfile, T, V):
	raw_data, transformed_data = data_preprocessor(datafile)
	model = ML_model(raw_data, 50, 1.96)
	rnn_mdn = model.load_model(modelfile)
	srs = model.SRS(T, V, transformed_data[-1], raw_data[-1], rnn_mdn, 0.03, True)
	print(srs[-1])

def ML_model_hybrid_root_path_test():
	T = 200
	V = 1350
	t_splits = [100, 100]
	v_splits = [1300, 1350]

	datafile = 'google_stock.csv'
	modelfile = 'model/google_rnn_mdn_2_5_16_30.h5'

	raw_data, transformed_data = data_preprocessor(datafile)
	model = ML_model(raw_data, 50, 1.96)
	rnn_mdn = model.load_model(modelfile)
	model.initialize(t_splits, v_splits)
	model.MLSS_hybrid_root_path(T, V, transformed_data[-1], raw_data[-1], rnn_mdn,\
	 [3,3], t_splits, v_splits)

def ML_model_hybrid_test():
	T = 200
	V = 2500
	t_splits = [100, 100]
	v_splits = [2500, 2500]

	datafile = 'amazon_stock.csv'
	modelfile = 'model/amazon_rnn_mdn_2_5_32_100_gpu.h5'

	raw_data, transformed_data = data_preprocessor(datafile)
	model = ML_model(raw_data, 50, 1.96)
	rnn_mdn = model.load_model(modelfile)

	print(T,V,t_splits,v_splits)

	# print('====SRS===')
	# start = time.time()
	# srs = model.SRS(T, V, transformed_data[-1], raw_data[-1], rnn_mdn, 0.05)
	# end = time.time()
	# print(srs[-1], end-start)

	print('===MLSS===')
	start = time.time()
	mlss = model.MLSS_hybrid(T, V, transformed_data[-1], raw_data[-1], rnn_mdn, \
		3, t_splits, v_splits, 0.05)
	end = time.time()
	print(mlss[-1], end-start)



if __name__ == '__main__':
	# ML_model_train()
	# ML_model_test()
	# ML_model_SRS_test('google_stock.csv', 'model/google_rnn_mdn_2_5_16_30.h5',\
	# 	200, 1450)
	# ML_model_SRS_test('amazon_stock.csv', 'model/amazon_rnn_mdn_2_5_16_50.h5',\
	# 	200, 3500)
	ML_model_hybrid_test()
	# ML_model_hybrid_root_path_test()