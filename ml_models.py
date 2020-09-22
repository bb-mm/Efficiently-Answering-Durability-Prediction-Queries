import random
import numpy as np
from tensorflow import keras
from collections import defaultdict
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
		# self.hits = {}
		self.counter = defaultdict(list)
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
		model = keras.Sequential()

		# Add two LSTM layers
		model.add(keras.layers.LSTM(HIDDEN_UNITS, batch_input_shape=(None,self.lags-1,NUMBER_DIM), return_sequences=True))
		model.add(keras.layers.LSTM(HIDDEN_UNITS))

		# Here's the MDN layer
		model.add(mdn.MDN(NUMBER_DIM, NUMBER_MIXTURES))

		# Now we compile the MDN RNN - need to use a special loss function with the right number of dimensions and mixtures.
		model.compile(loss=mdn.get_mixture_loss_func(NUMBER_DIM, NUMBER_MIXTURES), optimizer=keras.optimizers.Adam())

		# Let's see what we have:
		model.summary()

		history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=[keras.callbacks.TerminateOnNaN()])

		model.save('model/{}_rnn_mdn_{}_{}_{}_{}.h5'.format(model_name, NUMBER_DIM, NUMBER_MIXTURES, batch_size, epochs))
		
		plt.figure()
		plt.plot(history.history['loss'])
		plt.title('loss_{}_{}_{}_{}'.format(NUMBER_DIM, NUMBER_MIXTURES, batch_size, epochs))
		plt.show()

	def load_model(self, model_name):
		decoder = keras.Sequential()
		decoder.add(keras.layers.LSTM(HIDDEN_UNITS, batch_input_shape=(1,1,NUMBER_DIM), return_sequences=True, stateful=True))
		decoder.add(keras.layers.LSTM(HIDDEN_UNITS, stateful=True))
		decoder.add(mdn.MDN(NUMBER_DIM, NUMBER_MIXTURES))
		decoder.compile(loss=mdn.get_mixture_loss_func(NUMBER_DIM,NUMBER_MIXTURES), optimizer=keras.optimizers.Adam())
		#decoder.summary()
		decoder.load_weights(model_name) # load weights independently from file
		print('Model Loaded!')
		return decoder

	def simulate(self, T, V, state, base, decoder, plot=False):
		predicted_sequence = []
		if T == 0:
			return predicted_sequence, state, 0
		
		start = time.time()
		for t in range(T):
			input_state = np.array(state).reshape(1, 1, NUMBER_DIM)
			params = decoder.predict(input_state)
			sample_value = mdn.sample_from_output(params[0], NUMBER_DIM, NUMBER_MIXTURES)
			sample_value.reshape(NUMBER_DIM,)
			# covert to actual value changes
			diff = np.sign(sample_value[0][0]) * sample_value[0][1]
			base += diff
			predicted_sequence.append(base)
			state = sample_value
			if base >= V:
				break
		end = time.time()
			
			
		if plot:
			print(predicted_sequence)
			plt.plot(range(0, len(self.data)), self.data, color='b')
			plt.plot(range(len(self.data), len(self.data) + len(predicted_sequence)), \
				predicted_sequence, color='black', label='ML-model')
			#plt.hlines(V, 0, T, color='r')
			plt.legend()
			plt.show()

		return predicted_sequence, state, end-start

	def SRS(self, T, V, state, base, rnn_model, ci_threshold, verbose=False):
		total_cost = 0
		total_time = 0
		estimate = 0
		ci = 1
		samples = []

		history = []
		while ci > ci_threshold or estimate == 0 or estimate == 1 or ci == 0:
			for i in range(20):
				# print(state, base)
				s, _, simulation_time = self.simulate(T, V, state, base, rnn_model)
				t = len(s)
				total_cost += t
				total_time += simulation_time
				# print(len(s), s[-1])
				# if t > 0 and s[-1] >= V:
				if t >= T:
					samples.append(0)
				else:
					samples.append(1)

			estimate = np.mean(samples)
			ci = self.Z * np.sqrt((estimate*(1-estimate))/len(samples))
			
			history.append((estimate, ci, total_cost, total_time))

			if verbose:
				print(history[-1])

		return history

	def SRS_v2(self, T, V, state, base, rnn_model, ground_truth, relative_error):
		total_cost = 0
		total_time = 0
		estimate = 0
		ci = 1
		samples = []

		history = []
		while True:
			for i in range(20):
				s, _, simulation_time = self.simulate(T, V, state, base, rnn_model)
				t = len(s)
				total_cost += t
				total_time += simulation_time

				if t >= T:
					samples.append(0)
				else:
					samples.append(1)

			estimate = np.mean(samples)
			var = np.sqrt(estimate*(1-estimate) / len(samples))
			history.append((estimate, var/ground_truth, total_cost, total_time))
			if var/ground_truth > 0  and var/ground_truth <= relative_error:
				break

		return history

	# def initialize(self, t_boundaries, v_boundaries):		
	# 	self.hits.clear()
	# 	for i in range(len(v_boundaries)):
	# 		self.hits[i] = []

	def MLSS_hybird_dfs(self, T, V, state, base, rnn_model, idx, splits, v_boundaries):
		s, t = state
		cost = 0
		hits = 0
		time = 0
		success_cnt = 0
		for i in range(splits):
			if idx == len(v_boundaries)-1:
				# print('---start: ', idx, base, state)
				q, last_state, simulation_time = self.simulate(T-t, v_boundaries[idx], s, base, rnn_model)
				# print('===end: ', idx, q[-1], last_state)
				cost += len(q)
				time += simulation_time
				# if len(q) >= t_boundaries[idx]:
				# if t + len(q) >= np.sum(t_boundaries):
				if len(q) > 0 and q[-1] >= v_boundaries[idx]:
					success_cnt += 1
					hits += 1
			else:
				# print('---start: ', idx, base, state)
				q, last_state, simulation_time = self.simulate(T-t, v_boundaries[idx], s, base, rnn_model)
				# print('===end: ', idx, q[-1], last_state)
				cost += len(q)
				time += simulation_time

				# if len(q) >= t_boundaries[idx]:
				if len(q) > 0 and q[-1] >= v_boundaries[idx]:
					success_cnt += 1
					c,h,st = self.MLSS_hybird_dfs(T, V, (last_state, t+len(q)), q[-1], rnn_model, idx+1, \
						splits, v_boundaries)
					cost += c
					hits += h
					time += st
		self.counter[idx].append(success_cnt)
		return cost, hits, time

	def MLSS_hybrid_root_path(self, T, V, state, base, rnn_model, splits, v_boundaries):
		total_cost = 0
		total_hits = 0
		total_time = 0
		
		idx = 0
		# print('---start: ', idx, base, state)
		self.counter[idx].append(1)
		s, last_state, simulation_time = self.simulate(T, v_boundaries[idx], state, base, rnn_model)
		# print('===end: ', idx, s[-1], last_state)
		t = len(s)
		total_cost += t
		total_time += simulation_time

		if s[-1] > v_boundaries[idx]:
			c, h, st = self.MLSS_hybird_dfs(T, V, (last_state,t), s[-1], rnn_model, idx+1, splits, v_boundaries)
			total_cost += c
			total_hits += h
			total_time += st

		return total_cost, total_hits, total_time

	def MLSS_hybrid(self, T, V, state, base, rnn_model, splits, v_boundaries, ci_threshold, verbose=False):
		self.counter.clear()
		
		total_cost = 0
		total_time = 0
		estimate = 1
		ci = 1

		m = len(v_boundaries)

		root_paths = 0
		root_path_hits = []

		# self.initialize(t_boundaries, v_boundaries)

		history = []

		while ci > ci_threshold or estimate == 0 or estimate == 1 or ci == 0:
			for i in range(10):
				# print(state, base)
				c, h, st = self.MLSS_hybrid_root_path(T, V, state, base, rnn_model, splits, v_boundaries)
				
				root_paths += 1
				total_cost += c
				total_time += st
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
			history.append((estimate, ci, total_cost, total_time))

			if verbose == True:
				print(history[-1])
		# calculate average path cost and root-path-hits variance
		avg_path_cost = total_cost / len(root_path_hits)
		plan_eval = avg_path_cost * np.var(root_path_hits) / splits ** (2*(m-1))
		history.append((estimate, ci, total_cost, total_time, plan_eval))
		return history
	
	def MLSS_v2(self, T, V, state, base, rnn_model, splits, v_boundaries, ground_truth, relative_error):
		self.counter.clear()
		
		total_cost = 0
		total_time = 0
		estimate = 1
		ci = 1

		m = len(v_boundaries)

		root_paths = 0
		root_path_hits = []

		history = []

		while True:
			for i in range(10):
				c, h, st = self.MLSS_hybrid_root_path(T, V, state, base, rnn_model, splits, v_boundaries)
				
				root_paths += 1
				total_cost += c
				total_time += st
				root_path_hits.append(h)

			# estimation
			estimate = np.sum(root_path_hits) / (len(root_path_hits) * splits ** (m-1))
			var = np.sqrt(np.var(root_path_hits) / (len(root_path_hits) * splits ** (2*(m-1))))
			history.append((estimate, var/ground_truth, total_cost, total_time))
			if var/ground_truth > 0  and var/ground_truth <= relative_error:
				break
		return history
	
	# simulation effiency comparison between SRS and MLSS
	def rare_event_efficiency(self, T, V, splits, v_boundaries, state, base, rnn_model, relative_error, ground_truth=None):
		m = len(v_boundaries)
		if ground_truth == None:
			samples = []
			for i in range(100):
				c,h,st = self.MLSS_hybrid_root_path(T, V, state, base, rnn_model, splits, v_boundaries)
				samples.append(h)
			ground_truth = np.sum(samples) / (len(samples) * splits ** (m-1))
		print('ground-truth: {}'.format(ground_truth))

		mlss_history = []
		srs_history = []

		srs_total_cost = 0
		srs_total_time = 0
		mlss_total_cost = 0
		mlss_total_time = 0
		target_hits = 0
		root_path_hits = []
		
		# SRS
		while True:
			for i in range(10):
				s, _, simulation_time = self.simulate(T, V, state, base, rnn_model)
				t = len(s)
				srs_total_cost += t
				srs_total_time += simulation_time
				if t >= T:
					samples.append(0)
				else:
					samples.append(1)
			estimate = np.mean(samples)
			if estimate == 0:
				continue
			var = np.sqrt(estimate*(1-estimate) / len(samples))
			srs_history.append((estimate, var/ground_truth, srs_total_cost, srs_total_time))
			if len(srs_history) % 10 == 0:
				print('srs:', srs_history[-1])
			if var/ground_truth > 0  and var/ground_truth <= relative_error:
				break
		# MLSS	
		while True:
			for i in range(5):
				c,h,st = self.MLSS_hybrid_root_path(T, V, state, base, rnn_model, splits, v_boundaries)

				target_hits += h
				mlss_total_cost += c
				mlss_total_time += st
				root_path_hits.append(h)

			estimate = target_hits / (len(root_path_hits) * splits ** (m-1))
			var = np.sqrt(np.var(root_path_hits) / (len(root_path_hits) * splits ** (2*(m-1))))
			mlss_history.append((estimate, var/ground_truth, mlss_total_cost, mlss_total_time))
			if len(mlss_history) % 10 == 0:
				print('mlss,', mlss_history[-1])
			if var/ground_truth > 0  and var/ground_truth <= relative_error:
				break
		
		return srs_history, mlss_history

def ML_model_test():
	raw_data, transformed_data = data_preprocessor('google_stock.csv')
	model = ML_model(raw_data, 50, 1.96)
	# model.train_rnn(50, 32, 'google')
	rnn_mdn = model.load_model('model/google_rnn_mdn_2_5_16_50.h5')

	predicated_traces = []
	for i in range(10):
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
	V = 1450
	t_splits = [100, 100]
	v_splits = [1350, 1450]

	datafile = 'amazon_stock.csv'
	modelfile = 'model/amazon_rnn_mdn_2_5_16_50.h5'

	raw_data, transformed_data = data_preprocessor(datafile)
	model = ML_model(raw_data, 50, 1.96)
	rnn_mdn = model.load_model(modelfile)
	model.initialize(t_splits, v_splits)
	model.MLSS_hybrid_root_path(T, V, transformed_data[-1], raw_data[-1], rnn_mdn,\
	 [3,3], v_splits)

def ML_model_hybrid_test():
	ci_threshold = 0.05
	splits = 3
	# Google 86% setting
	# T = 200
	# V = 1450
	# t_splits = [200, 0]
	# v_splits = [1350, 1450]

	# Google 95% setting
	# T = 200
	# V = 1470
	# t_splits = [200, 0]
	# v_splits = [1430, 1470]

	# Google 12% setting
	# T = 200
	# V = 1280
	# t_splits = [160, 40]
	# v_splits = [1280, 1280]

	# Google 1% setting
	# T = 200
	# V = 1275
	# t_splits = [150, 50]
	# v_splits = [1275, 1275]

	# Google 14% setting
	# T = 500
	# V = 1300
	# t_splits = [400, 100]
	# v_splits = [1300, 1300]

	# Google 10% setting
	# T = 500
	# V = 1290
	# t_splits = [400, 100]
	# v_splits = [1290, 1290]

	# Google 85% setting
	# T = 500
	# V = 1550
	# t_splits = [500, 0]
	# v_splits = [1450, 1550]

	# Google 95% setting
	T = 500
	V = 1580
	t_splits = [500, 0]
	v_splits = [1400, 1500, 1580]

	# Amazon 19% setting
	# T = 200
	# V = 3000
	# t_splits = [150, 50]
	# v_splits = [3000, 3000]

	# Amazon 5% setting
	# T = 200
	# V = 2950
	# t_splits = [150, 50]
	# v_splits = [2950, 2950]

	# Amazon 90% setting
	# T = 200
	# V = 3500
	# t_splits = [200, 0]
	# v_splits = [3300, 3500]

	# Amazon 95% setting
	# T = 200
	# V = 3570
	# t_splits = [200, 0]
	# v_splits = [3350, 3570]

	# Amazon 18% setting
	# T = 500
	# V = 4000
	# t_splits = [400, 100]
	# v_splits = [4000, 4000]

	# Amazon 5% setting
	# T = 500
	# V = 3800
	# t_splits = [400, 100]
	# v_splits = [3800, 3800]

	# Amazon 89% setting
	# T = 500
	# V = 4900
	# t_splits = [500, 0]
	# v_splits = [4500, 4900]

	# Amazon 
	# T = 500
	# V = 4950
	# t_splits = [500, 0]
	# v_splits = [4750, 4950]


	datafile = 'google_stock.csv'
	modelfile = 'model/google_rnn_mdn_2_5_16_30.h5'
	
	# datafile = 'amazon_stock.csv'
	# modelfile = 'model/amazon_rnn_mdn_2_5_16_50.h5'

	raw_data, transformed_data = data_preprocessor(datafile)
	model = ML_model(raw_data, 50, 1.96)
	rnn_mdn = model.load_model(modelfile)

	print(T, V, t_splits, v_splits, ci_threshold)

	# print('====SRS===')
	# srs = model.SRS(T, V, transformed_data[-1], raw_data[-1], rnn_mdn, ci_threshold, True)
	# print(srs[-1])
	

	print('===MLSS===')
	mlss = model.MLSS_hybrid(T, V, transformed_data[-1], raw_data[-1], rnn_mdn, \
		3, v_splits, ci_threshold, True)
	print(mlss[-1])
	# for idx in model.counter:
	# 	if idx == 0:
	# 		print(idx, len(model.counter[idx+1]) / len(model.counter[idx]))
	# 	else:
	# 		print(idx, np.sum(model.counter[idx]) / (len(model.counter[idx]) * splits))

# def relative_error_test():
# 	splits = 3
# 	relative_error = 0.1
# 	T = 200
# 	V = 1550
# 	v_splits = [1400, 1480, 1540, 1600]



def rare_event_test():
	splits = 3
	relative_error = 0.1
	T = 200
	V = 1550
	v_splits = [1305.1741890228486, 1459.0811656021651, 1550]
	# v_splits = [1400, 1480, 1540, 1600]
	# V = 4000
	# v_splits = [3000, 3400, 3700, 4000]

	print(T, V, v_splits, relative_error)

	datafile = 'google_stock.csv'
	modelfile = 'model/google_rnn_mdn_2_5_16_30.h5'
	# datafile = 'amazon_stock.csv'
	# modelfile = 'model/amazon_rnn_mdn_2_5_16_50.h5'

	raw_data, transformed_data = data_preprocessor(datafile)
	model = ML_model(raw_data, 50, 1.96)
	rnn_mdn = model.load_model(modelfile)
	srs_history, mlss_history = model.rare_event_efficiency(T, V, splits, v_splits, \
			transformed_data[-1], raw_data[-1], rnn_mdn, relative_error)

	# print('SRS:', srs_history[-1])
	# print('MLSS:', mlss_history[-1])

	print('======SRS History======')
	for item in srs_history:
		print(item)
	
	print('======MLSS History======')
	for item in mlss_history:
		print(item)

def relative_error_test():
	splits = 3
	relative_error = 0.1
	ground_truth = 0.005
	T = 200
	V = 1600
	# v_splits = [1184.2097974088554, 1373.2347605271157, 1472.6506221393076, 1600]
	v_splits = [1296.9129656526768, 1451.422772775328, 1600]
	
	print(T, V, v_splits, ground_truth, relative_error)

	datafile = 'google_stock.csv'
	modelfile = 'model/google_rnn_mdn_2_5_16_30.h5'

	raw_data, transformed_data = data_preprocessor(datafile)
	model = ML_model(raw_data, 50, 1.96)
	rnn_mdn = model.load_model(modelfile)

	mlss = model.MLSS_v2(T, V, transformed_data[-1], raw_data[-1], rnn_mdn, \
		3, v_splits, ground_truth, relative_error)
	
	print(mlss)


if __name__ == '__main__':
	# ML_model_test()
	# ML_model_SRS_test('google_stock.csv', 'model/google_rnn_mdn_2_5_16_30.h5',\
	# 	200, 1450)
	# ML_model_SRS_test('amazon_stock.csv', 'model/amazon_rnn_mdn_2_5_16_50.h5',\
	# 	200, 3500)
	# ML_model_hybrid_test()
	# ML_model_hybrid_root_path_test()
	# rare_event_test()
	relative_error_test()