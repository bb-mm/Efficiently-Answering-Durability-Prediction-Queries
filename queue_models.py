import random
import numpy as np
import time
from collections import defaultdict
from matplotlib import pyplot as plt
import json

class tandem_queue_model:
	def __init__(self, params, Z):
		self.lamda, self.mu_1, self.mu_2 = params
		self.Z = Z
		self.counter = defaultdict(list)
		self.level_cost = defaultdict(list)

	def poisson_arrival(self, T):
		arrival_times = set()
		_arrival_time = 0
		while True:
			#Get the next probability value from Uniform(0,1)
			p = np.random.uniform(0,1,1)[0]

			#Plug it into the inverse of the CDF of Exponential(_lamnbda)
			_inter_arrival_time = -np.log(p)/self.lamda

			#Add the inter-arrival time to the running sum
			_arrival_time = _arrival_time + _inter_arrival_time

			arrival_times.add(int(_arrival_time))

			if _arrival_time > T:
				break

		return arrival_times

	def simulate(self, T, V, state, plot=False):
		q1_sequence = []
		q2_sequence = []

		if T == 0:
			return q1_sequence, q2_sequence, 0
		
		arrival_instants = self.poisson_arrival(T)
		deque_1_time = int(np.random.exponential(self.mu_1,1)[0])+1
		deque_2_time = int(np.random.exponential(self.mu_2,1)[0])+1

		queue_1, queue_2 = state

		start = time.time()
		for t in range(T):
		
			# join queue 1
			if t in arrival_instants:
				queue_1 += 1
			if queue_1 == 0:
				deque_1_time += 1
			# quit queue 1 and join queue 2
			if queue_1 > 0 and t == deque_1_time:
				queue_1 -= 1
				queue_2 += 1
				deque_1_time += int(np.random.exponential(self.mu_1,1)[0])+1
			# quit queue 2
			if queue_2 == 0:
				deque_2_time += 1
			if queue_2 > 0 and t == deque_2_time:
				queue_2 -= 1
				deque_2_time += int(np.random.exponential(self.mu_2,1)[0])+1

			q1_sequence.append(queue_1)
			q2_sequence.append(queue_2)

			if queue_2 >= V:
				break
		end = time.time()

		if plot:
			print(q2_sequence)
			plt.plot(q2_sequence, color='black', label='Queue 2')
			plt.hlines(V, 0, T, color='r')
			plt.legend()
			plt.show()

		return q1_sequence, q2_sequence, end-start

	# SRS for given confidence interval
	def SRS(self, T, V, ci_threshold):
		total_cost = 0
		total_time = 0
		estimate = 0
		ci = 1
		samples = []

		history = []
		while ci > ci_threshold or estimate == 0 or estimate == 1 or ci == 0:
			for i in range(10):
				s1, s2, simulation_time = self.simulate(T, V, (0,0))
				t = len(s2)
				total_cost += t
				total_time += simulation_time
				# if t > 0 and s2[-1] >= V:
				if t >= T:
					samples.append(0)
				else:
					samples.append(1)

			estimate = np.mean(samples)
			ci = self.Z * np.sqrt((estimate*(1-estimate))/len(samples))
			
			history.append((estimate, ci, total_cost, total_time))

		return history
	
	# SRS for given relative error
	def SRS_v2(self, T, V, ground_truth, relative_error):
		total_cost = 0
		total_time = 0
		estimate = 0
		ci = 1
		samples = []

		history = []
		while True:
			for i in range(10):
				s1, s2, simulation_time = self.simulate(T, V, (0,0))
				t = len(s2)
				total_cost += t
				total_time += simulation_time
				# if t > 0 and s2[-1] >= V:
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

	def MLSS_dfs(self, T, V, state, idx, splits, boundaries):
		s1, s2, t = state
		cost = 0
		time = 0
		hits = 0
		success_cnt = 0
		for i in range(splits):
			if idx == len(boundaries)-1:
				q1, q2, simulation_time = self.simulate(T - t, boundaries[idx], (s1, s2))
				cost += len(q2)
				self.level_cost[idx].append(len(q2))
				#print(idx, t, q2)
				time += simulation_time
				if len(q2) > 0 and q2[-1] >= boundaries[idx]:
					hits += 1
					success_cnt += 1
			else:
				q1, q2, simulation_time = self.simulate(T - t, boundaries[idx], (s1, s2))
				cost += len(q2)
				self.level_cost[idx].append(len(q2))
				time += simulation_time
				#print(idx, t, q2)
				if len(q2) > 0 and q2[-1] >= boundaries[idx]:
					success_cnt += 1
					c,h,st = self.MLSS_dfs(T, V, (q1[-1], q2[-1], t+len(q2)), idx+1, splits, boundaries)
					cost += c
					time += st
					hits += h
		self.counter[idx].append(success_cnt)
		return cost, hits, time

	# simulation of one root path using MLSS
	def MLSS_root_path(self, T, V, splits, boundaries):
		total_cost = 0
		total_time = 0
		total_hits = 0

		idx = 0
		self.counter[idx].append(1)
		s1, s2, simulation_time = self.simulate(T, boundaries[idx], (0,0))
		t = len(s2)
		total_cost += t
		self.level_cost[idx].append(t)
		total_time += simulation_time
		
		if s2[-1] >= boundaries[idx]:
			c, h, st = self.MLSS_dfs(T, V, (s1[-1], s2[-1], t), idx+1, splits, boundaries)
			total_cost += c
			total_time += st
			total_hits += h

		return total_cost, total_hits, total_time

	# MLSS simulation for given confidence interval
	def MLSS(self, T, V, splits, v_boundaries, ci_threshold):
		self.counter.clear()
		self.level_cost.clear()

		total_cost = 0
		total_time = 0
		estimate = 1
		ci = 1

		m = len(v_boundaries)

		target_hits = 0
		root_paths = 0

		root_path_hits = []

		history = []
		while ci > ci_threshold or estimate == 0 or estimate == 1 or ci == 0:
			for i in range(5):
				c,h,st = self.MLSS_root_path(T, V, splits, v_boundaries)

				target_hits += h
				total_cost += c
				total_time += st
				root_path_hits.append(h)
				root_paths += 1

			# estimation
			estimate = target_hits / (len(root_path_hits) * splits ** (m-1))
			estimated_var = np.var(root_path_hits) / (len(root_path_hits) * splits ** (2*(m-1)))
			ci = self.Z * np.sqrt(estimated_var)
			history.append((estimate, ci, total_cost, total_time))
		# calculate average path cost and root-path-hits variance
		avg_path_cost = total_cost / len(root_path_hits)
		plan_eval = avg_path_cost * np.var(root_path_hits) / splits ** (2*(m-1))
		history.append((estimate, ci, total_cost, total_time, plan_eval))
		return history
	
	# MLSS simulation for given relative error
	def MLSS_v2(self, T, V, splits, v_boundaries, ground_truth, relative_error):
		self.counter.clear()
		self.level_cost.clear()

		total_cost = 0
		total_time = 0
		estimate = 1
		ci = 1

		m = len(v_boundaries)

		target_hits = 0
		root_paths = 0

		root_path_hits = []

		history = []
		while True:
			for i in range(5):
				c,h,st = self.MLSS_root_path(T, V, splits, v_boundaries)

				target_hits += h
				total_cost += c
				total_time += st
				root_path_hits.append(h)
				root_paths += 1

			# estimation
			estimate = target_hits / (len(root_path_hits) * splits ** (m-1))
			var = np.sqrt(np.var(root_path_hits) / (len(root_path_hits) * splits ** (2*(m-1))))
			history.append((estimate, var/ground_truth, total_cost, total_time))

			if var/ground_truth > 0  and var/ground_truth <= relative_error:
				break
		return history
	
	# simulation effiency comparison between SRS and MLSS
	def rare_event_efficiency(self, T, V, splits, v_boundaries, relative_error, ground_truth=None):
		m = len(v_boundaries)
		if ground_truth == None:
			samples = []
			for i in range(1000):
				c,h,st = self.MLSS_root_path(T, V, splits, v_boundaries)
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
		
		while True:
			for i in range(10):
				s1, s2, simulation_time = self.simulate(T, V, (0,0))
				t = len(s2)
				srs_total_cost += t
				srs_total_time += simulation_time
				if t >= T:
					samples.append(0)
				else:
					samples.append(1)
			estimate = np.mean(samples)
			var = np.sqrt(estimate*(1-estimate) / len(samples))
			srs_history.append((estimate, var/ground_truth, srs_total_cost, srs_total_time))
			if len(srs_history) % 500 == 0:
				print('srs:', srs_history[-1])
			if var/ground_truth > 0  and var/ground_truth <= relative_error:
				break
			
		while True:
			for i in range(5):
				c,h,st = self.MLSS_root_path(T, V, splits, v_boundaries)

				target_hits += h
				mlss_total_cost += c
				mlss_total_time += st
				root_path_hits.append(h)

			estimate = target_hits / (len(root_path_hits) * splits ** (m-1))
			var = np.sqrt(np.var(root_path_hits) / (len(root_path_hits) * splits ** (2*(m-1))))
			mlss_history.append((estimate, var/ground_truth, mlss_total_cost, mlss_total_time))
			if len(mlss_history) % 100 == 0:
				print('mlss,', mlss_history[-1])
			if var/ground_truth > 0  and var/ground_truth <= relative_error:
				break
		
		return srs_history, mlss_history

def tandem_queue_hybrid_test():
	tandem_queue = tandem_queue_model((0.5, 2, 2), 1.96)
	T = 500
	V = 26
	splits = 3

	v_splits = [12, 19, 26]
	print(v_splits)

	srs = tandem_queue.SRS(T, V, 0.01)
	print(srs[-1])
	
	mlss = tandem_queue.MLSS(T, V, splits, v_splits, 0.01)
	print(mlss[-1])

	for idx in tandem_queue.counter:
		if idx == 0:
			print('{}->{}'.format(0, v_splits[idx]), len(tandem_queue.counter[idx+1]) / len(tandem_queue.counter[idx]), len(tandem_queue.counter[idx]))
			print('cost:',np.mean(tandem_queue.level_cost[idx]))
		else:
			print('{}->{}'.format(v_splits[idx-1], v_splits[idx]), np.sum(tandem_queue.counter[idx]) / (len(tandem_queue.counter[idx]) * splits), \
			len(tandem_queue.counter[idx]), np.sum(tandem_queue.counter[idx]))
			print('cost:',np.mean(tandem_queue.level_cost[idx]))

def tandem_queue_avg_test():
	tandem_queue = tandem_queue_model((0.5, 2, 2), 1.96)
	ground_truth = 0.0016
	# ground_truth = 1-0.8821480374423912
	relative_error = 0.1
	T = 500
	V = 40
	splits = 3
	# v_splits = [13, 17, 21, 24, 26]
	# v_splits = [13, 18, 22, 26]
	# v_splits = [19, 26]
	# v_splits = [15, 21, 26]
	# v_splits = [16, 26]
	# v_splits = [13, 16, 20]
	# v_splits = [16, 22]
	# v_splits = [13, 16, 19, 22]
	# v_splits = [14, 18, 22]
	# v_splits = [12, 15, 17, 20, 22]

	# v_splits = [12.882426409856528, 14.244729227125239, 22.794850376007133, 26]

	# v_splits = [28, 40]
	# v_splits = [22, 32, 40]
	# v_splits = [18, 26, 33, 40]
	# v_splits = [18, 24, 30, 35, 40]
	# v_splits = [16, 21, 26, 31, 36, 40]
	# v_splits = [15, 21, 25, 29, 32, 36, 40]
	v_splits = [14, 19, 23, 27, 30, 33, 37, 40]
	
	print(v_splits, ground_truth, relative_error)

	srs_avg = []
	mlss_avg = []

	for i in range(100):
		if i % 10 == 0:
			print('\r{}'.format(i), end='')
		# srs = tandem_queue.SRS(T, V, 0.01)
		# srs_avg.append(srs[-1])
		
		# mlss = tandem_queue.MLSS(T, V, splits, v_splits, 0.01)
		mlss = tandem_queue.MLSS_v2(T, V, splits, v_splits, ground_truth, relative_error)
		# if np.abs(mlss[-1][0] - ground_truth) <= 0.01:
		mlss_avg.append(mlss[-1])
		if i == 0:
			for idx in tandem_queue.counter:
				if idx == 0:
					print('{}->{}'.format(0, v_splits[idx]), len(tandem_queue.counter[idx+1]) / len(tandem_queue.counter[idx]), len(tandem_queue.counter[idx]))
					print('cost:',np.mean(tandem_queue.level_cost[idx]))
				else:
					print('{}->{}'.format(v_splits[idx-1], v_splits[idx]), np.sum(tandem_queue.counter[idx]) / (len(tandem_queue.counter[idx]) * splits), \
					len(tandem_queue.counter[idx]), np.sum(tandem_queue.counter[idx]))
					print('cost:',np.mean(tandem_queue.level_cost[idx]))
	
	# print('srs: {}/{}, {}/{}, {}/{}'.format(np.mean([item[0] for item in srs_avg]), np.std([item[0] for item in srs_avg]), \
	# 									np.mean([item[2] for item in srs_avg]), np.std([item[2] for item in srs_avg]), \
	# 									np.mean([item[3] for item in srs_avg]), np.std([item[3] for item in srs_avg])))
	print('mlss: {}/{}, {}/{}, {}/{}'.format(np.mean([item[0] for item in mlss_avg]), np.std([item[0] for item in mlss_avg]), \
										np.mean([item[2] for item in mlss_avg]), np.std([item[2] for item in mlss_avg]), \
										np.mean([item[3] for item in mlss_avg]), np.std([item[3] for item in mlss_avg])))

def tandem_queue_balance_growth():
	tandem_queue = tandem_queue_model((0.5, 2, 2), 1.96)
	# ground_truth = 0.05211185375428542
	ground_truth = 1-0.8821480374423912

	T = 500
	V = 26
	splits = 3

	v_s1 = [22, 26]
	v_s2 = [16, 26]
	v_s3 = [19, 26]
	v_s4 = [10, 26]

	# v_s1 = [15, 21, 26]
	# v_s2 = [15, 18, 26]
	# v_s3 = [15, 24, 26]
	# v_s4 = [10, 21, 26]

	print(v_s1, v_s2, v_s3, v_s4)

	mlss1_avg = []
	mlss2_avg = []
	mlss3_avg = []
	mlss4_avg = []

	for i in range(100):
		if i % 10 == 0:
			print('\r{}'.format(i), end='')
		
		mlss = tandem_queue.MLSS(T, V, splits, v_s1, 0.01)
		if np.abs(mlss[-1][0] - ground_truth) <= 0.01:
			mlss1_avg.append(mlss[-1])
		mlss = tandem_queue.MLSS(T, V, splits, v_s2, 0.01)
		if np.abs(mlss[-1][0] - ground_truth) <= 0.01:
			mlss2_avg.append(mlss[-1])
		mlss = tandem_queue.MLSS(T, V, splits, v_s3, 0.01)
		if np.abs(mlss[-1][0] - ground_truth) <= 0.01:
			mlss3_avg.append(mlss[-1])
		mlss = tandem_queue.MLSS(T, V, splits, v_s4, 0.01)
		if np.abs(mlss[-1][0] - ground_truth) <= 0.01:
			mlss4_avg.append(mlss[-1])
	
	print('{}: {}/{}, {}/{}, {}/{}'.format(v_s1, np.mean([item[0] for item in mlss1_avg]), np.std([item[0] for item in mlss1_avg]), \
										np.mean([item[2] for item in mlss1_avg]), np.std([item[2] for item in mlss1_avg]), \
										np.mean([item[3] for item in mlss1_avg]), np.std([item[3] for item in mlss1_avg])))
	print('{}: {}/{}, {}/{}, {}/{}'.format(v_s2, np.mean([item[0] for item in mlss2_avg]), np.std([item[0] for item in mlss2_avg]), \
										np.mean([item[2] for item in mlss2_avg]), np.std([item[2] for item in mlss2_avg]), \
										np.mean([item[3] for item in mlss2_avg]), np.std([item[3] for item in mlss2_avg])))
	print('{}: {}/{}, {}/{}, {}/{}'.format(v_s3, np.mean([item[0] for item in mlss3_avg]), np.std([item[0] for item in mlss3_avg]), \
										np.mean([item[2] for item in mlss3_avg]), np.std([item[2] for item in mlss3_avg]), \
										np.mean([item[3] for item in mlss3_avg]), np.std([item[3] for item in mlss3_avg])))
	print('{}: {}/{}, {}/{}, {}/{}'.format(v_s4, np.mean([item[0] for item in mlss4_avg]), np.std([item[0] for item in mlss4_avg]), \
										np.mean([item[2] for item in mlss4_avg]), np.std([item[2] for item in mlss4_avg]), \
										np.mean([item[3] for item in mlss4_avg]), np.std([item[3] for item in mlss4_avg])))


def relative_error_test():
	tandem_queue = tandem_queue_model((0.5, 2, 2), 1.96)
	splits = 3
	T = 500
	V = 45
	v_splits = [15.017152638593565, 39.343153880124746, 45]
	relative_error = 0.1
	ground_truth = 0.0004

	print(v_splits, ground_truth, relative_error)

	mlss_history = []

	for i in range(10):
		mlss = tandem_queue.MLSS_v2(T, V, splits, v_splits, ground_truth, relative_error)
		mlss_history.append(mlss[-1])
	
	print('mlss: {}/{}, {}/{}, {}/{}'.format(np.mean([item[0] for item in mlss_history]), np.std([item[0] for item in mlss_history]), \
										np.mean([item[2] for item in mlss_history]), np.std([item[2] for item in mlss_history]), \
										np.mean([item[3] for item in mlss_history]), np.std([item[3] for item in mlss_history])))

def rare_event_test():
	tandem_queue = tandem_queue_model((0.5, 2, 2), 1.96)
	splits = 3
	T = 500
	# V = 40
	# v_splits = [20, 30, 40]
	V = 30
	v_splits = [11.932329404360516, 17.501181000065912, 25.890728884628423, 30]
	relative_error = 0.1
	srs_history, mlss_history = \
		tandem_queue.rare_event_efficiency(T, V, splits, v_splits, relative_error)
	
	# print('srs:', srs_history[-1])
	# print('mlss:', mlss_history[-1])

	# return srs_history, mlss_history
	print(srs_history)
	print(mlss_history)

def estimation_trace_test():
	tandem_queue = tandem_queue_model((0.5, 2, 2), 1.96)
	splits = 3
	T = 500
	# V = 40
	# v_splits = [20, 30, 40]
	V = 30
	v_splits = [11.932329404360516, 17.501181000065912, 25.890728884628423, 30]
	
	srs = tandem_queue.SRS(T, V, 0.01)
	mlss = tandem_queue.MLSS(T, V, splits, v_splits, 0.01)

	# srs = tandem_queue.SRS_v2(T, V, 0.001, 0.1)
	# mlss = tandem_queue.MLSS_v2(T, V, splits, v_splits, 0.001, 0.1)

	# print('====SRS====')
	# print(srs)
	# print('====MLSS====')
	# print(mlss)

	with open('data/queue-{}-{}-srs.json'.format(T, V), 'w') as f:
		json.dump(srs, f)
	with open('data/queue-{}-{}-mlss.json'.format(T, V), 'w') as f:
		json.dump(mlss, f)

if __name__ == '__main__':
	# tandem_queue_model_test()
	# tandem_queue_split_by_time_test()
	# tandem_queue_hybrid_test()
	tandem_queue_avg_test()
	# tandem_queue_balance_growth()
	# rare_event_test()
	# estimation_trace_test()
	# relative_error_test()




