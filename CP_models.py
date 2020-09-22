# Compound Poisson (CP) process, typical insurance risk management
import random
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import time
import json

class cp_model:
	# U(t) = u + c*t - S(t), where S(t) is a compound poisson process
	# with uniform jump distribution [lo, hi]
	def __init__(self, params, c, Z):
		self.lamda, self.lo, self.hi = params
		self.c = c
		self.Z = Z
		# maintain the intermidiate counters for each level
		self.counter = defaultdict(list)
	
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

	def simulate(self, T, V, state):
		supply, demand = state

		sequence = []

		if T == 0:
			return sequence, (supply, demand), 0
		
		arrival_instants = self.poisson_arrival(T)

		start = time.time()
		for t in range(T):
			supply += self.c
			if t in arrival_instants:
				demand += np.random.uniform(self.lo, self.hi, 1)[0]

			sequence.append(supply - demand)

			if supply - demand >= V:
				break
		end = time.time()

		return sequence, (supply, demand), end-start

	# SRS for given confidence interval
	def SRS(self, T, V, state, ci_threshold):
		total_cost = 0
		total_time = 0
		estimate = 0
		ci = 1
		samples = []

		history = []
		while ci > ci_threshold or estimate == 0 or estimate == 1 or ci == 0:
			for i in range(10):
				s, _, simulation_time = self.simulate(T, V, state)
				t = len(s)
				total_cost += t
				total_time += simulation_time
				# if t > 0 and s2[-1] >= V:
				if t >= T:
					# not ruin
					samples.append(0)
				else:
					# ruin
					samples.append(1)

			estimate = np.mean(samples)
			ci = self.Z * np.sqrt((estimate*(1-estimate))/len(samples))
			
			history.append((estimate, ci, total_cost, total_time))

		return history
	
	# SRS for given relative error
	def SRS_v2(self, T, V, state, ground_truth, relative_error):
		total_cost = 0
		total_time = 0
		estimate = 0
		ci = 1
		samples = []

		history = []
		while True:
			for i in range(10):
				s, _, simulation_time = self.simulate(T, V, state)
				t = len(s)
				total_cost += t
				total_time += simulation_time
				# if t > 0 and s2[-1] >= V:
				if t >= T:
					# not ruin
					samples.append(0)
				else:
					# ruin
					samples.append(1)
			
			estimate = np.mean(samples)
			var = np.sqrt(estimate*(1-estimate) / len(samples))
			history.append((estimate, var/ground_truth, total_cost, total_time))
			if var/ground_truth > 0  and var/ground_truth <= relative_error:
				break
		return history

	def MLSS_hybrid_dfs(self, T, V, state, idx, splits, v_boundaries):
		s, t = state
		cost = 0
		hits = 0
		time = 0
		success_cnt = 0
		for i in range(splits):
			if idx == len(v_boundaries)-1:
				# q, _ = self.simulate(np.sum(t_boundaries)-t, v_boundaries[idx], s)
				q, last_state, simulation_time = self.simulate(T-t, v_boundaries[idx], s)
				cost += len(q)
				time += simulation_time
				#print('{}-{}-{}-{}-{}'.format(idx, np.sum(t_boundaries)-t, v_boundaries[idx], len(q2), q2[-1]))
				# if t+len(q) >= np.sum(t_boundaries):
				if len(q) > 0 and q[-1] >= v_boundaries[idx]:
					hits += 1
					success_cnt += 1
			else:
				# q, last_state = self.simulate(np.sum(t_boundaries[:idx+1])-t, v_boundaries[idx], s)
				q, last_state, simulation_time = self.simulate(T-t, v_boundaries[idx], s)
				cost += len(q)
				time += simulation_time
				# if q[-1] >= V:
				# 	continue
				# if t + len(q) >= T:
				# 	continue
				if len(q) > 0 and q[-1] >= v_boundaries[idx]:
					success_cnt += 1
					c,h,st = self.MLSS_hybrid_dfs(T, V, (last_state, t+len(q)), idx+1, \
						splits, v_boundaries)
					cost += c
					hits += h
					time += st
		self.counter[idx].append(success_cnt)
		return cost, hits, time

	def MLSS_hybrid_root_path(self, T, V, splits, initial_state, v_boundaries):
		total_cost = 0
		total_hits = 0
		total_time = 0

		idx = 0
		self.counter[idx].append(1)
		# s, last_state = self.simulate(t_boundaries[idx], v_boundaries[idx], initial_state)
		s, last_state, simulation_time = self.simulate(T, v_boundaries[idx], initial_state)
		t = len(s)
		total_cost += t
		total_time += simulation_time
		# if s[-1] >= V:
		# 	return total_cost, total_hits

		# if t >= T:
		# 	return total_cost, 1

		if s[-1] >= v_boundaries[idx]:
			c, h, st = self.MLSS_hybrid_dfs(T, V, (last_state,t), idx+1, splits, v_boundaries)
			total_cost += c
			total_hits += h
			total_time += st

		return total_cost, total_hits, total_time

	# MLSS for given confidence interval
	def MLSS_hybrid(self, T, V, splits, initial_state, v_boundaries, ci_threshold):
		self.counter.clear()
		
		total_cost = 0
		total_time = 0
		estimate = 1
		ci = 1

		m = len(v_boundaries)

		root_paths = 0
		root_path_hits = []

		history = []

		while ci > ci_threshold or estimate == 0 or estimate == 1 or ci == 0:
			for i in range(5):
				root_paths += 1
				c, h, st = self.MLSS_hybrid_root_path(T, V, splits, initial_state, v_boundaries)
				total_cost += c
				total_time += st
				root_path_hits.append(h)

			# estimation
			estimated_var = 0

			estimate = np.sum(root_path_hits) / (len(root_path_hits) * splits ** (m-1))
			estimated_var = np.var(root_path_hits) / (len(root_path_hits) * splits ** (2*(m-1)))

			ci = self.Z * np.sqrt(estimated_var)
			history.append((estimate, ci, total_cost, total_time))
		# calculate average path cost and root-path-hits variance
		avg_path_cost = total_cost / len(root_path_hits)
		plan_eval = avg_path_cost * np.var(root_path_hits) / splits ** (2*(m-1))
		history.append((estimate, ci, total_cost, total_time, plan_eval))
		return history
	

	# MLSS for given relative error
	def MLSS_v2(self, T, V, splits, initial_state, v_boundaries, ground_truth, relative_error):
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
			for i in range(5):
				root_paths += 1
				c, h, st = self.MLSS_hybrid_root_path(T, V, splits, initial_state, v_boundaries)
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
	def rare_event_efficiency(self, T, V, splits, v_boundaries, initial_state, relative_error, ground_truth=None):
		m = len(v_boundaries)
		if ground_truth == None:
			samples = []
			for i in range(1000):
				c,h,st = self.MLSS_hybrid_root_path(T, V, splits, initial_state, v_boundaries)
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
				s, _, simulation_time = self.simulate(T, V, initial_state)
				t = len(s)
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
		# MLSS	
		while True:
			for i in range(5):
				c,h,st = self.MLSS_hybrid_root_path(T, V, splits, initial_state, v_boundaries)

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

T = 500
V = 350
payment = 4.5
t_splits = [500, 0]
# v_splits = [220, 265, 310, 350]
# v_splits = [220, 350]
v_splits = [200, 308, 350]
splits = 3


def model_test():
	model = cp_model((0.8, 5, 10), payment, 1.96)
	for i in range(10):
		s,_ = model.simulate(500, V, (15, 0))
		plt.plot(s)
	plt.show()

def SRS_test():
	# choose 4 or 5
	model = cp_model((0.8, 5, 10), payment, 1.96)
	srs = model.SRS(T, V, (15, 0) ,0.01)
	print(srs[-1])

def MLSS_test():
	model = cp_model((0.8, 5, 10), payment, 1.96)
	mlss = model.MLSS_hybrid(T, V, splits, (15, 0), v_splits, 0.01)
	print(v_splits, mlss[-1])

	tau = mlss[-1][0]
	# p1 = len(model.counter[1]) / len(model.counter[0])
	# p2 = np.sum(model.counter[1]) / (len(model.counter[1]) * splits)
	# var_pi = np.var([v/splits for v in model.counter[1]])
	# avg_path_cost = mlss[-1][2] / len(model.counter[0])

	for idx in model.counter:
		if idx == 0:
			print('{}->{}'.format(0, v_splits[idx]), len(model.counter[idx+1]) / len(model.counter[idx]), len(model.counter[idx]))
		else:
			print('{}->{}'.format(v_splits[idx-1], v_splits[idx]), np.sum(model.counter[idx]) / (len(model.counter[idx]) * splits), \
				len(model.counter[idx]), np.sum(model.counter[idx]))
		# print(idx, len(model.counter[idx]))
		# if idx > 0:
		# 	print('sum',np.sum(model.counter[idx]))
		# 	print('p1:', p1)
		# 	#print('p1 optimal',np.sqrt(mlss[-1][0]**2 / np.var([v/splits for v in model.counter[idx]])))
		# 	print('p2:', p2)
		# 	#print('p2 optinmal',np.sqrt(np.var([v/splits for v in model.counter[idx]])))
		# 	print(tau*p2 + p1*var_pi, avg_path_cost, avg_path_cost * (tau*p2 + p1*var_pi))
			
			

def counter_test():
	model = cp_model((0.8, 5, 10), payment, 1.96)
	for i in range(10):
		model.MLSS_hybrid_root_path(T, V, splits, (15, 0), v_splits)
	
	for idx in model.counter:
		print(idx, len(model.counter[idx]))
		print(model.counter[idx])

def avg_test():
	# ground_truth = 0.05406737112202787
	ground_truth = 0.003
	relative_error = 0.1
	T = 500
	V = 450
	payment = 4.5
	# v_splits = [220, 270, 315, 350]
	# v_splits = [275, 350]
	# v_splits = [240, 300, 350]
	# v_splits = [210, 250 , 285, 320, 350]
	# v_splits = [250, 350]

	# v_splits = [240, 300]
	# v_splits = [210, 260, 300]
	# v_splits = [205, 245, 270, 300]
	# v_splits = [200, 242, 265, 283, 300]

	# v_splits = [202.98251384158772, 252.68688174852014, 350]

	# v_splits = [340, 450]
	# v_splits = [300, 385, 450]
	# v_splits = [280, 350, 410, 450]
	# v_splits = [260, 320, 370, 410, 450]
	# v_splits = [240, 300, 340, 390, 420, 450]
	# v_splits = [220, 260, 300, 340, 380, 420, 450]
	v_splits = [200, 250, 290, 330, 370, 400, 430, 450]

	splits = 3

	print(v_splits, ground_truth, relative_error)

	model = cp_model((0.8, 5, 10), payment, 1.96)

	srs_avg = []
	mlss_avg = []

	for i in range(100):
		if i % 10 == 0:
			print('\r{}'.format(i), end='')
		# srs = model.SRS(T, V, (15, 0) ,0.01)
		# srs_avg.append(srs[-1])
		
		# mlss = model.MLSS_hybrid(T, V, splits, (15, 0), v_splits, 0.01)
		mlss = model.MLSS_v2(T, V, splits, (15,0), v_splits, ground_truth, relative_error)
		# if np.abs(mlss[-1][0] - ground_truth) <= 0.01:
		mlss_avg.append(mlss[-1])
		
		if i == 0:
			for idx in model.counter:
				if idx == 0:
					print('{}->{}'.format(0, v_splits[idx]), len(model.counter[idx+1]) / len(model.counter[idx]), len(model.counter[idx]))

				else:
					print('{}->{}'.format(v_splits[idx-1], v_splits[idx]), np.sum(model.counter[idx]) / (len(model.counter[idx]) * splits), \
					len(model.counter[idx]), np.sum(model.counter[idx]))
	
	# print('srs: {}/{}, {}/{}, {}/{}'.format(np.mean([item[0] for item in srs_avg]), np.std([item[0] for item in srs_avg]), \
	# 									np.mean([item[2] for item in srs_avg]), np.std([item[2] for item in srs_avg]), \
	# 									np.mean([item[3] for item in srs_avg]), np.std([item[3] for item in srs_avg])))
	print('mlss: {}/{}, {}/{}, {}/{}'.format(np.mean([item[0] for item in mlss_avg]), np.std([item[0] for item in mlss_avg]), \
										np.mean([item[2] for item in mlss_avg]), np.std([item[2] for item in mlss_avg]), \
										np.mean([item[3] for item in mlss_avg]), np.std([item[3] for item in mlss_avg])))

def balance_growth():
	T = 500
	V = 300
	# ground_truth = 0.05406737112202787
	ground_truth = 1-0.84479517498086
	payment = 4.5
	splits = 3

	model = cp_model((0.8, 5, 10), payment, 1.96)

	# v_s1 = [275, 350]
	# v_s2 = [300, 350]
	# v_s3 = [250, 350]
	# v_s4 = [200, 350]

	# v_s1 = [240, 300, 350]
	# v_s2 = [240, 320, 350]
	# v_s3 = [200, 300, 350]
	# v_s4 = [200, 320, 350]

	print(v_s1, v_s2, v_s3, v_s4)

	mlss1_avg = []
	mlss2_avg = []
	mlss3_avg = []
	mlss4_avg = []

	for i in range(100):
		if i % 10 == 0:
			print('\r{}'.format(i), end='')
		
		mlss = model.MLSS_hybrid(T, V, splits, (15, 0), v_s1, 0.01)
		if np.abs(mlss[-1][0] - ground_truth) <= 0.01:	
			mlss1_avg.append(mlss[-1])
		mlss = model.MLSS_hybrid(T, V, splits, (15, 0), v_s2, 0.01)
		if np.abs(mlss[-1][0] - ground_truth) <= 0.01:
			mlss2_avg.append(mlss[-1])
		mlss = model.MLSS_hybrid(T, V, splits, (15, 0), v_s3, 0.01)
		if np.abs(mlss[-1][0] - ground_truth) <= 0.01:
			mlss3_avg.append(mlss[-1])
		mlss = model.MLSS_hybrid(T, V, splits, (15, 0), v_s4, 0.01)
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

def rare_event_test():
	payment = 4.5
	splits = 3
	initial_state = (15, 0)
	relative_error = 0.1

	T = 500
	V = 500
	v_splits = [229.1493839704837, 341.6575309961904, 400.39481498610525, 500]
	# V = 450
	# v_splits = [156.41417460238358, 251.7845500767113, 332.35556145945907, 450]

	model = cp_model((0.8, 5, 10), payment, 1.96)
	model.rare_event_efficiency(T, V, splits, v_splits, initial_state, relative_error)

def relative_error_test():
	splits = 3
	T = 500
	V = 500
	v_splits = [284.4565775733977, 458.8749691981446, 500]
	payment = 4.5
	
	model = cp_model((0.8, 5, 10), payment, 1.96)
	ground_truth = 0.0003
	relative_error = 0.1
	mlss_history = []

	print(v_splits, ground_truth, relative_error)

	for i in range(10):
		mlss = model.MLSS_v2(T, V, splits, (15,0), v_splits, ground_truth,relative_error)
		mlss_history.append(mlss[-1])
	
	print('mlss: {}/{}, {}/{}, {}/{}'.format(np.mean([item[0] for item in mlss_history]), np.std([item[0] for item in mlss_history]), \
										np.mean([item[2] for item in mlss_history]), np.std([item[2] for item in mlss_history]), \
										np.mean([item[3] for item in mlss_history]), np.std([item[3] for item in mlss_history])))
def estimation_trace_test():
	
	splits = 3
	T = 500
	V = 500
	v_splits = [240, 300, 400, 500]
	payment = 4.5
	
	model = cp_model((0.8, 5, 10), payment, 1.96)
	
	# srs = model.SRS(T, V, (15, 0), 0.01)
	# mlss = model.MLSS_hybrid(T, V, splits, (15, 0), v_splits, 0.01)

	srs = model.SRS_v2(T, V, (15,0), 0.0003, 0.1)
	mlss = model.MLSS_v2(T, V, splits, (15,0), v_splits, 0.0003, 0.1)

	with open('data/cp-{}-{}-srs.json'.format(T, V), 'w') as f:
		json.dump(srs, f)
	with open('data/cp-{}-{}-mlss.json'.format(T, V), 'w') as f:
		json.dump(mlss, f)

if __name__ == '__main__':
	# model_test()
	# SRS_test()
	# MLSS_test()
	# counter_test()
	avg_test()
	# balance_growth()
	# rare_event_test()
	# estimation_trace_test()
	# relative_error_test()