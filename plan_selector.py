from queue_models import *
from ar_models import *
from ml_models import *
from CP_models import *
import operator
import argparse
import itertools
import math

N = 1000000000
WITNESS = 10
EPSILON = 0.2
datafile = 'google_stock.csv'
modelfile = 'model/google_rnn_mdn_2_5_16_30.h5'

class Selector:
	def __init__(self, levels, model_type, model, params, Z, seeds):
		self.levels = levels
		self.model = model
		self.model_type = model_type
		self.params = params
		self.Z = Z
		self.seeds = seeds
		self.weight = np.sqrt(2)
	

	# def initialize_plans(self, Ts, Vs):
	# 	for i in range(self.levels):
	# 		if i % 2 == 0:
	# 			self.counter.append(np.zeros(len(Ts)))
	# 		else:
	# 			self.counter.append(np.zeros(len(Vs)))
	def SRS(self, T, V, ci_threshold):
		if self.model_type == 'cp':
			initial_state = self.params
			srs = self.model.SRS(T, V, initial_state , ci_threshold)
		if self.model_type == 'queue':
			srs = self.model.SRS(T, V, ci_threshold)
		if self.model_type == 'nn':
			transformed_data, raw_data, rnn_mdn = self.params
			srs = self.model.SRS(T, V, transformed_data, raw_data, rnn_mdn, ci_threshold)
		return srs

	def run(self, T, V, t_bounds, v_bounds, ci_threshold, splits, trail_rounds):
		# initiliaze
		counter = {}
		# reward for each plan : 1 / (cost * path_var)
		cost = defaultdict(list)
		path_var = defaultdict(list)

		counter[0] = 0
		counter[1] = defaultdict(int)
		counter[2] = defaultdict(int)

		reward_t = defaultdict(list)
		reward_v = defaultdict(list)

		if self.model_type == 'cp':
			initial_state = self.params
		if self.model_type == 'nn':
			transformed_data, raw_data, rnn_mdn = self.params

		total_cost = 0
		estimate = 1
		ci = 1
		phase = 1

		root_paths = 0
		root_path_hits = []
		history = []

		t_splits_pool = []
		v_splits_pool = []

		t_candidates = itertools.combinations(t_bounds, self.levels-1)
		for item in t_candidates:
			item = list(item)
			item.append(T)
			t_splits_pool.append(item)

		v_candidates = itertools.combinations(v_bounds, self.levels-1)
		for item in v_candidates:
			item = list(item)
			item.append(V)
			v_splits_pool.append(item)

		print(t_splits_pool)
		print(v_splits_pool)

		while ci > ci_threshold or estimate == 0 or estimate == 1 or ci == 0:
			if phase == 1:
				# phase 1 : collection evidence by using round-robin trails for each setting
				print('Trail Round : collection info')
				
				for t_splits in t_splits_pool:
					# transform into time interval
					for i in range(1, len(t_splits)):
						t_splits[i] = t_splits[i] - t_splits[i-1]
					for v_splits in v_splits_pool:
						print('t-splits:{}, v-splits:{}'.format(t_splits, v_splits))
						for i in range(trail_rounds):
							print('\r{}'.format(i), end='')
							if self.model_type == 'cp':
								c, h = self.model.MLSS_hybrid_root_path(T, V, splits, initial_state, t_splits, v_splits)
							if self.model_type == 'queue':
								h, c, _ = self.model.MLSS_box_root_path(T, V, splits, t_splits, v_splits)
							if self.model_type == 'nn':
								c, h  = self.model.MLSS_hybrid_root_path(T, V, transformed_data, raw_data, rnn_mdn, \
									splits, t_splits, v_splits)

							root_paths += 1
							root_path_hits.append(h)
							total_cost += c
							# collect information for UCT-based selection
							path_id = '-'.join([str(t_splits[0]), str(v_splits[0])])
							cost[path_id].append(c)
							path_var[path_id].append(h)
							counter[0] += 1
							counter[1][t_splits[0]] += 1
							counter[2][v_splits[0]] += 1
				# calculate rewards for each partition
				for path in path_var:
					t, v = path.split('-')
					reward_t[int(t)] = N - (np.var(path_var[path]) * np.mean(cost[path]))
					reward_v[float(v)] = N - (np.var(path_var[path]) * np.mean(cost[path]))
				phase = 2
				print('reward T:', reward_t)
				print('reward V:', reward_v)
			elif phase == 2:
			# phase 2 : UCT algorithm to select best plan
				print('UCT Selection Round')
				best_t_splits = defaultdict(float)
				best_v_splits = defaultdict(float)
				# first select t
				for t in counter[1]:
					best_t_splits[t] = reward_t[t] * self.weight * np.sqrt(np.log(counter[0]) / counter[1][t])
				best_t = max(best_t_splits.items(), key=operator.itemgetter(1))[0]
				# then select v
				for v in counter[2]:
					best_v_splits[v] = reward_v[v] * self.weight * np.sqrt(np.log(counter[1][best_t]) / counter[2][v])
				best_v = max(best_v_splits.items(), key=operator.itemgetter(1))[0]

				# use the selection to run a few simulations
				t_splits = [best_t, T-best_t]
				v_splits = [best_v, V]
				path_id = '-'.join([str(best_t), str(best_v)])
				print('best selection:', t_splits, v_splits)
				for i in range(5):
					if self.model_type == 'cp':
						c, h = self.model.MLSS_hybrid_root_path(T, V, splits, initial_state, t_splits, v_splits)
					if self.model_type == 'queue':
						h, c, _ = self.model.MLSS_box_root_path(T, V, splits, t_splits, v_splits)
					if self.model_type == 'nn':
						c, h  = self.model.MLSS_hybrid_root_path(T, V, transformed_data, raw_data, rnn_mdn, \
							splits, t_splits, v_splits)

					root_paths += 1
					root_path_hits.append(h)
					total_cost += c
					# collect information for UCT-based selection
					
					cost[path_id].append(c)
					path_var[path_id].append(h)
					counter[0] += 1
					counter[1][best_t] += 1
					counter[2][best_v] += 1

				# update reward
				reward_t[best_t] = N - (np.var(path_var[path_id]) * np.mean(cost[path_id]))
				reward_v[best_v] = N - (np.var(path_var[path_id]) * np.mean(cost[path_id]))
			else:
				print('should not be here!')
			# repeat until CI is satisfied

			# estimation
			estimate = np.sum(root_path_hits) / (root_paths * splits ** (self.levels-1))
			estimated_var = np.var(root_path_hits) / (root_paths * splits ** (2*(self.levels-1)))
			ci = self.Z * np.sqrt(estimated_var)
			print(estimate, ci, total_cost)
			history.append((estimate, ci, total_cost))

		return history

	def run_v2(self, T, V, t_bounds, v_bounds, ci_threshold, splits, trail_rounds):
		# initiliaze
		counter = defaultdict(int)
		# reward for each plan : 1 / (cost * path_var)
		cost = defaultdict(list)
		path_var = defaultdict(list)

		rewards = defaultdict(float)

		if self.model_type == 'cp':
			initial_state = self.params
		if self.model_type == 'nn':
			transformed_data, raw_data, rnn_mdn = self.params

		total_cost = 0
		estimate = 1
		ci = 1
		phase = 1

		root_paths = 0
		root_path_hits = []
		history = []

		t_splits_pool = []
		v_splits_pool = []

		t_candidates = itertools.combinations(t_bounds, self.levels-1)
		for item in t_candidates:
			item = list(item)
			item.append(T)
			t_splits_pool.append(item)

		v_candidates = itertools.combinations(v_bounds, self.levels-1)
		for item in v_candidates:
			item = list(item)
			item.append(V)
			v_splits_pool.append(item)

		while ci > ci_threshold or estimate == 0 or estimate == 1 or ci == 0:
			if phase == 1:
				# phase 1 : collection evidence by using round-robin trails for each setting
				for t_splits in t_splits_pool:
					# transform into time interval
					for i in range(1, len(t_splits)):
						t_splits[i] = t_splits[i] - t_splits[i-1]
					for v_splits in v_splits_pool:
						for i in range(trail_rounds):
							if self.model_type == 'cp':
								c, h = self.model.MLSS_hybrid_root_path(T, V, splits, initial_state, t_splits, v_splits)
							if self.model_type == 'queue':
								h, c, _ = self.model.MLSS_box_root_path(T, V, splits, t_splits, v_splits)
							if self.model_type == 'nn':
								c, h  = self.model.MLSS_hybrid_root_path(T, V, transformed_data, raw_data, rnn_mdn, \
									splits, t_splits, v_splits)

							root_paths += 1
							root_path_hits.append(h)
							total_cost += c
							# collect information for UCT-based selection
							path_id = '-'.join([str(t_splits[0]), str(v_splits[0])])
							cost[path_id].append(c)
							path_var[path_id].append(h)
							counter[path_id] += 1
				
				# calculate rewards for each partition
				for path in path_var:
					if np.var(path_var[path]) == 0:
						continue
					rewards[path] = np.var(path_var[path]) * np.mean(cost[path])
				
				# second round of filtered candidates
				candidates = dict(sorted(rewards.items(), key=operator.itemgetter(1))[:int(len(rewards)*0.2)])
				
				for path in candidates:
					t, v = path.split('-')
					t = int(t)
					v = float(v)
					t_splits = [t, T-t]
					v_splits = [v, V]
					for i in range(trail_rounds*2):
						# print('\r{}'.format(i), end='')
						if self.model_type == 'cp':
							c, h = self.model.MLSS_hybrid_root_path(T, V, splits, initial_state, t_splits, v_splits)
						if self.model_type == 'queue':
							h, c, _ = self.model.MLSS_box_root_path(T, V, splits, t_splits, v_splits)
						if self.model_type == 'nn':
							c, h  = self.model.MLSS_hybrid_root_path(T, V, transformed_data, raw_data, rnn_mdn, \
								splits, t_splits, v_splits)
						root_paths += 1
						root_path_hits.append(h)
						total_cost += c
						cost[path].append(c)
						path_var[path].append(h)
						counter[path] += 1
				
				# update rewards
				for path in candidates:
					candidates[path] = np.var(path_var[path]) * np.mean(cost[path])
				# print('candidates:', candidates)
				best_plan = min(candidates.items(), key=operator.itemgetter(1))[0]
				print(best_plan, total_cost)
				best_t, best_v = best_plan.split('-')
				best_t = int(best_t)
				best_v = float(best_v)
				# use the selection to run a few simulations
				best_t_splits = [best_t, T-best_t]
				best_v_splits = [best_v, V]
				# print('best selection:', best_t_splits, best_v_splits)
				phase = 2
			elif phase == 2:
			# stick with the best plan
				for i in range(5):
					if self.model_type == 'cp':
						c, h = self.model.MLSS_hybrid_root_path(T, V, splits, initial_state, best_t_splits, best_v_splits)
					if self.model_type == 'queue':
						h, c, _ = self.model.MLSS_box_root_path(T, V, splits, best_t_splits, best_v_splits)
					if self.model_type == 'nn':
						c, h  = self.model.MLSS_hybrid_root_path(T, V, transformed_data, raw_data, rnn_mdn, \
							splits, best_t_splits, best_v_splits)

					root_paths += 1
					root_path_hits.append(h)
					path_var[best_plan].append(h)
					cost[best_plan].append(c)
					total_cost += c
					# collect information for UCT-based selection
			else:
				print('should not be here!')
				break
			# repeat until CI is satisfied

			# estimation
			estimate = np.sum(root_path_hits) / (root_paths * splits ** (self.levels-1))
			estimated_var = np.var(root_path_hits) / (root_paths * splits ** (2*(self.levels-1)))
			ci = self.Z * np.sqrt(estimated_var)
			# print('original estimation:', estimate, ci, total_cost)
			history.append((estimate, ci, total_cost))

		return history
	
	def run_v3(self, T, V, t_bounds, v_bounds, ci_threshold, splits, trail_rounds):
		# linear combination of estimation from different plans
		# initiliaze
		counter = defaultdict(int)
		weight = {}
		# reward for each plan : 1 / (cost * path_var)
		cost = defaultdict(list)
		path_var = defaultdict(list)

		rewards = defaultdict(float)

		if self.model_type == 'cp':
			initial_state = self.params
		if self.model_type == 'nn':
			transformed_data, raw_data, rnn_mdn = self.params

		total_cost = 0
		total_time = 0
		estimate = 1
		ci = 1
		phase = 1

		root_paths = 0
		root_path_hits = []
		history = []

		t_splits_pool = []
		v_splits_pool = []

		t_candidates = itertools.combinations(t_bounds, self.levels-1)
		for item in t_candidates:
			item = list(item)
			item.append(T)
			t_splits_pool.append(item)

		v_candidates = itertools.combinations(v_bounds, self.levels-1)
		for item in v_candidates:
			item = list(item)
			item.append(V)
			v_splits_pool.append(item)

		while ci > ci_threshold or estimate == 0 or estimate == 1 or ci == 0:
			if phase == 1:
				# phase 1 : collection evidence by using round-robin trails for each setting
				for t_splits in t_splits_pool:
					# transform into time interval
					for i in range(1, len(t_splits)):
						t_splits[i] = t_splits[i] - t_splits[i-1]
					for v_splits in v_splits_pool:
						for i in range(trail_rounds):
							# print('\r{}'.format(i), end='')
							
							if self.model_type == 'cp':
								start = time.time()
								c, h = self.model.MLSS_hybrid_root_path(T, V, splits, initial_state, t_splits, v_splits)
								end = time.time()
							if self.model_type == 'queue':
								start = time.time()
								h, c, _ = self.model.MLSS_box_root_path(T, V, splits, t_splits, v_splits)
								end = time.time()
							if self.model_type == 'nn':
								start = time.time()
								c, h  = self.model.MLSS_hybrid_root_path(T, V, transformed_data, raw_data, rnn_mdn, \
									splits, t_splits, v_splits)
								end = time.time()

							total_time += end-start

							root_paths += 1
							root_path_hits.append(h)
							total_cost += c
							# collect information for UCT-based selection
							path_id = '-'.join([str(t_splits[0]), str(v_splits[0])])
							cost[path_id].append(c)
							path_var[path_id].append(h)
							counter[path_id] += 1
				phase = 2
				# update rewards
				for path in path_var:
					rewards[path] = np.var(path_var[path]) * np.mean(cost[path])
				# update weight
				for path in path_var:
					weight[path] = len(path_var[path])
				
			elif phase == 2:
			# choose the current best plan
				best_plan = min(rewards.items(), key=operator.itemgetter(1))[0]
				best_t, best_v = best_plan.split('-')
				best_t = int(best_t)
				best_v = float(best_v)
				# use the selection to run a few simulations
				best_t_splits = [best_t, T-best_t]
				best_v_splits = [best_v, V]
				# print('best selection:', best_t_splits, best_v_splits)
				for i in range(5):
					
					if self.model_type == 'cp':
						start = time.time()
						c, h = self.model.MLSS_hybrid_root_path(T, V, splits, initial_state, best_t_splits, best_v_splits)
						end = time.time()
					if self.model_type == 'queue':
						start = time.time()
						h, c, _ = self.model.MLSS_box_root_path(T, V, splits, best_t_splits, best_v_splits)
						end = time.time()
					if self.model_type == 'nn':
						start = time.time()
						c, h  = self.model.MLSS_hybrid_root_path(T, V, transformed_data, raw_data, rnn_mdn, \
							splits, best_t_splits, best_v_splits)
						end = time.time()

					total_time += end - start

					root_paths += 1
					root_path_hits.append(h)
					path_var[best_plan].append(h)
					cost[best_plan].append(c)
					total_cost += c

				# update rewards
				rewards[best_plan] = np.var(path_var[best_plan]) * np.mean(cost[best_plan])
				# update weight
				weight[best_plan] = len(path_var[best_plan])
			else:
				print('should not be here!')
				break
			# repeat until CI is satisfied
			# estimation with linear combination
			# print('weight sum: ', np.sum([weight[k] for k in weight]), root_paths)
			estimate = 0
			estimated_var = 0
			for path in weight:
				w = weight[path]/root_paths
				estimate += w * np.sum(path_var[path]) / (len(path_var[path]) * splits ** (self.levels-1))
				estimated_var += w**2 * np.var(path_var[path]) / (len(path_var[path]) * splits ** (2*(self.levels-1)))
			ci = self.Z * np.sqrt(estimated_var)
			# print(estimate, ci, total_cost)
			history.append((estimate, ci, total_cost, total_time))
		return history
	
	def order_estimate(self, T, V, v_bound, initial_state, splits, trails):
		cost = 0
		simulation_time = 0
		# get order estimate for a plan using SRS
		max_value = []
		if v_bound is None:
			for i in range(trails):
				if self.model_type == 'cp':
					s,_,time = self.model.simulate(T, V, initial_state)
				if self.model_type == 'queue':
					_,s,time = self.model.simulate(T, V, initial_state)
				if self.model_type == 'nn':
					s,_,time = self.model.simulate(T, V, transformed_data[-1], raw_data[-1], rnn_mdn)
				cost += len(s)
				simulation_time += time
				max_value.append(np.max(s))
		
		else:
			for i in range(trails):
				# get entrance state
				if self.model_type == 'cp':
					s,state,time = self.model.simulate(T, v_bound, initial_state)
				if self.model_type == 'queue':
					s1,s,time = self.model.simulate(T, v_bound, initial_state)
					state = (s1[-1], s[-1])
				if self.model_type == 'nn':
					s,state,time = self.model.simulate(T, v_bound, transformed_data[-1], raw_data[-1], rnn_mdn)
				cost += len(s)
				simulation_time += time
				# then get order estimate
				if self.model_type == 'cp':
					s,_,time = self.model.simulate(T-len(s), V, state)
				if self.model_type == 'queue':
					_,s,time = self.model.simulate(T-len(s), V, state)
				if self.model_type == 'nn':
					s,_,time = self.model.simulate(T-len(s), V, state, s[-1], rnn_mdn)
				cost += len(s)
				simulation_time += time
				max_value.append(np.max(s))
		
		return np.sort(max_value), cost, simulation_time

	# using order statistics to get level probability estimate
	def run_dynamic(self, T, V, v_bounds, ci_threshold, splits, trail_rounds):
		total_cost = 0
		total_time = 0

		if self.model_type == 'cp':
			initial_state = self.params
		if self.model_type == 'nn':
			transformed_data, raw_data, rnn_mdn = self.params
		if self.model_type == 'queue':
			initial_state = (0,0)
		
		orders, c, t = self.order_estimate(T, V, None, initial_state, splits, trail_rounds)

		d = math.floor(trail_rounds * (1-0.22))
		f = trail_rounds * (1-0.22) - d
		level_value = (1-f) * orders[d] + f * orders[d+1]

		print(level_value)
	
	# greedy partition the space level-by-level
	# always partition the region with lower probability
	def greedy_selector(self, T, V, v_start, v_end, splits, trail_rounds, ground_truth):
		if self.model_type == 'cp':
			initial_state = self.params
		if self.model_type == 'nn':
			transformed_data, raw_data, rnn_mdn = self.params
		if self.model_type == 'queue':
			initial_state = (0,0)
		
		selection = [v_end]
		start = v_start
		end = v_end
		all_estimates = {}
		level_weights = {}
		final_root_path_hits = []

		total_cost = 0
		total_time = 0

		level_eval = [100000]
		level_partitions = []
		
		# while len(selection) < self.levels:
		while True:
			path_eval = {}
			counter = {}
			candidate = np.random.uniform(start, end, self.seeds)
			print('===candidate:{}-{}/{}==='.format(start, end, candidate))
			m = len(selection)
			total_root_path_hits = []
			total_root_path_cost = []
			for value in candidate:
				partition = copy.deepcopy(selection)
				partition.append(value)
				partition.sort()
				#print(partition)
				root_path_hits = []
				root_path_cost = []
				# for i in range(trail_rounds):
				# while len(root_path_hits) == 0 or np.sum(root_path_hits) < WITNESS:
				while np.sum(root_path_cost) < T*trail_rounds:
					if self.model_type == 'cp':
						c,h,st = self.model.MLSS_hybrid_root_path(T, V, splits, initial_state, partition)
					if self.model_type == 'queue':
						c,h,st = self.model.MLSS_root_path(T, V, splits, partition)
					if self.model_type == 'nn':
						c,h,st = self.model.MLSS_hybrid_root_path(T, V, transformed_data, raw_data, rnn_mdn, splits, partition)
					total_cost += c
					total_time += st
					root_path_hits.append(h)
					root_path_cost.append(c)
					total_root_path_hits.append(h)
					total_root_path_cost.append(c)
				if np.var(root_path_hits) > 0:
					path_eval[value] = np.var(root_path_hits) * np.mean(root_path_cost)
					counter[value] = len(root_path_hits)
				#print(np.var(root_path_hits), np.mean(root_path_cost), path_eval[value])
			# calculate the current estimate and variance
			estimate = np.sum(total_root_path_hits) / (len(total_root_path_hits) * splits ** m)
			var = np.var(total_root_path_hits) / (len(total_root_path_hits) * splits ** (2*m))
			print('estimate:{}, var:{}'.format(estimate, var))
			all_estimates[m] = (estimate, var)
			final_root_path_hits.append(total_root_path_hits)
			# find the best partition
			best_partition = min(path_eval.items(), key=operator.itemgetter(1))[0]
			level_weights[m] = counter[best_partition]
			print('best partition:', best_partition)
			selection.append(best_partition)
			selection.sort()
			# remeber the best level info
			level_eval.append(np.var(total_root_path_hits) * np.mean(total_root_path_cost) / (splits ** (2*m)))
			level_partitions.append(copy.deepcopy(selection))
			# find the direction for next partition
			max_diff = selection[0] - v_start
			start = v_start
			end = selection[0]
			for i in range(1, len(selection)):
				if np.abs(selection[i] - selection[i-1]) > max_diff:
					max_diff = np.abs(selection[i] - selection[i-1])
					start = selection[i-1]
					end = selection[i]
			
			# remeber for final selection plan
			# if len(selection) == self.levels:
			# 	final_root_path_hits = copy.deepcopy(total_root_path_hits)

			# automatically find the best partitions
			# if the eval function start to increase, then stop
			if level_eval[-1] > level_eval[-2] and (level_eval[-1] - level_eval[-2]) / level_eval[-2] > EPSILON:
				break
		
		print(level_eval)
		# print(level_partitions)

		selection = level_partitions[-2]
		print(selection)

		estimates = [all_estimates[key][0] for key in sorted(all_estimates.keys())]
		vars = [all_estimates[key][1] for key in sorted(all_estimates.keys())]
		weights = [level_weights[key] for key in sorted(level_weights.keys())]

		total_weights = np.sum(weights)
		combined_estimate = np.sum([item[0]*item[1]/total_weights for item in zip(estimates, weights)])
		combined_var = np.sum([item[0]*(item[1]/total_weights)**2 for item in zip(vars, weights)])
		
		print(combined_estimate, total_cost, total_time)

		# # stick to the best partition plan
		# relative_error = 1
		# combined_estimate = 0
		# combined_var = 0
		# while relative_error > 0.1:
		# 	for i in range(5):
		# 		if self.model_type == 'cp':
		# 			c,h,st = self.model.MLSS_hybrid_root_path(T, V, splits, initial_state, selection)
		# 		if self.model_type == 'queue':
		# 			c,h,st = self.model.MLSS_root_path(T, V, splits, selection)
		# 		if self.model_type == 'nn':
		# 			c,h,st = self.model.MLSS_hybrid_root_path(T, V, transformed_data, raw_data, rnn_mdn, splits, selection)
		# 		total_cost += c
		# 		total_time += st
		# 		final_root_path_hits[-2].append(h)
		# 		weights[-2] += 1

		# 	#print(estimates, vars, weight)
		# 	estimates[-2] = np.sum(final_root_path_hits[-2]) / (len(final_root_path_hits[-2]) * splits ** (len(selection) - 1))
		# 	vars[-2] = np.var(final_root_path_hits[-2]) / (len(final_root_path_hits[-2]) * splits ** (2*(len(selection) - 1)))

		# 	total_weights = np.sum(weights)
		# 	combined_estimate = np.sum([item[0]*item[1]/total_weights for item in zip(estimates, weights)])
		# 	combined_var = np.sum([item[0]*(item[1]/total_weights)**2 for item in zip(vars, weights)])

		# 	relative_error = np.sqrt(combined_var) / ground_truth
			
		# print(combined_estimate, total_cost, total_time)

	# greedy partition the space level-by-level
	# always partition the region with lower probability
	def greedy_selector_v2(self, T, V, v_start, v_end, splits, trail_rounds, ground_truth):
		if self.model_type == 'cp':
			initial_state = self.params
		if self.model_type == 'nn':
			transformed_data, raw_data, rnn_mdn = self.params
		if self.model_type == 'queue':
			initial_state = (0,0)
		
		selection = [v_end]
		start = v_start
		end = v_end
		all_estimates = {}
		level_weights = {}
		final_root_path_hits = []

		total_cost = 0
		total_time = 0
		while len(selection) < self.levels:
			path_eval = {}
			counter = {}
			candidate = np.random.uniform(start, end, self.seeds)
			print('===candidate:{}-{}/{}==='.format(start, end, candidate))
			m = len(selection)
			total_root_path_hits = []
			for value in candidate:
				partition = copy.deepcopy(selection)
				partition.append(value)
				partition.sort()
				#print(partition)
				root_path_hits = []
				root_path_cost = []
				# for i in range(trail_rounds):
				# while len(root_path_hits) == 0 or np.sum(root_path_hits) < WITNESS:
				while np.sum(root_path_cost) < T*trail_rounds:
					if self.model_type == 'cp':
						c,h,st = self.model.MLSS_hybrid_root_path(T, V, splits, initial_state, partition)
					if self.model_type == 'queue':
						c,h,st = self.model.MLSS_root_path(T, V, splits, partition)
					if self.model_type == 'nn':
						c,h,st = self.model.MLSS_hybrid_root_path(T, V, transformed_data, raw_data, rnn_mdn, splits, partition)
					total_cost += c
					total_time += st
					root_path_hits.append(h)
					root_path_cost.append(c)
					total_root_path_hits.append(h)
				if np.var(root_path_hits) > 0:
					path_eval[value] = np.var(root_path_hits) * np.mean(root_path_cost)
					counter[value] = len(root_path_hits)
				#print(np.var(root_path_hits), np.mean(root_path_cost), path_eval[value])
			# calculate the current estimate and variance
			estimate = np.sum(total_root_path_hits) / (len(total_root_path_hits) * splits ** m)
			var = np.var(total_root_path_hits) / (len(total_root_path_hits) * splits ** (2*m))
			print('estimate:{}, var:{}'.format(estimate, var))
			all_estimates[m] = (estimate, var)
			# find the best partition
			best_partition = min(path_eval.items(), key=operator.itemgetter(1))[0]
			level_weights[m] = counter[best_partition]
			print('best partition:', best_partition)
			selection.append(best_partition)
			selection.sort()
			# find the direction for next partition
			max_diff = selection[0] - v_start
			start = v_start
			end = selection[0]
			for i in range(1, len(selection)):
				if np.abs(selection[i] - selection[i-1]) > max_diff:
					max_diff = np.abs(selection[i] - selection[i-1])
					start = selection[i-1]
					end = selection[i]
			
			# remeber for final selection plan
			if len(selection) == self.levels:
				final_root_path_hits = copy.deepcopy(total_root_path_hits)
		
		selection.sort()
		print(selection)

		estimates = [all_estimates[key][0] for key in sorted(all_estimates.keys())]
		vars = [all_estimates[key][1] for key in sorted(all_estimates.keys())]
		weights = [level_weights[key] for key in sorted(level_weights)]

		print(total_cost, total_time)

'''
		# stick to the best partition plan
		relative_error = 1
		combined_estimate = 0
		combined_var = 0
		while relative_error > 0.1:
			for i in range(5):
				if self.model_type == 'cp':
					c,h,st = self.model.MLSS_hybrid_root_path(T, V, splits, initial_state, selection)
				if self.model_type == 'queue':
					c,h,st = self.model.MLSS_root_path(T, V, splits, selection)
				if self.model_type == 'nn':
					c,h,st = self.model.MLSS_hybrid_root_path(T, V, transformed_data, raw_data, rnn_mdn, splits, selection)
				total_cost += c
				total_time += st
				final_root_path_hits.append(h)
				weights[-1] += 1

			#print(estimates, vars, weight)
			estimates[-1] = np.sum(final_root_path_hits) / (len(final_root_path_hits) * splits ** (len(selection) - 1))
			vars[-1] = np.var(final_root_path_hits) / (len(final_root_path_hits) * splits ** (2*(len(selection) - 1)))

			total_weights = np.sum(weights)
			combined_estimate = np.sum([item[0]*item[1]/total_weights for item in zip(estimates, weights)])
			combined_var = np.sum([item[0]*(item[1]/total_weights)**2 for item in zip(vars, weights)])

			relative_error = np.sqrt(combined_var) / ground_truth
			
		print(combined_estimate, total_cost, total_time)
'''



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model', type=str, help='model type')
	parser.add_argument('T', type=int, help='length of the series')
	parser.add_argument('V', type=float, help='upper boundary')
	parser.add_argument('ci_threshold', type=float, help='confidence interval')
	parser.add_argument('trails', type=int, help='number of trails rounds')
	# parser.add_argument('witness', type=int, help='number of witness hits')
	parser.add_argument('levels', type=int, help='number of level partitions')
	parser.add_argument('seeds', type=int, help='number of candidates to explore')
	parser.add_argument('--splits',  type=int, help='splits for each value partition')
	parser.add_argument('--v_start',  type=int, help='starting point of parition space')
	parser.add_argument('--v_end',  type=int, help='ending point of parition space')
	parser.add_argument('--truth', type=float, help='ground truth probability')
	# parser.add_argument('--t_bounds',  metavar='N', type=int, nargs='+',
    #                 help='candidate bonds on time')
	# parser.add_argument('--v_bounds',  metavar='N', type=float, nargs='+',
    #                 help='candidate bonds on value')
	args = parser.parse_args()

	model_type = args.model
	T = args.T
	V = args.V
	ci_threshold = args.ci_threshold
	trail_rounds = args.trails
	# witness = args.witness
	num_levels = args.levels
	seeds = args.seeds
	splits = args.splits
	v_start = args.v_start
	v_end = args.v_end
	ground_truth = args.truth
	# t_bounds = args.t_bounds
	# v_bounds = args.v_bounds

	print(T, V, ci_threshold, seeds, splits, v_start, v_end)

	if model_type == 'queue':
		print('Loading Queue Model...')
		ts_model = tandem_queue_model((0.5, 2, 2), 1.96)
		params = None
	if model_type == 'cp':
		print('Loading Compound-Poisson model')
		ts_model = cp_model((0.8, 5, 10), 4.5, 1.96)
		params = (15, 0)
	if model_type == 'nn':
		print('Loading RNN-MDN model')
		print(datafile, modelfile)
		raw_data, transformed_data = data_preprocessor(datafile)
		ts_model = ML_model(raw_data, 50, 1.96)
		rnn_mdn = ts_model.load_model(modelfile)
		params = (transformed_data[-1], raw_data[-1], rnn_mdn)

	selector_test = Selector(num_levels, model_type, ts_model, params, 1.96, seeds)
	# selector_test.run_dynamic(T, V, None, ci_threshold, splits, trail_rounds)
	selector_test.greedy_selector(T, V, v_start, v_end, splits, trail_rounds, ground_truth)
	# selector_test.greedy_selector_v2(T, V, v_start, v_end, splits, trail_rounds, ground_truth)

	# single run test
	# mlss = selector_test.run_v3(T, V, t_bounds, v_bounds, ci_threshold, splits, trail_rounds)
	# print(mlss[-1])
	# srs = selector_test.SRS(T, V, ci_threshold)
	# print(srs[-1])
