from queue_models import *
from ml_models import *
from CP_models import *
from plan_selector import *
import sys
import argparse
import time
import itertools

# datafile = 'google_stock.csv'
# modelfile = 'model/google_rnn_mdn_2_5_16_30.h5'

datafile = 'amazon_stock.csv'
modelfile = 'model/amazon_rnn_mdn_2_5_16_50.h5'
error_bound = 2

def splitting_test(T, V, v_splits, ci_threshold, ground_truth, relative_error, trails, model):
	simulation_cost = defaultdict(list)
	simulation_time = defaultdict(list)

	# define time series model
	if model == 'cp':
		print('Loading Compound-Poisson model')
		ts_model = cp_model((0.8, 5, 10), 4.5, 1.96)
	if model == 'queue':
		print('Loading Queue Model...')
		ts_model = tandem_queue_model((0.5, 2, 2), 1.96)
	if model == 'nn':
		print('Loading RNN-MDN Model')
		print(datafile, modelfile)
		raw_data, transformed_data = data_preprocessor(datafile)
		ts_model = ML_model(raw_data, 50, 1.96)
		rnn_mdn = ts_model.load_model(modelfile)
	
	for split in range(1, 8):
		if ci_threshold > 0:
			print('split:{}, confidence interval:{}'.format(split, ci_threshold))
			for i in range(trails):
				if model == 'queue':
					mlss = ts_model.MLSS(T, V, split, v_splits, ci_threshold)
				if model == 'cp':
					mlss = ts_model.MLSS_hybrid(T, V, split, (15, 0), v_splits, ci_threshold)
				if model == 'nn':
					mlss = ts_model.MLSS_hybrid(T, V, transformed_data[-1], raw_data[-1], rnn_mdn, \
							split, v_splits, ci_threshold)
				simulation_cost[split].append(mlss[-1][2])
				simulation_time[split].append(mlss[-1][3])
		else:
			print('relative error:{}/{}'.format(ground_truth, relative_error))
			for i in range(trails):
				if model == 'queue':
					mlss = ts_model.MLSS_v2(T, V, split, v_splits, ground_truth, relative_error)
				if model == 'cp':
					mlss = ts_model.MLSS_v2(T, V, split, (15,0), v_splits, ground_truth, relative_error)
				if model == 'nn':
					mlss = ts_model.MLSS_hybrid(T, V, transformed_data[-1], raw_data[-1], rnn_mdn, \
							split, v_splits, ground_truth, relative_error)
				simulation_cost[split].append(mlss[-1][2])
				simulation_time[split].append(mlss[-1][3])
	
	for key in simulation_cost:
		print('====={}====='.format(key))
		print('{}/{}'.format(np.mean(simulation_cost[key]), np.std(simulation_cost[key])))
		print('{}/{}'.format(np.mean(simulation_time[key]), np.std(simulation_time[key])))

def overall_efficiency_test(T, V, v_splits, splits, ci_threshold, ground_truth, relative_error, trails, model):
	srs_history = []
	mlss_history = []

	if model == 'queue':
		print('Loading Queue Model...')
		ts_model = tandem_queue_model((0.5, 2, 2), 1.96)
		if ci_threshold > 0:
			print('confidence interval:{}'.format(ci_threshold))
			for i in range(trails):
				print('\r{}'.format(i), end='')
				srs = ts_model.SRS(T, V, ci_threshold)
				srs_history.append(srs[-1])
				
				mlss = ts_model.MLSS(T, V, splits, v_splits, ci_threshold)
				mlss_history.append(mlss[-1])
		else:
			print('relative error:{}/{}'.format(ground_truth, relative_error))
			for i in range(trails):
				print('\r{}'.format(i), end='')
				srs = ts_model.SRS_v2(T, V, ground_truth, relative_error)
				srs_history.append(srs[-1])
				
				mlss = ts_model.MLSS_v2(T, V, splits, v_splits, ground_truth, relative_error)
				mlss_history.append(mlss[-1])
	
	if model == 'cp':
		print('Loading Compound-Poisson model')
		ts_model = cp_model((0.8, 5, 10), 4.5, 1.96)
		if ci_threshold > 0:
			print('confidence interval:{}'.format(ci_threshold))
			for i in range(trails):
				print('\r{}'.format(i), end='')
				srs = ts_model.SRS(T, V, (15, 0) , ci_threshold)
				srs_history.append(srs[-1])
				
				mlss = ts_model.MLSS_hybrid(T, V, splits, (15, 0), v_splits, ci_threshold)
				mlss_history.append(mlss[-1])
		else:
			print('relative error:{}/{}'.format(ground_truth, relative_error))
			for i in range(trails):
				print('\r{}'.format(i), end='')
				srs = ts_model.SRS_v2(T, V, (15,0), ground_truth, relative_error)
				srs_history.append(srs[-1])
				
				mlss = ts_model.MLSS_v2(T, V, splits, (15,0), v_splits, ground_truth, relative_error)
				mlss_history.append(mlss[-1])

	if model == 'nn':
		print('Loading RNN-MDN Model')
		raw_data, transformed_data = data_preprocessor(datafile)
		ts_model = ML_model(raw_data, 50, 1.96)
		rnn_mdn = ts_model.load_model(modelfile)
		for i in range(trails):
			print('\r{}'.format(i), end='')
			srs = ts_model.SRS(T, V, transformed_data[-1], raw_data[-1], rnn_mdn, ci_threshold)
			srs_history.append(srs[-1])
			
			mlss = ts_model.MLSS_hybrid(T, V, transformed_data[-1], raw_data[-1], rnn_mdn, \
		splits, t_splits, v_splits, ci_threshold)
			mlss_history.append(mlss[-1])


	avg_srs_est = np.mean([item[0] for item in srs_history])
	std_srs_est = np.std([item[0] for item in srs_history])
	avg_srs_error = np.mean([item[1] for item in srs_history])
	std_srs_error = np.std([item[1] for item in srs_history])
	avg_srs_cost = np.mean([item[2] for item in srs_history])
	std_srs_cost = np.std([item[2] for item in srs_history])
	avg_srs_time = np.mean([item[3] for item in srs_history])
	std_srs_time = np.std([item[3] for item in srs_history])
	
	estimate = np.array([item[0] for item in mlss_history])
	filter_condition = np.where(abs(estimate - np.mean(estimate)) < 2 * np.std(estimate))
	mlss_history = np.array(mlss_history)
	print(len(mlss_history[filter_condition]) / len(mlss_history))

	avg_mlss_est = np.mean([item[0] for item in mlss_history[filter_condition]])
	std_mlss_est = np.std([item[0] for item in mlss_history[filter_condition]])
	avg_mlss_error = np.mean([item[1] for item in mlss_history[filter_condition]])
	std_mlss_error = np.std([item[1] for item in mlss_history[filter_condition]])
	avg_mlss_cost = np.mean([item[2] for item in mlss_history[filter_condition]])
	std_mlss_cost = np.std([item[2] for item in mlss_history[filter_condition]])
	avg_mlss_time = np.mean([item[3] for item in mlss_history[filter_condition]])
	std_mlss_time = np.std([item[3] for item in mlss_history[filter_condition]])

	print('=====SRS=====')
	print('estimate avg: {}, std:{}'.format(avg_srs_est, std_srs_est))
	print('cost avg: {}, std:{}'.format(avg_srs_cost, std_srs_cost))
	print('avg time: {}, std:{}'.format(avg_srs_time, std_srs_time))
	print('error: {}/{}'.format(avg_srs_error, std_srs_error))
	
	print('=====MLSS=====')
	print('estimate avg: {}, std:{}'.format(avg_mlss_est, std_mlss_est))
	print('cost avg: {}, std:{}'.format(avg_mlss_cost, std_mlss_cost))
	print('avg time: {}, std:{}'.format(avg_mlss_time, std_mlss_time))
	print('error: {}/{}'.format(avg_mlss_error, std_mlss_error))

def search_for_optimal(T, V, t_bounds, v_bounds, splits, levels, ci_threshold, trails, model):
	# generate all possible level partitions
	t_splits_pool = []
	v_splits_pool = []

	t_candidates = itertools.combinations(t_bounds, levels-1)
	for item in t_candidates:
		item = list(item)
		item.append(T)
		t_splits_pool.append(item)

	v_candidates = itertools.combinations(v_bounds, levels-1)
	for item in v_candidates:
		item = list(item)
		item.append(V)
		v_splits_pool.append(item)

	print(t_splits_pool)
	print(v_splits_pool)

	# define time series model
	if model == 'cp':
		print('Loading Compound-Poisson model')
		ts_model = cp_model((0.8, 5, 10), 4.5, 1.96)
	if model == 'queue':
		print('Loading Queue Model...')
		ts_model = tandem_queue_model((0.5, 2, 2), 1.96)
	if model == 'ar':
		print('Loading AR Model')
		ts_model = AR_model((-0.3, 0.4, 0.5), 5, (0, 2), 1.96)
	if model == 'nn':
		print('Loading RNN-MDN Model')
		print(datafile, modelfile)
		raw_data, transformed_data = data_preprocessor(datafile)
		ts_model = ML_model(raw_data, 50, 1.96)
		rnn_mdn = ts_model.load_model(modelfile)

	# SRS baseline
	srs_history = []
	srs_time = []
	for i in range(trails):
		print('\rSRS:{}'.format(i), end='')
			
		start = time.time()
		if model == 'cp':
			srs = ts_model.SRS(T, V, (15, 0) , ci_threshold)
		if model == 'queue':
			srs = ts_model.SRS(T, V, ci_threshold)
		if model == 'ar':
			srs = ts_model.SRS(T, V, (1,2,3), 0, ci_threshold)
		if model == 'nn':
			srs = ts_model.SRS(T, V, transformed_data[-1], raw_data[-1], rnn_mdn, ci_threshold)
		end = time.time()

		srs_history.append(srs[-1])
		srs_time.append(end-start)

	avg_srs_est = np.mean([item[0] for item in srs_history])
	std_srs_est = np.std([item[0] for item in srs_history])
	avg_srs_cost = np.mean([item[2] for item in srs_history])
	std_srs_cost = np.std([item[2] for item in srs_history])

	print('=====SRS=====')
	print('estimate avg: {}, std:{}'.format(avg_srs_est, std_srs_est))
	print('cost avg: {}, std:{}'.format(avg_srs_cost, std_srs_cost))
	print('avg time: {}'.format(np.mean(srs_time)))

	best_mlss = 10*np.mean(srs_time)
	best_splits = None
	# Search for Optimal Performance
	for t_splits in t_splits_pool:
		# transform into time interval
		for i in range(1, len(t_splits)):
			t_splits[i] = t_splits[i] - t_splits[i-1]
		for v_splits in v_splits_pool:
			print('t-splits:{}, v-splits:{}'.format(t_splits, v_splits))
			
			mlss_opt_history = []
			mlss_opt_time = []

			for i in range(trails):
				if model == 'cp':
					start = time.time()
					mlss_opt = ts_model.MLSS_hybrid(T, V, splits, (15, 0) ,t_splits, v_splits, ci_threshold)
					end = time.time()
				if model == 'queue':
					start = time.time()
					mlss_opt = ts_model.MLSS(T, V, splits, t_splits, v_splits, ci_threshold, split_type=0)
					end = time.time()
				if model == 'ar':
					start = time.time()
					mlss_opt = ts_model.MLSS_hybrid(T, V, splits, (1,2,3), t_splits, v_splits, ci_threshold)
					end = time.time()
				if model == 'nn':
					start = time.time()
					mlss_opt = ts_model.MLSS_hybrid(T, V, transformed_data[-1], raw_data[-1], rnn_mdn, \
				splits, t_splits, v_splits, ci_threshold)
					end = time.time()

				mlss_opt_history.append(mlss_opt[-1])
				mlss_opt_time.append(end-start)
			
			avg_mlss_opt_est = np.mean([item[0] for item in mlss_opt_history])
			std_mlss_opt_est = np.std([item[0] for item in mlss_opt_history])
			avg_mlss_opt_cost = np.mean([item[2] for item in mlss_opt_history])
			std_mlss_opt_cost = np.std([item[2] for item in mlss_opt_history])

			if np.mean(mlss_opt_time) < best_mlss:
				best_mlss = np.mean(mlss_opt_time)
				best_splits = (t_splits, v_splits)
				print('*******NEW BEST********')
				print('estimate avg: {}, std:{}'.format(avg_mlss_opt_est, std_mlss_opt_est))
				print('cost avg: {}, std:{}'.format(avg_mlss_opt_cost, std_mlss_opt_cost))
				print('avg time: {}'.format(np.mean(mlss_opt_time)))
			
def auto_opt(T, V, t_bounds, v_bounds, splits, levels, ci_threshold, trails, trail_rounds, model_type):
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

	selector_test = Selector(2, model_type, ts_model, params, 1.96)

	mlss_history = []
	mlss_time = []

	for i in range(trails):
		print('\rMLSS:{}'.format(i), end='')
		mlss = selector_test.run_v3(T, V, t_bounds, v_bounds, ci_threshold, splits, trail_rounds)
		mlss_history.append(mlss[-1])
		mlss_time.append(mlss[-1][-1])

	avg_mlss_est = np.mean([item[0] for item in mlss_history])
	std_mlss_est = np.std([item[0] for item in mlss_history])
	avg_mlss_cost = np.mean([item[2] for item in mlss_history])
	std_mlss_cost = np.std([item[2] for item in mlss_history])

	print('estimate avg: {}, std:{}'.format(avg_mlss_est, std_mlss_est))
	print('cost avg: {}, std:{}'.format(avg_mlss_cost, std_mlss_cost))
	print('avg time: {}'.format(np.mean(mlss_time)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model', type=str, help='model type')
	parser.add_argument('T', type=int, help='length of the series')
	parser.add_argument('V', type=float, help='upper boundary')
	parser.add_argument('ci_threshold', type=float, help='confidence interval')
	parser.add_argument('ground_truth', type=float, help='ground truth probability')
	parser.add_argument('relative_error', type=float, help='relative error')
	parser.add_argument('trails', type=int, help='number of trails to report average')
	parser.add_argument('exp_type', type=str, help='experiment type')
	parser.add_argument('--v_splits', metavar='N', type=float, nargs='+',
                    help='splitting levels on value')
	parser.add_argument('--splits',  type=int, help='splits for each value partition')
	# parser.add_argument('--exploration', type=int, help='number of exploration trails')
	

	args = parser.parse_args()

	model = args.model
	T = args.T
	V = args.V
	ci_threshold = args.ci_threshold
	ground_truth = args.ground_truth
	relative_error = args.relative_error
	v_splits = args.v_splits
	splits = args.splits
	trails = args.trails
	exp_type = args.exp_type

	print('model:{}, T:{}, V:{}, CI:{}'.format(model, T, V, ci_threshold))
	print('v-splits:{}, splits:{}'.format(v_splits, splits))
	print('trails:{}, experiment type:{}'.format(trails, exp_type))

	if exp_type == 'efficiency':
		# overall_efficiency_test(T, V, t_splits, v_splits, splits, ci_threshold, trails, model)
		overall_efficiency_test(T, V, v_splits, splits, ci_threshold, ground_truth, relative_error, trails, model)
	
	if exp_type == 'splitting':
		splitting_test(T, V, v_splits, ci_threshold, ground_truth, relative_error, trails, model)
	
	if exp_type == 'search':
		search_for_optimal(T, V, t_bounds, v_bounds, splits, levels, ci_threshold, trails, model)

	if exp_type == 'auto':
		exploration_rounds = args.exploration
		auto_opt(T, V, t_bounds, v_bounds, splits, levels, ci_threshold, trails, exploration_rounds, model)