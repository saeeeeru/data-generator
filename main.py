import random, argparse

import numpy as np

from generate import simulatedDataGenerator

PLT = False
WRITE = True

def perser():
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument('-C', '--n_class', type=int, default=4, help='number of classification labels')
	p.add_argument('-N', '--n_sample', type=int, default=800, help='number of samples per one label')
	p.add_argument('-R', '--n_regime', type=int, default=3, help='number of regime')
	p.add_argument('-K', '--n_sensor', type=int, default=6, help='number of sensors')
	p.add_argument('-DT', '--dt', type=float, default=0.01, help='sampling rate[s]')
	p.add_argument('--segment_mean_length', type=int, default=1000, help='mean length of each segment')
	p.add_argument('--segment_variance_length', type=int, default=200, help='variance of length between each segment')
	p.add_argument('--population_mean_coef', type=int, default=20, help='population mean coefficient of sinusoid')
	p.add_argument('--population_variance_coef', type=int, default=15, help='population variance coefficient of sinusoid')
	p.add_argument('--population_mean_freq', type=int, default=50, help='population mean frequency of sinusoid')
	p.add_argument('--population_variance_freq', type=int, default=25, help='population variance frequency of sinusoid')
	p.add_argument('--noise', type=int, default=2, help='coefficient of noise term')

	args = p.parse_args()

	return args

def generate_param_setting(args):
	param_setting = {'n_class':args.n_class,
			'N':args.n_sample, 
			'R':args.n_regime, 
			'dt':args.dt, 
			'K':args.n_sensor, 
			'segment_mean_length':args.segment_mean_length, 
			'segment_variance_length':args.segment_variance_length, 
			'population_mean_coef':args.population_mean_coef, 
			'population_variance_coef':args.population_variance_coef, 
			'population_mean_freq':args.population_mean_freq, 
			'population_variance_freq':args.population_variance_freq, 
			'noise':args.noise,
			'PLT':PLT,
			'WRITE':WRITE}

	return param_setting


def main():
	# random.seed(42)
	# np.random.seed(42)

	args = perser()
	param_setting = generate_param_setting(args)
	print('The simulated data parameter setting is')
	print('---------------------------------------')
	for key, value in param_setting.items():
		print('{key} : {value}'.format(**locals()))
	print('---------------------------------------')

	data_generator = simulatedDataGenerator(param_setting)
	data_generator.generate_dataset()


if __name__ == '__main__':
	main()