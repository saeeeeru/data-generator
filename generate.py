import os, string, shutil

import numpy as np
import matplotlib.pyplot as plt

def plot_seq_fft(multi_seq, dt, length):
	t = np.arange(0, length*dt, dt)[:length]
	freq = np.linspace(0, 1.0/dt, length)
	K = multi_seq.shape[0]
	min = multi_seq.min()
	max = multi_seq.max()
	fig, axes = plt.subplots(K, 2)
	for k in range(K):
		seq = multi_seq[k,:]
		F = np.fft.fft(seq)
		Amp = np.abs(F)
		axes[k, 0].plot(t, seq, label='f(n)')
		axes[k, 0].set_ylim(min, max)
		axes[k, 1].plot(freq[:int(length/2)], Amp[:int(length/2)], label='|F(k)|')
	fig.suptitle('left: Time * Value, right: frequency * Amplitude')
	plt.show()


def get_segment_membership(R, M_max):
	"""
		This is a function for getting segment membership.

		Args: 
			- R : number of regimes
			- M_max : number of segments

		return: 
			segment_membership : list of assigned regimeID
	"""

	segment_membership = []
	M = np.random.random_integers(R, M_max)
	for m in range(M):
		r = np.random.random_integers(R)-1
		if m == 0:
			segment_membership.append(r)
		else:
			while r == segment_membership[m-1]:
				r = np.random.random_integers(R)-1
			else:
				segment_membership.append(r)

	return segment_membership

class simulatedDataGenerator():
	"""
		This is a class for generating simulated dataset.
	"""

	def __init__(self, param_setting):
		self.__dict__ = param_setting.copy()

	def generate_individual_param(self):
		# generate parameters for one user

		params = []
		for r in range(self.R):
			sample_coef, sample_freq = [], []
			n_sinusoid = np.random.random_integers(1,5)
			for sinusoid in range(n_sinusoid):
				sample_coef.append(np.random.normal(loc = self.population_mean_coef, scale = self.population_variance_coef, size = self.K))
				sample_freq.append(np.random.normal(loc = self.population_mean_freq, scale = self.population_variance_freq, size = self.K))

			d = {'coef':sample_coef,'freq':sample_freq}
			params.append(d)

		return params


	def compute_segment(self, r, param, length):
		# compute one segment

		segment = np.zeros((self.K, length))
		t = np.arange(0, length*self.dt, self.dt)[:length]
		for k in range(self.K):
			for sinusoid in range(len(param['coef'])):
				# synthesize sinusoids of all frequencies 
				segment[k,:] += 0.5*(r+1)*param['coef'][sinusoid][k]*np.sin(2*np.pi*param['freq'][sinusoid][k]*t) + self.noise*np.random.randn((length))

			

		if r == 0 and self.PLT:
			for k in range(self.K):
				print('sensor {0}'.format(k))
				for sinusoid in range(len(param['coef'])):
					print('coef: {0}, freq: {1}'.format(param['coef'][sinusoid][k], param['freq'][sinusoid][k]))

		if self.PLT: plot_seq_fft(segment, self.dt, length)

		return segment

	def generate_sample_data(self, sample_param, length_param):
		# generate one data for one user

		# segment_membership = get_segment_membership(self.R, 5)
		segment_membership = [1,0,2]
		for m, r in enumerate(segment_membership):
			segment = self.compute_segment(r, sample_param[r], length_param[r])
			if m == 0:
				sample = segment
			else:
				sample = np.hstack((sample, segment))

		# add time axis
		length = sample.shape[1]
		time = np.linspace(0,length*self.dt,length).reshape((1,-1))
		sample = np.vstack((time,sample))

		return sample

	def generate_individual_data(self):
		# generate dataset for one user

		individual_param = self.generate_individual_param()
		individual_dataset = []
		for n in range(self.N):

			sample_param = individual_param.copy()
			length_param = np.random.normal(loc = self.segment_mean_length, scale = self.segment_variance_length, size = self.R).astype(int).tolist()

			individual_dataset.append(self.generate_sample_data(sample_param,length_param))

			'''
			plt.plot(individual_dataset[n][1:,:].T)
			plt.show()
			plt.close()
			'''

		return individual_dataset

	def generate_dataset(self):
		dataset = []
		for i in range(self.n_class):
			dataset.append(self.generate_individual_data())

		if self.WRITE: self.write_dataset(dataset)

	def write_dataset(self, dataset):
		labels = list(string.ascii_lowercase)
		# set the output directory
		# out_dir = os.path.join('..','deepplait','_dat','simulated')
		out_dir = os.path.join('.','_dat','simulated')

		shutil.rmtree(out_dir)
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		for i, individual_dataset in enumerate(dataset):
			for n, data in enumerate(individual_dataset):

				'''
				plt.figure()
				plt.plot(data.T[:,1:])
				plt.title(labels[i]+'-'+str(n))
				# plt.savefig(os.path.join('.',labels[i]+'-'+str(n)+'.png'))
				plt.show()
				plt.close()
				'''

				np.savetxt(os.path.join(out_dir,labels[i]+'-'+str(n)),data.T)

if __name__ == '__main__':
	print(get_segment_membership(3,5))