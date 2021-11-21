import importlib
import numpy as np
import lensingModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
importlib.reload(lensingModel)



def plot_sn_vs_counts(minmass, maxmass, nmasses, zs):
	noiselevel = 0.3
	test_ns = np.logspace(0, 8, 1000)
	noise_for_ns = noiselevel / np.sqrt(test_ns)

	masses = np.logspace(minmass, maxmass, nmasses)
	ns_for_masses = []
	for mass in masses:
		ns_for_zs = []
		for z in zs:
			signal_noise = lensingModel.filtered_model_center([z], 0, mass, mode='mass') / noise_for_ns

			ns_for_5sig_detection = np.min(test_ns[np.where(signal_noise > 5)])
			ns_for_zs.append(ns_for_5sig_detection)


		ns_for_masses.append(ns_for_zs)


	logmasses = np.log10(masses)
	plt.figure(figsize=(9, 7))
	for j in range(len(ns_for_masses)):
		print(logmasses[j] * np.ones(len(zs)), ns_for_masses[j])
		plt.scatter(logmasses[j]*np.ones(len(zs)), ns_for_masses[j])

	plt.xlabel('Halo Mass')
	plt.ylabel('Source Number for $5 \sigma \kappa$ Detection')
	plt.yscale('log')
	plt.savefig('plots/lensing_detection_potential.pdf')
	plt.close('all')

plot_sn_vs_counts(11, 15, 10, np.linspace(0.5, 4, 5))

