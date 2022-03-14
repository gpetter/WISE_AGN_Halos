import numpy as np
import redshift_dists
import clusteringModel
import importlib
import sys
import resampling
import plotting
import sample
from functools import partial
from astropy.table import Table
import lensingModel
from source import bias_tools
importlib.reload(lensingModel)
importlib.reload(bias_tools)
importlib.reload(redshift_dists)
importlib.reload(clusteringModel)
importlib.reload(resampling)
importlib.reload(plotting)
importlib.reload(sample)

def fit_lensing_of_bin(binnum, samplename, mode='bias'):
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	binnedtab = tab[np.where(tab['bin'] == binnum)]
	frac, zs = redshift_dists.get_redshifts(binnedtab, sample='cosmos')

	xpower = np.load('results/lensing_xcorrs/%s_%s.npy' % (samplename, binnum), allow_pickle=True)

	power = xpower[0]
	power_err = resampling.covariance_matrix(xpower[1:], power)
	power_err = np.std(xpower[1:], axis=0)


	midzs, dndz = redshift_dists.redshift_dist(zs, nbins=10)
	scales = np.load('results/lensing_xcorrs/scales.npy', allow_pickle=True)






	if mode == 'bias':
		b, b_err = lensingModel.fit_bias(power, power_err, midzs, dndz, mode=mode)
		mass, massuperr, massloerr = bias_tools.avg_bias_to_mass(b, midzs, dndz, b_err)
		avgmasserr = np.mean([massloerr, massuperr])
		np.array([b, b_err]).dump('results/lensing_xcorrs/bias/%s_%s.npy' % (samplename, binnum))
		np.array([mass, avgmasserr]).dump('results/lensing_xcorrs/mass/%s_%s.npy' % (samplename, binnum))
		modcf = lensingModel.biased_binned_x_power_spectrum(None, b, midzs, dndz)
	else:
		mass, mass_err = lensingModel.fit_mass(power, power_err, midzs, dndz)
		print(mass_err)

		b, b_err = bias_tools.mass_to_avg_bias(mass, midzs, dndz, log_merr=[mass_err, mass_err])

		np.array([b, b_err]).dump('results/lensing_xcorrs/bias/%s_%s.npy' % (samplename, binnum))
		np.array([mass, mass_err]).dump('results/lensing_xcorrs/mass/%s_%s.npy' % (samplename, binnum))
		modcf = lensingModel.mass_biased_x_power_spectrum(None, mass, midzs, dndz)
	unbiased_cf = lensingModel.binned_x_power_spectrum(midzs, dndz)
	plotting.plot_each_lensing_fit(binnum, int(np.max(tab['bin'])), scales, power, power_err, modcf, unbiased_cf)
	return [b, b_err, mass, mass_err, mass_err]


def fit_lensing_by_bin(pool, samplename, mode='bias'):
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	maxbin = int(np.max(tab['bin']))
	binnums = np.arange(1, maxbin+1)
	partial_fit = partial(fit_lensing_of_bin, samplename=samplename, mode=mode)
	bs = list(pool.map(partial_fit, binnums))
	medcolors = sample.get_median_colors(tab)
	np.array([medcolors, np.array(bs)[:, 0], np.array(bs)[:, 1]]).dump('results/lensing_xcorrs/bias/%s.npy' %
	                                                                   samplename)
	np.array([medcolors, np.array(bs)[:, 2], np.array(bs)[:, 3], np.array(bs)[:, 4]]).dump(
				'results/lensing_xcorrs/mass/%s.npy' % samplename)

	"""
	if mode == 'bias':
		plotting.bias_v_color(medcolors, np.array(bs)[:, 0], np.array(bs)[:, 1])
		plotting.mass_v_color(medcolors, np.array(bs)[:, 2], np.array(bs)[:, 3] - np.array(bs)[:, 2],
		                      np.array(bs)[:, 2] - np.array(bs)[:, 4])"""
	plotting.bias_v_color(samplename)
	plotting.mass_v_color(samplename)
	pool.close()





if __name__ == "__main__":
	import schwimmbad

	# use different executor based on command line arguments
	# lets code run either serially (python measure_clustering.py)
	# or with multiprocessing to do bootstraps in parallel (python measure_clustering.py --ncores=5)
	# or with MPI
	from argparse import ArgumentParser
	parser = ArgumentParser(description="Schwimmbad example.")

	group = parser.add_mutually_exclusive_group()
	group.add_argument("--ncores", dest="n_cores", default=1,
	                   type=int, help="Number of processes (uses multiprocessing).")
	group.add_argument("--mpi", dest="mpi", default=False,
	                   action="store_true", help="Run with MPI.")
	args = parser.parse_args()

	pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
	if args.mpi:
		if not pool.is_master():
			pool.wait()
			sys.exit(0)

	fit_lensing_by_bin(pool, 'catwise', mode='mass')