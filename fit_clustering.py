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
from source import bias_tools
import mcmc
importlib.reload(mcmc)
importlib.reload(bias_tools)
importlib.reload(redshift_dists)
importlib.reload(clusteringModel)
importlib.reload(resampling)
importlib.reload(plotting)
importlib.reload(sample)

def fit_clustering_of_bin(binnum, samplename, mode='bias'):
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	binnedtab = tab[np.where(tab['bin'] == binnum)]
	frac, zs = redshift_dists.get_redshifts(binnedtab, sample='cosmos')

	clustering = np.load('clustering/%s_%s.npy' % (samplename, binnum), allow_pickle=True)
	w = clustering[0]
	werr = resampling.covariance_matrix(clustering[1:], w)
	werr = np.std(clustering[1:], axis=0)

	midzs, dndz = redshift_dists.redshift_dist(zs, nbins=10)
	scales = np.load('clustering/scales.npy', allow_pickle=True)

	# either fit correlation function with HOD framework
	if mode == 'hod':
		mcmc.sample_space(nwalkers=32, ndim=3, niter=500, anglebins=scales, y=w, yerr=werr, zs=midzs, dndz=dndz,
		                  modeltype='3param')
		modcf = clusteringModel.angular_corr_func_in_bins(scales, midzs, dndz, hodparams=[12., 1., 12.5],
		                                                  hodmodel='3param')


		dm_cf = clusteringModel.angular_corr_func_in_bins(scales, midzs, dndz)
		##### FIXXX the scales to be averages in bins !!!!!!!!!!!!!
		plotting.plot_each_cf_fit(binnum, np.logspace(np.log10(np.min(scales)), np.log10(np.max(scales)), len(modcf)),
		                          w, werr, modcf, dm_mod=dm_cf)


	# or just fit the two-halo term as a biased dark matter tracer
	else:

		b, b_err = clusteringModel.fit_bias(scales, w, werr, midzs, dndz, mode=mode)

		masses, massuperr, massloerr = bias_tools.avg_bias_to_mass(b, midzs, dndz, b_err)
		np.array([b, b_err]).dump('%s/%s_%s.npy' % (mode, samplename, binnum))
		if mode == 'bias':
			modcf = clusteringModel.biased_ang_cf(scales, b, midzs, dndz)
		else:
			modcf = clusteringModel.mass_biased_ang_cf(scales, b, midzs, dndz)
		plotting.plot_each_cf_fit(binnum, scales[:len(scales)-1], w, werr, modcf)
		return [b, b_err, np.log10(masses), np.log10(massuperr), np.log10(massloerr)]


def fit_clustering_by_bin(pool, samplename, mode='bias'):
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	maxbin = int(np.max(tab['bin']))
	binnums = np.arange(1, maxbin+1)
	partial_fit = partial(fit_clustering_of_bin, samplename=samplename, mode=mode)
	bs = list(pool.map(partial_fit, binnums))

	medcolors = sample.get_median_colors(tab)
	if mode == 'bias':
		plotting.bias_v_color(medcolors, np.array(bs)[:, 0], np.array(bs)[:, 1])
		plotting.mass_v_color(medcolors, np.array(bs)[:, 2], np.array(bs)[:, 3] - np.array(bs)[:, 2],
		                      np.array(bs)[:, 2] - np.array(bs)[:, 4])
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

	fit_clustering_by_bin(pool, 'catwise_r90', mode='hod')


