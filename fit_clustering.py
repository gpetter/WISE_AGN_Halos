
from astropy.table import Table
import numpy as np


def fit_clustering_of_bin(binnum, samplename, mode='bias', hodmodel=None, n_mcmc=None):
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	binnedtab = tab[np.where(tab['bin'] == binnum)]
	if samplename == 'eBOSS':
		frac, zs = 1, tab['Z']
	else:
		frac, zs = redshift_dists.get_redshifts(binnedtab, sample='cosmos')


	clustering = np.load('results/clustering/%s_%s.npy' % (samplename, binnum), allow_pickle=True)
	w = clustering[0]

	werr = resampling.covariance_matrix(clustering[1:], w)
	werr = np.std(clustering[1:], axis=0)

	midzs, dndz = redshift_dists.redshift_dist(zs, nbins=10)
	scales = np.load('results/clustering/scales.npy', allow_pickle=True)

	#zspace = np.linspace(np.min(zs), np.max(zs), 200)
	#interp_dndz = np.interp(zspace, midzs, dndz)
	zspace, interp_dndz = midzs, dndz

	# either fit correlation function with HOD framework
	if mode == 'hod':
		if hodmodel == '3param':
			ndim = 3
			nderived = 3
		elif hodmodel == '2param':
			ndim = 2
			nderived = 3
		else:
			ndim = 1
			nderived = 2
		try:
			initial_fit, fiterrs = clusteringModel.initial_hod_fit(theta_data=scales, w_data=w, w_errs=werr,
			                            zs=zspace, dn_dz=interp_dndz, hodmodel=hodmodel)

		except:
			print('Initial fit failed')
			initial_fit = None

		centervals, lowerrs, higherss = mcmc.sample_space(binnum, int(np.max(tab['bin'])), nwalkers=32, ndim=ndim,
		                niter=n_mcmc, anglebins=scales, y=w, yerr=werr, zs=zspace,
		                dndz=interp_dndz, modeltype=hodmodel, initial_params=initial_fit)
		onemodcf = clusteringModel.angular_corr_func_in_bins(scales, zs=zspace, dn_dz_1=interp_dndz,
		                hodparams=[centervals[0], centervals[1], centervals[2]], hodmodel=hodmodel,
		                term='one')
		twomodcf = clusteringModel.angular_corr_func_in_bins(scales, zs=zspace, dn_dz_1=interp_dndz,
		                hodparams=[centervals[0], centervals[1], centervals[2]],
		                hodmodel=hodmodel, term='two')
		bothmodcf = clusteringModel.angular_corr_func_in_bins(scales, zs=zspace, dn_dz_1=interp_dndz, hodparams=[
			centervals[0], centervals[1], centervals[2]], hodmodel=hodmodel, term='both')

		dmmod = clusteringModel.angular_corr_func_in_bins(scales, zs=zspace, dn_dz_1=interp_dndz, hodmodel='dm')


		#dm_cf = clusteringModel.angular_corr_func_in_bins(scales, midzs, dndz)
		##### FIXXX the scales to be averages in bins !!!!!!!!!!!!!
		plotting.plot_each_cf_fit(binnum, int(np.max(tab['bin'])), np.logspace(np.log10(np.min(scales)),
		                np.log10(np.max(scales)),len(onemodcf)), w, werr, onemodcf,
		                dmmod * centervals[(nderived - 2) + ndim] ** 2, bothmodcf,
		                dm_mod=dmmod)

		b, berr, masses, massuperr, masslowerr = centervals[(nderived - 2) + ndim], higherss[(nderived - 2) + ndim], \
		                        centervals[(nderived - 1) + ndim], higherss[(nderived - 1) + ndim], \
		                                         lowerrs[(nderived - 1) + ndim]
		out_hod_params = centervals[:ndim]

		return [b, berr, masses, massuperr, masslowerr, out_hod_params]


	# or just fit the two-halo term as a biased dark matter tracer
	else:
		from source import bias_tools
		b, b_err = clusteringModel.fit_bias(scales, w, werr, midzs, dndz, mode=mode)


		if mode == 'bias':
			modcf = clusteringModel.biased_ang_cf(scales, b, midzs, dndz)
			masses, massuperr, massloerr = bias_tools.avg_bias_to_mass(b, midzs, dndz, b_err)
		else:
			modcf = clusteringModel.mass_biased_ang_cf(scales, b, midzs, dndz)
			masses, massuperr, massloerr = b, b_err, b_err
		#plotting.plot_each_cf_fit(binnum, scales[:len(scales)-1], w, werr, modcf)
		return [b, b_err, masses, massuperr, massloerr]


def fit_clustering_by_bin(pool, samplename, mode='bias', hodmodel=None, n_mcmc=None):
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	maxbin = int(np.max(tab['bin']))
	binnums = np.arange(1, maxbin+1)
	partial_fit = partial(fit_clustering_of_bin, samplename=samplename, mode=mode,
	                      hodmodel=hodmodel, n_mcmc=n_mcmc)
	bs = list(pool.map(partial_fit, binnums))

	medcolors = sample.get_median_colors(tab)
	"""if mode == 'bias':
		plotting.bias_v_color(medcolors, np.array(bs)[:, 0], np.array(bs)[:, 1])
		plotting.mass_v_color(medcolors, np.array(bs)[:, 2], np.array(bs)[:, 3] - np.array(bs)[:, 2],
		                      np.array(bs)[:, 2] - np.array(bs)[:, 4])"""
	#plotting.bias_v_color(medcolors, np.array(bs)[:, 0], np.array(bs)[:, 1])

	#plotting.mass_v_color(medcolors, np.array(bs)[:, 2], np.array(bs)[:, 3], np.array(bs)[:, 4])

	np.array([medcolors, np.array(bs)[:, 0], np.array(bs)[:, 1]]).dump('results/clustering/bias/%s.npy' % samplename)
	np.array([medcolors, np.array(bs)[:, 2], np.array(bs)[:, 3], np.array(bs)[:, 4]]).dump(
		'results/clustering/mass/%s.npy' % samplename)
	if mode == 'hod':
		tothods = []
		for j in range(maxbin):
			tothods.append(hod_model.hod_total((np.array(bs)[:, 5])[j], modeltype=hodmodel))
		plotting.plot_hods(mass_space, tothods)
	plotting.mass_v_color(samplename)
	plotting.bias_v_color(samplename)


	pool.close()





if __name__ == "__main__":




	k_space = np.logspace(-5, 3, 1000)
	mass_space = np.logspace(10, 15, 50)

	k_space.dump('power_spectra/k_space.npy')
	mass_space.dump('power_spectra/m_space.npy')


	import hod_model

	import schwimmbad
	import redshift_dists
	import clusteringModel

	import sys
	import resampling
	import plotting
	import sample
	from functools import partial

	from source import bias_tools
	import mcmc





	#import hod_model
	#hod_model.write_ukm()

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

	fit_clustering_by_bin(pool, 'catwise', mode='hod', hodmodel='2param', n_mcmc=50)


