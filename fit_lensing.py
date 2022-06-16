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
import hm_calcs

def fit_lensing_of_bin(binnum, samplename, mode='bias', lensname='planck', hodmodel=None, n_mcmc=100, nwalkers=32,
                       pool=None):
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	nbins = int(np.max(tab['bin']))
	binnedtab = tab[np.where(tab['bin'] == binnum)]
	frac, zs, zspec, zphot = redshift_dists.get_redshifts(binnum, zthresh=4)

	xpower = np.load('results/lensing_xcorrs/%s_%s_%s.npy' % (samplename, lensname, binnum), allow_pickle=True)

	power = xpower[0]
	power_err = resampling.covariance_matrix(xpower[1:], power)
	power_err = np.std(xpower[1:], axis=0)


	midzs, dndz = redshift_dists.dndz_from_z_list(zs, bins=10)
	scales = np.load('results/lensing_xcorrs/%s_scales.npy' % lensname, allow_pickle=True)

	# zspace = np.linspace(np.min(zs), np.max(zs), 200)
	# interp_dndz = np.interp(zspace, midzs, dndz)
	zspace, interp_dndz = midzs, dndz


	if mode == 'hod':
		import mcmc
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
			initial_fit, fiterrs = lensingModel.initial_hod_fit(theta_data=scales, pow_data=power, pow_errs=power_err,
			                            zs=zspace, dn_dz=interp_dndz, hodmodel=hodmodel)
			print('Initial fit parameters:', initial_fit)

		except:
			print('Initial fit failed')
			initial_fit = None

		centervals, lowerrs, higherss = mcmc.sample_lens_space(binnum, nbins, nwalkers=nwalkers, ndim=ndim,
		                                                     niter=n_mcmc, ell_bins=None, y=power, yerr=power_err,
		                                                       zs=zspace,
		                                                     dndz=interp_dndz, modeltype=hodmodel,
		                                                     initial_params=initial_fit, pool=pool)
		b, berr, mass, massuperr, masslowerr = centervals[(nderived - 2) + ndim], higherss[(nderived - 2) + ndim], \
		                                         centervals[(nderived - 1) + ndim], higherss[(nderived - 1) + ndim], \
		                                         lowerrs[(nderived - 1) + ndim]

		halomodobj = hm_calcs.halomodel(zs=zspace)
		# get ang correlation function of dark matter as HOD not set yet
		dmmod = halomodobj.get_c_ell_kg(dndz=(zspace, dndz), ls=(1 + np.arange(3000)))
		# onemodcf = clusteringModel.angular_corr_func_in_bins(scales, zs=zspace, dn_dz_1=interp_dndz,
		#                hodparams=[centervals[0], centervals[1], centervals[2]], hodmodel=hodmodel,
		#                term='one')

		halomodobj.set_powspec(hodparams=[centervals[0], centervals[1], centervals[2]],
		                       modeltype=hodmodel, get2h=False)
		onemodcf = halomodobj.get_c_ell_kg(dndz=(zspace, dndz), ls=(1 + np.arange(3000)))

		# twomodcf = clusteringModel.angular_corr_func_in_bins(scales, zs=zspace, dn_dz_1=interp_dndz,
		#                hodparams=[centervals[0], centervals[1], centervals[2]],
		#                hodmodel=hodmodel, term='two')
		halomodobj.set_powspec(hodparams=[centervals[0], centervals[1], centervals[2]],
		                       modeltype=hodmodel, get1h=False)
		twomodcf = halomodobj.get_c_ell_kg(dndz=(zspace, dndz), ls=(1 + np.arange(3000)))
		# bothmodcf = clusteringModel.angular_corr_func_in_bins(scales, zs=zspace, dn_dz_1=interp_dndz, hodparams=[
		#	centervals[0], centervals[1], centervals[2]], hodmodel=hodmodel, term='both')

		# dmmod = clusteringModel.angular_corr_func_in_bins(scales, zs=zspace, dn_dz_1=interp_dndz, hodmodel='dm')
		halomodobj.set_powspec(hodparams=[centervals[0], centervals[1], centervals[2]],
		                       modeltype=hodmodel)
		modcf = halomodobj.get_c_ell_kg(dndz=(zspace, dndz), ls=(1 + np.arange(3000)))
	else:
		mass, mass_err = lensingModel.fit_mass(power, power_err, midzs, dndz)
		massuperr, masslowerr = mass_err, mass_err

		b, berr = bias_tools.mass_to_avg_bias(mass, midzs, dndz, log_merr=[mass_err, mass_err])
		hmobj = hm_calcs.halomodel(zs=midzs)

		np.array([b, berr]).dump('results/lensing_xcorrs/bias/%s_%s.npy' % (samplename, binnum))
		np.array([mass, mass_err]).dump('results/lensing_xcorrs/mass/%s_%s.npy' % (samplename, binnum))



		#modcf = lensingModel.mass_biased_x_power_spectrum(None, mass, midzs, dndz)
		#hmobj.set_powspec(hodparams=[12.4, 0.7], modeltype='2param')
		hmobj.set_powspec(log_meff=mass)
		modcf = hmobj.get_c_ell_kg(dndz=(midzs, dndz), ls=np.arange(3000) + 1)

		hmobj = hm_calcs.halomodel(zs=midzs)
		dmmod = hmobj.get_c_ell_kg(dndz=(midzs, dndz), ls=np.arange(3000)+1)
	plotting.plot_each_lensing_fit(binnum, int(np.max(tab['bin'])), scales, power, power_err, modcf, dmmod)
	return [b, berr, mass, massuperr, masslowerr, modcf, dmmod]


def fit_lensing_by_bin(pool, samplename, mode='bias', lensname='planck', hodmodel=None, n_mcmc=100):
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	maxbin = int(np.max(tab['bin']))
	binnums = np.arange(1, maxbin+1)
	partial_fit = partial(fit_lensing_of_bin, samplename=samplename, mode=mode, lensname=lensname,
	                      hodmodel=hodmodel, n_mcmc=n_mcmc)
	bs = list(pool.map(partial_fit, binnums))
	bs = np.array(bs)
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
	modcfs, unbiasedcfs = bs[:, 5], bs[:, 6]
	plotting.plot_all_lens_fits(nbins=len(modcfs), lensname=lensname, modcfs=modcfs, unbiasedcfs=unbiasedcfs)
	plotting.bias_v_color(samplename)
	plotting.mass_v_color(samplename)
	pool.close()





if __name__ == "__main__":
	import schwimmbad

	# use different executor based on command line arguments
	# lets code run either serially (python measure_clustering.py)
	# or with multiprocessing to do bootstraps in parallel (python measure_clustering.py --ncores=5)
	# or with MPI mpirun -np 10 python fit_lensing.py
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

	fit_lensing_by_bin(pool, 'catwise', mode='hod', lensname='planck', hodmodel='2param', n_mcmc=200)