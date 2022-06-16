import mcfit
import numpy as np
from colossus.cosmology import cosmology
from scipy.optimize import curve_fit
from scipy import stats
cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()
import astropy.cosmology.units as cu

import camb
from functools import partial
import pickle
import astropy.constants as const
import astropy.units as u
from source import bias_tools
from source import interpolate_tools
import hod_model
import hm_calcs


# define k space (includes little h)
#kmax = 1000
#k_grid = np.logspace(-5, np.log10(kmax), 1000) * (u.littleh / u.Mpc)
k_grid = np.load('power_spectra/k_space.npy', allow_pickle=True)




# set CAMB cosmology, generate power spectrum at 150 redshifts, return interpolator which can estimate power spectrum
# at any given redshift
def camb_matter_power_interpolator(zs, nonlinear=True):
	pars = camb.CAMBparams()
	pars.set_cosmology(H0=cosmo.H0, ombh2=cosmo.Ombh2, omch2=(cosmo.Omh2-cosmo.Ombh2), omk=cosmo.Ok0)
	pars.InitPower.set_params(ns=cosmo.ns)
	pk_interp = camb.get_matter_power_interpolator(pars, zs=np.linspace(np.min(zs), np.max(zs), 150),
	                        kmax=np.max(np.log10(k_grid)), nonlinear=nonlinear)
	return pk_interp


# write out power spectra in table for speed
# run this once with your desired minz, maxz, k grid
def write_power_spectra(minz, maxz, nonlinear):
	zs = np.linspace(minz, maxz, 1000)
	pk_interp = camb_matter_power_interpolator(zs, nonlinear)
	pks = pk_interp.P(zs, k_grid)
	writedict = {'zs': zs, 'ks': np.array(k_grid), 'Pk': pks}
	pickle.dump(writedict, open('power_spectra/nonlin_%s.p' % nonlinear, 'wb'))


# calculate power spectrum at given redshifts either by reading from table or using interpolator above
def power_spec_at_zs(zs, read=True, dimensionless=False):
	if read:
		pickled_powspectra = pickle.load(open('power_spectra/nonlin_False.p', 'rb'))
		if np.min(zs) < np.min(pickled_powspectra['zs']) or np.max(zs) > np.max(pickled_powspectra['zs']):
			print('Error: zs out of tabulated range')
			return
		z_idxs = np.digitize(zs, pickled_powspectra['zs']) - 1
		pks = pickled_powspectra['Pk'][z_idxs]
	else:
		pk_interp = camb_matter_power_interpolator(zs)
		pks = pk_interp.P(zs, k_grid)

	if dimensionless:
		pks = k_grid ** 3 / (2 * (np.pi ** 2)) * pks
	return pks


# DiPompeo 2017
def angular_corr_func(thetas, zs, dn_dz_1, dn_dz_2=None, hodparams=None, hodmodel=None, term='both'):

	# if not doing a cross correlation, term is dn/dz^2
	if dn_dz_2 is None:
		dn_dz_2 = dn_dz_1

	# thetas from degrees to radians
	thetas = (thetas*u.deg).to('radian').value

	# if specifying the model from an HOD
	if hodmodel is None:
		hodmodel = 'dm'

	#if hodmodel == '1param':
	#	onespec, twospec = [], []
	#	for j in range(len(zs)):
	#		onespec.append(np.zeros(len(k_grid)))
	#		twospec.append(cosmo.matterPowerSpectrum(k_grid.value, zs[j]))
	#else:
	onespec, twospec = hod_model.power_spectra_for_zs(zs, hodparams, modeltype=hodmodel)




	if term == 'both':
		powspec = np.array(onespec) + np.array(twospec)
	elif term == 'one':
		powspec = np.array(onespec)
	else:
		powspec = np.array(twospec)

	# Hankel transform of P(k,z) gives theta*xi grid, and the result of the k integral of Dipompeo+17 Eq. 2
	thetachis, dipomp_int = mcfit.Hankel(k_grid, lowring=True)(powspec, axis=1)

	# 2D grid of thetas * chi(z) to interpolate model power spectra onto
	input_theta_chis = np.outer(cosmo.comovingDistance(np.zeros(len(zs)), np.array(zs)), thetas)



	# for each redshift, chi(z), interpolate the result of the above integral onto theta*chi(z) grid
	interped_dipomp = []
	for j in range(len(zs)):
		interped_dipomp.append(interpolate_tools.log_interp1d(thetachis, dipomp_int[j])(input_theta_chis[j]))

	interped_dipomp = np.array(interped_dipomp)

	# convert H(z)/c from 1/Mpc to h/Mpc in order to cancel units of k
	dz_d_chi = (apcosmo.H(zs) / const.c).to(u.littleh / u.Mpc, cu.with_H0(apcosmo.H0)).value
	# product of redshift distributions, and dz/dchi
	differentials = dz_d_chi * dn_dz_1 * dn_dz_2

	z_int = 1 / (2 * np.pi) * np.trapz(differentials * np.transpose(interped_dipomp), x=zs, axis=1)



	return z_int







def angular_corr_func_in_bins(thetabins, zs, dn_dz_1, dn_dz_2=None, hodparams=None, hodmodel=None,
                              term='both'):
	thetagrid = np.logspace(np.log10(np.min(thetabins)), np.log10(np.max(thetabins)), 100)
	angcf = angular_corr_func(thetagrid, zs, dn_dz_1, dn_dz_2=dn_dz_2, hodparams=hodparams, hodmodel=hodmodel,
	                          term=term)
	# check why below gives different answer in last bin

	#binidxs = np.digitize(thetagrid, thetabins)

	#w_avg = []
	#for j in range(1, len(thetabins)):
	#	w_avg.append(np.mean(angcf[np.where(binidxs == j)]))
	return stats.binned_statistic(thetagrid, angcf, statistic='mean', bins=thetabins)[0]
	#print(w_avg, w_avg2, w_avg / w_avg2)
	#return np.array(w_avg)



def biased_ang_cf(thetas, b, zs, dn_dz_1, dn_dz_2=None):
	return (b ** 2) * angular_corr_func_in_bins(thetabins=thetas, zs=zs, dn_dz_1=dn_dz_1,
	                                            dn_dz_2=dn_dz_2)


def mass_biased_ang_cf(thetas, log_m, zs, dn_dz_1, dn_dz_2=None):
	b = bias_tools.mass_to_avg_bias(10**log_m, zs, dn_dz_1)
	return biased_ang_cf(thetas, b, zs, dn_dz_1, dn_dz_2)


def fit_bias(theta_data, w_data, w_errs, zs, dn_dz, mode='bias'):
	if mode == 'bias':
		partialfun = partial(biased_ang_cf, zs=zs, dn_dz_1=dn_dz)
		popt, pcov = curve_fit(partialfun, theta_data, w_data, sigma=w_errs, absolute_sigma=True)
	else:
		partialfun = partial(mass_biased_ang_cf, zs=zs, dn_dz_1=dn_dz)
		popt, pcov = curve_fit(partialfun, theta_data, w_data, sigma=w_errs, absolute_sigma=True, bounds=[11, 14])
	return popt[0], np.sqrt(pcov)[0][0]

# function taking 3 HOD parameters and returning angular CF. Used for initial least squares fit to give MCMC a good
# starting point
def three_param_hod_ang_cf(thetabins, m_min, alpha, m_1, zs, dn_dz_1):
	halomod_obj = hm_calcs.halomodel(zs=zs)
	halomod_obj.set_powspec(hodparams=[m_min, alpha, m_1], modeltype='3param')
	return halomod_obj.get_binned_ang_cf(dndz=[zs, dn_dz_1], theta_bins=thetabins)
# function taking 2 HOD parameters and returning angular CF. Used for initial least squares fit to give MCMC a good
# starting point
def two_param_hod_ang_cf(thetabins, m_min, alpha, zs, dn_dz_1):
	halomod_obj = hm_calcs.halomodel(zs=zs)
	halomod_obj.set_powspec(hodparams=[m_min, alpha], modeltype='2param')
	return halomod_obj.get_binned_ang_cf(dndz=[zs, dn_dz_1], theta_bins=thetabins)

# function taking 2 HOD parameters and returning angular CF. Used for initial least squares fit to give MCMC a good
# starting point
def one_param_hod_ang_cf(thetabins, m_min, zs, dn_dz_1):
	halomod_obj = hm_calcs.halomodel(zs=zs)
	halomod_obj.set_powspec(hodparams=[m_min], modeltype='1param')
	return halomod_obj.get_binned_ang_cf(dndz=[zs, dn_dz_1], theta_bins=thetabins)


# fitting CF with HOD model with least squares to give initial guess before running MCMC
def initial_hod_fit(theta_data, w_data, w_errs, zs, dn_dz, hodmodel):
	if hodmodel == '3param':
		partialfun = partial(three_param_hod_ang_cf, zs=zs, dn_dz_1=dn_dz)
		bounds = ([11., 0., 11.5], [15., 2., 15.])
	elif hodmodel == '2param':
		partialfun = partial(two_param_hod_ang_cf, zs=zs, dn_dz_1=dn_dz)
		bounds = ([11., 0.], [15., 2.])
	elif hodmodel == '1param':
		partialfun = partial(one_param_hod_ang_cf, zs=zs, dn_dz_1=dn_dz)
		bounds = ([11.], [15.])
	else:
		return 'error'

	popt, pcov = curve_fit(partialfun, theta_data, w_data, sigma=w_errs, absolute_sigma=True, bounds=bounds)

	return popt, np.sqrt(np.diag(pcov))


# assume that a fraction f of obscured quasars are just Type1 QSOs edge on, and assume a bias for the galaxy-obscured
# objects, then calculate the effective bias of a mixed sample
def bias_with_torus_fraction(b_qso2, torus_fraction):
	h=1

