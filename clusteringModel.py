import mcfit
import numpy as np
from colossus.cosmology import cosmology
from scipy.optimize import curve_fit
from scipy import stats
cosmo = cosmology.setCosmology('planck18')
from colossus.lss import bias
apcosmo = cosmo.toAstropy()

import camb
import matplotlib.pyplot as plt
from astropy.io import fits
#import mcfit
from functools import partial
import pickle
import scipy as sp
from scipy.special import j0
import astropy.constants as const
import astropy.units as u

import importlib
import redshift_dists
import resampling
from source import bias_tools
from source import interpolate_tools

importlib.reload(interpolate_tools)

importlib.reload(bias_tools)
importlib.reload(resampling)
importlib.reload(redshift_dists)



# define k space (includes little h)
kmax = 1000
k_grid = np.logspace(-4, np.log10(kmax), 1000) * (u.littleh / u.Mpc)




# set CAMB cosmology, generate power spectrum at 150 redshifts, return interpolator which can estimate power spectrum
# at any given redshift
def camb_matter_power_interpolator(zs, nonlinear=True):
	pars = camb.CAMBparams()
	pars.set_cosmology(H0=cosmo.H0, ombh2=cosmo.Ombh2, omch2=(cosmo.Omh2-cosmo.Ombh2), omk=cosmo.Ok0)
	pars.InitPower.set_params(ns=cosmo.ns)
	pk_interp = camb.get_matter_power_interpolator(pars, zs=np.linspace(np.min(zs), np.max(zs), 150), kmax=kmax,
	                                               nonlinear=nonlinear)
	return pk_interp


# write out power spectra in table for speed
# run this once with your desired minz, maxz, k grid
def write_power_spectra(minz, maxz, nonlinear):
	zs = np.linspace(minz, maxz, 1000)
	pk_interp = camb_matter_power_interpolator(zs, nonlinear)
	pks = pk_interp.P(zs, k_grid.value)
	writedict = {'zs': zs, 'ks': np.array(k_grid.value), 'Pk': pks}
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
		pks = pk_interp.P(zs, k_grid.value)

	if dimensionless:
		pks = k_grid.value**3 / (2 * (np.pi ** 2)) * pks
	return pks


# DiPompeo 2017
def angular_corr_func(thetas, zs, dn_dz_1, dn_dz_2=None, hodparams=None, hodmodel=None, term='both'):
	#import hod_model

	#importlib.reload(hod_model)

	# if not doing a cross correlation, term is dn/dz^2
	if dn_dz_2 is None:
		dn_dz_2 = dn_dz_1

	# thetas from degrees to radians
	thetas = (thetas*u.deg).to('radian').value



	if hodparams is not None:


		if hodmodel == '1param':
			onespec, twospec = [], []
			for j in range(len(zs)):
				onespec.append(np.zeros(len(k_grid)))
				twospec.append(cosmo.matterPowerSpectrum(k_grid.value, zs[j]))
		else:
			onespec, twospec = hod_model.power_spectra_for_zs(k_grid, zs, hodparams, modeltype=hodmodel)


		if term == 'both':
			powspec = np.array(onespec) + np.array(twospec)
		elif term == 'one':
			powspec = np.array(onespec)
		else:
			powspec = np.array(twospec)







		# Hankel transform of P(k,z) gives theta*xi grid, and the result of the k integral of Dipompeo+17 Eq. 2
		thetachis, dipomp_int = mcfit.Hankel(k_grid, lowring=True)(powspec, axis=1)

		input_theta_chis = np.outer(cosmo.comovingDistance(np.zeros(len(zs)), np.array(zs)), thetas)



		# for each redshift, chi(z), interpolate the result of the above integral onto theta*chi(z) grid
		interped_dipomp = []
		for j in range(len(zs)):
			interped_dipomp.append(interpolate_tools.log_interp1d(thetachis, dipomp_int[j])(input_theta_chis[j]))

		interped_dipomp = np.array(interped_dipomp)



		# Not sure if this is right
		# I think you need to convert H(z)/c from 1/Mpc to h/Mpc in order to cancel units of k, but not sure
		dz_d_chi = (apcosmo.H(zs) / const.c).to(u.littleh / u.Mpc, u.with_H0(apcosmo.H0)).value
		# product of redshift distributions, and dz/dchi
		differentials = dz_d_chi * dn_dz_1 * dn_dz_2

		z_int = 1 / (2 * np.pi) * np.trapz(differentials * np.transpose(interped_dipomp), x=zs, axis=1)

		return z_int

		"""rs, xis = mcfit.P2xi(k_grid, lowring=True)(powspec, axis=1)
				#rs = np.logspace(-3, 2.2, 500)
				#xis = xi_of_r(rs)
				first_bad_r_idx = np.where(rs > 150)[0][0]
				xis[:, first_bad_r_idx:] = 0
		chis = cosmo.comovingDistance(np.zeros(len(zs)), zs)

		u_grid = np.diff(chis)

		

		xi_at_r_grid = []
		for j in range(len(zs)):
			new_r_grid = np.sqrt(np.add.outer(u_grid ** 2, (cosmo.comovingDistance(0, zs[j]) ** 2) * (thetas ** 2)))
			interpedxi = log_interp1d(rs, xis[j])(new_r_grid)

			interpedxi[np.where(np.isnan(interpedxi))] = 0.
			xi_at_r_grid.append(interpedxi)


		firstintegral = np.trapz(xi_at_r_grid, dx=u_grid[0], axis=1)








		norm = 1. / np.trapz((chis ** 2) * dn_dz_1, x=chis)
		phi = norm * dn_dz_1
		z_int = 2 * np.trapz(((chis ** 4) * (phi ** 2)) * np.transpose(firstintegral), x=chis, axis=1)
		effb = bias_tools.mass_to_avg_bias(10 ** hodparams[0], zs, dn_dz_1)
		z_int = (effb ** 2) * z_int
		return z_int


		# Not sure if this is right
		# I think you need to convert H(z)/c from 1/Mpc to h/Mpc in order to cancel units of k, but not sure
		#dz_d_chi = (apcosmo.H(zs) / const.c).to(u.littleh / u.Mpc, u.with_H0(apcosmo.H0)).value
		dz_d_chi = (apcosmo.H(zs) / const.c).to(u.littleh / u.Mpc, u.with_H0(apcosmo.H0)).value
		# product of redshift distributions, and dz/dchi
		differentials = dz_d_chi * dn_dz_1 * dn_dz_2

		z_int = 2 * np.trapz(differentials * np.transpose(firstintegral), x=zs, axis=1)

		effb = bias_tools.mass_to_avg_bias(10**hodparams[0], zs, dn_dz_1)
		z_int = (effb ** 2) * z_int


		return z_int"""





	else:
		powspec = power_spec_at_zs(zs, read=True, dimensionless=False)
		powspec = []
		for z in zs:
			powspec.append(cosmo.matterPowerSpectrum(k_grid.value, z) )
		powspec = np.array(powspec) * (u.Mpc / u.littleh) ** 3

		first_term = powspec * k_grid


		# everything inside Bessel function
		# has 3 dimensions, k, theta, and z
		# therefore need to do outer product of two arrays, then broadcast 3rd array to 3D and multiply
		besselterm = j0(cosmo.comovingDistance(np.zeros(len(zs)), zs)[:, None, None] * np.outer(k_grid.value, thetas))

		# Not sure if this is right
		# I think you need to convert H(z)/c from 1/Mpc to h/Mpc in order to cancel units of k, but not sure
		dz_d_chi = (apcosmo.H(zs) / const.c).to(u.littleh/u.Mpc, u.with_H0(apcosmo.H0)).value

		# product of redshift distributions, and dz/dchi
		differentials = dz_d_chi * dn_dz_1 * dn_dz_2

		# total integrand is all terms multiplied out. This is a 3D array
		integrand = 1. / (2 * np.pi) * differentials * np.transpose(first_term) * np.transpose(
			besselterm)

		# do k integral first along k axis
		k_int = np.trapz(integrand, k_grid.value, axis=1)
		# then integrate along z axis
		return np.trapz(k_int, zs, axis=1)



def angular_corr_func_in_bins(thetabins, zs, dn_dz_1, dn_dz_2=None, hodparams=None, hodmodel=None, term='both'):
	thetagrid = np.logspace(np.log10(np.min(thetabins)), np.log10(np.max(thetabins)), 100)
	angcf = angular_corr_func(thetagrid, zs, dn_dz_1, dn_dz_2, hodparams=hodparams, hodmodel=hodmodel, term=term)
	# check why below gives different answer in last bin

	#binidxs = np.digitize(thetagrid, thetabins)

	#w_avg = []
	#for j in range(1, len(thetabins)):
	#	w_avg.append(np.mean(angcf[np.where(binidxs == j)]))
	return stats.binned_statistic(thetagrid, angcf, statistic='mean', bins=thetabins)[0]
	#print(w_avg, w_avg2, w_avg / w_avg2)
	#return np.array(w_avg)



def biased_ang_cf(thetas, b, zs, dn_dz_1, dn_dz_2=None):
	#return (b**2) * angular_corr_func(thetas=thetas, zs=zs, dn_dz_1=dn_dz_1, dn_dz_2=dn_dz_2)
	return (b ** 2) * angular_corr_func_in_bins(thetabins=thetas, zs=zs, dn_dz_1=dn_dz_1, dn_dz_2=dn_dz_2)


def mass_biased_ang_cf(thetas, m, zs, dn_dz_1, dn_dz_2=None):
	b = bias_tools.mass_to_avg_bias(m, zs, dn_dz_1)
	return biased_ang_cf(thetas, b, zs, dn_dz_1, dn_dz_2)


def fit_bias(theta_data, w_data, w_errs, zs, dn_dz, mode='bias'):
	if mode == 'bias':
		partialfun = partial(biased_ang_cf, zs=zs, dn_dz_1=dn_dz)
	else:
		partialfun = partial(mass_biased_ang_cf, zs=zs, dn_dz_1=dn_dz)
	popt, pcov = curve_fit(partialfun, theta_data, w_data, sigma=w_errs, absolute_sigma=True)
	return popt[0], np.sqrt(pcov)[0][0]

# function taking 3 HOD parameters and returning angular CF. Used for initial least squares fit to give MCMC a good
# starting point
def three_param_hod_ang_cf(thetabins, m_min, alpha, m_1, zs, dn_dz_1):
	return angular_corr_func_in_bins(thetabins=thetabins, zs=zs, dn_dz_1=dn_dz_1, hodparams=[m_min, alpha, m_1],
	                                 hodmodel='3param')
# function taking 2 HOD parameters and returning angular CF. Used for initial least squares fit to give MCMC a good
# starting point
def two_param_hod_ang_cf(thetabins, m_min, alpha, zs, dn_dz_1):
	return angular_corr_func_in_bins(thetabins=thetabins, zs=zs, dn_dz_1=dn_dz_1, hodparams=[m_min, alpha],
	                                 hodmodel='2param')

# function taking 2 HOD parameters and returning angular CF. Used for initial least squares fit to give MCMC a good
# starting point
def one_param_hod_ang_cf(thetabins, m_min, zs, dn_dz_1):
	return angular_corr_func_in_bins(thetabins=thetabins, zs=zs, dn_dz_1=dn_dz_1, hodparams=[m_min],
	                                 hodmodel='1param')


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





zlist = np.linspace(0.1, 3, 100)
dndz = np.ones(len(zlist))
norm = 1 / np.trapz(zlist * dndz, x=zlist)
dndz = dndz * norm
angles = np.logspace(-2, 1, 500)
newcf = angular_corr_func(angles, zs=zlist, dn_dz_1=dndz, hodparams=[12.], hodmodel='1param')
oldcf = angular_corr_func(angles, zs=zlist, dn_dz_1=dndz, hodparams=None)



plt.figure(figsize=(8,7))
plt.plot(angles, oldcf, label="old")
plt.plot(angles, newcf, label="new")
#plt.plot(angles, oldercf, label='older')

#plt.plot(newcf[0], newcf[1])
#plt.plot(newcf[0], oldercf[:, 0])

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig('cf.png')
plt.close('all')

