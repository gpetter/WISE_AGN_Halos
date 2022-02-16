import healpy as hp
import numpy as np
from astropy import constants as const
import astropy.units as u
from colossus.halo import concentration
from colossus.halo import profile_nfw
from colossus.cosmology import cosmology
from colossus.lss import bias
from scipy.special import j0
from scipy.optimize import curve_fit
from astropy.io import fits
import pymaster as nmt
import fitting
from functools import partial
import importlib
import redshift_dists
from source import bias_tools
import clusteringModel
importlib.reload(clusteringModel)
importlib.reload(redshift_dists)
importlib.reload(bias_tools)
importlib.reload(fitting)

cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()


# defining the critical surface density for lensing
def sigma_crit(z):
	return ((const.c ** 2) / (4. * np.pi * const.G) * (apcosmo.angular_diameter_distance(1100.) / (
		(apcosmo.angular_diameter_distance(z) * apcosmo.angular_diameter_distance_z1z2(z, 1100.))))).decompose().to(
		u.solMass * u.littleh / u.kpc ** 2, u.with_H0(apcosmo.H0))


# calculate the concentration of a halo given a mass and redshift using the Ludlow+16 model
def calc_concentration(m_200, z):

	c_200 = concentration.modelLudlow16(m_200, z)
	if c_200[1]:
		return c_200[0]
	else:
		return np.nan


# converting a radial NFW profile to an angular surface mass density profile given a halo mass and redshift
def nfw_sigma(theta, m_200, z):

	# define a NFW profile in terms of the halo mass and concentration parameter c_200
	p_nfw = profile_nfw.NFWProfile(M=m_200, c=calc_concentration(m_200, z), z=z, mdef='200c')
	# angular diameter distance
	d_a = apcosmo.angular_diameter_distance(z).to(u.kpc/u.littleh, u.with_H0(apcosmo.H0)).value
	return p_nfw.surfaceDensity(r=theta*d_a)


# Takes in a redshift and halo mass and returns the convergence predicted at an angle theta due to a NFW halo
def kappa_1_halo(theta, mass_or_bias, z, mode='mass'):
	if mode == 'mass':
		m_200 = mass_or_bias
	elif mode == 'bias':
		m_200 = bias_tools.bias_to_mass(mass_or_bias, z)
	else:
		return
	return nfw_sigma(theta, m_200, z)/sigma_crit(z).value





# estimate lensing convergence due to correlated large scale structure to a DM halo of a given mass
def two_halo_term(theta, mass_or_bias, z, mode='mass'):

	d_a = apcosmo.angular_diameter_distance(z).to(u.kpc/u.littleh, u.with_H0(apcosmo.H0))     # kpc/h

	if mode == 'mass':
		# calculate halo bias through Tinker+10 model
		bh = bias.haloBias(M=mass_or_bias, z=z, mdef='200c', model='tinker10')
	elif mode == 'bias':
		bh = mass_or_bias
	else:
		return

	# the average (matter) density of the universe

	rho_avg = cosmo.rho_m(z)*u.solMass*(u.littleh**2)/(u.kpc**3)
	# amalgamation of constants outside the integral
	a = (rho_avg/(((1.+z)**3)*sigma_crit(z)*d_a**2))*bh/(2*np.pi)

	# scales
	ks = np.logspace(-5, 0, 200)*u.littleh/u.Mpc
	ls = ks*(1+z)*(apcosmo.angular_diameter_distance(z).to(u.Mpc/u.littleh, u.with_H0(apcosmo.H0)))

	# do an outer product of the thetas and ls for integration
	ltheta = np.outer(theta, ls)

	# compute matter power spectrum at comoving wavenumbers k
	mps = (cosmo.matterPowerSpectrum(ks.value, z=z)*(u.Mpc/u.littleh)**3).to((u.kpc/u.littleh)**3)

	# Eq. 13 in OOguri and Hamana 2011
	integrand = a*ls*j0(ltheta)*mps
	return np.trapz(integrand, x=ls)


# integrate kappa across redshift distribution dn/dz
def int_kappa(theta, mass_or_bias, terms, zdist, zbins=100, mode='mass'):

	# bin up redshift distribution of sample to integrate kappa over
	hist = np.histogram(zdist, zbins, density=True)

	avg_kappa = []
	zs = hist[1]
	dz = zs[1] - zs[0]
	# chop off last entry which is a rightmost bound of the z distribution
	zs = np.resize(zs, zs.size-1) + dz/2


	dndz = hist[0]

	for i in range(len(dndz)):
		z = zs[i] + dz/2
		if terms == 'one':
			avg_kappa.append(kappa_1_halo(theta, mass_or_bias, z, mode=mode)*dndz[i])
		elif terms == 'two':
			avg_kappa.append(two_halo_term(theta, mass_or_bias, z, mode=mode)*dndz[i])
		elif terms == 'both':
			avg_kappa.append((kappa_1_halo(theta, mass_or_bias, z, mode=mode) + two_halo_term(theta, mass_or_bias, z,
			                                            mode=mode))*dndz[i])
		else:
			return False
	avg_kappa = np.array(avg_kappa)

	return np.trapz(avg_kappa, dx=dz, axis=0)


# apply same filter applied to the map to the model
# gaussian with small l modes zeroed
def filter_model(zdist, mass_or_bias, mode='mass'):
	theta_list_rad = (np.arange(0.1, 360, 0.1) * u.arcmin).to('rad').value

	bothmodel = int_kappa(theta_list_rad, mass_or_bias, 'both', zdist, mode=mode)
	kmodel = hp.beam2bl(bothmodel, theta_list_rad, lmax=4096)

	k_space_filter = hp.gauss_beam((15 * u.arcmin).to('rad').value, lmax=4096)
	k_space_filter[:100] = 0

	kconvolved = np.array(kmodel) * np.array(k_space_filter)

	return hp.bl2beam(kconvolved, theta_list_rad)


def filtered_model_at_theta(zdist, mass_or_bias, inputthetas, mode='mass'):
	theta_list = np.arange(0.1, 360, 0.1)
	model = filter_model(zdist, mass_or_bias, mode=mode)
	flat_thetas = inputthetas.flatten()
	kappa_vals = []
	for j in range(len(flat_thetas)):
		kappa_vals.append(model[np.abs(theta_list - flat_thetas[j]).argmin()])

	return np.array(kappa_vals).reshape(inputthetas.shape)









# simulate a stacked map usin
def model_stacked_map(zdist, mass_or_bias, imsize=240, reso=1.5, mode='mass'):

	center = imsize/2 - 0.5
	x_arr, y_arr = np.mgrid[0:imsize, 0:imsize]
	radii_theta = np.sqrt(((x_arr - center) * reso) ** 2 + ((y_arr - center) * reso) ** 2)
	model_kappas = filtered_model_at_theta(zdist, mass_or_bias, radii_theta, mode=mode)
	return model_kappas




def filtered_model_center(zdist, obs_theta, mass_or_bias, reso=1.5, imsize=240, mode='mass'):

	modelmap = model_stacked_map(zdist, mass_or_bias, reso=reso, imsize=imsize, mode=mode)
	return modelmap[int(imsize/2), int(imsize/2)]


# calculate average model value in same bins as measured
def filtered_model_in_bins(zdist, obs_thetas, mass_or_bias, binsize=12, reso=1.5, imsize=240, maxtheta=180,
                           mode='mass'):

	modelmap = model_stacked_map(zdist, mass_or_bias, reso=reso, imsize=imsize, mode=mode)

	profile = fitting.measure_profile(modelmap, binsize, reso, maxtheta=maxtheta)
	return profile


# used Liu et al. 2015 (Cross-correlation of Planck CMB lensing ...)
# return lensing kernel for CMB (if no source redshifts provided
# or lensing kernel for galaxy weak lensing if source redshifts are provided
def lensing_kernel(lens_zs, source_zs=None):
	prefactor = 3. / 2. * apcosmo.Om0 * ((apcosmo.H0/const.c) ** 2) * (1 + lens_zs) * \
	            apcosmo.comoving_distance(lens_zs)

	if source_zs is None:
		wk = prefactor * (apcosmo.comoving_distance(1100)-apcosmo.comoving_distance(
			lens_zs)) / apcosmo.comoving_distance(1100)
	else:

		source_dist = np.histogram(source_zs, 200, density=True)
		dndz, source_bins = source_dist[0], source_dist[1]
		source_bin_centers = (source_bins[1:] + source_bins[:-1])/2

		integrals = []
		for lens_z in lens_zs:
			idxs = np.where(source_bin_centers > lens_z)

			if len(idxs[0]) > 1:
				new_source_zs, new_dndz = source_bin_centers[idxs], dndz[idxs]
				integrand = new_dndz * (apcosmo.comoving_distance(new_source_zs) - apcosmo.comoving_distance(lens_z))\
				            / apcosmo.comoving_distance(new_source_zs)

				integrals.append(np.trapz(integrand, new_source_zs))
			else:
				integrals.append(0)
		wk = prefactor * np.array(integrals)


	return wk

def dx_dz_lensing_kernel(lens_zs, source_zs=None):
	return const.c / apcosmo.H(lens_zs) * lensing_kernel(lens_zs, source_zs)



def dm_halo_kernel(lens_zs, dndz, b_of_z):
	return b_of_z * apcosmo.H(lens_zs) / const.c * dndz


def x_power_spectrum(ls, zs, dndz, log_eff_mass=None):
	kmax = 1000
	k_grid = (np.logspace(-5, np.log10(kmax), 1000) * (u.littleh / u.Mpc))

	lenskern = lensing_kernel(zs)
	if log_eff_mass is not None:
		b_of_z = bias.haloBias(M=10**log_eff_mass, z=zs, mdef='200c', model='tinker10')
	else:
		b_of_z = np.ones(len(zs))
	qsokern = dm_halo_kernel(zs, dndz, b_of_z)

	integrand = const.c * lenskern * qsokern / ((apcosmo.comoving_distance(zs) ** 2) * apcosmo.H(zs))
	ks = np.outer(1. / cosmo.comovingDistance(np.zeros(len(zs)), zs), (ls + 1/2.))
	#pk_of_z = np.array(clusteringModel.power_spec_at_zs(zs))


	ps_at_ks = []
	for j in range(len(zs)):
		pk_z = cosmo.matterPowerSpectrum(k_grid.value, z=zs[j])
		ps_at_ks.append(np.interp(ks[j], k_grid.value, pk_z))
	ps_at_ks = np.array(ps_at_ks) * (u.Mpc / u.littleh) ** 3


	integrand = integrand[:, None] * ps_at_ks

	integral = np.trapz(integrand, zs, axis=0)

	return integral.to(u.dimensionless_unscaled, u.with_H0(apcosmo.H0))


	
def binned_x_power_spectrum(zs, dndz, log_eff_mass=None):
	xpower = x_power_spectrum(np.arange(2048)+1, zs, dndz, log_eff_mass=log_eff_mass)


	wsp = nmt.NmtWorkspace()
	wsp.read_from('masks/workspace.fits')

	binned_xpow = wsp.decouple_cell(wsp.couple_cell([xpower]))
	return binned_xpow[0]

def biased_binned_x_power_spectrum(foo, b, zs, dndz):
	return b * binned_x_power_spectrum(zs, dndz)

def mass_biased_x_power_spectrum(foo, log_m, zs, dndz):
	#b = bias_tools.mass_to_avg_bias(m, zs, dndz)
	return binned_x_power_spectrum(zs, dndz, log_eff_mass=log_m)

def fit_bias(data, errs, zs, dndz):
	partialfun = partial(biased_binned_x_power_spectrum, zs=zs, dndz=dndz)

	popt, pcov = curve_fit(partialfun, np.ones(len(data)), data, sigma=errs, absolute_sigma=True)
	return popt[0], np.sqrt(pcov)[0][0]

def fit_mass(data, errs, zs, dndz):
	partialfun = partial(mass_biased_x_power_spectrum, zs=zs, dndz=dndz)
	popt, pcov = curve_fit(partialfun, np.ones(len(data)), data, sigma=errs, absolute_sigma=True, bounds=[11, 14])
	return popt[0], np.sqrt(pcov)[0][0]


def mass_to_avg_bias(m_per_h, zs):
	medz = np.median(zs)
	bh = bias.haloBias(M=m_per_h, z=medz, mdef='200c', model='tinker10')
	return bh

def kappa_mass_relation(zdist, eval_kappas):
	masses = np.logspace(11, 13.5, 20)
	kappas = []
	for j in range(len(masses)):
		kappas.append(filtered_model_center(zdist, 0, masses[j]))
	logmasses = np.log10(masses)
	polycoeffs = np.polyfit(kappas, logmasses, 2)
	return np.polyval(polycoeffs, eval_kappas)


