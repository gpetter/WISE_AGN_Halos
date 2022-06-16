from astropy.table import Table, vstack
import numpy as np
from scipy.special import gamma
from scipy.optimize import curve_fit
from functools import partial
import healpixhelper
import healpy as hp
import masking
import glob
import twoPointCFs
from source import interpolate_tools
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck18')

# defined under equation 3 in Matthews and Newman 2010 clustering redshifts
def h_of_gamma(little_gamma):
	return gamma(0.5) * gamma((little_gamma - 1) / 2.) / gamma(little_gamma / 2.)

def combine_lss_catalogs(n_randoms_to_n_data=10, nside_mask=256, remove_redshift_outliers=False):
	lowz_north_data = Table.read('../data/lss/BOSS_LOWZ/LOWZ_North.fits')
	lowz_south_data = Table.read('../data/lss/BOSS_LOWZ/LOWZ_South.fits')
	lowz_north_randoms = Table.read('../data/lss/BOSS_LOWZ/randoms_LOWZ_North.fits')
	lowz_south_randoms = Table.read('../data/lss/BOSS_LOWZ/randoms_LOWZ_South.fits')


	lowz_north_data['WEIGHT'] = lowz_north_data['WEIGHT_SYSTOT'] * (lowz_north_data['WEIGHT_CP'] + lowz_north_data[
		'WEIGHT_NOZ'] - 1)

	lowz_south_data['WEIGHT'] = lowz_south_data['WEIGHT_SYSTOT'] * (lowz_south_data['WEIGHT_CP'] + lowz_south_data[
		'WEIGHT_NOZ'] - 1)

	lowz_north_randoms['WEIGHT'] = np.ones(len(lowz_north_randoms))
	lowz_south_randoms['WEIGHT'] = np.ones(len(lowz_south_randoms))

	lowz_north_data = lowz_north_data['RA', 'DEC', 'Z', 'WEIGHT']
	lowz_south_data = lowz_south_data['RA', 'DEC', 'Z', 'WEIGHT']
	lowz_north_randoms = lowz_north_randoms['RA', 'DEC', 'Z', 'WEIGHT']
	lowz_south_randoms = lowz_south_randoms['RA', 'DEC', 'Z', 'WEIGHT']
	lowz_hpx_map = healpixhelper.healpix_density_map(list(lowz_north_randoms['RA']) + list(lowz_south_randoms['RA']),
	                                                 list(lowz_north_randoms['DEC']) + list(lowz_south_randoms['DEC']),
	                                                 nsides=nside_mask)
	if remove_redshift_outliers:
		lowz_north_data = lowz_north_data[np.where((lowz_north_data['Z'] > 0.1) & (lowz_north_data['Z'] < 0.5))]
		lowz_south_data = lowz_south_data[np.where((lowz_south_data['Z'] > 0.1) & (lowz_south_data['Z'] < 0.5))]
		lowz_north_randoms = lowz_north_randoms[np.where((lowz_north_randoms['Z'] > 0.1) &
		                                                 (lowz_north_randoms['Z'] < 0.5))]
		lowz_south_randoms = lowz_south_randoms[np.where((lowz_south_randoms['Z'] > 0.1) &
		                                           (lowz_south_randoms['Z'] < 0.5))]

	lowz_north_randoms = lowz_north_randoms[:n_randoms_to_n_data*len(lowz_north_data)]
	lowz_south_randoms = lowz_south_randoms[:n_randoms_to_n_data*len(lowz_south_data)]



	lowz_data = vstack([lowz_north_data, lowz_south_data])
	lowz_randoms = vstack([lowz_north_randoms, lowz_south_randoms])


	lowz_data.write('../data/lss/Combined/lowz_data.fits', format='fits', overwrite=True)
	lowz_randoms.write('../data/lss/Combined/lowz_randoms.fits', format='fits', overwrite=True)
	del lowz_north_data, lowz_north_randoms, lowz_south_data, lowz_south_randoms

	cmass_north_data = Table.read('../data/lss/BOSS_CMASS/CMASS_North.fits')
	cmass_south_data = Table.read('../data/lss/BOSS_CMASS/CMASS_South.fits')
	cmass_north_randoms = Table.read('../data/lss/BOSS_CMASS/randoms_CMASS_North.fits')
	cmass_south_randoms = Table.read('../data/lss/BOSS_CMASS/randoms_CMASS_South.fits')

	cmass_north_data['WEIGHT'] = cmass_north_data['WEIGHT_SYSTOT'] * (cmass_north_data['WEIGHT_CP'] + cmass_north_data[
		'WEIGHT_NOZ'] - 1)
	cmass_south_data['WEIGHT'] = cmass_south_data['WEIGHT_SYSTOT'] * (cmass_south_data['WEIGHT_CP'] + cmass_south_data[
		'WEIGHT_NOZ'] - 1)
	cmass_north_randoms['WEIGHT'] = np.ones(len(cmass_north_randoms))
	cmass_south_randoms['WEIGHT'] = np.ones(len(cmass_south_randoms))

	cmass_north_data = cmass_north_data['RA', 'DEC', 'Z', 'WEIGHT']
	cmass_south_data = cmass_south_data['RA', 'DEC', 'Z', 'WEIGHT']
	cmass_north_randoms = cmass_north_randoms['RA', 'DEC', 'Z', 'WEIGHT']
	cmass_south_randoms = cmass_south_randoms['RA', 'DEC', 'Z', 'WEIGHT']
	cmass_hpx_map = healpixhelper.healpix_density_map(list(cmass_north_randoms['RA']) +
	                                                  list(cmass_south_randoms['RA']),
	                                                 list(cmass_north_randoms['DEC']) +
	                                                  list(cmass_south_randoms['DEC']),
	                                                 nsides=nside_mask)
	if remove_redshift_outliers:
		cmass_north_data = cmass_north_data[np.where((cmass_north_data['Z'] > 0.1) & (cmass_north_data['Z'] < 0.8))]
		cmass_south_data = cmass_south_data[np.where((cmass_south_data['Z'] > 0.1) & (cmass_south_data['Z'] < 0.8))]
		cmass_north_randoms = cmass_north_randoms[np.where((cmass_north_randoms['Z'] > 0.1) &
		                                                 (cmass_north_randoms['Z'] < 0.8))]
		cmass_south_randoms = cmass_south_randoms[np.where((cmass_south_randoms['Z'] > 0.1) &
		                                           (cmass_south_randoms['Z'] < 0.8))]


	cmass_north_randoms = cmass_north_randoms[:n_randoms_to_n_data * len(cmass_north_data)]
	cmass_south_randoms = cmass_south_randoms[:n_randoms_to_n_data * len(cmass_south_data)]

	cmass_data = vstack([cmass_north_data, cmass_south_data])
	cmass_randoms = vstack([cmass_north_randoms, cmass_south_randoms])

	cmass_data.write('../data/lss/Combined/cmass_data.fits', format='fits', overwrite=True)
	cmass_randoms.write('../data/lss/Combined/cmass_randoms.fits', format='fits', overwrite=True)
	del cmass_north_data, cmass_north_randoms, cmass_south_data, cmass_south_randoms


	eboss_qso_north_data = Table.read('../data/lss/eBOSS_QSO/NGC_comov.fits')
	eboss_qso_south_data = Table.read('../data/lss/eBOSS_QSO/SGC_comov.fits')
	eboss_qso_north_randoms = Table.read('../data/lss/eBOSS_QSO/randoms_NGC_comov.fits')
	eboss_qso_south_randoms = Table.read('../data/lss/eBOSS_QSO/randoms_SGC_comov.fits')

	eboss_qso_north_data['WEIGHT'] = eboss_qso_north_data['WEIGHT_SYSTOT'] * eboss_qso_north_data['WEIGHT_NOZ'] * \
	                                 eboss_qso_north_data['WEIGHT_CP']
	eboss_qso_south_data['WEIGHT'] = eboss_qso_south_data['WEIGHT_SYSTOT'] * eboss_qso_south_data['WEIGHT_NOZ'] * \
	                                 eboss_qso_south_data['WEIGHT_CP']

	eboss_qso_north_randoms['WEIGHT'] = eboss_qso_north_randoms['WEIGHT_SYSTOT'] * \
	                                    eboss_qso_north_randoms['WEIGHT_NOZ'] * \
	                                    eboss_qso_north_randoms['WEIGHT_CP']
	eboss_qso_south_randoms['WEIGHT'] = eboss_qso_south_randoms['WEIGHT_SYSTOT'] * \
	                                    eboss_qso_south_randoms['WEIGHT_NOZ'] * \
	                                    eboss_qso_south_randoms['WEIGHT_CP']
	eboss_qso_north_data = eboss_qso_north_data['RA', 'DEC', 'Z', 'WEIGHT']
	eboss_qso_south_data = eboss_qso_south_data['RA', 'DEC', 'Z', 'WEIGHT']
	eboss_qso_north_randoms = eboss_qso_north_randoms['RA', 'DEC', 'Z', 'WEIGHT']
	eboss_qso_south_randoms = eboss_qso_south_randoms['RA', 'DEC', 'Z', 'WEIGHT']
	eboss_qso_hpx_map = healpixhelper.healpix_density_map(list(eboss_qso_north_randoms['RA']) +
	                                                 list(eboss_qso_south_randoms['RA']),
	                                                 list(eboss_qso_north_randoms['DEC']) +
	                                                 list(eboss_qso_south_randoms['DEC']),
	                                                 nsides=nside_mask)

	eboss_qso_north_randoms = eboss_qso_north_randoms[:n_randoms_to_n_data * len(eboss_qso_north_data)]
	eboss_qso_south_randoms = eboss_qso_south_randoms[:n_randoms_to_n_data * len(eboss_qso_south_data)]

	eboss_qso_data = vstack([eboss_qso_north_data, eboss_qso_south_data])
	eboss_qso_randoms = vstack([eboss_qso_north_randoms, eboss_qso_south_randoms])

	eboss_qso_data.write('../data/lss/Combined/eBOSS_QSO.fits', format='fits', overwrite=True)
	eboss_qso_randoms.write('../data/lss/Combined/eBOSS_QSO_randoms.fits', format='fits', overwrite=True)
	del eboss_qso_north_randoms, eboss_qso_south_randoms, eboss_qso_north_data, eboss_qso_south_data

	lrg_north_data = Table.read('../data/lss/LRG/NGC_comov.fits')
	lrg_south_data = Table.read('../data/lss/LRG/SGC_comov.fits')
	lrg_north_randoms = Table.read('../data/lss/LRG/randoms_NGC_comov.fits')
	lrg_south_randoms = Table.read('../data/lss/LRG/randoms_SGC_comov.fits')

	lrg_north_data['WEIGHT'] = lrg_north_data['WEIGHT_SYSTOT'] * lrg_north_data['WEIGHT_NOZ'] * \
	                           lrg_north_data['WEIGHT_CP']
	lrg_south_data['WEIGHT'] = lrg_south_data['WEIGHT_SYSTOT'] * lrg_south_data['WEIGHT_NOZ'] * \
	                           lrg_south_data['WEIGHT_CP']

	lrg_north_randoms['WEIGHT'] = lrg_north_randoms['WEIGHT_SYSTOT'] * \
	                              lrg_north_randoms['WEIGHT_NOZ'] * \
	                              lrg_north_randoms['WEIGHT_CP']
	lrg_south_randoms['WEIGHT'] = lrg_south_randoms['WEIGHT_SYSTOT'] * \
	                              lrg_south_randoms['WEIGHT_NOZ'] * \
	                              lrg_south_randoms['WEIGHT_CP']
	lrg_north_data = lrg_north_data['RA', 'DEC', 'Z', 'WEIGHT']
	lrg_south_data = lrg_south_data['RA', 'DEC', 'Z', 'WEIGHT']
	lrg_north_randoms = lrg_north_randoms['RA', 'DEC', 'Z', 'WEIGHT']
	lrg_south_randoms = lrg_south_randoms['RA', 'DEC', 'Z', 'WEIGHT']
	lrg_hpx_map = healpixhelper.healpix_density_map(list(lrg_north_randoms['RA']) +
	                                                      list(lrg_south_randoms['RA']),
	                                                      list(lrg_north_randoms['DEC']) +
	                                                      list(lrg_south_randoms['DEC']),
	                                                      nsides=nside_mask)

	lrg_north_randoms = lrg_north_randoms[:n_randoms_to_n_data * len(lrg_north_data)]
	lrg_south_randoms = lrg_south_randoms[:n_randoms_to_n_data * len(lrg_south_data)]

	lrg_data = vstack([lrg_north_data, lrg_south_data])
	lrg_randoms = vstack([lrg_north_randoms, lrg_south_randoms])

	lrg_data.write('../data/lss/Combined/LRG.fits', format='fits', overwrite=True)
	lrg_randoms.write('../data/lss/Combined/LRG_randoms.fits', format='fits', overwrite=True)




	boss_qso_data = Table.read('../data/lss/BOSS_QSO/data.fits')
	boss_qso_randoms = Table.read('../data/lss/BOSS_QSO/randoms.fits')
	boss_qso_hpx_map = healpixhelper.healpix_density_map(boss_qso_randoms['RA'], boss_qso_randoms['DEC'],
	                                                     nsides=nside_mask)
	boss_qso_data['WEIGHT'], boss_qso_randoms['WEIGHT'] = np.ones(len(boss_qso_data)), np.ones(len(boss_qso_randoms))
	boss_qso_data = boss_qso_data['RA', 'DEC', 'Z', 'WEIGHT']
	boss_qso_randoms = boss_qso_randoms['RA', 'DEC', 'Z', 'WEIGHT']

	if remove_redshift_outliers:
		boss_qso_data = boss_qso_data[np.where(boss_qso_data['Z'] > 2.2)]
		boss_qso_randoms = boss_qso_randoms[np.where(boss_qso_randoms['Z'] > 2.2)]

	boss_qso_randoms = boss_qso_randoms[:n_randoms_to_n_data * len(boss_qso_data)]
	boss_qso_data.write('../data/lss/Combined/BOSS_QSO.fits', format='fits', overwrite=True)
	boss_qso_randoms.write('../data/lss/Combined/BOSS_QSO_randoms.fits', format='fits', overwrite=True)

	#combined_hpx_map = lowz_hpx_map * cmass_hpx_map * eboss_qso_hpx_map * lrg_hpx_map * boss_qso_hpx_map
	#hp.write_map('../data/lss/Combined/all_hpx_map.fits', combined_hpx_map, overwrite=True)







	fulldata = vstack([lowz_data, cmass_data, eboss_qso_data, lrg_data, boss_qso_data])
	del lowz_data, cmass_data, eboss_qso_data, lrg_data, boss_qso_data
	fullrandoms = vstack([lowz_randoms, cmass_randoms, eboss_qso_randoms, lrg_randoms, boss_qso_randoms])
	del lowz_randoms, cmass_randoms, eboss_qso_randoms, lrg_randoms
	#fulldata = fulldata[np.where(combined_hpx_map[hp.ang2pix(nside=nside_mask, theta=fulldata['RA'], phi=fulldata[
	# 'DEC'],
	#                                                         lonlat=True)] > 0)]
	#fulldata = masking.mask_tab(fulldata)
	#fullrandoms = fullrandoms[np.where(combined_hpx_map[hp.ang2pix(nside=nside_mask, theta=fullrandoms['RA'],
	#                                                               phi=fullrandoms['DEC'], lonlat=True)] > 0)]
	#fullrandoms = masking.mask_tab(fullrandoms)
	fulldata.write('../data/lss/Combined/all.fits', format='fits', overwrite=True)
	fullrandoms.write('../data/lss/Combined/all_randoms.fits', format='fits', overwrite=True)
	del fulldata, fullrandoms



	wise_agn_cat = Table.read('catalogs/derived/catwise_binned.fits')
	#wise_agn_cat = wise_agn_cat[np.where(combined_hpx_map[hp.ang2pix(nside=nside_mask,
	#                                    theta=wise_agn_cat['RA'], phi=wise_agn_cat['DEC'], lonlat=True)] > 0)]
	wise_agn_cat['WEIGHT'] = np.ones(len(wise_agn_cat))
	wise_agn_cat = wise_agn_cat['RA', 'DEC', 'WEIGHT', 'bin']
	wise_agn_cat.write('../data/lss/Combined/wise_agn.fits', format='fits', overwrite=True)

	wise_random_files = glob.glob('catalogs/derived/catwise_randoms_*')
	for j, filename in enumerate(wise_random_files):
		binsize = len(wise_agn_cat[np.where(wise_agn_cat['bin'] == (j+1))])
		randtab = Table.read(filename)
		randtab = randtab[:n_randoms_to_n_data * binsize]

		#randtab = randtab[np.where(combined_hpx_map[hp.ang2pix(nside=nside_mask,
	    #                                theta=randtab['RA'], phi=randtab['DEC'], lonlat=True)] > 0)]
		randtab['WEIGHT'] = np.ones(len(randtab))
		randtab = randtab['RA', 'DEC', 'WEIGHT']
		randtab.write('../data/lss/Combined/wise_randoms_%s.fits' % (j+1), format='fits', overwrite=True)


def power_law_projected_cf_model(rp, r0, littlegamma):
	return h_of_gamma(littlegamma) * rp * (r0 / rp) ** littlegamma

def power_law_angular_cf_model(theta, a, gamma):
	return a * theta ** (1 - gamma)

def fit_projected_cf(rps, wps, wperrs):
	popt, pcov = curve_fit(power_law_projected_cf_model, rps, wps, sigma=wperrs)
	return popt[0], popt[1]

def fit_angular_ccf(thetas, wtheta_sp, werr_sp, gamma_sp=None):
	if gamma_sp is not None:
		partial_ang_cf_model = partial(power_law_angular_cf_model, gamma_sp=gamma_sp)
		popt, pcov = curve_fit(partial_ang_cf_model, thetas, wtheta_sp, sigma=werr_sp)
	else:
		popt, pcov = curve_fit(power_law_angular_cf_model, thetas, wtheta_sp, sigma=werr_sp)
	return popt[0], popt[1]

def fit_angular_autocf(thetas, wtheta_pp, werr_pp):
	popt, pcov = curve_fit(power_law_angular_cf_model, thetas, wtheta_pp, sigma=werr_pp)
	return popt[0], popt[1]

def photometric_redshift_distribution(z, a_sp_z, gamma_sp, r0_sp):
	dchidz = 1 # h/c?
	ang_dist_term = 1
	return dchidz * a_sp_z / (ang_dist_term * h_of_gamma(gamma_sp) * r0_sp ** gamma_sp)

# step 6 in paper
def r0_sp_to_power_gamma_sp(r0_ss, gamma_ss, gamma_pp, r0_pp=None):
	if r0_pp is None:
		# assume r0 of photometric sample is similar to r0 of spectroscopic sample
		r0_pp = r0_ss

	return np.sqrt((r0_ss ** gamma_ss) * (r0_pp ** gamma_pp))

def gamma_sp_estimate(gamma_ss, gamma_pp):
	return (gamma_ss + gamma_pp) / 2.

def limber_equation(theta, r0_pp, gamma_pp, zs, dndz_pp):
	big_h = h_of_gamma(gamma_pp)
	prefactor = big_h * theta ** (1 - gamma_pp)
	angdist_term = 1
	dchi_dz = 1
	integrand = (dndz_pp ** 2) * (r0_pp ** gamma_pp) * angdist_term / dchi_dz
	integral = np.trapz(integrand, x=zs)
	return prefactor * integral


def fit_limber_for_r0_pp(theta_pp, w_pp, werr_pp, gamma_pp, zs, dndz_pp):
	partiallimber = partial(limber_equation, gamma_pp=gamma_pp, zs=zs, dndz_pp=dndz_pp)
	popt, pcov = curve_fit(partiallimber, theta_pp, w_pp, sigma=werr_pp)
	return popt[0]


def full_routine_for_z(z, rps, wps, wperrs, theta_sp, w_sp, w_err_sp, gamma_pp, r0_pp=None):
	r0_ss, gamma_ss = fit_projected_cf(rps, wps, wperrs)
	r0_sp_to_gamma_sp_guess = r0_sp_to_power_gamma_sp(r0_ss=r0_ss, gamma_ss=gamma_ss, gamma_pp=gamma_pp, r0_pp=r0_pp)
	gamma_sp_guess = gamma_sp_estimate(gamma_ss, gamma_pp)
	a_sp, gamma_sp = fit_angular_ccf(thetas=theta_sp, wtheta_sp=w_sp, werr_sp=w_err_sp, gamma_sp=gamma_sp_guess)
	r0_sp_guess = r0_sp_to_gamma_sp_guess ** (1. / gamma_sp)
	initial_dndz_pp = photometric_redshift_distribution(z=z, a_sp_z=a_sp, gamma_sp=gamma_sp, r0_sp=r0_sp_guess)
	return initial_dndz_pp




def measure_all_cfs(zbins):
	rps_z, wps_z, wperr_z, theta_sp_z, w_sp_z, werr_sp_z = [], [], [], [], [], []
	for j in range(zbins - 1):
		zmin, zmax = zbins[j], zbins[j+1]

		rps, wps, wperrs = measure_autocorr(zmin, zmax)
		rps_z.append(rps), wps_z.append(wps), wperr_z.append(wperrs)
		theta_sp, w_sp, w_err_sp = measure_ang_xcorr(zmin, zmax)
		theta_sp_z.append(theta_sp), wps_z.append(w_sp), werr_sp_z.append(w_err_sp)
	return rps_z, wps_z, wperr_z, theta_sp_z, w_sp_z, werr_sp_z

def full_routine_for_all_zs(zbins):
	theta_pp, w_pp, werr_pp = measure_pp_ang_autocf()
	a_pp, gamma_pp = fit_angular_autocf(thetas=theta_pp, wtheta_pp=w_pp, werr_pp=werr_pp)
	rps_z, wps_z, wperr_z, theta_sp_z, w_sp_z, werr_sp_z = measure_all_cfs(zbins)

	r0_pp_estimate=None
	for i in range(100):
		dndz_pp, zcenters = [], []

		for j in range(zbins - 1):
			zmin, zmax = zbins[j], zbins[j+1]
			zcentre = (zmax - zmin) / 2.
			zcenters.append(zcentre)
			rps, wps, wperrs = rps_z[j], wps_z[j], wperr_z[j]
			theta_sp, w_sp, w_err_sp = theta_sp_z[j], w_sp_z[j], werr_sp_z[j]
			dndz_pp.append(full_routine_for_z(zcentre, rps=rps, wps=wps, wperrs=wperrs, theta_sp=theta_sp, w_sp=w_sp,
			                                  w_err_sp=w_err_sp, gamma_pp=gamma_pp, r0_pp=r0_pp_estimate))

		dndz_pp, zcenters = np.array(dndz_pp), np.array(zcenters)

		# set r0_pp to new value and start over
		r0_pp_estimate = fit_limber_for_r0_pp(theta_pp=theta_pp, w_pp=w_pp, werr_pp=werr_pp, gamma_pp=gamma_pp,
		                                      zs=zcenters, dndz_pp=dndz_pp)
	return dndz_pp




def spec_sample_bias_for_z(z):
	# Laurent 2017 quasar bias evolution
	if z > 0.9:
		return 0.278 * (((1+z)**2) - 6.565) + 2.393
	elif (z > 0.7) & (z <= 0.9):
		# Zhai 2017 LRG bias
		return 2.3
	else:
		return 1.85



# Krolewski et al 2020 (unwise tomography) used cross-correlations with SDSS spec samples
# they measured the bias as funciton of redshift for LOWZ and CMASS
def krolewski_bias(sample, z_inputs):
	if sample == 'lowz':
		zbincenters = np.linspace(0.025, 0.475, 10)
		bias = np.array([1.34, 1.37, 1.52, 1.73, 1.89, 2.01, 2.01, 2.06, 2.25, 2.46])
	elif sample == 'cmass':
		zbincenters = np.linspace(0.125, 0.775, 14)
		bias = np.array([1.36, 2.82, 1.54, 2.11, 1.99, 2.24, 2.05, 2.08, 2.06, 2.17, 2.22, 2.39, 2.52, 2.73])
	return np.interp()




def menard_ideal_estimator(zbins, ntheta_bins, minscale=0.5, maxscale=10., ls=True):
	specz_table = Table.read('../data/lss/Combined/all.fits')
	specz_randoms = Table.read('../data/lss/Combined/all_randoms.fits')


	wise_agntab = Table.read('../data/lss/Combined/wise_agn.fits')
	zcenters = interpolate_tools.bin_centers(zbins, method='mean')

	import matplotlib.pyplot as plt
	for j in range(int(np.max(wise_agntab['bin']))):
		dndz, uperrs, lowerrs = [], [], []
		binagntab = wise_agntab[np.where(wise_agntab['bin'] == (j+1))]
		#binagntab = Table.read('../data/lss/Combined/wise_randoms_%s.fits' % (j + 1))
		#binagntab = binagntab[:int(len(binagntab) / 10)]

		for k in range(len(zbins)-1):
			zmin, zmax = zbins[k], zbins[k + 1]

			ang_diam_dist = np.float(cosmo.angularDiameterDistance(zcenters[k]))

			min_theta_at_z = minscale / ang_diam_dist * 180. / np.pi    # degrees

			max_theta_at_z = maxscale / ang_diam_dist * 180. / np.pi    # degrees
			thetas_at_z = np.logspace(np.log10(min_theta_at_z), np.log10(max_theta_at_z), ntheta_bins)


			# don't consider very small scales < 15 arcsec
			if min_theta_at_z * 3600. < 6.:
				print('probing scale %s arcsec, within WISE PSF, beware' % (min_theta_at_z * 3600.))




			zbinspectab = specz_table[np.where((specz_table['Z'] > zmin) & (specz_table['Z'] <= zmax))]
			rand_zbinspectab = specz_randoms[np.where((specz_randoms['Z'] > zmin) & (specz_randoms['Z'] <= zmax))]

			if ls:
				agnrandtab = Table.read('../data/lss/Combined/wise_randoms_%s.fits' % (j + 1))
				refrandras, refranddecs, refrandweights = agnrandtab['RA'], agnrandtab['DEC'], agnrandtab['WEIGHT']
			else:
				refrandras, refranddecs, refrandweights = None, None, None



			w_sp, werr_sp = twoPointCFs.ang_cross_corr_from_coords(refras=binagntab['RA'], refdecs=binagntab['DEC'],
			                                              ras=zbinspectab['RA'], decs=zbinspectab['DEC'],
			                                              refrandras=refrandras, refranddecs=refranddecs,
			                                              minscale=min_theta_at_z, maxscale=max_theta_at_z,
			                                              randras=rand_zbinspectab['RA'],
			                                              randdecs=rand_zbinspectab['DEC'],
			                                              refweights=binagntab['WEIGHT'],
			                                              weights=zbinspectab['WEIGHT'],
			                                              refrandweights=refrandweights,
			                                              randweights=rand_zbinspectab['WEIGHT'],
			                                              nbins=ntheta_bins)





			plt.close('all')
			plt.figure(figsize=(8, 7))
			plt.scatter(thetas_at_z, w_sp)
			plt.errorbar(thetas_at_z, w_sp, yerr=werr_sp, fmt='none')
			plt.xlabel(r'$\theta$ [deg]', fontsize=20)
			plt.ylabel(r'$w_{\theta}$', fontsize=20)
			plt.xscale('log')
			plt.yscale('log')
			plt.ylim(1e-4, 1e-1)

			plt.savefig('plots/clustering_redshifts/each_xcorr/%s/%s_%s.pdf' % (j+1, j + 1, k))

			plt.close('all')



			# weight function to optimize S/N by using clustering information at all scales equally, since w
			# \propto theta^1-gamma
			weight_function = thetas_at_z ** (-0.8)
			# normalize such that integral of weight funciton is 1
			normalized_weight_function = 1 / np.trapz(weight_function, x=thetas_at_z) * weight_function


			integrated_w = np.trapz(w_sp * normalized_weight_function, x=thetas_at_z)
			upper_integrated_w = np.trapz((w_sp + werr_sp) * normalized_weight_function, x=thetas_at_z) - integrated_w

			lower_integrated_w = integrated_w - np.trapz((w_sp - werr_sp) * normalized_weight_function, x=thetas_at_z)


			bias_factor = (cosmo.correlationFunction(5, z=zcenters[k]) *
			                               spec_sample_bias_for_z(zcenters[k]) * zcenters[k] ** 1)



			dndz.append(integrated_w / bias_factor), uperrs.append(upper_integrated_w / bias_factor)
			lowerrs.append(lower_integrated_w / bias_factor)

		dndz = 1 / np.trapz(np.array(dndz), x=np.array(zcenters)) * np.array(dndz)
		np.array([zcenters, dndz]).dump('redshifts/clustering/%s.npy' % (j+1))
		plt.close('all')
		plt.figure(figsize=(8,7))
		plt.scatter(zcenters, dndz)
		plt.errorbar(zcenters, dndz, yerr=[lowerrs, uperrs], fmt='none')
		plt.xlim(0, 3.5)
		plt.savefig('plots/clustering_redshifts/%s.pdf' % (j+1))
		plt.close('all')

#combine_lss_catalogs(remove_redshift_outliers=True)
menard_ideal_estimator(np.linspace(1, 3.5, 25), 20, minscale=0.05, maxscale=2.5, ls=True)
