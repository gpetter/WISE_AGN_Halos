import numpy as np
import mcfit
import matplotlib.pyplot as plt
from colossus.halo import profile_nfw
from colossus.lss import bias
from colossus.cosmology import cosmology
import scipy as sp
import scipy.interpolate
from colossus.lss import mass_function
import astropy.units as u
from colossus.halo import concentration
import time
from scipy import special

from halotools.empirical_models import PrebuiltHodModelFactory
from scipy.special import sici
cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()



param_keys = {'logMmin': 0, 'alpha': 1, 'logM1': 2}

mass_grid = np.logspace(10.5, 15, 50)

r_grid = np.logspace(-2, 6, 10000) * (u.kpc / u.littleh)

#k_grid = np.logspace(-5, 3, 1000) * (u.littleh / u.Mpc)

def log_interp1d(xx, yy, kind='linear'):
	logx = np.log10(xx)
	logy = np.log10(yy)
	lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, fill_value=0., bounds_error=False)
	log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
	return log_interp

def concentration_from_mass(masses, z):
	return concentration.concentration(masses, '200c', z, model='ludlow16')


# take Fourier transform of NFW density profile
def transformed_nfw_profile(k_grid, masses, z, analytic=True):
	#c = concentration_from_mass(m, z)
	#p_nfw = profile_nfw.NFWProfile(M=m, c=c, z=z, mdef='200c')


	# use analytic Fourier transform of NFW profile given by Scoccimarro 2001
	# more clear in Equation 81 in Cooray+Sheth 2002
	if analytic:
		cs = concentration_from_mass(masses, z)
		rs_s = []
		for j in range(len(mass_grid)):
			rs_s.append(profile_nfw.NFWProfile(M=mass_grid[j], c=cs[j], z=z, mdef='200c').fundamentalParameters(
				M=mass_grid[j], c=cs[j], z=z, mdef='200c')[1])


		prefactor = 1 / (np.log(1 + cs) - cs / (1 + cs))
		#rho_s, r_s = p_nfw.fundamentalParameters(M=m, c=c, z=z, mdef='200c')
		rs_s = np.array(rs_s) * u.kpc / u.littleh
		k_r_s = (np.outer(k_grid, rs_s.to(u.Mpc / u.littleh))).value

		term1 = np.sin(k_r_s) * (sici((1+cs) * k_r_s)[0] - sici(k_r_s)[0]) - np.sin(cs * k_r_s) / ((1 + cs) * k_r_s) + \
		        np.cos(k_r_s) * (sici((1 + cs) * k_r_s)[1] - sici(k_r_s)[1])
		uk = prefactor * term1

	# !!!! this method doesn't work right now, use analytic form
	else:
		rho_of_r = p_nfw.density(r=r_grid.value) * (u.solMass * u.littleh ** 2) * (1 / (u.kpc ** 3)).to(
			1 / (u.Mpc ** 3))

		# Fourier transforming changes units
		rho_of_r = (u.Mpc / u.littleh) ** 3 * rho_of_r

		k, uk = mcfit.xi2P(r_grid.to(u.Mpc / u.littleh).value)(rho_of_r)

		# interpolate result onto k grid
		uk_interp = log_interp1d(k, uk)
		uk = uk_interp(k_grid.value) * (u.solMass / u.littleh) / m


	return uk


# number density of halos at z per log mass interval
def halo_mass_function(masses, z):
	return mass_function.massFunction(masses, z, mdef='200c', model='tinker08', q_in='M', q_out='dndlnM') * (
		u.littleh / u.Mpc) ** 3




def two_param_hod(masses, logm_min, alpha):
	n_cen = np.heaviside(np.log10(masses) - logm_min, 1)
	n_sat = (n_cen * (masses / (10 ** logm_min)) ** alpha)
	return n_cen, n_sat


# Zheng 2005 model
def three_param_hod(masses, logm_min, alpha, logm1):
	# fix softening parameter
	sigma = 0.25
	n_cen = 1 / 2. * (1 + special.erf((np.log10(masses) - logm_min) / sigma))
	n_sat = (((masses - (10 ** logm_min))/ (10 ** logm1)) ** alpha)
	n_sat[np.where(np.log10(masses) <= logm_min)] = 0.


	return n_cen, n_sat


# number of central AGN
def n_central(masses, params, modelname='zheng07'):
	if modelname == 'zheng07':
		model = PrebuiltHodModelFactory('zheng07')
		mean_ncen = model.mean_occupation_centrals(prim_haloprop=masses)
	elif modelname == '2param':
		mean_ncen = two_param_hod(masses, params[param_keys['logMmin']], params[param_keys['alpha']])[0]
	elif modelname == '3param':
		mean_ncen = three_param_hod(masses, params[param_keys['logMmin']], params[param_keys['alpha']],
		                            params[param_keys['logM1']])[0]
	else:
		return 'error'


	return mean_ncen


# number of satellites
def n_satellites(masses, params, modelname='zheng07'):
	if modelname == 'zheng07':
		model = PrebuiltHodModelFactory('zheng07')
		mean_nsat = model.mean_occupation_satellites(prim_haloprop=masses)
	elif modelname == '2param':
		mean_nsat = two_param_hod(masses, params[param_keys['logMmin']], params[param_keys['alpha']])[1]
	elif modelname == '3param':
		mean_nsat = three_param_hod(masses, params[param_keys['logMmin']], params[param_keys['alpha']],
		                            params[param_keys['logM1']])[1]
	else:
		return 'error'

	return mean_nsat


# sum of one and two halo terms
def hod_total(masses, params, modeltype='zheng07'):
	return n_central(masses, params, modeltype) + n_satellites(masses, params, modeltype)


# integral of HOD over halo mass function gives average number density of AGN
def avg_number_density(hmf, hod):
	#hmf = halo_mass_function(mass_grid, z)
	#hod = hod_total(mass_grid, params, modeltype)

	return np.trapz(hmf * hod, x=np.log(mass_grid))


def one_halo_power_spectrum(k_grid, hmf, u_of_k_for_m, n_cen, n_sat, z, params, modeltype='zheng07'):
	force_turnover = False


	integrand = hmf * (2 * n_cen * n_sat * u_of_k_for_m + ((n_sat * u_of_k_for_m) ** 2))
	avg_dens = avg_number_density(hmf, (n_cen+n_sat))
	integral = (1 / avg_dens ** 2) * np.trapz(integrand, dx=np.log(mass_grid)[1] - np.log(mass_grid)[0])
	if force_turnover:
		integral[np.where(k_grid.value < 1e-2)] = 0.
	return integral


def halo_halo_power_spectrum(k_grid, m1, m2, z):
	b1, b2 = bias.haloBias(m1, z, mdef='200c', model='tinker10'), bias.haloBias(m2, z, mdef='200c', model='tinker10')

	pk = cosmo.matterPowerSpectrum(k_grid.value, z) * (u.Mpc / u.littleh) ** 3
	return np.outer(b1 * b2, pk)


def two_halo_power_spectrum(k_grid, hmf, u_of_k_for_m, n_cen, n_sat, z, params, modeltype='zheng07'):
	avg_dens = avg_number_density(hmf, (n_cen + n_sat))

	# do full double integral. !!!! doesn't work, use single integral
	decompose = False
	if decompose:
		def inner_integral(m_outer):
			first_term = n_cen + n_sat * u_of_k_for_m[:, np.where(mass_grid == m_outer)[0]]

			second_term = n_cen + n_sat * u_of_k_for_m
			p_hhs = halo_halo_power_spectrum(k_grid, mass_grid, m_outer, z)
			p_hhs = np.transpose(p_hhs)

			return np.trapz(hmf * first_term * second_term * np.array(p_hhs), x=np.log(mass_grid))

		inners = []
		for m in mass_grid:
			inners.append(inner_integral(m))

		return 1 / avg_dens ** 2 * np.trapz(hmf * np.transpose(inners), x=np.log(mass_grid))

	else:
		bias_grid = bias.haloBias(mass_grid, z, mdef='200c', model='tinker10')
		integrand = hmf * bias_grid * (n_cen + n_sat) * u_of_k_for_m
		integral = np.trapz(integrand, x=np.log(mass_grid))
		return (1 / avg_dens ** 2) * (integral ** 2) * cosmo.matterPowerSpectrum(k_grid.value, z) * \
		       (u.Mpc / u.littleh) ** 3





def effective_bias(hmf, hod, avg_dens, zs, dndz):

	bofm_z = []
	for z in zs:
		bofm_z.append(bias.haloBias(mass_grid, model='tinker10', z=z, mdef='200c'))

	beff_zs = 1. / avg_dens * np.trapz(bofm_z * hod * hmf, x=np.log(mass_grid))

	return np.average(beff_zs, weights=dndz)


def effective_mass(hmf, hod, avg_dens, dndz):


	meff_zs = 1. / avg_dens * np.trapz(mass_grid * hod * hmf, x=np.log(mass_grid))


	return np.average(meff_zs, weights=dndz)


def satellite_fraction(hmf, n_sat, avg_dens, dndz):

	fsat_zs = 1. / avg_dens * np.trapz(n_sat * hmf, x=np.log(mass_grid))
	return np.average(fsat_zs, weights=dndz)


def u_of_k_for_m_for_zs(k_grid, masses, zs):
	uk_m_z = []
	for z in zs:
		uk_m_z.append(transformed_nfw_profile(k_grid, masses, z))
	return np.array(uk_m_z)

def hmf_for_zs(masses, zs):
	hmfs = []
	for z in zs:
		hmfs.append(halo_mass_function(masses, z))
	return np.array(hmfs)


def power_spectra_for_zs(k_grid, zs, params, modeltype='zheng07'):
	hmf_of_z = hmf_for_zs(mass_grid, zs)
	uk_m_z = u_of_k_for_m_for_zs(k_grid, mass_grid, zs)
	n_cen, n_sat = n_central(mass_grid, params, modeltype), n_satellites(mass_grid, params, modeltype)

	onehalos, twohalos = [], []

	for j in range(len(zs)):
		onehalos.append(one_halo_power_spectrum(k_grid, hmf_of_z[j], uk_m_z[j], n_cen, n_sat, zs[j], params, modeltype))
		twohalos.append(two_halo_power_spectrum(k_grid, hmf_of_z[j], uk_m_z[j], n_cen, n_sat, zs[j], params, modeltype))

	return onehalos, twohalos

def derived_parameters(zs, dndz, params, modeltype='zheng07'):
	hmf_of_z = hmf_for_zs(mass_grid, zs)
	n_cen, n_sat = n_central(mass_grid, params, modeltype), n_satellites(mass_grid, params, modeltype)
	avg_dens_of_z = avg_number_density(hmf_of_z, (n_cen + n_sat))

	beff = effective_bias(hmf_of_z, (n_cen + n_sat), avg_dens_of_z, zs, dndz)
	meff = effective_mass(hmf_of_z, (n_cen + n_sat), avg_dens_of_z, dndz)
	f_sat = satellite_fraction(hmf_of_z, n_sat, avg_dens_of_z, dndz)

	return f_sat, beff, meff






def redshift_averaged_power_spectra(k_grid, zs, dndz, params, modeltype='zheng07'):
	onehalos, twohalos = power_spectra_for_zs(k_grid, zs, params, modeltype)
	avg_onehalo = np.average(onehalos, weights=dndz, axis=0)
	avg_twohalo = np.average(twohalos, weights=dndz, axis=0)

	return avg_onehalo, avg_twohalo

def halomod_powspec():
	import halomod
	hm_smt3 = halomod.TracerHaloModel(
		z=np.random.rand(1)[0],  # Redshift
		Mmin=9, Mmax=15., transfer_model='EH', hod_model='Zheng05', hm_logk_min=-5, hm_logk_max=3,
		hod_params={'M_min': 12., 'M_1': 12.5, 'alpha': 1., 'sig_logm': 0.25, 'M_0': 12.},
		exclusion_model=None, halo_concentration_model='Ludlow16', mdef_model='SOCritical', hm_dlog10k=0.008,
		force_1halo_turnover=False
	)
	hm2 = halomod.integrate_corr.AngularCF(
		z=np.random.rand(1)[0],  # Redshift
		Mmin=9, Mmax=15., transfer_model='EH', hod_model='Zheng05', hm_logk_min=-5, hm_logk_max=3,
		hod_params={'M_min': 12., 'M_1': 12.5, 'alpha': 1., 'sig_logm': 0.25, 'M_0': 12.},
		exclusion_model=None, halo_concentration_model='Ludlow16', mdef_model='SOCritical', hm_dlog10k=0.008,
		force_1halo_turnover=False)


def plot_hod(params, modeltype):
	n_cen = n_central(mass_grid, params, modeltype)
	n_sat = n_satellites(mass_grid, params, modeltype)
	masses = np.log10(mass_grid)
	plt.figure(figsize=(8,7))
	plt.plot(masses, n_cen, ls='--', c='k', label='Central')
	plt.plot(masses, n_sat, ls='-.', c='k', label='Satellite')
	plt.plot(masses, n_cen + n_sat, c='k', label='Total')
	#plt.plot(np.log10(hm_smt3.m), hm_smt3.total_occupation)
	plt.xlabel('log$_{10}(M_h)$', fontsize=20)
	plt.ylabel(r'$\langle N \rangle$', fontsize=20)
	plt.yscale('log')
	plt.legend(fontsize=15)
	plt.ylim(1e-3, plt.ylim()[1])

	plt.savefig('plots/hod.pdf')
	plt.close('all')




def plot_both_pow_specs(k_grid, params, modeltype):

	pk = cosmo.matterPowerSpectrum(k_grid.value, 0)
	onepk, twopk = redshift_averaged_power_spectra(k_grid, [0.01,0.01], [0.5, 0.5], params, modeltype)
	totpk = onepk + twopk

	plt.figure(figsize=(8,7))
	plt.plot(k_grid, pk, c='r', ls='-', label='$P_{mm}$')
	plt.plot(k_grid, onepk, c='k', ls='--')
	plt.plot(k_grid, twopk, c='k', ls='-.')
	plt.plot(k_grid, totpk, c='k', ls='-')

	#plt.plot(hm_smt3.k_hm, hm_smt3.power_2h_auto_tracer, c='b')


	plt.legend(fontsize=20)
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('$k$ [h/Mpc]', fontsize=20)
	plt.ylabel('$P_k$', fontsize=20)
	plt.ylim(1e-5, 1e5)
	plt.savefig('plots/power_spectra.pdf')
	plt.close('all')


#plot_hod([12., 1, 12.5], modeltype='3param')
#k_grid_tmp = np.logspace(-5, 3, 1000) * (u.littleh / u.Mpc)
#plot_both_pow_specs(k_grid_tmp, [12., 1., 12.5], modeltype='3param')