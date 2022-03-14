import numpy as np

from colossus.halo import profile_nfw
from colossus.lss import bias
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import astropy.units as u
from colossus.halo import concentration

from scipy import special
from scipy.special import sici



cosmo = cosmology.setCosmology('planck15')
apcosmo = cosmo.toAstropy()



param_keys = {'logMmin': 0, 'alpha': 1, 'logM1': 2}

mass_grid = np.load('power_spectra/m_space.npy', allow_pickle=True)
k_grid = np.load('power_spectra/k_space.npy', allow_pickle=True)

ukms = np.load('power_spectra/ukm.npy', allow_pickle=True)


# mass-concentration relation
def concentration_from_mass(masses, z):
	return concentration.concentration(masses, '200c', z, model='ludlow16')

# use halomod u(k,m) by writing tables and interpolating onto them
def write_ukm():
	import halomod
	lnkmin = np.log(np.min(k_grid))
	lnkmax = np.log(np.max(k_grid))
	dlnk = (lnkmax - lnkmin) / 1000.
	zlist = np.linspace(0, 4, 400)
	uks = []
	for z in zlist:
		mmin, mmax = np.min(np.log10(mass_grid)), np.max(np.log10(mass_grid))

		hm_smt3 = halomod.TracerHaloModel(
			z=z, cosmo_model='Planck15',
			Mmin=mmin, Mmax=mmax, transfer_model='CAMB', hod_model='Zheng05', hm_logk_min=-5, hm_logk_max=3,
			hod_params={'M_min': 12., 'M_1': 12.5, 'alpha': 1., 'sig_logm': 0.25, 'M_0': 12.}, takahashi=False,
			exclusion_model=None, halo_concentration_model='Ludlow16Empirical', mdef_model='SOCritical',
			hm_dlog10k=0.008, dlog10m=(mmax - mmin)/len(mass_grid), lnk_min=lnkmin, lnk_max=lnkmax, dlnk=dlnk,
			force_1halo_turnover=False, mdef_params={'overdensity': 200}
		)
		uks.append(hm_smt3.halo_profile_ukm)
	with open('power_spectra/ukm.npy', 'wb') as f:
		np.save(f, np.array(uks))



# take Fourier transform of NFW density profile
# currently can't figure out how to get same result as halomod, so interpolating onto their solution
def transformed_nfw_profile(masses, z, analytic=False):
	#c = concentration_from_mass(m, z)
	#p_nfw = profile_nfw.NFWProfile(M=m, c=c, z=z, mdef='200c')


	# use analytic Fourier transform of NFW profile given by Scoccimarro 2001
	# more clear in Equation 81 in Cooray+Sheth 2002
	if analytic:
		cs = concentration_from_mass(masses, z)

		rs_s = []
		for j in range(len(mass_grid)):
			rs_s.append(profile_nfw.NFWProfile.fundamentalParameters(
				M=mass_grid[j], c=cs[j], z=0, mdef='200c')[1])


		prefactor = 1 / (np.log(1 + cs) - cs / (1 + cs))
		#rho_s, r_s = p_nfw.fundamentalParameters(M=m, c=c, z=z, mdef='200c')
		rs_s = (np.array(rs_s) * u.kpc / u.littleh).to(u.Mpc / u.littleh) * np.pi

		k_r_s = (np.outer(k_grid, rs_s)).value

		# sin and cosine integrals of terms
		si_1c_krs, ci_1c_krs = sici((1+cs) * k_r_s)
		si_krs, ci_krs = sici(k_r_s)

		term1 = np.sin(k_r_s) * (si_1c_krs - si_krs) - np.sin(cs * k_r_s) / ((1 + cs) * k_r_s) + \
		        np.cos(k_r_s) * (ci_1c_krs - ci_krs)
		uk = prefactor * term1
	else:

		"""lnkmin = np.log(10**(-5.))
		lnkmax = np.log(10**(3.))
		dlnk = (lnkmax - lnkmin) / 1000.

		hm_smt3 = halomod.TracerHaloModel(
			z=z, cosmo_model='Planck15',
			Mmin=9, Mmax=15., transfer_model='CAMB', hod_model='Zheng05', hm_logk_min=-5, hm_logk_max=3,
			hod_params={'M_min': 12., 'M_1': 12.5, 'alpha': 1., 'sig_logm': 0.25, 'M_0': 12.}, takahashi=False,
			exclusion_model=None, halo_concentration_model='Ludlow16Empirical', mdef_model='SOCritical',
			hm_dlog10k=0.008, dlog10m=0.12, lnk_min=lnkmin, lnk_max=lnkmax, dlnk=dlnk,
			force_1halo_turnover=False, mdef_params={'overdensity': 200}
		)
		uk = hm_smt3.halo_profile_ukm"""

		uk = ukms[np.argmin(np.abs(z - np.linspace(0, 4, 400)))]

	# !!!! this method doesn't work right now, use analytic form
	r_grid = np.logspace(-2, 6, 10000) * (u.kpc / u.littleh)
	"""else:
		rho_of_r = p_nfw.density(r=r_grid.value) * (u.solMass * u.littleh ** 2) * (1 / (u.kpc ** 3)).to(
			1 / (u.Mpc ** 3))

		# Fourier transforming changes units
		rho_of_r = (u.Mpc / u.littleh) ** 3 * rho_of_r

		k, uk = mcfit.xi2P(r_grid.to(u.Mpc / u.littleh).value)(rho_of_r)

		# interpolate result onto k grid
		uk_interp = log_interp1d(k, uk)
		uk = uk_interp(k_grid.value) * (u.solMass / u.littleh) / m"""


	return uk



# number density of halos at z per log mass interval
def halo_mass_function(masses, z):
	return mass_function.massFunction(masses, z, mdef='200c', model='tinker08', q_in='M', q_out='dndlnM') * (
		u.littleh / u.Mpc) ** 3




"""def two_param_hod(masses, logm_min, alpha):
	n_cen = np.heaviside(np.log10(masses) - logm_min, 1)
	n_sat = (n_cen * (masses / (10 ** logm_min)) ** alpha)
	return n_cen, n_sat"""


# Zheng 2005 model
def three_param_hod(masses, logm_min, alpha, logm1):
	# fix softening parameter
	sigma = 0.25
	n_cen = 1 / 2. * (1 + special.erf((np.log10(masses) - logm_min) / sigma))
	n_sat = (((masses - (10 ** logm_min))/ (10 ** logm1)) ** alpha)
	n_sat[np.where(np.log10(masses) <= logm_min)] = 0.

	return n_cen, n_sat

# a two parameter version of Zheng model where M_1 is fixed to be a constant multiple of M_min
def two_param_hod(masses, logm_min, alpha):
	# fix M1 to be 20x M_min (Georgakakis 2018)
	n_cen, n_sat = three_param_hod(masses, logm_min, alpha, logm_min + np.log10(10))
	return n_cen, n_sat

def one_param_hod(masses, logm_min):
	#ncen = np.ones(len(masses))
	#ncen[np.where(np.log10(masses) < logm_min)] = 0.
	ncen = np.zeros(len(masses))
	closeidx = np.abs(np.log10(masses) - logm_min).argmin()
	ncen[closeidx] = 1
	ncen[closeidx + 1] = 1
	nsat = np.zeros(len(masses))
	return ncen, nsat


# number of central AGN
def n_central(masses, params, modelname='zheng07'):
	if modelname == 'zheng07':
		from halotools.empirical_models import PrebuiltHodModelFactory
		model = PrebuiltHodModelFactory('zheng07')
		mean_ncen = model.mean_occupation_centrals(prim_haloprop=masses)
	elif modelname == '2param':
		mean_ncen = two_param_hod(masses, params[param_keys['logMmin']], params[param_keys['alpha']])[0]
	elif modelname == '3param':
		mean_ncen = three_param_hod(masses, params[param_keys['logMmin']], params[param_keys['alpha']],
		                            params[param_keys['logM1']])[0]
	elif modelname == '1param':
		mean_ncen = one_param_hod(masses, params[param_keys['logMmin']])[0]
	elif modelname == 'dm':
		return np.zeros(len(masses))
	else:
		return None


	return mean_ncen


# number of satellites
def n_satellites(masses, params, modelname='zheng07'):
	if modelname == 'zheng07':
		from halotools.empirical_models import PrebuiltHodModelFactory
		model = PrebuiltHodModelFactory('zheng07')
		mean_nsat = model.mean_occupation_satellites(prim_haloprop=masses)
	elif modelname == '2param':
		mean_nsat = two_param_hod(masses, params[param_keys['logMmin']], params[param_keys['alpha']])[1]
	elif modelname == '3param':
		mean_nsat = three_param_hod(masses, params[param_keys['logMmin']], params[param_keys['alpha']],
		                            params[param_keys['logM1']])[1]
	elif modelname == '1param':
		mean_nsat = one_param_hod(masses, params[param_keys['logMmin']])[1]
	elif modelname == 'dm':
		return np.zeros(len(masses))
	else:
		return None

	return mean_nsat


# sum of one and two halo terms
def hod_total(params, modeltype='zheng07'):
	return n_central(mass_grid, params, modeltype) + n_satellites(mass_grid, params, modeltype)


# integral of HOD over halo mass function gives average number density of AGN
def avg_number_density(hmf, hod):
	#hmf = halo_mass_function(mass_grid, z)
	#hod = hod_total(mass_grid, params, modeltype)

	return np.trapz(hmf * hod, x=np.log(mass_grid))


def one_halo_power_spectrum(hmf, u_of_k_for_m, n_cen, n_sat, z, params, modeltype='zheng07'):
	if modeltype == 'dm':
		return np.zeros(len(k_grid))
	else:
		force_turnover = False


		integrand = hmf * (2 * n_cen * n_sat * u_of_k_for_m + ((n_sat * u_of_k_for_m) ** 2))
		avg_dens = avg_number_density(hmf, (n_cen+n_sat))
		integral = (1 / avg_dens ** 2) * np.trapz(integrand, dx=np.log(mass_grid)[1] - np.log(mass_grid)[0])
		if force_turnover:
			integral[np.where(k_grid < 1e-2)] = 0.
		return integral


def halo_halo_power_spectrum(m1, m2, z):
	b1, b2 = bias.haloBias(m1, z, mdef='200c', model='tinker10'), bias.haloBias(m2, z, mdef='200c', model='tinker10')

	pk = cosmo.matterPowerSpectrum(k_grid, z) * (u.Mpc / u.littleh) ** 3
	return np.outer(b1 * b2, pk)


def b_eff_of_z(hmf, hod, avg_dens, z):
	beff_z = 1. / avg_dens * np.trapz(bias.haloBias(mass_grid, model='tinker10', z=z, mdef='200c') *
	                                   hod * hmf, x=np.log(mass_grid))
	return beff_z

def two_halo_power_spectrum(hmf, u_of_k_for_m, n_cen, n_sat, z, modeltype='zheng07'):
	avg_dens = avg_number_density(hmf, (n_cen + n_sat))

	# do full double integral. !!!! doesn't work, use single integral
	decompose = False
	if decompose:
		def inner_integral(m_outer):
			first_term = n_cen + n_sat * u_of_k_for_m[:, np.where(mass_grid == m_outer)[0]]

			second_term = n_cen + n_sat * u_of_k_for_m
			p_hhs = halo_halo_power_spectrum(mass_grid, m_outer, z)
			p_hhs = np.transpose(p_hhs)

			return np.trapz(hmf * first_term * second_term * np.array(p_hhs), x=np.log(mass_grid))

		inners = []
		for m in mass_grid:
			inners.append(inner_integral(m))

		return 1 / avg_dens ** 2 * np.trapz(hmf * np.transpose(inners), x=np.log(mass_grid))


	else:
		if modeltype == 'dm':
			return cosmo.matterPowerSpectrum(k_grid, z) * (u.Mpc / u.littleh) ** 3
		else:
			bias_grid = bias.haloBias(mass_grid, z, mdef='200c', model='tinker10')
			integrand = hmf * bias_grid * (n_cen + n_sat) * u_of_k_for_m
			integral = np.trapz(integrand, x=np.log(mass_grid))
			#return (1 / avg_dens ** 2) * (integral ** 2) * cosmo.matterPowerSpectrum(k_grid, z) * \
			#       (u.Mpc / u.littleh) ** 3
			return (b_eff_of_z(hmf=hmf, hod=(n_cen + n_sat), avg_dens=avg_dens, z=z) ** 2) * \
			       cosmo.matterPowerSpectrum( k_grid, z) * (u.Mpc / u.littleh) ** 3






def effective_bias(hmf, hod, avg_dens, zs, dndz):

	bofm_z = []
	for z in zs:
		bofm_z.append(bias.haloBias(mass_grid, model='tinker10', z=z, mdef='200c'))


	beff_zs = 1. / avg_dens * np.trapz(bofm_z * hod * hmf, x=np.log(mass_grid))


	return np.average(beff_zs, weights=(dndz ** 2))
	#return np.trapz(beff_zs * dndz, x=zs)


def log_effective_mass(hmf, hod, avg_dens, zs, dndz):


	meff_zs = 1. / avg_dens * np.trapz(mass_grid * hod * hmf, x=np.log(mass_grid))


	return (np.average(np.log10(meff_zs), weights=(dndz ** 2)))
	#return np.log10(np.trapz(meff_zs * dndz, x=zs))


def satellite_fraction(hmf, n_sat, avg_dens, dndz):

	fsat_zs = 1. / avg_dens * np.trapz(n_sat * hmf, x=np.log(mass_grid))
	return np.average(fsat_zs, weights=dndz)


def u_of_k_for_m_for_zs(masses, zs):
	uk_m_z = []
	for z in zs:
		uk_m_z.append(transformed_nfw_profile(masses, z))
	return np.array(uk_m_z)

def hmf_for_zs(masses, zs):
	hmfs = []
	for z in zs:
		hmfs.append(halo_mass_function(masses, z))
	return np.array(hmfs)


def power_spectra_for_zs(zs, params, modeltype='zheng07'):
	hmf_of_z = hmf_for_zs(mass_grid, zs)
	uk_m_z = u_of_k_for_m_for_zs(mass_grid, zs)
	n_cen, n_sat = n_central(mass_grid, params, modeltype), n_satellites(mass_grid, params, modeltype)

	onehalos, twohalos = [], []

	for j in range(len(zs)):
		onehalos.append(one_halo_power_spectrum(hmf_of_z[j], uk_m_z[j], n_cen, n_sat, zs[j], params, modeltype))
		twohalos.append(two_halo_power_spectrum(hmf_of_z[j], uk_m_z[j], n_cen, n_sat, zs[j], modeltype))

	return onehalos, twohalos

def derived_parameters(zs, dndz, params, modeltype='zheng07'):
	hmf_of_z = hmf_for_zs(mass_grid, zs)
	n_cen, n_sat = n_central(mass_grid, params, modeltype), n_satellites(mass_grid, params, modeltype)
	avg_dens_of_z = avg_number_density(hmf_of_z, (n_cen + n_sat))

	beff = effective_bias(hmf_of_z, (n_cen + n_sat), avg_dens_of_z, zs, dndz)
	meff = log_effective_mass(hmf_of_z, (n_cen + n_sat), avg_dens_of_z, zs, dndz)
	f_sat = satellite_fraction(hmf_of_z, n_sat, avg_dens_of_z, dndz)
	if modeltype == '1param':
		return beff, meff
	else:
		return f_sat, beff, meff






def redshift_averaged_power_spectra(zs, dndz, params, modeltype='zheng07'):
	onehalos, twohalos = power_spectra_for_zs(zs, params, modeltype)
	avg_onehalo = np.average(onehalos, weights=dndz, axis=0)
	avg_twohalo = np.average(twohalos, weights=dndz, axis=0)

	return avg_onehalo, avg_twohalo

def compare_angcfs():
	import halomod

	import matplotlib.pyplot as plt
	import clusteringModel
	import importlib
	importlib.reload(clusteringModel)
	thetagrid = np.logspace(-3, 0.1, 1000)
	zmiddle = 3

	hm2 = halomod.integrate_corr.AngularCF(
		z=zmiddle,
		zmin=zmiddle - 0.001, zmax=zmiddle + 0.001, znum=20,  # Redshift
		Mmin=10, Mmax=15., transfer_model='EH', hod_model='Zheng05', hm_logk_min=-5, hm_logk_max=3,
		theta_min=np.min(thetagrid)/180.*np.pi, theta_max=np.max(thetagrid)/180.*np.pi, theta_num=len(thetagrid),
		theta_log=True, cosmo_model='Planck15',
		hod_params={'M_min': 12., 'M_1': 12.5, 'alpha': 1., 'sig_logm': 0.25, 'M_0': 12.},
		exclusion_model=None, halo_concentration_model='Ludlow16', mdef_model='SOCritical', hm_dlog10k=0.008,
		force_1halo_turnover=False, bias_model='Tinker10')

	zlist = np.linspace(zmiddle - 0.001, zmiddle + 0.001, 1000)
	zhist = np.histogram(zlist, density=True, bins=20)
	myzs, mydndz = zhist[1][:len(zhist[1]) - 1], zhist[0]
	print('b_eff_halomod: %s' % hm2.bias_effective_tracer)
	print('b_eff_mine: %s' % derived_parameters(myzs, mydndz, params=[12., 1., 12.5], modeltype='3param')[1])

	print('m_eff_halomod: %s' % hm2.mass_effective)
	print('m_eff_mine: %s' % derived_parameters(myzs, mydndz, params=[12., 1., 12.5], modeltype='3param')[2])

	mymodel = clusteringModel.angular_corr_func(thetagrid, myzs, mydndz,
		hodparams=[12., 1., 12.5], hodmodel='3param')


	plt.figure(figsize=(8,7))
	plt.plot(hm2.theta*180./np.pi, hm2.angular_corr_gal, label='halomod')



	#plt.plot(hm_smt3.r, hm_smt3.corr_1h_auto_tracer, label='1h')
	#plt.plot(hm_smt3.r, hm_smt3.corr_2h_auto_tracer, label='2h')
	#plt.plot(np.logspace(-3, 0, 30), mymodel, label='my 1h')

	plt.plot(thetagrid, mymodel, label='my model')
	plt.xscale('log')
	plt.yscale('log')
	#plt.ylim(1e-2, 1e3)
	plt.legend()
	plt.xlabel(r'$\theta$', fontsize=20)
	plt.ylabel(r'$w(\theta)$', fontsize=20)
	plt.savefig('plots/halomod/modelcf_test.pdf')

	plt.close('all')
	plt.figure(figsize=(8, 7))
	plt.plot(thetagrid, mymodel / hm2.angular_corr_gal, label='my model')
	plt.xscale('log')
	plt.xlabel(r'$\theta$', fontsize=20)
	plt.ylabel(r'$w(\theta)$ ratio', fontsize=20)
	plt.savefig('plots/halomod/modelcf_ratio.pdf')
	plt.close('all')



def compare_ukm():
	import halomod
	import matplotlib.pyplot as plt
	plt.figure(figsize=(8, 7))
	lnkmin = np.log(10 ** (-5.))
	lnkmax = np.log(10 ** (3.))
	dlnk = (lnkmax - lnkmin) / 1000.
	hm_smt3 = halomod.TracerHaloModel(
		z=3., cosmo_model='Planck15', halo_profile_model="NFW", dlog10m=0.12, lnk_min=np.log(10 ** (-5.)),
		lnk_max=np.log(10 ** (3.)), Mmin=9, Mmax=15., transfer_model='CAMB', hod_model='Zheng05', hm_logk_min=-5,
		hm_logk_max=3, dlnk=dlnk,
		hod_params={'M_min': 12., 'M_1': 12.5, 'alpha': 1., 'sig_logm': 0.25, 'M_0': 12.}, takahashi=False,
		exclusion_model=None, halo_concentration_model='Ludlow16Empirical', mdef_model='SOCritical', hm_dlog10k=0.008,
		force_1halo_turnover=False, mdef_params={'overdensity': 200}
	)
	k_grid = hm_smt3.k_hm * u.littleh / u.Mpc
	#from halomod.concentration import Ludlow16
	#lud = Ludlow16()
	#nf = halomod.profiles.NFW(cm_relation=lud)
	#print(nf.scale_radius(m=1e12, at_z=True))

	ukm = transformed_nfw_profile(mass_grid, 3)
	#ukm2 = transformed_nfw_profile(mass_grid, 2)

	plt.plot(k_grid, ukm[:, 0])
	#plt.plot(k_grid, ukm2[:, 0])
	plt.plot(hm_smt3.k, hm_smt3.halo_profile_ukm[:, 0], c='k', ls='--')


	plt.legend()
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('$k$', fontsize=20)
	plt.ylabel('$u(k, M)$', fontsize=20)
	plt.xlim(1e-1, 1e4)
	plt.ylim(1e-3, 2)
	plt.savefig('plots/halomod/power_spectra.pdf')
	plt.close('all')


def compare_nfwruns():
	import halomod
	import matplotlib.pyplot as plt
	plt.figure(figsize=(8, 7))
	hm_smt3 = halomod.TracerHaloModel(
		z=0., cosmo_model='Planck15',
		Mmin=9, Mmax=15., transfer_model='CAMB', hod_model='Zheng05', hm_logk_min=-5, hm_logk_max=3,
		hod_params={'M_min': 12., 'M_1': 12.5, 'alpha': 1., 'sig_logm': 0.25, 'M_0': 12.}, takahashi=False,
		exclusion_model=None, halo_concentration_model='Ludlow16Empirical', mdef_model='SOCritical', hm_dlog10k=0.008,
		force_1halo_turnover=False, mdef_params={'overdensity': 200}
	)

	plt.plot(np.logspace(-2, 0, 100), 1000 * profile_nfw.NFWProfile(M=1e12, c=concentration_from_mass(np.array([1e12]),
	                                                                                                  0)[0],
	                                                                z=0, mdef='200c').density(np.logspace(-5, -2, 100)),
	         label='my model')
	plt.plot(hm_smt3.r, hm_smt3.halo_profile.rho(hm_smt3.r, np.array([1e12])),
	         c=concentration_from_mass(np.array([1e12]),
	                                   0)[0])

	plt.legend()
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('$k$', fontsize=20)
	plt.ylabel('$u(k, M)$', fontsize=20)
	# plt.ylim(1e-1, 1e1)
	plt.savefig('plots/halomod/nfw.pdf')
	plt.close('all')

def compare_powspec():
	import halomod
	import matplotlib.pyplot as plt
	plt.close('all')
	plt.figure(figsize=(8, 7))
	hm_smt3 = halomod.TracerHaloModel(
		z=1.5, cosmo_model='Planck15',
		Mmin=9, Mmax=15., transfer_model='CAMB', hod_model='Zheng05', hm_logk_min=-5, hm_logk_max=3,
		hod_params={'M_min': 12., 'M_1': 12.5, 'alpha': 1., 'sig_logm': 0.25, 'M_0': 12.}, takahashi=False,
		exclusion_model=None, halo_concentration_model='Ludlow16Empirical', mdef_model='SOCritical', hm_dlog10k=0.008,
		force_1halo_turnover=False, mdef_params={'overdensity': 200}
	)
	newk_grid = hm_smt3.k_hm * u.littleh / u.Mpc
	#onepk, twopk = redshift_averaged_power_spectra(k_grid, [2, 2], [0.5, 0.5], [12., 1., 12.5], '3param')
	onepk, twopk = power_spectra_for_zs([1.5], params=[12., 1., 12.5], modeltype='3param')


	plt.plot(hm_smt3.k_hm, hm_smt3.power_auto_tracer, c='b')

	totpk = np.array(onepk[0]) + np.array(twopk[0])
	#plt.plot(k_grid, totpk / hm_smt3.power_auto_tracer)
	plt.plot(newk_grid, totpk, c='k', ls='--')
	plt.yscale('log')
	plt.xscale('log')
	plt.savefig('plots/halomod/power_spectra.pdf')
	plt.close('all')
	#plt.plot(k_grid, onepk, c='k', ls='-.')
	#plt.plot(hm_smt3.k_hm, hm_smt3.power_1h_auto_tracer, label='HALOMOD-1h z=2', linewidth=5)
	#plt.plot(k_grid, onepk, c='k', ls='--', label='my 1h z=2')
	#plt.plot(hm_smt3.k, hm_smt3.halo_profile_ukm[:,0], linewidth=5, label='HALOMOD z=0')


def plot_hod(params, modeltype):
	import matplotlib.pyplot as plt
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


#compare_angcfs()