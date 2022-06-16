import pyccl as ccl
import numpy as np
import astropy.cosmology.units as cu
from astropy import units as u
import mcfit
from source import interpolate_tools
import astropy.constants as const
from colossus.cosmology import cosmology
col_cosmo = cosmology.setCosmology('planck18')
apcosmo = col_cosmo.toAstropy()
import abel
from scipy.interpolate import interp1d


def z_to_a(zs):
	return 1. / (1. + zs)


def dndz_from_z_list(zs, bins):
	dndz, zbins = np.histogram(zs, bins=bins, density=True)
	zcenters = interpolate_tools.bin_centers(zbins, method='mean')
	dndz = dndz / np.trapz(dndz, x=zcenters)
	return (zcenters, dndz)

def comoving_dist(zs, cosmo):
	a_arr = z_to_a(zs=zs)
	return cosmo.comoving_radial_distance(a=a_arr)

def hubble_parameter(zs, cosmo, littleh):
	a_arr = z_to_a(zs=zs)
	return cosmo.h_over_h0(a=a_arr) * littleh * 100.


def remove_littleh_hodparams(hodparams, littleh):
	logh = np.log10(littleh)
	hodparams[0] -= logh
	hodparams[2] -= logh
	hodparams[3] -= logh
	return hodparams




class HOD_model(object):
	def __init__(self):
		self.littleh = apcosmo.H0.value/100.
		self.cosmo = ccl.Cosmology(Omega_c=apcosmo.Odm0, Omega_b=apcosmo.Ob0,
							h=self.littleh, n_s=0.9665, sigma8=0.8102,
							transfer_function='eisenstein_hu', matter_power_spectrum='linear')

		self.cosmo.cosmo.spline_params.ELL_MAX_CORR = 1000000
		self.cosmo.cosmo.spline_params.N_ELL_CORR = 5000


		self.mdef = ccl.halos.massdef.MassDef200c(c_m='Duffy08')
		self.c_m_relation = ccl.halos.concentration.ConcentrationDuffy08(mdef=self.mdef)
		self.hmf_mod = ccl.halos.hmfunc.MassFuncTinker08(cosmo=self.cosmo, mass_def=self.mdef)
		self.bias_mod = ccl.halos.hbias.HaloBiasTinker10(cosmo=self.cosmo, mass_def=self.mdef)
		self.hmc = ccl.halos.halo_model.HMCalculator(cosmo=self.cosmo, massfunc=self.hmf_mod,
													 hbias=self.bias_mod, mass_def=self.mdef)
		self.k_space = np.logspace(-4, 2, 1024)
		# flip this around since a is inversely propto z, so a is increasing
		self.z_space = np.flip(np.linspace(0.01, 4., 16))
		self.a_space = z_to_a(self.z_space)

		# need to figure out what to do with f_c parameter, central fraction
		self.two_pt_hod_profile = ccl.halos.profiles_2pt.Profile2ptHOD()



	# kill off 1 halo power at scales less than kcut
	# a is dummy scalefactor, so cutoff doesn't change with redshift
	def large_scale_1h_suppresion_func(self, a, kcut=1e-2):
		return kcut * np.ones(len(self.k_space))

	# smooth transition
	# a is dummy scalefactor, so cutoff doesn't change with redshift
	def smooth_1h_2h_transition_func(self, a, smooth_alpha=1):
		return smooth_alpha * np.ones(len(self.k_space))

	def nfw_profile(self):
		return ccl.halos.profiles.HaloProfileNFW(c_M_relation=self.c_m_relation)

	# set up density profile with HOD parameters
	def hod_profile(self, hodparams):
		return ccl.halos.profiles.HaloProfileHOD(c_M_relation=self.c_m_relation, lMmin_0=hodparams[0],
		                siglM_0=hodparams[1], lM0_0=hodparams[2], lM1_0=hodparams[3], alpha_0=hodparams[4])



	# calculate P(k, z) with given hod parameters
	def hod_pk_a(self, hodparams, zs):
		a_arr = z_to_a(np.flip(zs))
		return ccl.halos.halo_model.halomod_power_spectrum(cosmo=self.cosmo, hmc=self.hmc, k=self.k_space,
		                        a=a_arr, prof=self.hod_profile(hodparams), prof_2pt=self.two_pt_hod_profile,
		                        supress_1h=self.large_scale_1h_suppresion_func, normprof1=True, normprof2=True,
		                        smooth_transition=self.smooth_1h_2h_transition_func)


	def hod_pk_2d_obj(self, hodparams, zs):
		a_arr = z_to_a(np.flip(zs))

		return ccl.halos.halo_model.halomod_Pk2D(cosmo=self.cosmo, hmc=self.hmc, prof=self.hod_profile(hodparams),
		                        prof_2pt=self.two_pt_hod_profile, a_arr=a_arr, lk_arr=np.log(self.k_space),
		                        normprof1=True, normprof2=True,
		                        supress_1h=self.large_scale_1h_suppresion_func,
		                        smooth_transition=self.smooth_1h_2h_transition_func)

	def linear_pk_a(self):
		pk_as = []
		for this_a in self.a_space:
			pk_as.append(ccl.power.linear_matter_power(cosmo=self.cosmo, k=self.k_space, a=this_a))
		return np.array(pk_as)


	# ccl doesn't have built in tracer class for HOD
	# Just have a density tracer with no bias, then use the HOD power spectrum in the integral later
	def unbiased_density_tracer(self, dndz):
		return ccl.NumberCountsTracer(cosmo=self.cosmo, has_rsd=False, dndz=dndz, bias=(dndz[0], np.ones(len(dndz[0]))))

	# do an angular autocorrelation of HOD halos
	# if thetas given, transform C_ell to W(theta)
	def hod_corr_gg(self, hodparams, dndz, thetas=None):
		pk_a_hod = self.hod_pk_2d_obj(hodparams, dndz[0])
		#hodtracer = self.hod_tracer(hodparams, dndz)
		gal_tracer = self.unbiased_density_tracer(dndz=dndz)
		ell_modes = 1+np.arange(500000)

		cl_gg = ccl.cls.angular_cl(cosmo=self.cosmo, cltracer1=gal_tracer, cltracer2=gal_tracer, ell=ell_modes,
		                           p_of_k_a=pk_a_hod)

		if thetas is not None:
			w_theta = ccl.correlations.correlation(cosmo=self.cosmo, ell=ell_modes, C_ell=cl_gg,
			                                       theta=thetas, type='NN', method='fftlog')
			return w_theta
		else:
			return cl_gg

	def custom_hod_ang_cf(self, hodparams, dndz, thetas):
		thetas = (thetas * u.deg).to('radian').value
		hod_power_a = self.hod_pk_a(hodparams=hodparams, zs=dndz[0])

		hod_power_z = np.flipud(hod_power_a)


		thetachis, dipomp_int = mcfit.Hankel(self.k_space, lowring=True)(hod_power_z, axis=1, extrap=True)
		#return thetachis, dipomp_int

		#neg_idxs = np.where(dipomp_int < 0)

		#dipomp_int = np.abs(dipomp_int)




		#print(np.shape(np.dstack((thetas_for_zs, dipomp_int))))


		#linfit = interp1d(dndz[0], np.shape(np.dstack((thetas_for_zs, dipomp_int))), axis=1)
		input_theta_chis = np.outer(comoving_dist(dndz[0], cosmo=self.cosmo), thetas)

		# 2D grid of thetas * chi(z) to interpolate model power spectra onto
		#input_theta_chis = np.outer(apcosmo.comoving_distance(dndz[0]).value, thetas)

		# for each redshift, chi(z), interpolate the result of the above integral onto theta*chi(z) grid
		interped_dipomp = []
		for j in range(len(dndz[0])):
			# trick for interpolation for function varying in log space (propto r^-2)
			# multiply result of Hankel transform by theta*chi, this makes it pretty smooth in linear space
			# then divide it back out at end
			flatpower = dipomp_int[j] * thetachis
			interped_dipomp.append(interp1d(thetachis, flatpower)(input_theta_chis[j]) /
			                       input_theta_chis[j])


		interped_dipomp = np.array(interped_dipomp)



		#newinterp = interpolate_tools.log_interp2d(thetachis, dndz[0], dipomp_int)
		#print(newinterp(input_theta_chis, dndz[0]))

		# convert H(z)/c from 1/Mpc to h/Mpc in order to cancel units of k
		#dz_d_chi = (apcosmo.H(dndz[0]) / const.c).to(1. / u.Mpc).value
		dz_d_chi = (hubble_parameter(zs=dndz[0], cosmo=self.cosmo, littleh=self.littleh) / 2.99e5)

		# product of redshift distributions, and dz/dchi
		differentials = dz_d_chi * (dndz[1]) ** 2

		z_int = 1 / (2 * np.pi) * np.trapz(differentials * np.transpose(interped_dipomp), x=dndz[0], axis=1)

		return z_int

	def custom_hod_xi_r(self, hodparams, dndz, radii, projected=False):


		hod_power_a = self.hod_pk_a(hodparams=hodparams, zs=dndz[0])

		hod_power_z = np.flipud(hod_power_a)


		rgrid, xis = mcfit.P2xi(self.k_space, lowring=True)(hod_power_z, axis=1, extrap=True)

		if projected:
			xis = np.array(abel.direct.direct_transform(xis, r=rgrid, direction='forward', backend='python'))
		# am i sure this is okay for the projected correlation function? am i actually getting it at r_p?

		# trick to make interpolation work for logarithmically varying xi (propto r^-2)
		# multiply xi by r to make smooth in linear space, then divide r back out at end
		interpedxis = interp1d(rgrid, xis*rgrid)(radii) / radii


		return np.trapz((dndz[1] ** 2) * np.transpose(interpedxis) , x=dndz[0], axis=1)

	# cross correlation spectrum of galaxy density and CMB lensing
	def hod_c_ell_kg(self, hodparams, dndz):
		pk_a_hod = self.hod_pk_2d_obj(hodparams=hodparams, zs=dndz[0])
		hodtracer = self.unbiased_density_tracer(dndz=dndz)
		cmbtracer = ccl.tracers.CMBLensingTracer(cosmo=self.cosmo, z_source=1090.)
		ell_modes = 1 + np.arange(3000)

		c_ell_kg = ccl.cls.angular_cl(cosmo=self.cosmo, cltracer1=hodtracer, cltracer2=cmbtracer, ell=ell_modes,
		                           p_of_k_a=pk_a_hod)

		return c_ell_kg

	def hod_xi_of_r_z(self, hodparams, r_grid, z):
		hod_pk_a = self.hod_pk_2d_obj(hodparams=hodparams)
		a_return = z_to_a(z)
		return ccl.correlations.correlation_3d(cosmo=self.cosmo, a=a_return, r=r_grid, p_of_k_a=hod_pk_a)

