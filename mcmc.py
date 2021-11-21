import emcee
import numpy as np
import clusteringModel
import importlib
import hod_model
import corner
import matplotlib.pyplot as plt
importlib.reload(hod_model)
importlib.reload(clusteringModel)


# log prior function
def ln_prior(theta):
	if len(theta) == 2:
		mmin, alpha = theta
		# make flat priors on minimum mass and satellite power law
		if (mmin > 11.) & (mmin < 14.) & (alpha > 0) & (alpha < 2):
			return 0.
		else:
			return -np.inf
	elif len(theta) == 3:
		mmin, alpha, m1 = theta
		# make flat priors on minimum mass and satellite power law
		if (mmin > 11.) & (mmin < 14.) & (alpha > 0) & (alpha < 2) & (m1 > mmin) & (m1 < 13.):
			return 0.
		else:

			return -np.inf



# log likelihood function
def ln_likelihood(residual, yerr):

	# if yerr is a covariance matrix, likelihood function is r.T * inv(Cov) * r
	err_shape = np.shape(yerr)
	try:
		# trying to access the second axis will throw an error if yerr is 1D, catch this error and use 1D least squares
		foo = err_shape[1]
		return -0.5 * np.dot(residual.T, np.dot(np.linalg.inv(yerr), residual))[0][0]

	# if yerr is 1D, above will throw error, use least squares
	except:
		return -0.5 * np.sum((residual / yerr) ** 2)


# log probability is prior plus likelihood
def ln_prob(theta, anglebins, y, yerr, zs, dndz, modeltype, derived_keys=('f_sat', 'b_eff', 'm_eff')):
	prior = ln_prior(theta)

	# keep track of derived parameters like satellite fraction, effective bias, effective mass
	derived = (hod_model.derived_parameters(zs, dndz, theta, modeltype))
	"""if 'f_sat' in derived_keys:
		derived += (hod_model.satellite_fraction(zs, dndz, theta, modeltype),)
	if 'b_eff' in derived_keys:
		derived += (hod_model.effective_bias(zs, dndz, theta, modeltype),)
	if 'm_eff' in derived_keys:
		derived += (hod_model.effective_mass(zs, dndz, theta, modeltype),)"""

	# get model prediciton for given parameter set
	modelprediction = clusteringModel.angular_corr_func_in_bins(anglebins, zs, dndz, hodparams=theta,
	                                                            hodmodel=modeltype)

	# residual is data - model
	residual = y - modelprediction

	likely = ln_likelihood(residual, yerr)
	prob = prior + likely

	# return log_prob, along with derived parameters for this parameter set
	return (prob,) + derived





def sample_space(nwalkers, ndim, anglebins, y, yerr, zs, dndz, modeltype, derived_keys=('f_sat', 'b_eff', 'm_eff')):
	blobs_dtype = [("f_sat", float), ("b_eff", float), ("m_eff", float)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, args=[anglebins, y, yerr, zs, dndz, modeltype,
	                                                                derived_keys], blobs_dtype=blobs_dtype)





	pos = np.array([12., 1., 12.5]) + 1e-1 * np.random.normal(size=(sampler.nwalkers, sampler.ndim))
	sampler.run_mcmc(pos, 500, progress=True)


	flatchain = sampler.get_chain(discard=50, flat=True)
	blobs = sampler.get_blobs(discard=50, flat=True)

	flatchain = np.hstack((
		flatchain,
		np.atleast_2d(blobs['f_sat']).T,
		np.atleast_2d(blobs['b_eff']).T,
		np.atleast_2d(blobs['m_eff']).T
	))

	plt.close('all')

	corner.corner(
		flatchain,
		labels=[r'log $M_{\rm min}$', r'$\alpha$', 'log $M1$', r'$f_{\rm sat}$', r'$b_{\mathrm{eff}}$',
		        r'log $M_{\mathrm{eff}}$'],
		quantiles=(0.16, 0.84),
		show_titles=True,
		# range=lim,
		levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4)),
		plot_datapoints=False,
		plot_density=False,
		fill_contours=True,
		color="blue",
		hist_kwargs={"color": "black"},
		smooth=0.5,
		smooth1d=0.5,
		truths=[12., 12.8, 1.05, None, None, None],
		truth_color='darkgray'
	)

	plt.savefig("plots/hod_params.pdf")
	plt.close('all')