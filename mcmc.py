import emcee
import numpy as np
import clusteringModel
import hod_model
import plotting



# log prior function
def ln_prior(theta):
	if len(theta) == 1:
		mmin = theta
		if (mmin > 11) & (mmin < 14):
			return 0.
		else:
			return -np.inf
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
		if (mmin > 11.) & (mmin < 14.) & (alpha > 0) & (alpha < 2) & (m1 > mmin) & (m1 < 14.5):
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
def ln_prob(theta, anglebins, y, yerr, zs, dndz, modeltype):
	prior = ln_prior(theta)

	# keep track of derived parameters like satellite fraction, effective bias, effective mass
	derived = (hod_model.derived_parameters(zs, dndz, theta, modeltype))


	# get model prediciton for given parameter set
	modelprediction = clusteringModel.angular_corr_func_in_bins(anglebins, zs=zs, dn_dz_1=dndz,
	                                                            hodparams=theta,
	                                                            hodmodel=modeltype)

	# residual is data - model
	residual = y - modelprediction

	likely = ln_likelihood(residual, yerr)
	prob = prior + likely

	# return log_prob, along with derived parameters for this parameter set
	return (prob,) + derived





def sample_space(binnum, nbins, nwalkers, ndim, niter, anglebins, y, yerr,
                 zs, dndz, modeltype, initial_params=None):

	blobs_dtype = [("f_sat", float), ("b_eff", float), ("m_eff", float)]
	if ndim == 1:
		blobs_dtype = blobs_dtype[1:]


	sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, args=[anglebins, y, yerr, zs, dndz,
	                                                               modeltype], blobs_dtype=blobs_dtype)


	if initial_params is None:
		initial_params = [12., 1., 12.5]
		initial_params = initial_params[:ndim]

	# start walkers near least squares fit position with random gaussian offsets
	pos = np.array(initial_params) + 2e-1 * np.random.normal(size=(sampler.nwalkers, sampler.ndim))

	sampler.run_mcmc(pos, niter, progress=True)


	flatchain = sampler.get_chain(discard=10, flat=True)
	blobs = sampler.get_blobs(discard=10, flat=True)

	if ndim > 1:
		flatchain = np.hstack((
			flatchain,
			np.atleast_2d(blobs['f_sat']).T,
			np.atleast_2d(blobs['b_eff']).T,
			np.atleast_2d(blobs['m_eff']).T
		))
	else:
		flatchain = np.hstack((
			flatchain,
			np.atleast_2d(blobs['b_eff']).T,
			np.atleast_2d(blobs['m_eff']).T
		))

	centervals, lowerrs, higherrs = [], [], []
	for i in range(ndim + len(blobs_dtype)):
		post = np.percentile(flatchain[:, i], [16, 50, 84])
		q = np.diff(post)
		centervals.append(post[1])
		lowerrs.append(q[0])
		higherrs.append(q[1])


	plotting.hod_corner(flatchain, ndim, binnum, nbins)

	return centervals, lowerrs, higherrs

