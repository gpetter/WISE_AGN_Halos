import numpy as np
import healpy as hp
from scipy import stats
from source import organization
organizer = organization.Organizer()



def montecarlo_spearman(xs, ys, yerrs):

	#realizations = np.random.normal(loc=ys, scale=yerrs, size=(1000, len(ys)))
	#spearmanranks = stats.spearmanr(xs, realizations, axis=1)[0]

	spearmanranks = []
	for j in range(1000):
		realization = np.random.normal(ys, yerrs)
		spearmanranks.append(stats.spearmanr(xs, realization)[0])
	return np.mean(spearmanranks), np.std(spearmanranks)




def bin_on_sky(ras, decs, njackknives, nside=None):
	from sklearn import cluster
	import time
	import matplotlib.pyplot as plt
	"""test_nsides = np.linspace(1, 50, 100)
	area_as_func_of_nside = hp.nside2pixarea(test_nsides, degrees=True)

	lowresdensity = healpixhelper.healpix_density_map(ras, decs, 32)
	skyfraction = len(lowresdensity[np.where(lowresdensity > 0)])/len(lowresdensity)

	footprintarea = 41252.96 * skyfraction  # deg^2

	approx_area_per_pixel = footprintarea / njackknives

	nside_for_area = int(np.interp(approx_area_per_pixel, np.flip(area_as_func_of_nside), np.flip(test_nsides)))

	if nside is not None:
		nside_for_area = nside

	pixidxs = hp.ang2pix(nside_for_area, ras, decs, lonlat=True)

	return pixidxs, nside_for_area"""
	nsides = 64
	occupiedpix = hp.ang2pix(nside=nsides, theta=ras, phi=decs, lonlat=True)
	uniquepix = np.unique(occupiedpix)
	hp_ra, hp_dec = hp.pix2ang(nside=nsides, ipix=uniquepix, lonlat=True)
	weighted_kmeans = False

	if weighted_kmeans:
		# Apply weights by adding more of the same pixels to the array
		hp_ra_w = np.copy(hp_ra)
		hp_dec_w = np.copy(hp_dec)
		for i in range(1, 20):
			mask = np.round(density / 50.) == i
			for j in range(i - 1):  # i-1 because there is already one copy
				hp_ra_w = np.concatenate([hp_ra_w, hp_ra[mask]])
				hp_dec_w = np.concatenate([hp_dec_w, hp_dec[mask]])
		print(len(hp_ra))
		print(len(hp_ra_w))
		coords = np.concatenate([[hp_ra_w], [hp_dec_w]])
		coords = coords.T
	else:
		coords = np.concatenate([[hp_ra], [hp_dec]])
		coords = coords.T

	kmeans = cluster.KMeans(n_clusters=njackknives, n_init=10)
	t0 = time.time()
	kmeans.fit(coords)
	elapsed_time = time.time() - t0
	print('time %.2fs' % (elapsed_time))

	if weighted_kmeans:
		labels = kmeans.labels_[:len(hp_ra)]
	else:
		labels = kmeans.labels_
	plt.figure(figsize=(12, 8))
	#plt.scatter(hp_ra, hp_dec, c=labels, s=5, cmap=plt.cm.nipy_spectral)
	blankmap = hp.UNSEEN * np.ones(hp.nside2npix(nsides))
	blankmap[uniquepix] = labels
	hp.mollview(blankmap, cmap=plt.cm.nipy_spectral)
	# plt.axis([360, 0, -21, 36])
	plt.savefig(organizer.plotdir + 'jackknives.pdf')
	plt.close('all')
	return blankmap






def bootstrap_sky_bins(ras, decs, refras, refdecs, randras, randdecs, njackknives):

	# divide sample up into subvolumes on sky
	refpixidxs, nside = bin_on_sky(refras, refdecs, njackknives)
	pixidxs, foo = bin_on_sky(ras, decs, njackknives, nside=nside)

	randpixidxs, randnside = bin_on_sky(randras, randdecs, njackknives, nside=nside)

	# Norberg+2009
	nsub_factor = 3

	# list of subvolume indices is the set of unique values of above array
	unique_pix = np.unique(refpixidxs)

	# bootstrap resample the subvolumes, oversample by factor 3
	boot_pix = np.random.choice(unique_pix, nsub_factor*len(unique_pix))

	idxs, refidxs, randidxs = [], [], []

	# for each subvolume
	for pixel in boot_pix:
		# find which objects lie in subvolume
		idxs_in_pixel = np.where(pixidxs == pixel)[0]
		idxs += list(idxs_in_pixel)

		refidxs += list(np.where(refpixidxs == pixel)[0])

		# add randoms in subvolume to list
		randidxs += list(np.where(randpixidxs == pixel)[0])

		# bootstrap resample these objects
		#booted_idxs_in_pixel = np.random.choice(idxs_in_pixel, len(idxs_in_pixel))
		# add bootstrapped objects to list
		#idxs += list(booted_idxs_in_pixel)


	# subsample
	#final_idxs = idxs
	final_idxs = np.random.choice(idxs, len(ras))
	final_refidxs = np.random.choice(refidxs, len(refras))
	final_randidxs = np.random.choice(randidxs, len(randras), replace=False)

	return final_idxs, final_refidxs, final_randidxs



def covariance_matrix(resampled_profiles, avg_profile):
	n_bins = len(resampled_profiles[0])
	n_realizations = len(resampled_profiles)
	c_ij = np.zeros((n_bins, n_bins))
	for i in range(n_bins):
		for j in range(n_bins):
			k_i = resampled_profiles[:, i]
			k_i_bar = avg_profile[i]

			k_j = resampled_profiles[:, j]
			k_j_bar = avg_profile[j]

			product = (k_i - k_i_bar) * (k_j - k_j_bar)
			sum = np.sum(product)
			c_ij[i, j] = 1 / (n_realizations - 1) * sum

	return c_ij

