import time

import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table, hstack, vstack
import importlib
import pandas as pd
import plotting
from source import coord_transforms
import healpixhelper
import matplotlib.pyplot as plt
from scipy import interpolate




# examine whether
def density_by_coord(datatab, randtab, coordframe, lonlat, ncoordbins, applyweights=True, quantilebins=None):

	if not applyweights:
		bootrandixs = np.random.choice(len(randtab), int(len(randtab) / 1.5), replace=False,
		                               p=randtab['weight']/np.sum(randtab['weight']))
		randtab = randtab[bootrandixs]


	coords = SkyCoord(datatab['RA'] * u.deg, datatab['DEC'] * u.deg)
	randcoords = SkyCoord(randtab['RA'] * u.deg, randtab['DEC'] * u.deg)
	if coordframe == 'C':
		lon, lat = coords.ra, coords.dec
		randlon, randlat = randcoords.ra, randcoords.dec
	elif coordframe == 'G':
		lon, lat = coords.galactic.l, coords.galactic.b
		randlon, randlat = randcoords.galactic.l, randcoords.galactic.b
	elif coordframe == 'E':
		lon, lat = coords.geocentricmeanecliptic.lon, coords.geocentricmeanecliptic.lat
		randlon, randlat = randcoords.geocentricmeanecliptic.lon, randcoords.geocentricmeanecliptic.lat
	else:
		return


	norm = float(len(randcoords)) / len(coords)

	if lonlat == 'lon':
		vals, randvals = lon.value, randlon.value
	elif lonlat == 'lat':
		vals, randvals = lat.value, randlat.value
	else:
		return



	if applyweights:
		maxval, minval = np.max(randvals), np.min(randvals)
		out, quantilebins = pd.qcut(randvals, ncoordbins, retbins=True)

		quantilebins[0] += -0.001
		quantilebins[-1] += 0.001
	else:
		quantilebins = quantilebins

	bincenters = (quantilebins[1:] + quantilebins[:-1]) / 2

	idxs, randidx = np.digitize(vals, quantilebins), np.digitize(randvals, quantilebins)


	idxs = idxs[np.where((idxs > 0) & (idxs <= len(bincenters)))]
	randidx = randidx[np.where((randidx > 0) & (randidx <= len(bincenters)))]

	datacounts, randcounts = np.bincount(idxs)[1:], np.bincount(randidx)[1:]


	ratios = datacounts / randcounts * norm

	# propogate Poisson uncertainties to ratios
	errors = np.sqrt(np.square(norm / randcounts * np.sqrt(datacounts)) + np.square(norm * datacounts / np.square(
		randcounts) * np.sqrt(randcounts)))


	if applyweights:
		ratio_interpolator = interpolate.interp1d(bincenters, ratios, fill_value='extrapolate')
		interped_ratios, interped_rand_ratios = ratio_interpolator(vals), ratio_interpolator(randvals)
		datatab['weight'] *= 1. / interped_ratios
		randtab['weight'] *= interped_rand_ratios


	#plotting.density_vs_coord(binnum, ratios, errors, bincenters, coordframe, lonlat)
	return quantilebins, bincenters, ratios, errors, datatab, randtab

def density_by_depth(datatab, randtab, depthbins, applyweights=True):

	if not applyweights:
		bootrandixs = np.random.choice(len(randtab), int(len(randtab) / 2), replace=False,
		                               p=randtab['weight']/np.sum(randtab['weight']))
		randtab = randtab[bootrandixs]

	# read in legacy survey depth map in healpix format
	depth_map = hp.read_map('masks/ls_depth.fits')
	nside = hp.npix2nside(len(depth_map))
	# convert ra dec to l, b, to compare to depth map
	lons, lats = healpixhelper.equatorial_to_galactic(datatab['RA'], datatab['DEC'])
	randlons, randlats = healpixhelper.equatorial_to_galactic(randtab['RA'], randtab['DEC'])
	# find indexes of depth map where AGN land
	pix = hp.ang2pix(nside, lons, lats, lonlat=True)
	randpix = hp.ang2pix(nside, randlons, randlats, lonlat=True)
	# a depth value for each AGN
	depthsforpix = depth_map[pix]
	randdepthsforpix = depth_map[randpix]

	depthbinidxs = np.digitize(depthsforpix, depthbins)
	randdepthbinidxs = np.digitize(randdepthsforpix, depthbins)

	norm = float(len(randtab)) / len(datatab)

	ratios, errors = [], []

	median_depthvals = []
	for j in range(1, len(depthbins)):
		# if considering the leftmost bin, accept anything to the left of the right edge of that bin
		if j == 1:
			inbinidxs = np.where(depthbinidxs <= j)
			randinbinidxs = np.where(randdepthbinidxs <= j)
			depthvalsinbin = depthsforpix[inbinidxs]
			randdepthvalsinbin = randdepthsforpix[randinbinidxs]
		# if considering the rightmost bin, accept anythign to the right of the left edge of that bin
		elif j == len(depthbins) - 1:
			inbinidxs = np.where(depthbinidxs >= j)
			randinbinidxs = np.where(randdepthbinidxs >= j)
			depthvalsinbin = depthsforpix[inbinidxs]
			randdepthvalsinbin = randdepthsforpix[randinbinidxs]
		else:
			inbinidxs = np.where(depthbinidxs == j)
			randinbinidxs = np.where(randdepthbinidxs == j)
			depthvalsinbin = depthsforpix[inbinidxs]
			randdepthvalsinbin = randdepthsforpix[randinbinidxs]

		median_depthvals.append(np.median(randdepthvalsinbin))

		ratio = norm * len(depthvalsinbin) / len(randdepthvalsinbin)
		error = np.sqrt(np.square(norm / len(randdepthvalsinbin) * np.sqrt(len(depthvalsinbin))) +
		                      np.square(norm * len(depthvalsinbin) / np.square(len(randdepthvalsinbin))
		                                * np.sqrt(len(randdepthvalsinbin))))
		if applyweights:
			datatab['weight'][inbinidxs] *= 1. / ratio
			randtab['weight'][randinbinidxs] *= ratio

		ratios.append(ratio)
		errors.append(error)


	return median_depthvals, ratios, errors, datatab, randtab



def correct_all_systematics(systnames):

	binnedtab = Table.read('catalogs/derived/catwise_binned.fits')


	nbins = int(np.max(binnedtab['bin']))



	for j in range(nbins):
		centers, corcenters, ratios, corratios, errs, corerrs = [], [], [], [], [], []
		bintab = binnedtab[np.where(binnedtab['bin'] == j+1)]
		randtab = Table.read('catalogs/derived/ls_randoms_1_filtered.fits')
		if 'RA' in systnames:
			print('Testing RA')
			rabins, racenters, raratios, raerrs, bintab, randtab = density_by_coord(bintab, randtab, 'C', lonlat='lon',
			                                                               ncoordbins=10)
			foojoo, corracenters, corraratios, corraerrs, foo, foobar = density_by_coord(bintab, randtab, 'C',
			                                                                             lonlat='lon',
			                                                                             ncoordbins=10,
			                                                                             applyweights=False,
			                                                                             quantilebins=rabins)
			centers.append(racenters), corcenters.append(corracenters), ratios.append(raratios), corratios.append(
				corraratios), errs.append(raerrs), corerrs.append(corraerrs)
		if 'DEC' in systnames:
			print('DEC')
			decbins, deccenters, decratios, decerrs, bintab, randtab = density_by_coord(bintab, randtab, 'C',
			                                                            lonlat='lat', ncoordbins=10)
			foojoo, cordeccenters, cordecratios, cordecerrs, foo, foobar = density_by_coord(bintab, randtab, 'C',
			                                                                                lonlat='lat',
			                                                                                ncoordbins=10,
			                                                                                applyweights=False,
			                                                                                quantilebins=decbins)
			centers.append(deccenters), corcenters.append(cordeccenters), ratios.append(decratios), corratios.append(
				cordecratios), errs.append(decerrs), corerrs.append(cordecerrs)
		if 'Ecliptic Longitude' in systnames:
			print('Eclip. lon.')
			elonbins, eloncenters, elonratios, elonerrs, bintab, randtab = density_by_coord(bintab, randtab, 'E',
			                                                                              lonlat='lon',
			                                                                   ncoordbins=10)
			foojoo, coreloncenters, corelonratios, corelonerrs, foo, foobar = density_by_coord(bintab, randtab, 'E',
			                                                                                   lonlat='lon',
			                                                                                   ncoordbins=10,
			                                                                                   applyweights=False,
			                                                                                   quantilebins=elonbins)
			centers.append(eloncenters), corcenters.append(coreloncenters), ratios.append(elonratios), corratios.append(
				corelonratios), errs.append(elonerrs), corerrs.append(corelonerrs)
		if 'Ecliptic Latitude' in systnames:
			print('Eclip. lat.')
			elatbins, elatcenters, elatratios, elaterrs, bintab, randtab = density_by_coord(bintab, randtab, 'E',
			                                                                              lonlat='lat',
			                                                                   ncoordbins=10)
			foojoo, corelatcenters, corelatratios, corelaterrs, foo, foobar = density_by_coord(bintab, randtab, 'E',
			                                                                                   lonlat='lat',
			                                                                                   ncoordbins=10,
			                                                                                   applyweights=False,
			                                                                                   quantilebins=elatbins)
			centers.append(elatcenters), corcenters.append(corelatcenters), ratios.append(elatratios), corratios.append(
				corelatratios), errs.append(elaterrs), corerrs.append(corelaterrs)
		if 'Galactic Longitude' in systnames:
			print('Galactic lon.')
			glonbins, gloncenters, glonratios, glonerrs, bintab, randtab = density_by_coord(bintab, randtab, 'G',
			                                                                              lonlat='lon',
			                                                                   ncoordbins=10)
			foojoo, corgloncenters, corglonratios, corglonerrs, foo, foobar = density_by_coord(bintab, randtab, 'G',
			                                                                                   lonlat='lon',
			                                                                                   ncoordbins=10,
			                                                                                   applyweights=False,
			                                                                                   quantilebins=glonbins)
			centers.append(gloncenters), corcenters.append(corgloncenters), ratios.append(glonratios), corratios.append(
				corglonratios), errs.append(glonerrs), corerrs.append(corglonerrs)
		if 'Galactic Latitude' in systnames:
			print('Galactic lat.')
			glatbins, glatcenters, glatratios, glaterrs, bintab, randtab = density_by_coord(bintab, randtab, 'G',
			                                                                              lonlat='lat',
			                                                                   ncoordbins=10)
			foojoo, corglatcenters, corglatratios, corglaterrs, foo, foobar = density_by_coord(bintab, randtab, 'G',
			                                                                                   lonlat='lat',
			                                                                                   ncoordbins=10,
			                                                                                   applyweights=False,
			                                                                                   quantilebins=glatbins)
			centers.append(glatcenters), corcenters.append(corglatcenters), ratios.append(glatratios), corratios.append(
				corglatratios), errs.append(glaterrs), corerrs.append(corglaterrs)
		print('r-band depth')
		rdepthcenters, rdepthratios, rdeptherrs, bintab, randtab = density_by_depth(bintab, randtab, [150, 1500,
		                                                                                               20000])
		centers.append(np.log10(rdepthcenters)), ratios.append(rdepthratios), errs.append(rdeptherrs)

		corrdepthcenters, corrdepthratios, corrdeptherrs, foo, foobar = density_by_depth(bintab, randtab,
		                                                                            [150, 1500, 20000],
		                                                                            applyweights=False)
		corcenters.append(np.log10(corrdepthcenters)), corratios.append(corrdepthratios), corerrs.append(corrdeptherrs)

		bintab.write('catalogs/derived/catwise_binned_%s.fits' % (j+1), format='fits', overwrite=True)

		bootrandixs = np.random.choice(len(randtab), int(len(bintab) * 20), replace=False,
		                               p=randtab['weight'] / np.sum(randtab['weight']))
		randtab = randtab[bootrandixs]
		randtab.write('catalogs/derived/catwise_randoms_%s.fits' % (j+1), format='fits', overwrite=True)




		plotting.all_clustering_systematics(nbins, j+1, centers, corcenters, ratios, corratios, errs, corerrs,
		                                    systnames)








def smoothed_density(catname, nside):
	cat = Table.read('catalogs/derived/%s_binned.fits' % catname)
	#randcat = Table.read('catalogs/derived/ls_randoms_1_filtered.fits')

	#randmap = np.array(healpixhelper.healpix_density_map(randl, randb, nside)).astype(np.float)
	totalmask = hp.read_map('masks/union.fits')
	for j in range(int(np.max(cat['bin']))):
		bincat = cat[np.where(cat['bin'] == j+1)]
		l, b = coord_transforms.sky_transform(bincat['RA'], bincat['DEC'], ['C', 'G'])
		datamap = np.array(healpixhelper.healpix_density_map(l, b, nside)).astype(np.float64)
		#ratiomap = datamap / randmap
		#badpix = np.where((np.isinf(ratiomap)) | np.isnan(ratiomap))[0]
		badpix = np.where(totalmask == 0)

		datamap[badpix] = np.nan
		smoothedratio = healpixhelper.masked_smoothing(datamap, fwhm_arcmin=300.)
		smoothedratio[badpix] = hp.UNSEEN

		plotting.data_vs_random_density(smoothedratio, j+1)

# assign weights to correct for varying optical imaging depth across the field
def correct_randoms():
	# read in table of AGN binned by r-W2 color
	tab = Table.read('catalogs/derived/catwise_binned.fits')
	randtab = Table.read('catalogs/derived/ls_randoms_1_filtered.fits')
	nbins = int(np.max(tab['bin']))


	# split sample up into two bins of imaging depth

	depthbinedges = [150, 1500, 20000]

	# separate table into regions with shallow and deep optical imaging
	tab_in_shallow_depth = tab[np.where(depthsforpix < 1500)]
	tab_in_deep = tab[np.where(depthsforpix > 1500)]
	randtab_in_shallow_depth = randtab[np.where(randdepthsforpix < 1500)]
	randtab_in_deep = randtab[np.where(randdepthsforpix > 1500)]

	med_depths = [np.median(np.log10(randtab_in_shallow_depth['PSFDEPTH_R'])),
	              np.median(np.log10(randtab_in_deep['PSFDEPTH_R']))]

	# coordinates of randoms in deep and shallow footprints
	randcoords_in_shallow = SkyCoord(randtab_in_shallow_depth['RA'] * u.deg, randtab_in_shallow_depth['DEC'] * u.deg)
	randcoords_in_deep = SkyCoord(randtab_in_deep['RA'] * u.deg, randtab_in_deep['DEC'] * u.deg)

	# convert RA DEC to other coordinate that presents systematic
	coordframe = 'E'
	latlon = 'lat'
	if coordframe == 'E':
		shallowrandlon, shallowrandlat = randcoords_in_shallow.geocentricmeanecliptic.lon, \
		                                 randcoords_in_shallow.geocentricmeanecliptic.lat
		deeprandlon, deeprandlat = randcoords_in_deep.geocentricmeanecliptic.lon, \
		                           randcoords_in_deep.geocentricmeanecliptic.lat
	if latlon == 'lat':
		shallowrandvals = shallowrandlat
		deeprandvals = deeprandlat
	else:
		shallowrandvals = shallowrandlon
		deeprandvals = deeprandlon


	tmp, deepcoordbins = pd.qcut(deeprandvals, 10, retbins=True)

	tmp, shallowcoordbins = pd.qcut(shallowrandvals, 10, retbins=True)
	deepbincenters = (deepcoordbins[1:] + deepcoordbins[:-1]) / 2
	shallowbincenters = (shallowcoordbins[1:] + shallowcoordbins[:-1]) / 2
	deepcoordbins[0] += -0.001
	deepcoordbins[-1] += 0.001
	shallowcoordbins[0] += -0.001
	shallowcoordbins[-1] += 0.001

	deepfrac = float(len(randtab_in_deep)) / len(randtab)
	shallowfrac = float(len(randtab_in_shallow_depth)) / len(randtab)






	# count total number of AGN in each depth bin
	fullcounts = np.bincount(np.digitize(depthsforpix, depthbinedges))[1:]


	avgdepths, relative_counts, errors = [], [], []
	# count number of AGN in same depth bins for each color bin, compare to counts for whole sample
	for i in range(1, nbins+1):
		binnedtab = tab[np.where(tab['bin'] == i)]



		lons, lats = healpixhelper.equatorial_to_galactic(binnedtab['RA'], binnedtab['DEC'])

		pix = hp.ang2pix(nside, lons, lats, lonlat=True)
		depthsforpix = depth_map[pix]

		shallowbintab = binnedtab[np.where(depthsforpix < 1500)]

		deepbintab = binnedtab[np.where(depthsforpix > 1500)]

		deepnormfactor, shallownormfactor = len(binnedtab) / 10 * deepfrac, len(binnedtab) / 10 * shallowfrac

		shallowbinlon, shallowbinlat = coord_transforms.sky_transform(shallowbintab['RA'], shallowbintab['DEC'],
		                                                              ['C', 'E'])
		deepbinlon, deepbinlat = coord_transforms.sky_transform(deepbintab['RA'], deepbintab['DEC'], ['C', 'E'])

		deepbincounts = np.bincount(np.digitize(deepbinlat, deepcoordbins))[1:]
		shallowbincounts = np.bincount(np.digitize(shallowbinlat, shallowcoordbins))[1:]

		shallowcorrections, deepcorrections = shallowbincounts / shallownormfactor, deepbincounts / deepnormfactor


		#interpolator = interpolate.interp2d(x=med_depths, y=[shallowbincenters, deepbincenters], z=ratio_matrix,
		#                                    fill_value=None)


		#shallowweights, deepweights = np.ones(len(randtab_in_shallow_depth)), np.ones(len(randtab_in_deep))
		shallowinterpolator = interpolate.interp1d(shallowbincenters, shallowcorrections, fill_value='extrapolate')
		deepinterpolator = interpolate.interp1d(deepbincenters, deepcorrections, fill_value='extrapolate')
		shallowweights = shallowinterpolator(shallowrandvals.value)
		deepweights = deepinterpolator(deeprandvals.value)
		#shallowweights = shallowcorrections[np.digitize(shallowrandvals.value, bins=shallowcoordbins) - 1]


		#deepweights = deepcorrections[np.digitize(deeprandvals.value, bins=deepcoordbins) - 1]
		randtab_in_shallow_depth['weight'] = shallowweights
		randtab_in_deep['weight'] = deepweights
		fullweighted_randtab = vstack([randtab_in_shallow_depth, randtab_in_deep])

		normed_weights = fullweighted_randtab['weight'] / np.sum(fullweighted_randtab['weight'])

		subsampled_rands = fullweighted_randtab[np.random.choice(np.arange(len(fullweighted_randtab)),
		                                                         20*len(binnedtab), replace=False, p=normed_weights)]

		subsampled_rands.write('catalogs/derived/catwise_randoms_%s.fits' % i, format='fits', overwrite=True)







		"""factor = len(binnedtab) / float(len(tab))

		counts = np.bincount(np.digitize(depthsforpix, depthbinedges))[1:]



		relative_counts.append(factor * fullcounts / counts)

		errors.append(np.sqrt(np.square(factor / counts * np.sqrt(fullcounts)) +
		        np.square(factor * fullcounts / np.square(counts) * np.sqrt(counts))))"""
	"""avgdepth = np.array([500, 3000])
	plotting.depth_v_density(avgdepth, relative_counts, errors)
	return relative_counts

def correct_systematics(mode='weights'):
	if mode == 'weights':
		return
	else:
		randcat = Table.read('catalogs/derived/ls_randoms_1_filtered.fits')

		rel_counts = correct_for_depth()
		for j in range(len(rel_counts)):
			init_weights = np.ones(len(randcat))
			init_weights[np.where(randcat['PSFDEPTH_R'] < 1500)] = (rel_counts[j][0])**10
			init_weights[np.where(randcat['PSFDEPTH_R'] > 1500)] = (rel_counts[j][1])**10
			init_weights = init_weights / np.sum(init_weights)
			sampleidxs = np.random.choice(np.arange(len(init_weights)), size=int(len(randcat) / 3), replace=False,
			                              p=init_weights)
			subsampledcat = randcat[sampleidxs]
			#subsampledcat.write('catalogs/derived/ls_randoms_filtered_%s.fits' % (j + 1), format='fits',
			# overwrite=True)


			latcenters, ratios = density_by_coord('catwise', j+1, subsampledcat, 'E', 'lat', 10)
			randelon, randelat = coord_transforms.sky_transform(subsampledcat['RA'], subsampledcat['DEC'],
			                                                    trans=['C', 'E'])
			interpfunc = interpolate.interp1d(latcenters, ratios, fill_value='extrapolate')
			newweights = interpfunc(randelat)
			newweights = newweights / np.sum(newweights)
			sampleidxs = np.random.choice(np.arange(len(subsampledcat)), size=int(len(randcat) / 5), replace=False,
			                              p=newweights)
			subsampledcat = subsampledcat[sampleidxs]
			subsampledcat.write('catalogs/derived/ls_randoms_filtered_%s.fits' % (j + 1), format='fits',
			                    overwrite=True)

			density_by_coord('catwise', j+1, subsampledcat, 'E', 'lat', 10)"""





