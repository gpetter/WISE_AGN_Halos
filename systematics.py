import time

import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table, hstack, vstack
import importlib
import pandas as pd
#import plotting
from source import coord_transforms
import healpixhelper
from scipy import interpolate
from scipy import stats



# examine whether
def density_by_coord(datatab, randtab, coordframe, lonlat, ncoordbins, applyweights=True, quantilebins=None):
	import time
	st = time.time()

	if not applyweights:
		bootrandixs = np.random.choice(len(randtab), int(len(randtab) / 1.5), replace=False,
		                               p=randtab['weight']/np.sum(randtab['weight']))
		randtab = randtab[bootrandixs]


	#coords = SkyCoord(datatab['RA'] * u.deg, datatab['DEC'] * u.deg)
	#randcoords = SkyCoord(randtab['RA'] * u.deg, randtab['DEC'] * u.deg)
	if coordframe == 'C':
		lon, lat = datatab['RA'], datatab['DEC']
		randlon, randlat = randtab['RA'], randtab['DEC']
	elif coordframe == 'G':
		lon, lat = datatab['l'], datatab['b']
		randlon, randlat = randtab['l'], randtab['b']
	elif coordframe == 'E':
		lon, lat = datatab['elon'], datatab['elat']
		randlon, randlat = randtab['elon'], randtab['elat']
	else:
		return



	norm = float(len(randlon)) / len(lon)

	if lonlat == 'lon':
		vals, randvals = lon, randlon
	elif lonlat == 'lat':
		vals, randvals = lat, randlat
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

	print(time.time() - st)


	idxs = idxs[np.where((idxs > 0) & (idxs <= len(bincenters)))]
	randidx = randidx[np.where((randidx > 0) & (randidx <= len(bincenters)))]

	print(time.time() - st)

	datacounts, randcounts = np.bincount(idxs)[1:], np.bincount(randidx)[1:]

	print(time.time() - st)


	ratios = datacounts / randcounts * norm

	# propogate Poisson uncertainties to ratios
	errors = np.sqrt(np.square(norm / randcounts * np.sqrt(datacounts)) + np.square(norm * datacounts / np.square(
		randcounts) * np.sqrt(randcounts)))

	print(time.time() - st)


	if applyweights:
		ratio_interpolator = interpolate.interp1d(bincenters, ratios, fill_value='extrapolate')
		interped_ratios, interped_rand_ratios = ratio_interpolator(vals), ratio_interpolator(randvals)
		datatab['weight'] *= (1. / interped_ratios) ** 2
		randtab['weight'] *= interped_rand_ratios ** 2

	print(time.time() - st)


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



		bootrandixs = np.random.choice(len(randtab), int(len(bintab) * 20), replace=False,
		                               p=randtab['weight'] / np.sum(randtab['weight']))
		randtab = randtab[bootrandixs]
		# just keep positions for clustering and weights for lensing
		bintab = bintab['RA', 'DEC', 'l', 'b', 'weight']
		# just keep positions for clustering analysis
		randtab = randtab['RA', 'DEC']

		try:

			bintab.write('/Volumes/grayson/Type1_2_Halos/catalogs/derived/catwise_binned_%s.fits' % (j + 1),
			             format='fits', overwrite=True)
			randtab.write('/Volumes/grayson/Type1_2_Halos/catalogs/derived/catwise_randoms_%s.fits' % (j + 1),
			              format='fits', overwrite=True)
			print('saving QSO positions to cluster')
		except:
			print('dartFS not mounted, saving to local')
			bintab.write('catalogs/derived/catwise_binned_%s.fits' % (j+1), format='fits', overwrite=True)
			randtab.write('catalogs/derived/catwise_randoms_%s.fits' % (j + 1), format='fits', overwrite=True)



		plotting.all_clustering_systematics(nbins, j+1, centers, corcenters, ratios, corratios, errs, corerrs,
		                                    systnames)




def elat_and_depth_weights(ncoordbins):
	binnedtab = Table.read('catalogs/derived/catwise_binned.fits')

	nbins = int(np.max(binnedtab['bin']))
	randtab = Table.read('catalogs/derived/ls_randoms_1_filtered.fits')
	randvals = randtab['elat']
	out, latbins = pd.qcut(randvals, ncoordbins, retbins=True)

	latbins[0] += -0.01
	latbins[-1] += 0.01

	depth_bins = [0, 1500, 100000]
	ls_depth_map = hp.read_map('masks/ls_depth.fits')
	nside = hp.npix2nside(len(ls_depth_map))
	randdepths = ls_depth_map[hp.ang2pix(nside, randtab['l'], randtab['b'], lonlat=True)]

	randhist = stats.binned_statistic_2d(randvals, randdepths, None, statistic='count',
	                                     bins=[latbins, depth_bins], expand_binnumbers=True)

	randidxs = np.array(randhist[3])-1



	for j in range(nbins+1):
		randtab['weight'] = np.ones(len(randtab))
		tab = binnedtab[np.where(binnedtab['bin'] == j)]
		if len(tab) == 0:
			tab = binnedtab
		randratio = len(tab)/float(len(randtab))


		lats = tab['elat']

		depths = ls_depth_map[hp.ang2pix(nside, tab['l'], tab['b'], lonlat=True)]


		thishist = stats.binned_statistic_2d(lats, depths, None, statistic='count',
		                                     bins=[latbins, depth_bins], expand_binnumbers=True)

		data_idxs = np.array(thishist[3]) - 1

		weight_matrix = 1 / randratio * thishist[0] / randhist[0]
		weight_matrix[np.where(np.logical_not(np.isfinite(weight_matrix)))] = 1.
		inv_weight = 1. / weight_matrix



		randweights = weight_matrix[randidxs[0], randidxs[1]]
		dataweights = inv_weight[data_idxs[0], data_idxs[1]]

		tab['weight'] = dataweights
		randtab['weight'] = randweights
		# just keep positions for clustering and weights for lensing
		tab = tab['RA', 'DEC', 'l', 'b', 'weight']
		tab.write('catalogs/derived/catwise_binned_%s.fits' % (j), format='fits', overwrite=True)

		bootrandixs = np.random.choice(len(randtab), int(len(tab) * 20), replace=False,
		                               p=randtab['weight'] / np.sum(randtab['weight']))
		thisrandtab = randtab[bootrandixs]


		# just keep positions for clustering analysis
		thisrandtab = thisrandtab['RA', 'DEC']
		thisrandtab.write('catalogs/derived/catwise_randoms_%s.fits' % (j), format='fits', overwrite=True)






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






