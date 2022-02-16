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
from scipy import interpolate
importlib.reload(healpixhelper)
importlib.reload(coord_transforms)
importlib.reload(plotting)


# examine whether
def density_by_coord(catname, binnum, rand, coordframe, lonlat, nbins):
	t = Table.read('catalogs/derived/%s_binned.fits' % catname)
	t = t[np.where(t['bin'] == binnum)]

	coords = SkyCoord(t['RA'] * u.deg, t['DEC'] * u.deg)
	randcoords = SkyCoord(rand['RA'] * u.deg, rand['DEC'] * u.deg)
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

	norm = float(len(randcoords) ) / len(coords)

	if lonlat == 'lon':
		vals, randvals = lon.value, randlon.value
	elif lonlat == 'lat':
		vals, randvals = lat.value, randlat.value
	else:
		return

	maxval, minval = np.max(randvals), np.min(randvals)
	out, quantilebins = pd.qcut(randvals, nbins, retbins=True)
	bincenters = (quantilebins[1:] + quantilebins[:-1])/2
	quantilebins[0] += -0.001
	quantilebins[-1] += 0.001



	idxs, randidx = np.digitize(vals, quantilebins), np.digitize(randvals, quantilebins)

	datacounts, randcounts = np.bincount(idxs)[1:], np.bincount(randidx)[1:]

	ratios = datacounts / randcounts * norm
	# propogate Poisson uncertainties to ratios
	errors = np.sqrt(np.square(norm / randcounts * np.sqrt(datacounts)) + np.square(norm * datacounts / np.square(
		randcounts) * np.sqrt(randcounts)))

	"""ratios = []

	for j in range(1, nbins):
		try:
			ratios.append(len(np.where(idxs == j)[0]) / len(np.where(randidx == j)[0]) * norm)

		except:
			ratios.append(np.nan)"""

	plotting.density_vs_coord(binnum, ratios, errors, bincenters, coordframe, lonlat)
	return bincenters, ratios

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
	# read in legacy survey depth map in healpix format
	depth_map = hp.read_map('masks/ls_depth.fits')
	nside = hp.npix2nside(len(depth_map))
	# convert ra dec to l, b, to compare to depth map
	lons, lats = healpixhelper.equatorial_to_galactic(tab['RA'], tab['DEC'])
	randlons, randlats = healpixhelper.equatorial_to_galactic(randtab['RA'], randtab['DEC'])
	# find indexes of depth map where AGN land
	pix = hp.ang2pix(nside, lons, lats, lonlat=True)
	randpix = hp.ang2pix(nside, randlons, randlats, lonlat=True)
	# a depth value for each AGN
	depthsforpix = depth_map[pix]
	randdepthsforpix = depth_map[randpix]

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





