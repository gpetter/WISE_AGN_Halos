import numpy as np
import healpy as hp
import convergence_map
from astropy.io import fits
import importlib
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import fileinput
import sys
import glob
import plotting
import time
from source import survival


# convert ras and decs to galactic l, b coordinates
def equatorial_to_galactic(ra, dec):
	ra_decs = SkyCoord(ra, dec, unit='deg', frame='icrs')
	ls = np.array(ra_decs.galactic.l.radian * u.rad.to('deg'))
	bs = np.array(ra_decs.galactic.b.radian * u.rad.to('deg'))
	return ls, bs


# given list of ras, decs, return indices of sources whose centers lie outside the masked region of the lensing map
def get_qsos_outside_mask(nsides, themap, ras, decs):
	ls, bs = equatorial_to_galactic(ras, decs)
	pixels = hp.ang2pix(nsides, ls, bs, lonlat=True)
	idxs = np.where(themap[pixels] != hp.UNSEEN)
	return idxs


# AzimuthalProj.projmap requires a vec2pix function for some reason, so define one where the nsides are fixed
def newvec2pix(x, y, z):
	return hp.vec2pix(nside=4096, x=x, y=y, z=z)


# perform one iteration of a stack
def stack_iteration(current_sum, current_weightsum, new_cutout, weight, prob_weight, imsize):
	# create an image filled with the value of the weight, set weights to zero where the true map is masked
	wmat = np.full((imsize, imsize), weight)
	wmat[np.isnan(new_cutout)] = 0

	# the weights for summing in the denominator are multiplied by the probabilty weight to account
	# for the fact that some sources aren't quasars and contribute no signal to the stack
	wmat_for_sum = wmat * prob_weight
	# the running total sum is the sum from last iteration plus the new cutout
	new_sum = np.nansum([current_sum, new_cutout], axis=0)
	new_weightsum = np.sum([current_weightsum, wmat_for_sum], axis=0)

	return new_sum, new_weightsum


def sum_projections(lon, lat, weights, prob_weights, imsize, reso, inmap, nstack):
	running_sum, weightsum = np.zeros((imsize, imsize)), np.zeros((imsize, imsize))
	for j in range(nstack):
		azproj = hp.projector.AzimuthalProj(rot=[lon[j], lat[j]], xsize=imsize, reso=reso, lamb=True)
		new_im = weights[j] * convergence_map.set_unseen_to_nan(azproj.projmap(inmap, vec2pix_func=newvec2pix))

		running_sum, weightsum = stack_iteration(running_sum, weightsum, new_im, weights[j], prob_weights[j], imsize)

	return running_sum, weightsum


# for parallelization of stacking procedure, this method will stack a "chunk" of the total stack
def stack_chunk(chunksize, nstack, lon, lat, inmap, weighting, prob_weights, imsize, reso, k):
	# if this is the last chunk in the stack, the number of sources in the chunk probably won't be = chunksize
	if (k * chunksize) + chunksize > nstack:
		stepsize = nstack % chunksize
	else:
		stepsize = chunksize
	highidx, lowidx = ((k * chunksize) + stepsize), (k * chunksize)

	totsum, weightsum = sum_projections(lon[lowidx:highidx], lat[lowidx:highidx], weighting[lowidx:highidx],
	                                    prob_weights[lowidx:highidx], imsize, reso, inmap, stepsize)

	return totsum, weightsum


# stack by computing an average iteratively. this method uses little memory but cannot be parallelized
def stack_projections(ras, decs, weights=None, prob_weights=None, imsize=240, outname=None, reso=1.5, inmap=None,
                      nstack=None, mode='normal', chunksize=500):
	# if no weights provided, weights set to one
	if weights is None:
		weights = np.ones(len(ras))
	if prob_weights is None:
		prob_weights = np.ones(len(ras))
	# if no limit to number of stacks provided, stack the entire set
	if nstack is None:
		nstack = len(ras)

	lons, lats = equatorial_to_galactic(ras, decs)

	if mode == 'normal':
		totsum, weightsum = sum_projections(lons, lats, weights, prob_weights, imsize, reso, inmap, nstack)
		finalstack = totsum/weightsum

	else:
		return

	if outname is not None:
		finalstack.dump('%s.npy' % outname)

	return finalstack


def fast_stack(lons, lats, inmap, weights=None, prob_weights=None, nsides=2048, iterations=500, bootstrap=False):
	if weights is None:
		weights = np.ones(len(lons))

	outerkappa = []


	inmap = convergence_map.set_unseen_to_nan(inmap)
	pix = hp.ang2pix(nsides, lons, lats, lonlat=True)
	neighborpix = hp.get_all_neighbours(nsides, pix)

	centerkappa = inmap[pix]
	neighborkappa = np.nanmean(inmap[neighborpix], axis=0)
	centerkappa = np.nanmean([centerkappa, neighborkappa], axis=0)

	weights[np.isnan(centerkappa)] = 0
	centerkappa[np.isnan(centerkappa)] = 0


	if prob_weights is not None:
		true_weights_for_sum = weights * np.array(prob_weights)
		weightsum = np.sum(true_weights_for_sum)
	else:
		weightsum = np.sum(weights)

	centerstack = np.sum(weights * centerkappa) / weightsum


	if iterations > 0:

		for x in range(iterations):
			bootidx = np.random.choice(len(lons), len(lons))
			outerkappa.append(np.nanmean(inmap[hp.ang2pix(nsides, lons[bootidx], lats[bootidx], lonlat=True)]))

		if bootstrap:
			return centerstack, outerkappa
		else:
			return centerstack, np.nanstd(outerkappa)
	else:
		return centerstack

# for a given sky position, choose which pre-calculated sky projection is centered closest to that position
def find_closest_cutout(l, b, fixedls, fixedbs):
	return np.argmin(hp.rotator.angdist([fixedls, fixedbs], [l, b], lonlat=True))


def stack_cutouts(ras, decs, weights, prob_weights, imsize, nstack, outname=None, bootstrap=False):
	# read in previously calculated projections covering large swaths of sky
	projectionlist = glob.glob('planckprojections/*')
	projections = np.array([np.load(filename, allow_pickle=True) for filename in projectionlist])

	if bootstrap:
		bootidxs = np.random.choice(len(ras), len(ras))
		ras, decs, weights, prob_weights = ras[bootidxs], decs[bootidxs], weights[bootidxs], prob_weights[bootidxs]
	# center longitudes/latitudes of projections
	projlons = [int(filename.split('/')[1].split('.')[0].split('_')[0]) for filename in projectionlist]
	projlats = [int(filename.split('/')[1].split('.')[0].split('_')[1]) for filename in projectionlist]

	# healpy projection objects used to create the projections
	# contains methods to convert from angular position of quasar to i,j position in projection
	projector_objects = [hp.projector.AzimuthalProj(rot=[projlons[i], projlats[i]], xsize=5000, reso=1.5, lamb=True)
	                     for i in range(len(projlons))]
	# convert ras and decs to galactic ls, bs
	lon, lat = equatorial_to_galactic(ras, decs)


	running_sum, weightsum = np.zeros((imsize, imsize)), np.zeros((imsize, imsize))
	# for each source
	for k in range(nstack):
		# choose the projection closest to the QSO's position
		cutoutidx = find_closest_cutout(lon[k], lat[k], projlons, projlats)
		cutout_to_use = projections[cutoutidx]
		projobj = projector_objects[cutoutidx]
		# find the i,j coordinates in the projection corresponding to angular positon in sky
		i, j = projobj.xy2ij(projobj.ang2xy(lon[k], lat[k], lonlat=True))
		# make cutout
		cut_at_position = cutout_to_use[int(i-imsize/2):int(i+imsize/2), int(j-imsize/2):int(j+imsize/2)]
		# stack
		running_sum, weightsum = stack_iteration(running_sum, weightsum, cut_at_position, weights[j], prob_weights[j],
		                                         imsize)
	finalstack = running_sum/weightsum
	if outname is not None:
		finalstack.dump('%s.npy' % outname)

	return finalstack


def stack_without_projection():
	cat = Table.read('catalogs/derived/catwise_r90_binned.fits')
	inmap = hp.read_map('lensing_maps/planck/smoothed_masked.fits')
	ras, decs = cat["RA"], cat["DEC"]
	lons, lats = equatorial_to_galactic(ras, decs)
	nside = hp.npix2nside(len(inmap))
	coords = SkyCoord(lons * u.deg, lats * u.deg)
	hpras, hpdecs = hp.pix2ang(nside, np.arange(len(inmap)), lonlat=True)
	st = time.time()
	coordvecs = hp.ang2vec(lons, lats, lonlat=True)

	for j in range(len(lons)):
		pixindisk = hp.query_disc(nside, coordvecs[j], np.pi / 90.)
		hpras, hpdecs = hp.pix2ang(nside, pixindisk, lonlat=True)
		hpcoords = SkyCoord(hpras * u.deg, hpdecs * u.deg)
		separations = coords[j].separation(hpcoords).to('arcmin').value
		if j % 1000 == 0:
			print(time.time() - st)







def quick_stack_suite(sample_name, kappa_name, bootstrap=False, nsides=2048):
	kappa_map = hp.read_map('lensing_maps/%s/smoothed_masked.fits' % kappa_name)
	tab = Table.read('catalogs/derived/%s_binned.fits' % sample_name)


	medcolors, peakkappas, kappa_errs = [], [], []
	for j in range(int(np.max(tab['bin']))):
		binnedtab = tab[np.where(tab['bin'] == j+1)]
		if kappa_name == 'planck':
			lons, lats = equatorial_to_galactic(binnedtab['RA'], binnedtab['DEC'])
		else:
			lons, lats = binnedtab['RA'], binnedtab['DEC']
		peakkap, kapstd = fast_stack(lons, lats, kappa_map, nsides=nsides)
		peakkappas.append(peakkap), kappa_errs.append(kapstd)

		#medcolors.append(survival.km_median(binnedtab['color'], np.logical_not(binnedtab['detect'])))
		medcolors.append(np.median(binnedtab['color']))
	plotting.plot_peakkappa_vs_bin(medcolors, peakkappas, kappa_errs)





# if running on local machine, can use this to stack using multiprocessing.
# Otherwise use stack_mpi.py to perform stack on a cluster computer
def stack_suite(sample_name, stack_map, stack_noise, reso=1.5, imsize=240, nsides=2048, mode='normal',
                nstack=None, bootstrap=False, temperature=False, random=False):

	if temperature:
		planck_map = hp.read_map('maps/smica_masked.fits')
		outname = 'stacks/%s_%s_temp' % (sample_name, color)
	outname = 'stacks/%s' % sample_name

	cat = fits.open('catalogs/derived/%s_binned.fits' % (sample_name))[1].data

	ras, decs = cat['RA'], cat['DEC']

	if nstack is None:
		nstack = len(ras)

	"""if (color == 'complete') or (color == 'RL'):
		weights = np.ones(len(cat))
	else:
		weights = cat['weight']"""
	weights = np.ones(len(cat))

	if random:
		ras = ras + np.random.uniform(2., 14., len(ras))
		outname = 'stacks/random_stack'



	if mode == 'fast':
		return fast_stack(ras, decs, planck_map, weights=weights, nsides=nsides, bootstrap=bootstrap)
	elif mode == 'cutout':
		return stack_cutouts(ras, decs, weights, np.ones(len(ras)), imsize, nstack, outname, bootstrap)
	elif mode == 'mpi':
		for line in fileinput.input(['stack_mpi.py'], inplace=True):
			if line.strip().startswith('sample_name = '):
				line = "\tsample_name = '%s'\n" % sample_name
			if line.strip().startswith('color = '):
				line = "\tcolor = '%s'\n" % color
			if line.strip().startswith('imsize = '):
				line = '\timsize = %s\n' % imsize
			if line.strip().startswith('reso = '):
				line = '\treso = %s\n' % reso
			if line.strip().startswith('stack_maps = '):
				line = '\tstack_maps = %s\n' % stack_map
			if line.strip().startswith('stack_noise = '):
				line = '\tstack_noise = %s\n' % stack_noise
			if line.strip().startswith('temperature = '):
				line = '\ttemperature = %s\n' % temperature
			sys.stdout.write(line)
	else:
		if stack_map:
			stack_projections(ras, decs, weights=weights, prob_weights=probs, inmap=planck_map, mode=mode,
			         outname=outname, reso=reso, imsize=imsize, nstack=nstack)

		if stack_noise:
			noisemaplist = glob.glob('noisemaps/maps/*.fits')
			for j in range(0, len(noisemaplist)):
				noisemap = hp.read_map('noisemaps/maps/%s.fits' % j)
				stack_projections(ras, decs, weights=weights, prob_weights=probs, inmap=noisemap, mode=mode,
				         outname='stacks/noise_stacks/%s_%s' % (j, color), reso=reso, imsize=imsize, nstack=nstack)




