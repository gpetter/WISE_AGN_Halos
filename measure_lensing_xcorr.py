import healpy as hp
import numpy as np
from astropy.table import Table
import healpixhelper
import importlib
import sys
import pymaster as nmt
import os
import glob
from functools import partial


import plotting
importlib.reload(plotting)
importlib.reload(healpixhelper)


def write_master_workspace(minl, maxl, nbins, apodize=None):
	if len(glob.glob('masks/workspace.fits')) > 0:
		print('Removing old Master matrix')
		os.remove('masks/workspace.fits')
	logbins = np.logspace(np.log10(minl), np.log10(maxl), nbins+1).astype(int)
	lowedges, highedges = logbins[:-1], (logbins - 1)[1:]

	mask = hp.read_map('masks/union.fits')

	if apodize is not None:
		mask = nmt.mask_apodization(mask, apodize, apotype='Smooth')



	f0 = nmt.NmtField(mask, [np.zeros(len(mask))])
	b = nmt.NmtBin.from_edges(lowedges, highedges)
	np.array(b.get_effective_ells()).dump('results/lensing_xcorrs/scales.npy')
	w = nmt.NmtWorkspace()
	w.compute_coupling_matrix(f0, f0, b)
	w.write_to('masks/workspace.fits')


def write_ls_density_mask():
	randomcat = Table.read('catalogs/randoms/ls_randoms/randoms_0-5_coords.fits')
	randras, randdecs = randomcat['RA'], randomcat['DEC']
	randlons, randlats = healpixhelper.equatorial_to_galactic(randras, randdecs)
	randdensity = healpixhelper.healpix_density_map(randlons, randlats, 2048)
	hp.write_map('masks/ls_density.fits', randdensity, overwrite=True)


def density_map(ras, decs, nside):
	randdensity = hp.read_map('masks/ls_density.fits')
	lons, lats = healpixhelper.equatorial_to_galactic(ras, decs)
	data_density = healpixhelper.healpix_density_map(lons, lats, nside)
	#density_ratio = data_density / randdensity * np.sum(randdensity) / np.sum(data_density)
	#density_ratio[np.isinf(density_ratio)] = np.nan
	density_ratio = np.array(data_density).astype(np.float32)
	mask = hp.read_map('masks/union.fits')
	density_ratio[np.where(np.logical_not(mask))] = np.nan

	return density_ratio


def density_contrast_map(ras, decs, nside):
	density = density_map(ras, decs, nside)

	meandensity = np.nanmean(density)

	contrast = (density - meandensity) / meandensity

	contrast[np.isnan(contrast) | np.logical_not(np.isfinite(contrast))] = hp.UNSEEN

	return contrast


def xcorr_of_bin(bootnum, dcmap, master=True):
	mask = hp.read_map('masks/union.fits')



	if bootnum > 0:
		lensmap = hp.read_map('lensing_maps/planck/noise/maps/%s.fits' % (bootnum - 1))
	else:
		lensmap = hp.read_map('lensing_maps/planck/smoothed_masked.fits')

	if master:
		lensfield = nmt.NmtField(mask, [lensmap])

		dcfield = nmt.NmtField(mask, [dcmap])
		wsp = nmt.NmtWorkspace()
		wsp.read_from('masks/workspace.fits')
		cl = wsp.decouple_cell(nmt.compute_coupled_cell(dcfield, lensfield))[0]

	else:
		cl = hp.anafast(dcmap, lensmap, lmax=2048)

	return cl


def xcorr_by_bin(pool, nboots, samplename, minscale, maxscale, nbins=10, master=True):
	boots = list(np.arange(nboots + 1))

	oldfiles = glob.glob('results/lensing_xcorrs/%s*' % samplename)
	for oldfile in oldfiles:
		os.remove(oldfile)

	scales = np.logspace(np.log10(minscale), np.log10(maxscale), nbins + 1)


	nside = hp.npix2nside(len(hp.read_map('masks/union.fits')))

	tab = Table.read('catalogs/derived/catwise_binned.fits')

	for j in range(int(np.max(tab['bin']))):
		binnedtab = tab[np.where(tab['bin'] == j + 1)]
		dcmap = density_contrast_map(binnedtab['RA'], binnedtab['DEC'], nside=nside)


		part_func = partial(xcorr_of_bin, dcmap=dcmap, master=master)
		cls = list(pool.map(part_func, boots))


		if master:
			binnedcls = cls
		else:
			lmodes = np.arange(len(cls[0])) + 1
			idxs = np.digitize(lmodes, scales)
			binnedcls = []
			for k in range(1, nbins+1):
				binnedcls.append(np.nanmean(cls[0][np.where(idxs == k)]))


		np.array(binnedcls).dump('results/lensing_xcorrs/%s_%s.npy' % (samplename, j + 1))

	pool.close()

def visualize_xcorr():
	planckmap = hp.read_map('lensing_maps/planck/smoothed_masked.fits')
	tab = Table.read('catalogs/derived/catwise_binned.fits')
	dcmap = density_contrast_map(tab['RA'], tab['DEC'], nside=hp.npix2nside(len(planckmap)))
	smoothdcmap = hp.smoothing(dcmap, fwhm=1*np.pi/180.)
	smoothplanckmap = hp.smoothing(planckmap, fwhm=0.1 * np.pi/180.)


	dcproj = hp.gnomview(smoothdcmap, rot=[0, 80], xsize=2000, return_projected_map=True)
	planckproj = hp.gnomview(smoothplanckmap, rot=[0, 80], xsize=2000, return_projected_map=True)

	plotting.visualize_xcorr(dcproj, planckproj)


if __name__ == "__main__":
	samplename = 'catwise'

	oldfiles = glob.glob('results/lensing_xcorrs/%s_*' % samplename)
	for file in oldfiles:
		os.remove(file)

	import schwimmbad
	lmin, lmax, n_l_bins = 50, 1000, 7
	write_master_workspace(lmin, lmax, n_l_bins)

	# use different executor based on command line arguments
	# lets code run either serially (python measure_clustering.py)
	# or with multiprocessing to do bootstraps in parallel (python measure_clustering.py --ncores=5)
	# or with MPI
	from argparse import ArgumentParser
	parser = ArgumentParser(description="Schwimmbad example.")


	group = parser.add_mutually_exclusive_group()
	group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
	group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
	args = parser.parse_args()

	pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
	if args.mpi:
		if not pool.is_master():
			pool.wait()
			sys.exit(0)

	#visualize_xcorr()
	xcorr_by_bin(pool, 10, samplename, lmin, lmax, n_l_bins, master=True)
	plotting.lensing_xcorrs(samplename)



