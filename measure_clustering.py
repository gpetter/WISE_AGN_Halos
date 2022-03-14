from astropy.table import Table
import numpy as np
import importlib
import healpy as hp

import myCorrfunc
import twoPointCFs
import plotting
import sys
from functools import partial
import resampling

importlib.reload(resampling)
importlib.reload(twoPointCFs)




# perform one clustering measurement inside a given bin for a particular bootstrap iteration
def clustering_of_bin(bootnum, ras, decs, scales):

	randcat = Table.read('catalogs/derived/ls_randoms_1_filtered.fits')
	randcat = randcat[:15*len(ras)]
	randras, randdecs = randcat['RA'], randcat['DEC']
	if bootnum > 0:
		idxs, foo, randidxs = resampling.bootstrap_sky_bins(ras, decs, ras, decs, randcat['RA'], randcat['DEC'], 10)
		ras, decs, randras, randdecs = ras[idxs], decs[idxs], randras[randidxs], randdecs[randidxs]


	wtheta, foo = twoPointCFs.angular_corr_from_coords(ras, decs, randras, randdecs, scales)

	return wtheta

def clustering_of_patch(patchval, patchmap, nside, ras, decs, randras, randdecs, weights, randweights, scales):

	idxs_in_patch = np.where(patchmap[hp.ang2pix(nside, ras, decs, lonlat=True)] == patchval)
	ras, decs = ras[idxs_in_patch], decs[idxs_in_patch]


	randidxs_in_patch = np.where(patchmap[hp.ang2pix(nside, randras, randdecs, lonlat=True)] == patchval)
	randras, randdecs = randras[randidxs_in_patch], randdecs[randidxs_in_patch]

	n_data, n_rands = np.sum(weights), np.sum(randweights)

	ddcounts = twoPointCFs.data_counts(scales, ras, decs, weights, fulldict=False)
	drcounts = twoPointCFs.data_random_counts(scales, ras, decs, randras, randdecs, weights, randweights,
	                                          fulldict=False)
	rrcounts = twoPointCFs.random_counts(scales, randras, randdecs, randweights, fulldict=False)

	return [ddcounts, drcounts, rrcounts, n_data, n_rands]





# loop through each bin, map cores to bootstrap iterations or individual patches on sky
def clustering_by_bin(pool, nboots, samplename, minscale, maxscale, nbins=10, oversample=3, resample_patches=True):
	boots = list(np.arange(nboots + 1))

	# write angular scales to file
	scales = np.logspace(minscale, maxscale, nbins+1)
	scales.dump('results/clustering/scales.npy')
	# read in data
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	# for each color bin
	for j in range(int(np.max(tab['bin']))):
		binnedtab = tab[np.where(tab['bin'] == j+1)]

		npatches = 30
		# count pairs in each sky patch, then bootstrap/jackknife at the end
		if resample_patches:
			# bin the footprint into patches with equal area
			patchmap = resampling.bin_on_sky(binnedtab['RA'], binnedtab['DEC'], njackknives=npatches)
			# get IDs for each patch
			unique_patchvals = np.unique(patchmap)
			# remove masked patches
			unique_patchvals = unique_patchvals[np.where(unique_patchvals > -1e30)]

			randcat = Table.read('catalogs/derived/%s_randoms_%s.fits' % (samplename, j+1))
			randcat = randcat[:15*len(tab)]

			wtheta, poissonerr = twoPointCFs.angular_corr_from_coords(binnedtab['RA'], binnedtab['DEC'],
			                            randcat['RA'], randcat['DEC'], scales)
			wthetas = [wtheta]
			wthetas.append(poissonerr)



			part_func = partial(clustering_of_patch, nside=hp.npix2nside(len(patchmap)), patchmap=patchmap,
				ras=binnedtab['RA'], decs=binnedtab['DEC'], randras=randcat['RA'], randdecs=randcat['DEC'],
				weights=binnedtab['weight'], randweights=randcat['weight'], scales=scales)
			# map cores to different patches, count pairs within
			counts = list(pool.map(part_func, unique_patchvals))
			counts = np.array(counts)

			# separate out DD, DR, RR counts from returned array
			dd_counts, dr_counts, rr_counts = counts[:, 0], counts[:, 1], counts[:, 2]
			ndata, nrands = counts[:, 3], counts[:, 4]

			# for each bootstrap, randomly select patches (oversampled by factor of 3) and sum counts in those patches
			for k in range(nboots):
				boot_patches = np.random.choice(np.arange(npatches), oversample * npatches)
				bootndata, bootnrands = ndata[boot_patches], nrands[boot_patches]
				totdata, totrands = np.sum(bootndata) / np.float(oversample), np.sum(bootnrands) / np.float(oversample)
				boot_ddcounts, boot_drcounts, bootrrcounts = dd_counts[boot_patches], dr_counts[boot_patches], \
				                                             rr_counts[boot_patches]
				totddcounts, totdrcounts, totrrcounts = np.sum(boot_ddcounts, axis=0), np.sum(boot_drcounts, axis=0),\
				                                        np.sum(bootrrcounts, axis=0)

				wthetas.append(myCorrfunc.convert_raw_counts_to_cf(totdata, totdata, totrands,
				            totrands, totddcounts, totdrcounts, totdrcounts, totrrcounts))



		# otherwise, rerun whole sky pair counting for each resampling
		else:
			part_func = partial(clustering_of_bin, ras=binnedtab['RA'], decs=binnedtab['DEC'], scales=scales)
			wthetas = list(pool.map(part_func, boots))
		wthetas = np.array(wthetas)
		wthetas.dump('results/clustering/%s_%s.npy' % (samplename, j + 1))



	pool.close()



if __name__ == "__main__":
	import schwimmbad
	import glob
	import os
	samplename = 'catwise'
	oldfiles = glob.glob('results/clustering/%s_*' % samplename)
	for file in oldfiles:
		os.remove(file)
	if len(glob.glob('results/clustering/scales.npy')) > 0:
		os.remove('results/clustering/scales.npy')

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

	clustering_by_bin(pool, 500, samplename, -2.75, -0.25, nbins=15, oversample=1)
	plotting.plot_ang_autocorrs(samplename)
	plotting.cf_err_comparison(samplename)


