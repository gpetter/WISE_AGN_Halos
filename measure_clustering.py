from astropy.table import Table
import numpy as np
import importlib
import random_catalogs
import twoPointCFs
import plotting
import sys
from functools import partial
import resampling
importlib.reload(resampling)
importlib.reload(twoPointCFs)
importlib.reload(random_catalogs)
importlib.reload(plotting)


# perform one clustering measurement inside a given bin for a particular bootstrap iteration
def clustering_of_bin(bootnum, ras, decs, scales):

	randcat = Table.read('catalogs/derived/ls_randoms_1_filtered.fits')
	randcat = randcat[:7*len(ras)]
	randras, randdecs = randcat['RA'], randcat['DEC']
	if bootnum > 0:
		idxs, foo, randidxs = resampling.bootstrap_sky_bins(ras, decs, ras, decs, randcat['RA'], randcat['DEC'], 10)
		ras, decs, randras, randdecs = ras[idxs], decs[idxs], randras[randidxs], randdecs[randidxs]


	wtheta = twoPointCFs.angular_corr_from_coords(ras, decs, randras, randdecs, scales)

	return wtheta


# loop through each bin, map cores to bootstrap iterations
def clustering_by_bin(pool, nboots, samplename, minscale, maxscale, nbins=10):
	boots = list(np.arange(nboots + 1))

	scales = np.logspace(minscale, maxscale, nbins+1)
	scales.dump('clustering/scales.npy')
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	for j in range(int(np.max(tab['bin']))):
		binnedtab = tab[np.where(tab['bin'] == j+1)]

		part_func = partial(clustering_of_bin, ras=binnedtab['RA'], decs=binnedtab['DEC'], scales=scales)
		wthetas = list(pool.map(part_func, boots))
		np.array(wthetas).dump('clustering/%s_%s.npy' % (samplename, j+1))
	pool.close()


	#plotting.plot_ang_autocorrs(scales[:len(scales)-1], wthetas)


if __name__ == "__main__":
	import schwimmbad

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

	clustering_by_bin(pool, 10, 'catwise_r90', -3., 0., 15)

