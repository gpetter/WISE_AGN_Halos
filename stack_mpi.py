##########
# stack_mpi.py
# author: Grayson Petter
# code for performing a stack of many cmb projections in parallel using MPI for deployment on a cluster computer
##########

import numpy as np
import healpy as hp
import time
from functools import partial
from astropy.coordinates import SkyCoord
from math import ceil
from astropy import units as u
import convergence_map
import stacking
import importlib
from astropy.table import Table







# stacks many projections by breaking up list into chunks and processing each chunk in parallel
def stack_mp(stackmap, ras, decs, pool, weighting=None, prob_weights=None, nstack=None, outname=None, imsize=240, chunksize=500, reso=1.5):
	if weighting is None:
		weighting = np.ones(len(ras))
	if nstack is None:
		nstack = len(ras)

	# the number of chunks is the number of stacks divided by the chunk size rounded up to the nearest integer
	nchunks = ceil(nstack/chunksize)

	lons, lats = stacking.equatorial_to_galactic(ras, decs)

	starttime = time.time()

	# fill in all arguments to stack_chunk function but the index,
	# Pool.map() requires function to only take one paramter
	stack_chunk_partial = partial(stacking.stack_chunk, chunksize, nstack, lons, lats, stackmap, weighting, prob_weights, imsize, reso)
	# do the stacking in chunks, map the stacks to different cores for parallel processing
	# use mpi processing for cluster or multiprocessing for personal computer

	chunksum, chunkweightsum = zip(*pool.map(stack_chunk_partial, range(nchunks)))

	totsum = np.sum(chunksum, axis=0)
	weightsum = np.sum(chunkweightsum, axis=0)
	finalstack = totsum / weightsum

	finalstack.dump('%s.npy' % outname)
	print(time.time()-starttime)


if __name__ == "__main__":

	import schwimmbad

	# use different executor based on command line arguments
	# lets code run either serially (python stack_mpi.py)
	# or with multiprocessing to do bootstraps in parallel (python stack_mpi.py --ncores=5)
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

	imsize = 100
	reso = 2.


	planck_map = hp.read_map('lensing_maps/planck/smoothed_masked.fits', dtype=np.single)

	outname = 'lens_stacks/catwise_stack'

	cat = Table.read('catalogs/derived/catwise_binned.fits')


	for j in range(int(np.max(cat['bin']))):
		colorcat = cat[np.where(cat['bin'] == (j+1))]
		stack_mp(planck_map, colorcat['RA'], colorcat['DEC'], pool, weighting=np.ones(len(colorcat)),
		         prob_weights=np.ones(len(colorcat)), outname=(outname + '%s' % j), imsize=imsize, reso=reso)



	pool.close()
