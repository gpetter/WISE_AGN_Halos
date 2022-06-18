from astropy.table import Table
import numpy as np
from corrfunc_helper import twoPointCFs







# loop through each bin, map cores to bootstrap iterations or individual patches on sky
def clustering_by_bin(nboots, samplename, minscale, maxscale, nscalebins=10, nsamplebins=2, oversample=3, nthreads=1):

	# write angular scales to file
	scales = np.logspace(minscale, maxscale, nscalebins+1)
	scales.dump('results/clustering/scales.npy')

	# for each color bin
	for j in range(nsamplebins):
		binnedtab = Table.read('catalogs/derived/%s_binned_%s.fits' % (samplename, j+1))
		randtab = Table.read('catalogs/derived/%s_randoms_%s.fits' % (samplename, (j+1)))
		w, poissonerr, bootstraperr, covar = \
						twoPointCFs.autocorr_from_coords(coords=(binnedtab['RA'], binnedtab['DEC'], None),
						randcoords=(randtab['RA'], randtab['DEC'], None), scales=scales,
						nthreads=nthreads, nbootstrap=nboots, oversample=oversample)

		wthetas = np.array([w, poissonerr])
		wthetas.dump('results/clustering/%s_%s.npy' % (samplename, j + 1))





import glob
import os
samplename = 'catwise'
oldfiles = glob.glob('results/clustering/%s_*' % samplename)
for file in oldfiles:
	os.remove(file)
if len(glob.glob('results/clustering/scales.npy')) > 0:
	os.remove('results/clustering/scales.npy')



clustering_by_bin(500, samplename, -2.55, -0.5, nscalebins=15, oversample=1, nthreads=24)
#plotting.plot_ang_autocorrs(samplename)
#plotting.cf_err_comparison(samplename)


