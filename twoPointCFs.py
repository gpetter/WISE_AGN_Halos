

from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf
import myCorrfunc
import numpy as np
import importlib
importlib.reload(myCorrfunc)

def data_counts(scales, ras, decs, weights, fulldict=True):
	dd = DDtheta_mocks(1, 1, scales, ras, decs, weights1=weights)
	if fulldict:
		return dd
	else:
		if np.max(dd['weightavg']) > 0:
			return dd['npairs'] * dd['weightavg']
		else:
			return dd['npairs']

def data_random_counts(scales, ras, decs, randras, randdecs, weights, randweights, fulldict=True):
	dr = DDtheta_mocks(0, 1, scales, ras, decs, RA2=randras, DEC2=randdecs, weights1=weights,
	              weights2=randweights)
	if fulldict:
		return dr
	else:
		if np.max(dr['weightavg']) > 0:
			return dr['npairs'] * dr['weightavg']
		else:
			return dr['npairs']

def random_counts(scales, randras, randdecs, randweights, fulldict=True):
	rr = DDtheta_mocks(1, 1, scales, randras, randdecs, weights1=randweights)
	if fulldict:
		return rr
	else:
		if np.max(rr['weightavg']) > 0:
			return rr['npairs'] * rr['weightavg']
		else:
			return rr['npairs']

def angular_corr_from_coords(ras, decs, randras, randdecs, scales, weights=None, randweights=None,
                    nthreads=1, randcounts=None):

	# autocorrelation of catalog
	DD_counts = DDtheta_mocks(1, nthreads, scales, ras, decs, weights1=weights)

	# cross correlation between data and random catalog
	DR_counts = DDtheta_mocks(0, nthreads, scales, ras, decs, RA2=randras, DEC2=randdecs, weights1=weights,
	                          weights2=randweights)

	if randcounts is None:
		# autocorrelation of random points
		RR_counts = DDtheta_mocks(1, nthreads, scales, randras, randdecs, weights1=randweights)
	else:
		RR_counts = randcounts

	wtheta = convert_3d_counts_to_cf(len(ras), len(ras), len(randras), len(randras), DD_counts, DR_counts,
	                                 DR_counts, RR_counts)

	poisson_err = np.sqrt(2 * np.square(1 + wtheta) / DD_counts['npairs'])

	return wtheta, poisson_err


# angular cross correlation
def ang_cross_corr_from_coords(ras, decs, refras, refdecs, randras, randdecs, minscale, maxscale, weights=None,
                               refweights=None, randweights=None, nthreads=1, nbins=10):
	# set up logarithimically spaced bins in units of degrees
	bins = np.logspace(minscale, maxscale, (nbins + 1))

	# count pairs between sample and control sample
	DD_counts = DDtheta_mocks(0, nthreads, bins, ras, decs, RA2=refras, DEC2=refdecs, weights1=weights,
	                          weights2=refweights)

	# extract number counts
	dd = []
	for j in range(nbins):
		dd.append(DD_counts[j][3])


	# cross correlation between sample and random catalog
	DR_counts = np.array(DDtheta_mocks(0, nthreads, bins, ras, decs, RA2=randras, DEC2=randdecs, weights1=weights,
							  weights2=randweights))
	dr = []
	for j in range(nbins):
		dr.append(DR_counts[j][3])



	wtheta = np.array(dd)/np.array(dr) * (float(len(randras))) / float(len(refras)) - 1

	return wtheta

