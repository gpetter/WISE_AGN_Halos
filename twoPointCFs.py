

from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf
import myCorrfunc
import numpy as np

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
def ang_cross_corr_from_coords(ras, decs, refras, refdecs, randras, randdecs, minscale, maxscale, refrandras=None,
                               refranddecs=None,  weights=None,
                               refweights=None, randweights=None, refrandweights=None, nthreads=1, nbins=10):

	if (refweights is not None) | (weights is not None) | (randweights is not None) | (refrandweights is not None):
		weighttype = 'pair_product'
		n_data = np.sum(weights)
		n_ref = np.sum(refweights)
		n_rands = np.sum(randweights)
		n_refrands = np.sum(refrandweights)

	else:
		weighttype = None
		n_data = len(ras)
		n_ref = len(refras)
		n_rands = len(randras)
		n_refrands = len(refrandras)



	# set up logarithimically spaced bins in units of degrees
	bins = np.logspace(minscale, maxscale, (nbins + 1))

	# count pairs between sample and control sample
	D1D2_counts = DDtheta_mocks(0, nthreads, bins, ras, decs, RA2=refras, DEC2=refdecs, weights1=weights,
	                          weights2=refweights, weight_type=weighttype)

	D2R1_counts = DDtheta_mocks(0, nthreads, bins, refras, refdecs, RA2=randras, DEC2=randdecs, weights1=refweights,
	                            weights2=randweights)


	if refrandras is not None:
		D1R2_counts = DDtheta_mocks(0, nthreads, bins, ras, decs, RA2=refrandras, DEC2=refranddecs, weights1=weights,
		                            weights2=refrandweights, weight_type=weighttype)
		R1R2_counts = DDtheta_mocks(0, nthreads, bins, randras, randdecs, RA2=refrandras, DEC2=refranddecs,
		                            weights1=randweights, weights2=refrandweights, weight_type=weighttype)

		wtheta = myCorrfunc.convert_counts_to_cf(ND1=n_data, ND2=n_ref, NR1=n_rands, NR2=n_refrands,
		                                         D1D2=D1D2_counts, D1R2=D1R2_counts, D2R1=D2R1_counts,
		                                         R1R2=R1R2_counts, estimator='LS')
	else:
		wtheta = myCorrfunc.convert_counts_to_cf(ND1=n_data, ND2=None, NR1=n_rands, NR2=None,
		                                         D1D2=D1D2_counts, D1R2=D2R1_counts, D2R1=D2R1_counts,
		                                         R1R2=D2R1_counts, estimator='Peebles')


	w_poisson_err = (1 + wtheta) / np.sqrt(D1D2_counts['npairs'])

	return wtheta, w_poisson_err

