
import numpy as np
import scipy as sp

def log_interp1d(xx, yy, kind='linear', axis=-1):

	logx = np.log10(xx)
	logy = np.log10(yy)
	lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, fill_value=0., bounds_error=False, axis=axis)
	log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
	return log_interp


def bin_centers(binedges, method):
	if method == 'mean':
		return (binedges[1:] + binedges[:-1]) / 2
	elif method == 'geo_mean':
		return np.sqrt(binedges[1:] * binedges[:-1])
	else:
		return None