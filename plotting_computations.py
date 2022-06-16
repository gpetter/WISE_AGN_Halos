import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib.ticker import AutoMinorLocator


def two_scales(ax1, transforms, newlabel, analytic, axis='y', labelsize=20):
	if transforms is not None:
		if analytic:
			forward, inverse = transforms[0], transforms[1]
		else:
			def forward(x):
				# return np.interp(x, transforms[0], transforms[1])
				# return 10**x
				spl = InterpolatedUnivariateSpline(transforms[0], transforms[1])
				return spl(x)

			def inverse(x):
				# return np.interp(x, transforms[1], transforms[0])
				spl = InterpolatedUnivariateSpline(transforms[1], transforms[0])
				return spl(x)

		if axis == 'y':
			secax = ax1.secondary_yaxis('right', functions=(forward, inverse))
		else:
			secax = ax1.secondary_xaxis('top', functions=(forward, inverse))
		secax.yaxis.set_minor_locator(AutoMinorLocator())

		secax.set_ylabel(newlabel, fontsize=20, labelpad=15)
		secax.tick_params(axis=axis, which='major', labelsize=labelsize)
		secax.tick_params('both', labelsize=labelsize, which='major', length=8)
		secax.tick_params('both', which='minor', length=3)
		# secax.set_ylim(forward(ylim[0]), forward(ylim[1]))


def freedman_diaconis(vals):
	from scipy.stats import iqr
	from math import ceil
	statiqr = iqr(vals)
	h = 2 * statiqr * len(vals) ** (-1/3.)
	nbins = (np.max(vals) - np.min(vals)) / h

	return ceil(nbins)

