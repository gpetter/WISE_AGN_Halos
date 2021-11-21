import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
 

def bin_by_color(colors, nbins):

	# make empty list of lists
	binidxs = []
	for k in range(nbins):
		binidxs.append([])


	colorbins = pd.qcut(colors, nbins, retbins=True)[1]
	colorbins[len(colorbins) - 1] = colorbins[len(colorbins) - 1] + 0.001
	for j in range(nbins):
		binidxs[j] += list(
			np.where((colors < colorbins[j + 1]) & (colors >= colorbins[j]))[0])


	return binidxs



def bin_by_radio(rls):
	binidxs = [[], []]
	binidxs[0] = list(np.where(rls == 0))
	binidxs[1] = list(np.where(rls == 1))
	return binidxs

def bin_by_lum(lums, nlumbins):
	lumbins = pd.qcut(lums, nlumbins, retbins=True)[1]
	whichbins = np.digitize(lums, lumbins)
	indexarr = [list(np.where(whichbins == j + 1)) for j in range(nlumbins)]
	return indexarr



