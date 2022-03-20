import numpy as np

def donley_selection_flux(f1, f2, f3, f4):
	x = np.log10(f3/f1)
	y = np.log10(f4/f2)
	idxs = np.where((x >= 0.08) & (y >= 0.15) &
	                (y >= (1.21 * x) - 0.27) &
	                (y <= (1.21 * x) + 0.27) &
	                (f2 > f1) & (f3 > f2) & (f4 > f3))
	return idxs