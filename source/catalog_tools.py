from astropy.table import Table
from math import ceil

def prep_cat_for_opt_match():
	cat = Table.read('catalogs/catwise_r75pm.fits')
	cat = cat['RA', 'DEC']
	chunksize = 500000

	#for j in range(ceil(len(cat) / chunksize)):
	#	tab_j = cat[j*chunksize:]
	cat.write('catalogs/csv/catwise_r75pm.csv', format='csv', overwrite=True)