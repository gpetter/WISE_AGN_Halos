from astropy.io import fits
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import healpy as hp
import glob


def get_redshifts(tab, seplimit=2*u.arcsec, sample='bootes'):
	if sample == 'bootes':
		tab = tab[np.where((tab['DEC'] < 35.85) & (tab['RA'] > 216.15) & (tab['RA'] < 219.7) &
		                (((tab['DEC'] > 32.3) & (tab['RA'] < 218.25)) | ((tab['DEC'] > 32.9) &
		                (tab['RA'] >= 218.25) & (tab['RA'] < 218.95)) | ((tab['DEC'] >
		                33.5) & (tab['RA'] > 218.95))))]
		ztab = Table.read('catalogs/redshifts/bootes_lofar/LOFAR_Bootes_photo_zs.fits')
		coords = SkyCoord(tab['RA'] * u.deg, tab['DEC'] * u.deg)
		zcoords = SkyCoord(ztab['RA'] * u.deg, ztab['DEC'] * u.deg)

		idx, d2d, d3d = coords.match_to_catalog_sky(zcoords)
		ztab = ztab[idx]
		sep_constraint = d2d < seplimit
		ztab = ztab[sep_constraint]
		ztab = ztab[np.where(ztab['zbest'] > 0)]

		return len(ztab)/len(tab), np.array(ztab['zbest'])
	elif sample == 'cosmos':
		tabbin = tab['bin'][0]
		ztab = Table.read('catalogs/derived/catwise_r90_cosmos_zs.fits')
		binztab = ztab[np.where(ztab['bin'] == tabbin)]

		return 1, np.array(binztab['Z'])






def redshift_dist(zs, nbins, bin_avgs=True):

	# bin up redshift distribution of sample to integrate kappa over
	hist = np.histogram(zs, nbins, density=True)
	zbins = hist[1]
	if bin_avgs:
		dz = zbins[1] - zbins[0]
		# chop off last entry which is a rightmost bound of the z distribution, find center of bins by adding dz/2
		zbins = np.resize(zbins, zbins.size - 1) + dz / 2

	dndz = hist[0]

	return zbins, dndz


def redshift_overlap(zs1, zs2, bin_avgs=True):
	# find mininum and maximum redshift across two samples
	minzs1, maxzs1 = np.min(zs1), np.max(zs1)
	minzs2, maxzs2 = np.min(zs2), np.max(zs2)
	totmin, totmax = np.min([minzs1, minzs2]) - 0.0001, np.max([maxzs1, maxzs2]) + 0.0001

	# get z distributions for both samples across same grid
	hist1 = np.histogram(zs1, 200, density=True, range=[totmin, totmax])
	hist2 = np.histogram(zs2, 200, density=True, range=[totmin, totmax])

	# z values in each bin
	zbins = hist1[1]
	if bin_avgs:
		dz = zbins[1] - zbins[0]
		# chop off last entry which is a rightmost bound of the z distribution, find center of bins by adding dz/2
		zbins = np.resize(zbins, zbins.size - 1) + dz / 2

	# redshift distribution overlap is sqrt of product of two distributions
	dndz = np.sqrt(hist1[0] * hist2[0])

	return zbins, dndz



def match_to_spec_surveys(samplename, seplimit):
	cosmosonly = False
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	if cosmosonly:
		tab = tab[np.where((tab['RA'] > 145) & (tab['RA'] < 155) & (tab['DEC'] > 0) & (tab['DEC'] < 4))]
		cosmosmask = hp.read_map('masks/cosmos_good.fits')
		maskedtab = tab[np.where(cosmosmask[hp.ang2pix(hp.npix2nside(len(cosmosmask)), tab['RA'], tab['DEC'],
		                                               lonlat=True)] > 0)]
		maskedtab['Z'] = np.full(len(maskedtab), np.nan)
	else:
		maskedtab = tab
	tabcoord = SkyCoord(maskedtab['RA'] * u.deg, maskedtab['DEC'] * u.deg)


	bestmatch = False

	path = '../data/photozs/use/'
	folders = glob.glob(path + '*')
	idxs_w_redshifts = []
	for folder in folders:
		files = glob.glob(folder + '/*.fits')
		for file in files:
			spectab = Table.read(file)

			if cosmosonly:
				spectab = spectab[np.where((spectab['Zphot'] > 0.001) & (spectab['Zphot'] < 9.8))]
			speccoords = SkyCoord(spectab['RA'] * u.deg, spectab['DEC'] * u.deg)

			if bestmatch:
				idx, d2d, d3d = speccoords.match_to_catalog_sky(tabcoord)
				constraint = d2d < seplimit
				if cosmosonly:
					maskedtab['Z'][idx[constraint]] = spectab['Zphot'][constraint]
			else:
				idx, specidx, d2d, d3d = speccoords.search_around_sky(tabcoord, seplimit)
				idxs_w_redshifts += list(idx)


	path = '../data/specsurveys/use/'
	folders = glob.glob(path + '*')
	idxs_w_redshifts = []
	for folder in folders:
		files = glob.glob(folder + '/*.fits')
		for file in files:
			spectab = Table.read(file)
			if cosmosonly:
				spectab = spectab[np.where((spectab['RA'] > 145) & (spectab['RA'] < 155) & (spectab['DEC'] > 0) &
				                           (spectab['DEC'] < 4))]

			spectab = spectab[np.where((spectab['Zspec'] > 0.001) & (spectab['Zspec'] < 9.8))]
			speccoords = SkyCoord(spectab['RA'] * u.deg, spectab['DEC'] * u.deg)

			if bestmatch:
				idx, d2d, d3d = speccoords.match_to_catalog_sky(tabcoord)
				constraint = d2d < seplimit
				if cosmosonly:
					maskedtab['Z'][idx[constraint]] = spectab['Zspec'][constraint]
			else:
				idx, specidx, d2d, d3d = speccoords.search_around_sky(tabcoord, seplimit)
				idxs_w_redshifts += list(idx)
	if cosmosonly:
		maskedtab.write('catalogs/derived/%s_cosmos_zs.fits' % samplename, format='fits', overwrite=True)

	tab['hasz'] = np.zeros(len(tab))
	tab['hasz'][idxs_w_redshifts] = 1
	tab.write('catalogs/derived/%s_hasz.fits' % samplename, format='fits', overwrite=True)


#match_to_spec_surveys('catwise_r90', 2.5 * u.arcsec)