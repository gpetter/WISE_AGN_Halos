from astropy.io import fits
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
import healpy as hp
import glob
from source import interpolate_tools

"""def get_redshifts(binnedtab, seplimit=2.5*u.arcsec, sample='bootes'):
	from colossus.cosmology import cosmology
	cosmo = cosmology.setCosmology('planck18')


	ages_footprint = hp.read_map('catalogs/redshifts/AGES/ages_footprint.fits')
	# restrict table to only sources falling in the BOOTES field where AGES took spectra
	tab_in_bootes = binnedtab[np.where(ages_footprint[hp.ang2pix(hp.npix2nside(len(ages_footprint)), binnedtab['RA'],
	                                                         binnedtab['DEC'], lonlat=True)] > 0)]
	ages_tab = Table.read('catalogs/redshifts/AGES/redshifts.fits')
	tabcoords = SkyCoord(tab_in_bootes['RA'] * u.deg, tab_in_bootes['DEC'] * u.deg)
	zcoords = SkyCoord(ages_tab['RA'], ages_tab['DEC'])

	zidx, d2d, d3d = tabcoords.match_to_catalog_sky(zcoords)
	# find redshifts which are valid and matched to an AGN
	ages_tab = ages_tab[zidx]
	ages_tab = ages_tab[d2d < seplimit]
	ages_tab = ages_tab[np.where(ages_tab['z1'] > 0)]

	z_frac = len(ages_tab) / len(tab_in_bootes)
	# if AGES/BOOTES is > 90% complete
	if z_frac > 0.9:
		z_out = np.array(ages_tab['z1'])
	# if not, go to smaller, deeper cosmos field for redshifts
	else:
		print('AGES completeness is %s, reverting to COSMOS' % (z_frac))
		cosmosbinnedtab = binnedtab[np.where((binnedtab['RA'] > 149) & (binnedtab['RA'] < 153) &
		                        (binnedtab['DEC'] > -1) & (binnedtab['DEC'] < 4))]
		cosmosbinnedcoords = SkyCoord(cosmosbinnedtab['RA'] * u.deg, cosmosbinnedtab['DEC'] * u.deg)

		cosmosztab = Table.read('catalogs/derived/catwise_cosmos_zs.fits')
		cosmoszcoords = SkyCoord(cosmosztab['RA'] * u.deg, cosmosztab['DEC'] * u.deg)
		zidx, d2d, d3d = cosmosbinnedcoords.match_to_catalog_sky(cosmoszcoords)
		binztab = cosmosztab[zidx]
		binztab = binztab[d2d < 1*u.arcsec]
		#binztab = cosmosztab[np.where(cosmosztab['bin'] == binnedtab['bin'][0])]

		z_frac, z_out = float(len(np.where(binztab['phot_flag'] == 0)[0]))/len(binztab), np.array(binztab['Z'])

	z_out = z_out[np.where(np.isfinite(z_out))]

	cmdists = cosmo.comovingDistance(np.zeros(len(z_out)), z_out)
	print('Simon+07 says: Limber equation accurate to %s deg' % (260. / 60. * np.std(cmdists) / np.mean(cmdists)))

	return z_frac, z_out"""





def get_redshifts(bin, zthresh=4):
	from colossus.cosmology import cosmology
	cosmo = cosmology.setCosmology('planck18')

	ztab = Table.read('catalogs/derived/catwise_hasz.fits')
	ztab = ztab[np.where(ztab['bin'] == bin)]

	bootes_footprint = hp.read_map('masks/redshift_masks/bootes.fits')
	# restrict table to only sources falling in the BOOTES field where AGES took spectra
	tab_in_bootes = ztab[np.where(bootes_footprint[hp.ang2pix(hp.npix2nside(len(bootes_footprint)), ztab['RA'],
	                                                         ztab['DEC'], lonlat=True)] > 0)]

	cosmos_footprint = hp.read_map('masks/redshift_masks/cosmos.fits')
	tab_in_cosmos = ztab[np.where(cosmos_footprint[hp.ang2pix(hp.npix2nside(len(cosmos_footprint)), ztab['RA'],
	                                                               ztab['DEC'], lonlat=True)] > 0)]


	tottab = vstack((tab_in_cosmos, tab_in_bootes))

	zfrac = len(tottab[np.where((tottab['Z'] > 0) & (tottab['Z'] < zthresh))]) / len(tottab)
	tottab = tottab[np.where((tottab['Z'] > 0) & (tottab['Z'] < zthresh))]
	allzs = tottab['Z']
	photzs = tottab['Z'][np.where(tottab['phot_flag'] == 1)]
	speczs = tottab['Z'][np.where(tottab['phot_flag'] == 0)]

	return zfrac, allzs, speczs, photzs






def dndz_from_z_list(zs, bins):
	dndz, zbins = np.histogram(zs, bins=bins, density=True)
	zcenters = interpolate_tools.bin_centers(zbins, method='mean')
	dndz = dndz / np.trapz(dndz, x=zcenters)
	return (zcenters, dndz)


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
	# only match sources inside the cosmos footprint
	cosmosonly = False
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	if cosmosonly:
		# first cut down catalog to rough area of COSMOS to reduce healpix processing time
		tab = tab[np.where((tab['RA'] > 145) & (tab['RA'] < 155) & (tab['DEC'] > 0) & (tab['DEC'] < 4))]
		# read in cosmos healpix mask
		cosmosmask = hp.read_map('masks/cosmos_good.fits')
		# find sources inside the cosmos footprint
		maskedtab = tab[np.where(cosmosmask[hp.ang2pix(hp.npix2nside(len(cosmosmask)), tab['RA'], tab['DEC'],
		                                               lonlat=True)] > 0)]


	else:
		maskedtab = tab
	# set up empty array for redshifts
	maskedtab['Z'] = np.full(len(maskedtab), np.nan)
	# flag for photometric redshifts
	maskedtab['phot_flag'] = np.full(len(maskedtab), np.nan)
	maskedtab['z_source'] = np.full(len(maskedtab), np.nan)
	tabcoord = SkyCoord(maskedtab['RA'] * u.deg, maskedtab['DEC'] * u.deg)


	bestmatch = True
	# index tracking which catalog the redshift comes from
	k = 0
	# list of strings denoting paths to spectroscopic catalogs
	catlist = []

	# loop through redshift catalogs, matching and copying redshift to WISE AGN
	path = '../data/photozs/use/'
	folders = glob.glob(path + '*')
	idxs_w_redshifts = []
	for folder in folders:
		files = glob.glob(folder + '/*.fits')
		for file in files:
			spectab = Table.read(file)

			spectab = spectab[np.where((spectab['Zphot'] > 0.001) & (spectab['Zphot'] < 9.8))]
			speccoords = SkyCoord(spectab['RA'] * u.deg, spectab['DEC'] * u.deg)

			if bestmatch:
				idx, d2d, d3d = speccoords.match_to_catalog_sky(tabcoord)
				constraint = d2d < seplimit
				maskedtab['Z'][idx[constraint]] = spectab['Zphot'][constraint]
				# if copying photometric redshift, set flag
				maskedtab['phot_flag'][idx[constraint]] = 1
				# set integer key to the number corresponding to the redshift catalog
				maskedtab['z_source'][idx[constraint]] = k
				k += 1
				# append name of corresponding redshift catalog
				catlist.append(file)
				idxs_w_redshifts += list(idx[constraint])



			else:
				idx, specidx, d2d, d3d = speccoords.search_around_sky(tabcoord, seplimit)
				idxs_w_redshifts += list(idx)


	path = '../data/specsurveys/use/'
	folders = glob.glob(path + '*')
	idxs_w_redshifts = []
	for folder in folders:
		files = glob.glob(folder + '/*.fits')
		for file in files:
			print(file)
			spectab = Table.read(file)
			if cosmosonly:
				spectab = spectab[np.where((spectab['RA'] > 145) & (spectab['RA'] < 155) & (spectab['DEC'] > 0) &
				                           (spectab['DEC'] < 4))]

			spectab = spectab[np.where((spectab['Zspec'] > 0.001) & (spectab['Zspec'] < 9.8))]
			speccoords = SkyCoord(spectab['RA'] * u.deg, spectab['DEC'] * u.deg)

			if bestmatch:
				idx, d2d, d3d = speccoords.match_to_catalog_sky(tabcoord)
				constraint = d2d < seplimit

				maskedtab['Z'][idx[constraint]] = spectab['Zspec'][constraint]
				# if copying spec redshift, set flag
				maskedtab['phot_flag'][idx[constraint]] = 0
				maskedtab['z_source'][idx[constraint]] = k
				k += 1

				catlist.append(file)
				idxs_w_redshifts += list(idx[constraint])

			else:
				idx, specidx, d2d, d3d = speccoords.search_around_sky(tabcoord, seplimit)
				idxs_w_redshifts += list(idx)
	if cosmosonly:
		maskedtab.write('catalogs/derived/%s_cosmos_zs.fits' % samplename, format='fits', overwrite=True)

	with open('catalogs/derived/z_source_key.txt', 'w') as f:
		for j in range(len(catlist)):
			f.write('%s %s\n' % (j, catlist[j]))


	tab['hasz'] = np.zeros(len(tab))
	tab['hasz'][np.where(np.isfinite(tab['Z']))] = 1
	tab.write('catalogs/derived/%s_hasz.fits' % samplename, format='fits', overwrite=True)

