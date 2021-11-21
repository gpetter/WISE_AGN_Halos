import numpy as np
import importlib
from source import bin_agn
from astropy.table import Table, vstack
import plotting
from astropy.coordinates import SkyCoord
import healpy as hp
import astropy.units as u
import masking
from source import coord_transforms
import redshift_dists
importlib.reload(redshift_dists)
importlib.reload(coord_transforms)
importlib.reload(masking)
importlib.reload(plotting)
importlib.reload(bin_agn)


def get_median_colors(tab):
	medcolors = []
	for j in range(int(np.max(tab['bin']))):
		binnedtab = tab[np.where(tab['bin'] == j + 1)]
		# medcolors.append(survival.km_median(binnedtab['color'], np.logical_not(binnedtab['detect'])))
		medcolors.append(np.median(binnedtab['color']))
	return medcolors


def cut_lowz(tab, bands):
	x_colors = tab['dered_mag_%s' % bands[0]] - tab['W2mag']
	y_colors = tab['dered_mag_%s' % bands[1]] - tab['W2mag']
	tab = tab[np.where((y_colors > (x_colors * 0.8 + 0.4)) | (y_colors > (20 - (3 * x_colors))))]
	return tab


def filter_table(soln, bitmask=False, custom_mask=False, planckmask=False, w2cut=8, w1cut=None, pmsncut=None,
                                                    sepcut=None, nbcut=None, lowzcut=False):
	cat = Table.read('catalogs/derived/catwise_r90_ls.fits')
	#cat = cat[np.where(cat['PSFDEPTH_R'] > 100)]



	if pmsncut is not None:
		cat = cat[np.where((cat['pmRA']/cat['e_pmRA'] < pmsncut) & (cat['pmDE']/cat['e_pmDE'] < pmsncut))]
	if nbcut is not None:
		cat = cat[np.where(cat['nb'] < nbcut)]
	if sepcut is not None:
		cat['dered_mag_r'][np.where(cat['sep'] > sepcut)] = np.nan


	newcat = Table()
	newcat['RA'], newcat['DEC'] = cat['RA'], cat['DEC']
	newcat['W1mag'], newcat['W2mag'] = cat['W1%s' % soln], cat['W2%s' % soln]
	newcat['e_W1mag'], newcat['e_W2mag'] = cat['e_W1%s' % soln], cat['e_W2%s' % soln]
	newcat['rmag'] = cat['dered_mag_r']
	#newcat['r_depth'] = -2.5*(np.log10(5/np.sqrt(cat['PSFDEPTH_R']))-9)
	newcat['detect'] = np.zeros(len(newcat))
	newcat['detect'][np.where((newcat['rmag'] > 0) & (np.isfinite(newcat['rmag'])))] = 1
	newcat['ab_flags'] = cat['ab_flags']



	newcat = newcat[np.where(newcat['W2mag'] > w2cut)]
	if w1cut is not None:
		newcat = newcat[np.where(newcat['W1mag'] < w1cut)]

	newcat = newcat[np.where(((newcat['W1mag'] - newcat['W2mag'] > 0.65) & (newcat['W2mag'] <= 13.86))
	            | (newcat['W1mag'] - newcat['W2mag'] > 0.65*np.exp(0.153*np.square(newcat['W2mag'] - 13.86))))]

	randcat = Table.read('catalogs/derived/ls_randoms_1_masked.fits')

	masking.total_mask(150, True, True, True)

	newcat = masking.mask_tab(newcat)
	randcat = masking.mask_tab(randcat)

	if lowzcut:
		newcat = cut_lowz(newcat, ['r', 'W2'])

	newcat.write('catalogs/derived/catwise_r90_filtered.fits', format='fits', overwrite=True)
	randcat.write('catalogs/derived/ls_randoms_1_filtered.fits', format='fits', overwrite=True)



# match to deep HSC data to get color distribution of optically undetected sources in main sample
def undetected_dist(tab):
	coords = SkyCoord(tab['RA'] * u.deg, tab['DEC'] * u.deg)

	hsctab = Table.read('catalogs/raw/r90_pm_hsc_dud.fits')
	hsctab = hsctab[np.where(((hsctab['WISE_RA']) > 146) & (hsctab['WISE_RA'] < 153))]
	hsccoords = SkyCoord(hsctab['WISE_RA'] * u.deg, hsctab['WISE_DEC'] * u.deg)

	idx, hscidx, d2d, d3d = hsccoords.search_around_sky(coords, 1*u.arcsec)

	rmags = hsctab['rmag'][hscidx]
	w2mags = tab['W2mag'][idx]

	return rmags - w2mags - 3.313





def bin_sample(samplename, mode, nbins=3):

	cat = Table.read('catalogs/derived/catwise_r90_filtered.fits')

	detectidx = np.where(cat['detect'].astype(np.bool))
	nondetectidx = np.where(cat['detect'] == 0)

	if mode == 'color':
		cat['color'] = np.empty(len(cat))
		cat['color'][detectidx] = cat['rmag'][detectidx] - cat['W2mag'][detectidx] - 3.313
		#cat['color'][nondetectidx] = cat['r_depth'][nondetectidx] - cat['W2mag'][nondetectidx] - 3.313
		cat['color'][nondetectidx] = 99.

		indicesbybin = bin_agn.bin_by_color(cat['color'], nbins)
	cat['bin'] = np.zeros(len(cat))

	for j in range(len(indicesbybin)):
		cat['bin'][indicesbybin[j]] = j+1

	nondet_colors = undetected_dist(cat[nondetectidx])

	plotting.plot_color_dists(cat, nondet_colors)
	plotting.plot_assef_cut(cat)
	cat.write('catalogs/derived/%s_binned.fits' % samplename, format='fits', overwrite=True)


def long_wavelength_properties(samplename, bands):
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	medcolors = get_median_colors(tab)

	if 'w3w4' in bands:
		det_fracs_3, det_fracs_4 = [], []
		for j in range(int(np.max(tab['bin']))):
			binnedtab = tab[np.where(tab['bin'] == j+1)]
			det_fracs_3.append(len(binnedtab[np.where(binnedtab['e_W3mag'] > 0)]) / len(binnedtab))
			det_fracs_4.append(len(binnedtab[np.where(binnedtab['e_W4mag'] > 0)]) / len(binnedtab))



		plotting.w3_w4_det_fraction(medcolors, det_fracs_3, det_fracs_4)

		plotting.w3_w4_dists(tab)
	if 'mips_sep' in bands:
		elon, elat = coord_transforms.sky_transform(tab['RA'], tab['DEC'], ['C', 'E'])
		septab = tab[np.where((elon < 55.) & (elon > 36) & (elat < -73.25) & (elat > -74.8))]

		det_fracs, mips_mags = [], []
		mipstab = Table.read('catalogs/SEP/sep_mips24.fits')
		mipscoords = SkyCoord(mipstab['RA'] * u.deg, mipstab['DEC'] * u.deg)

		for j in range(int(np.max(septab['bin']))):
			binnedtab = septab[np.where(septab['bin'] == j + 1)]
			tabcoords = SkyCoord(binnedtab['RA'] * u.deg, binnedtab['DEC'] * u.deg)

			idx, d2d, d3d = tabcoords.match_to_catalog_sky(mipscoords)
			limit = (d2d < 2*u.arcsec)
			det_fracs.append(len(binnedtab[limit])/len(binnedtab))
			mips_mags.append(-2.5*np.log10(mipstab['flux'][idx[limit]]/(3631. * 1e6)))
		plotting.mips_fraction(medcolors, det_fracs)
		plotting.mips_dists(mips_mags)






def redshifts(samplename, z_sample='AGES'):
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)

	z_fractions, zs_by_bin = [], []
	if z_sample == 'AGES':
		ages_footprint = hp.read_map('catalogs/redshifts/AGES/ages_footprint.fits')
		tab_pix = hp.ang2pix(hp.npix2nside(len(ages_footprint)), tab['RA'], tab['DEC'], lonlat=True)
		tab = tab[np.where(ages_footprint[tab_pix] > 0)]

		z_tab = Table.read('catalogs/redshifts/AGES/redshifts.fits')
		with_specz = z_tab[np.where(z_tab['z1'] > 0)]
		zcoords = SkyCoord(with_specz['RA'], with_specz['DEC'])

		for j in range(int(np.max(tab['bin']))):
			binnedtab = tab[np.where(tab['bin'] == j+1)]

			binnedcoords = SkyCoord(binnedtab['RA'] * u.deg, binnedtab['DEC'] * u.deg)

			zidx, binnedidx, d2d, d3d = binnedcoords.search_around_sky(zcoords, 3 * u.arcsec)

			z_fractions.append(len(zidx)/len(binnedtab))
			zs_by_bin.append(with_specz[zidx]['z1'])

	elif z_sample == 'cosmos_laigle':

		in_footprint = np.where((tab['RA'] < 150.86) & (tab['RA'] > 149.388) & (tab['DEC'] < 2.938) & (tab['DEC'] >
		                                                                                               1.471))
		maxw1mag, maxw2mag = np.max(tab['W1mag']) + .75 + 2.699, np.max(tab['W2mag']) + .75 + 3.313
		z_tab = Table.read('catalogs/redshifts/cosmos_2015/cosmos_photozs.fits')
		# reduce COSMOS table to just sources bright enough in IRAC 1+2 to be detecetd with WISE
		z_tab = z_tab[np.where((z_tab['SPLASH_1_MAG'] < maxw1mag) & (z_tab['SPLASH_1_MAG'] > 0) &
		                       (z_tab['SPLASH_2_MAG'] < maxw2mag) & (z_tab['SPLASH_2_MAG'] > 0))]
		tab = tab[in_footprint]
		zcoords = SkyCoord(z_tab['RA'], z_tab['DEC'])





		for j in range(int(np.max(tab['bin']))):
			binnedtab = tab[np.where(tab['bin'] == j+1)]

			binnedcoords = SkyCoord(binnedtab['RA'] * u.deg, binnedtab['DEC'] * u.deg)

			zidx, binnedidx, d2d, d3d = binnedcoords.search_around_sky(zcoords, 2. * u.arcsec)

			z_fractions.append(len(zidx)/len(binnedtab))
			zs_by_bin.append(z_tab[zidx]['ZQ'])

	elif z_sample == 'cosmos_comp':
		z_tab = Table.read('catalogs/redshifts/cosmos_comp/catwise_r90_cosmos_zs.fits')
		tabcoords = SkyCoord(tab['RA'] * u.deg, tab['DEC'] * u.deg)
		zcoords = SkyCoord(z_tab['RA'], z_tab['DEC'])


		zidx, idx, d2d, d3d = tabcoords.search_around_sky(zcoords, 2. * u.arcsec)

		cosmos_tab = tab[idx]

		for j in range(int(np.max(tab['bin']))):
			binnedtab = cosmos_tab[np.where(cosmos_tab['bin'] == j+1)]

			binnedcoords = SkyCoord(binnedtab['RA'] * u.deg, binnedtab['DEC'] * u.deg)

			zidx, binnedidx, d2d, d3d = binnedcoords.search_around_sky(zcoords, 2. * u.arcsec)

			z_fractions.append(len(zidx)/len(binnedtab))
			zs_by_bin.append(z_tab[zidx]['z_best'])

	elif z_sample == 'lofar_bootes':
		for j in range(int(np.max(tab['bin']))):
			binnedtab = tab[np.where(tab['bin'] == j+1)]
			zfrac, zsinbin = redshift_dists.get_redshifts(binnedtab, 2 * u.arcsec)
			z_fractions.append(zfrac)
			zs_by_bin.append(zsinbin)

	elif z_sample == 'cosmos':
		for j in range(int(np.max(tab['bin']))):
			binnedtab = tab[np.where(tab['bin'] == j+1)]
			zfrac, zsinbin = redshift_dists.get_redshifts(binnedtab, sample='cosmos')
			z_fractions.append(zfrac)
			zs_by_bin.append(zsinbin)





	plotting.fraction_with_redshifts(get_median_colors(tab), z_fractions)
	plotting.redshift_dists(zs_by_bin)











