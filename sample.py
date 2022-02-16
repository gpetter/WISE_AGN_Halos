import numpy as np
from source import bin_agn
from astropy.table import Table, vstack
import plotting
from astropy.coordinates import SkyCoord
import healpy as hp
import astropy.units as u
import masking
from source import coord_transforms
import redshift_dists

vega_to_ab = {'g': 0, 'r': 0, 'z': 0, 'W1': 2.699, 'W2': 3.313}


def get_median_colors(tab):
	medcolors = []
	for j in range(int(np.max(tab['bin']))):
		binnedtab = tab[np.where(tab['bin'] == j + 1)]
		# medcolors.append(survival.km_median(binnedtab['color'], np.logical_not(binnedtab['detect'])))
		medcolors.append(np.median(binnedtab['color']))
	return medcolors


def cut_lowz(tab, bands):
	if bands[0] == 'r':
		#m1, b1, m2, b2 = 0.9, 0.1, -3, 21
		m1, b1, m2, b2 = 0.9, 0.1, 0, 5
	else:
		m1, b1, m2, b2 = 0.8, 0.1, 0, 4.8
	resolved_tab = tab[np.where(tab['Resolved'] > 1000)]
	unresolved_tab = tab[np.where(tab['Resolved'] < 1000)]
	plotting.low_z_cut_plot(bands[0], unresolved_tab['%smag' % bands[0]] - unresolved_tab['W2mag'],
	                        unresolved_tab['%smag' % bands[1]] - unresolved_tab['W2mag'],
	                        resolved_tab['%smag' % bands[0]] - resolved_tab['W2mag'],
	                        resolved_tab['%smag' % bands[1]] - resolved_tab['W2mag'], m1, b1, m2, b2)
	x_colors = tab['%smag' % bands[0]] - tab['W2mag']
	y_colors = tab['%smag' % bands[1]] - tab['W2mag']
	tab = tab[np.where((y_colors > (x_colors * m1 + b1)) | (y_colors > (b2 + (m2 * x_colors))))]
	return tab

def total_proper_motion(pmra, pmdec, dec, e_pmra, e_pmdec):
	dec_radian = dec * np.pi / 180.
	u = np.sqrt(np.square(pmdec) + np.square(pmra * np.cos(dec_radian)))
	u_err = np.sqrt(np.square(pmdec / u * e_pmdec) + np.square(pmra * np.square(np.cos(dec_radian)) / u * e_pmra))
	return u, u_err





# choose which magnitudes, AGN criteria, magnitude cuts, etc and perform masking
def filter_table(soln='mpro', criterion='r90', w2cut=9, w1cut=None, pmsncut=None, sepcut=None, nbcut=None,
                 lowzcut=False, highzcut=False, bands=['r', 'W2']):
	cat = Table.read('catalogs/catwise_r75pm_ls.fits')
	if criterion == 'r90':
		alpha, beta, gamma = 0.65, 0.153, 13.86
	elif criterion == 'r75':
		alpha, beta, gamma = 0.486, 0.092, 13.07
	elif criterion == 'stern':
		alpha, beta, gamma = 0.8, 0.0, 0.0
		cat = cat[np.where(cat['W2%s' % soln] < 15.05)]
	else:
		alpha, beta, gamma = 0.42, 0.085, 13.07

	cat = cat[np.where(((cat['W1%s' % soln] - cat['W2%s' % soln] > alpha) & (cat['W2%s' % soln] <= gamma))
	                         | (cat['W1%s' % soln] - cat['W2%s' % soln] > alpha * np.exp(
		beta * np.square(cat['W2%s' % soln] - gamma))))]

	cat = cat[np.where(cat['W2%s' % soln] > w2cut)]
	if w1cut is not None:
		cat = cat[np.where(cat['W1%s' % soln] < w1cut)]
	cat['pm'], cat['e_pm'] = total_proper_motion(cat['pmRA'], cat['pmDE'], cat['DEC'], cat['e_pmRA'], cat['e_pmDE'])


	if pmsncut is not None:
		#cat = cat[np.where((np.abs(cat['pmRA'])/cat['e_pmRA'] < pmsncut) & (np.abs(cat['pmDE'])/cat['e_pmDE'] <
		#                                                                   pmsncut))]

		goodidxs = np.where(cat['pm'] / cat['e_pm'] < pmsncut)
		print('%s percent of sources removed for %s sigma proper motion cut' %
		      (100. * (len(cat) - len(goodidxs[0]))/float(len(cat)), pmsncut))
		cat = cat[goodidxs]

	if nbcut is not None:
		cat = cat[np.where(cat['nb'] < nbcut)]
	if sepcut is not None:
		cat['dered_mag_%s' % bands[0]][np.where(cat['sep'] > sepcut)] = np.nan

	masking.total_mask(depth_cut=150, assef=1, unwise=1, planck=1, bright=1, ebv=1, ls_mask=0, zero_depth_mask=1)

	cat = masking.mask_tab(cat)
	#randcat = Table.read('catalogs/derived/ls_randoms_1_masked.fits')
	randcat = Table.read('catalogs/randoms/ls_randoms/ls_randoms_1.fits')
	randcat = masking.mask_tab(randcat)
	randcat = randcat[:int(3e7)]

	newcat = cat['RA', 'DEC', 'W1%s' % soln, 'W2%s' % soln,'e_W1%s' % soln, 'e_W2%s' % soln, 'dered_mag_g',
	             'dered_mag_r', 'dered_mag_z', 'W3mag', 'e_W3mag', 'W4mag', 'e_W4mag', 'ab_flags', 'pm', 'e_pm',
	             'nb', 'maskbits']
	oldnames = ('W1%s' % soln, 'W2%s' % soln,'e_W1%s' % soln, 'e_W2%s' % soln, 'dered_mag_g',
	             'dered_mag_r', 'dered_mag_z')
	newnames = ('W1mag', 'W2mag', 'e_W1mag', 'e_W2mag', 'gmag', 'rmag', 'zmag')
	newcat.rename_columns(oldnames, newnames)


	#newcat['forced_W3'] = -2.5 * np.log10(3.631e-6 / 31.674 * cat['snr_w3'] / np.sqrt(cat['psfdepth_w3']))
	newcat['flux_W3'] = 3.631e-6 / 31.674 * cat['snr_w3'] / np.sqrt(cat['psfdepth_w3'])
	newcat['flux_W4'] = 3.631e-6 / 8.363 * cat['snr_w4'] / np.sqrt(cat['psfdepth_w4'])
	#newcat['W4mag'] = cat['W4mag']
	#newcat['e_W4mag'] = cat['e_W4mag']
	#newcat['r_depth'] = -2.5*(np.log10(5/np.sqrt(cat['PSFDEPTH_R']))-9)
	newcat['detect'] = np.zeros(len(newcat))
	newcat['detect'][np.where((newcat['%smag' % bands[0]] > 0) & (np.isfinite(newcat['%smag' % bands[0]])))] = 1
	#newcat[np.where(newcat['detect'] == 0)] = np.nan
	#newcat['ab_flags'] = cat['ab_flags']
	newcat['Resolved'] = cat['dchisq_2'] - cat['dchisq_1']
	newcat['weight'] = np.ones(len(cat))
	randcat['weight'] = np.ones(len(randcat))



	if lowzcut:
		newcat = cut_lowz(newcat, ['r', 'z'])

	if highzcut:
		#newcat = newcat[np.where(np.logical_not(newcat['zmag'] - newcat['W2mag'] < 4))]
		newcat = newcat[np.where(np.logical_not(newcat['gmag'] - newcat['W2mag'] < 4.5))]

	newcat = newcat[np.where((newcat['%smag' % bands[1]] > 0) & (np.isfinite(newcat['%smag' % bands[1]])) & (
		np.isfinite(newcat['%smag' % bands[0]])) & (np.isfinite(newcat['%smag' % bands[0]])))]
	newcat.write('catalogs/derived/catwise_filtered.fits', format='fits', overwrite=True)
	randcat.write('catalogs/derived/ls_randoms_1_filtered.fits', format='fits', overwrite=True)



# match to deep HSC data to get color distribution of optically undetected sources in main sample
def undetected_dist(tab, optband='r'):
	coords = SkyCoord(tab['RA'] * u.deg, tab['DEC'] * u.deg)

	hsctab = Table.read('catalogs/raw/r90_pm_hsc_dud.fits')
	hsctab = hsctab[np.where(((hsctab['WISE_RA']) > 146) & (hsctab['WISE_RA'] < 153))]
	hsccoords = SkyCoord(hsctab['WISE_RA'] * u.deg, hsctab['WISE_DEC'] * u.deg)

	idx, hscidx, d2d, d3d = hsccoords.search_around_sky(coords, 1*u.arcsec)

	optmags = hsctab['%smag' % optband][hscidx]
	w2mags = tab['W2mag'][idx]

	return optmags - w2mags - 3.313





def bin_sample(samplename, mode, band1='r', band2='W2', nbins=3, combinebins=None):

	cat = Table.read('catalogs/derived/catwise_filtered.fits')

	detectidx = np.where(cat['detect'].astype(np.bool))
	nondetectidx = np.where(cat['detect'] == 0)

	if mode == 'color':
		cat['color'] = np.empty(len(cat))
		cat['color'][detectidx] = cat['%smag' % band1][detectidx] - cat['%smag' % band2][detectidx] - vega_to_ab[band2]
		#cat['color'][nondetectidx] = cat['r_depth'][nondetectidx] - cat['W2mag'][nondetectidx] - 3.313
		cat['color'][nondetectidx] = 99.


		indicesbybin = bin_agn.bin_by_color(cat['color'], nbins)
		nondet_colors = undetected_dist(cat[nondetectidx], band1)

		cat['bin'] = np.zeros(len(cat))

		for j in range(len(indicesbybin)):
			cat['bin'][indicesbybin[j]] = j + 1

		if combinebins is not None:
			cat['bin'][np.where(cat['bin'] < (combinebins + 1))] = 1
			for j in range(nbins - combinebins):
				cat['bin'][np.where(cat['bin'] == j + combinebins + 1)] = j + 2

		plotting.plot_color_dists(cat, nondet_colors, band1, band2)
		plotting.plot_assef_cut(cat)

	elif mode == 'gal_lat':
		lons, lats = coord_transforms.sky_transform(cat['RA'], cat['DEC'], trans=['C', 'G'])
		cat['bin'] = bin_agn.bin_by_gal_lat(lats) + 1



	cat.write('catalogs/derived/%s_binned.fits' % samplename, format='fits', overwrite=True)




def long_wavelength_properties(samplename, bands):
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	medcolors = get_median_colors(tab)

	if 'w1w2' in bands:
		plotting.w1_w2_dists(tab)

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
	if 'mateos' in bands:
		for j in range(int(np.max(tab['bin']))):
			binnedtab = tab[np.where(tab['bin'] == j + 1)]
			plotting.mateos_plot(binnedtab, '3')
			plotting.mateos_plot(binnedtab, '4')
	if 'donley' in bands:
		for j in range(int(np.max(tab['bin']))):
			binnedtab = tab[np.where(tab['bin'] == j + 1)]
			binnedtab = binnedtab[np.where((binnedtab['RA'] > 216.) & (binnedtab['RA'] < 218.88) &
			            (binnedtab['DEC'] > 32.14) & (binnedtab['DEC'] < 36.05))]
			tabcoords = SkyCoord(binnedtab['RA'] * u.deg, binnedtab['DEC'] * u.deg)
			full_sdwfs_tab = Table.read('catalogs/bootes/sdwfs_irac.fits')
			sdwfs_coords = SkyCoord(full_sdwfs_tab['RA'] * u.deg, full_sdwfs_tab['DEC'] * u.deg)
			sdidx, d2d, d3d = tabcoords.match_to_catalog_sky(sdwfs_coords)
			sdwfs_tab = full_sdwfs_tab[sdidx]
			binnedtab = binnedtab[d2d < 2*u.arcsec]
			sdwfs_tab = sdwfs_tab[d2d < 2*u.arcsec]

			plotting.stern05_plot(sdwfs_tab, full_sdwfs_tab, j+1)
			plotting.donley_plot(sdwfs_tab, full_sdwfs_tab, j+1)
	if 'radio' in bands:
		surveynames = ['LOTSS_DR1', 'FIRST', 'VLACOSMOS']

		for survey in surveynames:
			footprint = hp.read_map('../data/radio_cats/hpx_footprints/%s.fits' % survey)
			nside = hp.npix2nside(len(footprint))
			tab_in_footprint = tab[np.where(footprint[hp.ang2pix(nside, tab['RA'], tab['DEC'], lonlat=True)] == 1)]
			radiotab = Table.read('../data/radio_cats/%s.fits' % survey)
			radcoords = SkyCoord(radiotab['RA'] * u.deg, radiotab['DEC'] * u.deg)

			colors, det_fraction = [], []
			for j in range(int(np.max(tab['bin']))):
				binnedtab = tab_in_footprint[np.where(tab_in_footprint['bin'] == j + 1)]
				binnedcoords = SkyCoord(binnedtab['RA'] * u.deg, binnedtab['DEC'] * u.deg)

				radidx, tabidx, d2d, d3d = binnedcoords.search_around_sky(radcoords, 5 * u.arcsec)
				det_fraction.append(len(tabidx) / len(binnedtab))
				colors.append(np.nanmean(binnedtab['color']))
			plotting.radio_detection_fraction(colors, det_fraction, survey)
	if 'kim' in bands:
		for j in range(int(np.max(tab['bin']))):
			binnedtab = tab[np.where(tab['bin'] == j + 1)]
			binnedtab = binnedtab[np.where((binnedtab['RA'] > 215.8) & (binnedtab['RA'] < 220.5) &
			                               (binnedtab['DEC'] > 32.2) & (binnedtab['DEC'] < 36.1))]
			bootes_help_tab = Table.read('../data/HELP/bootes_catwisepm_r75.fits')
			tabcoords = SkyCoord(binnedtab['RA'] * u.deg, binnedtab['DEC'] * u.deg)
			helpcoords = SkyCoord(bootes_help_tab['RA'] * u.deg, bootes_help_tab['DEC'] * u.deg)
			helpidx, d2d, d3d = tabcoords.match_to_catalog_sky(helpcoords)
			binnedtab = binnedtab[d2d < 2 * u.arcsec]
			bootes_help_tab = bootes_help_tab[helpidx[d2d < 2 * u.arcsec]]
			ki2 = bootes_help_tab['m_ap_newfirm_k'] - bootes_help_tab['m_ap_irac_i2']
			i24 = bootes_help_tab['m_ap_irac_i2'] - bootes_help_tab['m_ap_irac_i4']
			i2mips = bootes_help_tab['m_ap_irac_i2'] + 2.5 * np.log10(bootes_help_tab['f_mips_24'] / 3.631e9)
			plotting.plot_kim_diagrams(j+1, ki2, i24, i2mips)










"""def redshifts(samplename, z_sample='AGES'):
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
	plotting.redshift_dists(zs_by_bin)"""



def redshifts(samplename):
	tab = Table.read('catalogs/derived/%s_binned.fits' % samplename)
	z_fractions, zs_by_bin = [], []
	for j in range(int(np.max(tab['bin']))):
		binnedtab = tab[np.where(tab['bin'] == j + 1)]
		frac, zs = redshift_dists.get_redshifts(binnedtab)
		z_fractions.append(frac)
		zs_by_bin.append(zs)


	plotting.fraction_with_redshifts(get_median_colors(tab), z_fractions)
	plotting.redshift_dists(zs_by_bin)









