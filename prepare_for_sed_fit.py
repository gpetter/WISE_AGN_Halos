import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import constants as con
from astropy.table import Table, vstack, hstack
from scipy.stats import sigmaclip

import plotting
from source import interpolate_tools

def write_ascii_table(match_radius=2*u.arcsec):
	tab = Table.read('catalogs/derived/catwise_r90_cosmos_zs.fits')
	tab = tab[np.where((tab['RA'] > 145) & (tab['RA'] < 155) & (tab['DEC'] > 0) & (tab['DEC'] < 4))]
	deblendedcat = Table.read('../data/COSMOS/cosmos_superdeblend_fir/COSMOS_Super_Deblended.fits')
	ircoords = SkyCoord(deblendedcat['RA'] * u.deg, deblendedcat['DEC'] * u.deg)
	tabcoords = SkyCoord(tab['RA'] * u.deg, tab['DEC'] * u.deg)

	iridx, d2d, d3d = tabcoords.match_to_catalog_sky(ircoords)
	irmatches = deblendedcat[iridx]
	irmatches = irmatches[d2d < match_radius]
	tab = tab[d2d < match_radius]

	cosmos2020_cat = Table.read('../data/COSMOS/cosmos2020/cosmos2020_tractorflux.fits')
	newircoords = SkyCoord(irmatches['RA'] * u.deg, irmatches['DEC'] * u.deg)
	cosmoscoords = SkyCoord(cosmos2020_cat['RA'] * u.deg, cosmos2020_cat['DEC'] * u.deg)
	cosidx, d2d, d3d = newircoords.match_to_catalog_sky(cosmoscoords)
	cosmos_matches = cosmos2020_cat[cosidx]
	cosmos_matches = cosmos_matches[d2d < 1 * u.arcsec]
	irmatches = irmatches[d2d < 1 * u.arcsec]
	tab = tab[d2d < 1 * u.arcsec]




	uvoptbandnames = ['NUV', 'u', 'g', 'r', 'i', 'z', 'y', 'Y', 'J', 'H', 'Ks']
	uvopterrnames = [x + '_e' for x in uvoptbandnames]
	cosmos_matches = cosmos_matches[uvoptbandnames + uvopterrnames]
	uvoptwavelengths = [0.231, 0.371, 0.485, 0.622, 0.77, 0.889, 0.976, 1.022, 1.253, 1.647, 2.156]
	irbandnames = ['IRAC1', 'IRAC2', 'IRAC3', 'IRAC4', 'MIPS24', 'PACS100', 'PACS160', 'SPIRE250', 'SPIRE350',
	               'SPIRE500']
	irwavelenghts = [3.6, 4.5, 5.8, 8., 24., 100., 160., 250., 350., 500.]
	combinedtab = hstack([tab, irmatches, cosmos_matches])

	columns = [combinedtab['ID'], combinedtab['Z']]

	for j in range(len(uvoptwavelengths)):
		columns.append(uvoptwavelengths[j] * np.ones(len(combinedtab)))
		flux = combinedtab[uvoptbandnames[j]] / 1000.
		flux[np.where(np.isnan(flux))] = -99
		fluxerr = combinedtab['%s_e' % uvoptbandnames[j]] / 1000.
		fluxerr[np.where(np.isnan(fluxerr))] = -99
		if uvoptbandnames[j] == 'NUV':
			flux[np.where(combinedtab['Z'] > 0.5)] = -99
			fluxerr[np.where(combinedtab['Z'] > 0.5)] = -99
		if uvoptbandnames[j] == 'u':
			flux[np.where(combinedtab['Z'] > 1.5)] = -99
			fluxerr[np.where(combinedtab['Z'] > 1.5)] = -99
		if uvoptbandnames[j] == 'g':
			flux[np.where(combinedtab['Z'] > 2.3)] = -99
			fluxerr[np.where(combinedtab['Z'] > 2.3)] = -99
		columns.append(flux)

		columns.append(fluxerr)



	for j in range(len(irwavelenghts)):
		columns.append(irwavelenghts[j] * np.ones(len(combinedtab)))
		flux = combinedtab[irbandnames[j]]
		flux[np.where((np.isnan(flux)) | (flux > 1e5))] = -99
		columns.append(flux)
		fluxerr = combinedtab['%s_e' % irbandnames[j]]
		fluxerr[np.where((np.isnan(fluxerr)) | (fluxerr > 1e5))] = -99
		columns.append(fluxerr)


	finaltab = hstack(columns)
	print(np.arange(len(finaltab[0])))
	finaltab.write('/Users/graysonpetter/Desktop/AGNfitter-AGNfitter-rX_v0.1/data/test.txt', format='ascii',
	                       overwrite=True)

	with open('/Users/graysonpetter/Desktop/AGNfitter-master/data/test.txt', 'r+') as f:
		lines = f.readlines()
	lines[0] = '#' + lines[0]
	with open('/Users/graysonpetter/Desktop/AGNfitter-master/data/test.txt', 'w') as f:
		f.writelines(lines)


#write_ascii_table()


def composite_sed(n_wavelength_bins, binnum, binnedtab, nbins, norm_micron):

	help_cat = Table.read('../data/HELP/bootes_catwisepm_r75.fits')
	help_cat['zbest'] = help_cat['redshift']
	help_cat['zbest'][np.where((help_cat['zspec'] > 0) & (help_cat['zspec'] < 10))] = \
				help_cat['zspec'][np.where((help_cat['zspec'] > 0) & (help_cat['zspec'] < 10))]
	fkey, ferr_key = 'f_', 'ferr_'
	help_cat = help_cat[np.where(help_cat['ferr_ap_irac_i2'] > 0)]
	help_cat = help_cat[np.where((help_cat['zbest'] > 0) & (help_cat['zbest'] < 3))]

	help_coords = SkyCoord(help_cat['RA_wise'] * u.deg, help_cat['DEC_wise'] * u.deg)



	binnedtab = binnedtab[np.where((binnedtab['RA'] < np.max(help_cat['ra_help']) + 0.5) &
	                               (binnedtab['RA'] > np.min(help_cat['ra_help']) - 0.5) &
	                               (binnedtab['DEC'] < np.max(help_cat['dec_help']) + 0.5) &
	                               (binnedtab['DEC'] > np.min(help_cat['dec_help']) - 0.5))]

	binnedcoords = SkyCoord(binnedtab['RA'] * u.deg, binnedtab['DEC'] * u.deg)

	helpidx, d2d, d3d = binnedcoords.match_to_catalog_sky(help_coords)
	help_cat = help_cat[helpidx]
	help_cat = help_cat[d2d < 1*u.arcsec]
	binnedtab = binnedtab[d2d < 1*u.arcsec]

	rest_wavelength_bins = np.logspace(-1.1, 2.75, n_wavelength_bins)


	zs = np.array(help_cat['zbest'])
	filters = ['lbc_u', 'mosaic_b', '90prime_g', 'gpc1_g', 'gpc1_r', '90prime_r', 'mosaic_r', 'gpc1_i', 'mosaic_i',
	           'gpc1_z', 'suprime_z', 'mosaic_z', '90prime_z', 'gpc1_y', 'lbc_y', 'ukidss_j', 'newfirm_j',
	           'newfirm_h', 'tifkam_ks', 'newfirm_k', 'irac_i1', 'irac_i2', 'irac_i3', 'irac_i4', 'mips_24',
	           'pacs_green', 'pacs_red', 'spire_250', 'spire_350', 'spire_500']
	aperture_list = ['ap_' if k<len(filters) - 6 else '' for k in range(len(filters))]

	fullfilters = [fkey + aperture_list[j] + filters[j] for j in range(len(filters))]
	fullerrfilters = [ferr_key + aperture_list[j] + filters[j] for j in range(len(filters))]

	fluxes = help_cat[fullfilters]
	fluxerrs = help_cat[fullerrfilters]

	newfluxes, newfluxerrs = [], []
	for j in range(len(fluxes)):
		newfluxes.append(np.array(list(fluxes[j])).astype(np.float64))
		newfluxerrs.append(np.array(list(fluxerrs[j])).astype(np.float64))

	fluxes, fluxerrs = np.array(newfluxes), np.array(newfluxerrs)

	obs_wavelengths = np.array([0.365, 0.445, 0.464, 0.464, 0.617, 0.617, 0.617, 0.752, 0.752, 0.866, 0.866, 0.866,
	                          0.866, 1.031, 1.031, 1.248, 1.248, 1.631, 2.201, 2.201, 3.6, 4.5, 5.8, 8.0, 24., 70.,
	                            160., 250, 350., 500.])
	#plotting.plot_each_sed(obs_wavelengths, fluxes[0], np.array(list(fluxerrs[0])))
	#rest_wavelengths = []

	#for j in range(zs):
	#	rest_wavelengths.append(obs_wavelengths / (1 + zs[j]))

	#rest_wavelengths = obs_wavelengths / (1 + zs)
	# rest wavelength is observed wavelength / (1 + z)
	rest_wavelengths = np.outer(1 / (1 + zs), obs_wavelengths)
	# observed frequency is c / wavelength
	obs_freqs = (con.c / (np.array(obs_wavelengths) * u.micron)).to('Hz').value

	obs_nu_f_nu = obs_freqs * fluxes
	obs_nu_ferr_nu = obs_freqs * fluxerrs
	#rest_freqs = np.transpose(np.transpose(obs_freqs) * (1 + zs))
	#rest_nu_f_nu = obs_nu_f_nu / (rest_freqs)


	norm_lums = []
	for j in range(len(obs_nu_f_nu)):
		good_filters = np.where(np.isfinite(fluxes[j]))
		norm_lums.append(interpolate_tools.log_interp1d(rest_wavelengths[j][good_filters],
		            obs_nu_f_nu[j][good_filters])(norm_micron))
	norm_lums = np.array(norm_lums)
	median_lum = np.median(norm_lums)
	lum_ratios = np.array(median_lum / norm_lums)
	non_outliers = np.where(lum_ratios < 5 * np.std(lum_ratios))
	lum_ratios = lum_ratios[non_outliers]
	obs_nu_f_nu = obs_nu_f_nu[non_outliers]
	obs_nu_ferr_nu = obs_nu_ferr_nu[non_outliers]
	rest_wavelengths = rest_wavelengths[non_outliers]


	obs_nu_f_nu = np.transpose(np.transpose(obs_nu_f_nu) * lum_ratios)


	wavelength_bin_idxs = []
	for j in range(len(obs_nu_f_nu)):
		wavelength_bin_idxs.append(np.digitize(rest_wavelengths[j], rest_wavelength_bins))
	wavelength_bin_idxs = np.array(wavelength_bin_idxs)

	binned_nu_f_nu = []
	for j in range(len(rest_wavelength_bins)):
		nfnu_in_bin = obs_nu_f_nu[np.where(wavelength_bin_idxs == j)]
		binned_nu_f_nu.append(np.nanmean(np.log10(nfnu_in_bin)))
	binned_nu_f_nu = 10 ** np.array(binned_nu_f_nu)



	plotting.plot_each_sed(binnum, nbins, rest_wavelengths, obs_nu_f_nu, obs_nu_ferr_nu)

	return rest_wavelength_bins, binned_nu_f_nu



	#rest_nu_f_nu = rest_nu_f_nu * lum_ratios



def all_composites(n_wavelength_bins, rest_wavelength_normalize):
	tab = Table.read('catalogs/derived/catwise_binned.fits')
	nbins = int(np.max(tab['bin']))

	binnedseds, normlums = [], []
	for j in range(nbins):
		binnedtab = tab[np.where(tab['bin'] == j + 1)]
		rest_wavelength_bins, binnedsed = composite_sed(n_wavelength_bins, j+1, binnedtab, nbins,
		                                                rest_wavelength_normalize)
		normlums.append(interpolate_tools.log_interp1d(rest_wavelength_bins, binnedsed)(rest_wavelength_normalize))
		binnedseds.append(binnedsed)
	mednormlums = np.nanmedian(normlums)
	lum_ratios = np.array(mednormlums / np.array(normlums))

	binnedseds = np.transpose(np.transpose(np.array(binnedseds)) * lum_ratios)



	plotting.plot_composite_sed(nbins, rest_wavelength_bins, np.array(binnedseds))


all_composites(20, 3)