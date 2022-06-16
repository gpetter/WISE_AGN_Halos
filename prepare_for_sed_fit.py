import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import constants as con
from astropy.table import Table, vstack, hstack
from scipy.stats import sigmaclip
import os
import healpy as hp
import glob
from astropy.io import fits
from mocpy import MOC


from colossus.cosmology import cosmology

cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()


from source import interpolate_tools

# names of filter names in the HELP catalog and corresponding filter names in CIGALE
help_to_cigale_filters = {'Hx': 'xray_boxcar_2to7keV',
                    'Fx': 'xray_boxcar_0p5to7keV', 'Sx': 'xray_boxcar_0p5to2keV',
                    'FUV': 'FUV', 'NUV': 'NUV',
					'u': 'LBC_U', 'g': 'PAN-STARRS_g',
                    'r': 'PAN-STARRS_r', 'i': 'PAN-STARRS_i',
                    'z': 'PAN-STARRS_z', 'y': 'PAN-STARRS_y',
                    'j': 'J', 'h': 'H', 'k': 'K',
                    'ks': 'Ks_2mass',
                    'lbc_u': 'LBC_U',
                    'mosaic_b': 'MOSAIC_B',
                    '90prime_g': '90PRIME_B', 'gpc1_g': 'PAN-STARRS_g',
                    'gpc1_r': 'PAN-STARRS_r',
                    '90prime_r': '90PRIME_R', 'mosaic_r': 'MOSAIC_R',
                    'gpc1_i': 'PAN-STARRS_i', 'mosaic_i': 'MOSAIC_I',
                    'gpc1_z': 'PAN-STARRS_z', 'suprime_z': 'SUBARU_z',
                    'mosaic_z': 'MOSAIC_Z', '90prime_z': '90PRIME_Z',
	                'gpc1_y': 'PAN-STARRS_y', 'lbc_y': 'LBC_Y',
	                'ukidss_j': 'UKIRT_WFCJ', 'newfirm_j': 'NEWFIRM_J',
	                'newfirm_h': 'NEWFIRM_H', 'tifkam_ks': 'TIFKAM_Ks',
	                'newfirm_k': 'NEWFIRM_K', 'irac_i1': 'IRAC1',
	                'irac_i2': 'IRAC2', 'irac_i3': 'IRAC3',
	                'irac_i4': 'IRAC4', 'mips_24': 'MIPS1',
	                'pacs_green': 'PACS_green', 'pacs_red': 'PACS_red',
	                'spire_250': 'PSW', 'spire_350': 'PMW',
	                'spire_500': 'PLW'}

# set of bands to use in Bootes field
bootes_filterset = ['FUV', 'NUV',
                        'lbc_u', 'mosaic_b', '90prime_g',
                        'gpc1_g', 'gpc1_r', '90prime_r',
                        'mosaic_r', 'gpc1_i', 'mosaic_i',
	                    'gpc1_z', 'suprime_z', 'mosaic_z',
	                    '90prime_z', 'gpc1_y', 'lbc_y',
	                    'ukidss_j', 'newfirm_j',
	                    'newfirm_h', 'tifkam_ks', 'newfirm_k',
	                    'irac_i1', 'irac_i2', 'irac_i3', 'irac_i4',
	                    'mips_24', 'pacs_green', 'pacs_red',
	                    'spire_250', 'spire_350', 'spire_500']

# set of bands in the "best" version of the HELP catalog
best_filterset = ['u', 'g', 'r', 'i', 'z', 'y', 'j', 'h', 'k', 'ks', 'irac_i1', 'irac_i2', 'irac_i3', 'irac_i4',
                  'mips_24', 'pacs_green', 'pacs_red', 'spire_250', 'spire_350', 'spire_500']

# fields to use in compsosite SED construction, deep and wide
#good_fields = ['bootes', 'cdfs_swire', 'elais_n1', 'elais_s1', 'lockman_swire', 'xmm_lss']
good_fields = ['xmm_lss']

fieldtranslations = {'s82': 'Herschel-Stripe-82', 'ssdf': 'SSDF', 'xmm_lss': 'XMM-LSS', 'bootes': 'Bootes',
                     'cdfs_swire': 'CDFS-SWIRE', 'cosmos': 'COSMOS', 'elais_n1': 'ELAIS-N1', 'elais_s1': 'ELAIS-S1',
                     'lockman_swire': 'Lockman SWIRE'}

# fluxes in muJy below which the HELP algorithm deems posterior estimates of flux become non-gaussian
mips_gauss_cut = {'akari_nep': 30, 'akari_sep': 40, 'bootes': 20,
		'cdfs_swire': 20, 'cosmos': 0,
		'elais_n1': 20, 'elais_n2': 20,
		'elais_s1': 30, 'lockman_swire': 20,
		'spire_nep': 20, 'xfls': 20,
		'xmm_lss': 0}

spire_250_gauss_cut = {'akari_nep': 5000, 'bootes': 5000,
		'cdfs_swire': 4000,
		'elais_n1': 4000, 'elais_n2': 4000,
		'elais_s1': 4000, 'lockman_swire': 4000,
		'spire_nep': 6000, 'xfls': 4000,
		'xmm_lss': 4000, 'cosmos': 0}

spire_350_gauss_cut = {'akari_nep': 5000, 'bootes': 5000,
		'cdfs_swire': 4000,
		'elais_n1': 4000, 'elais_n2': 4000,
		'elais_s1': 4000, 'lockman_swire': 4000,
		'spire_nep': 6000, 'xfls': 4000,
		'xmm_lss': 4000, 'cosmos': 0}

spire_500_gauss_cut = {'akari_nep': 6000, 'bootes': 10000,
		'cdfs_swire': 6000,
		'elais_n1': 4000, 'elais_n2': 4000,
		'elais_s1': 6000, 'lockman_swire': 6000,
		'spire_nep': 6000, 'xfls': 4000,
		'xmm_lss': 4000, 'cosmos': 0}

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



# take a photometry table and append galex fluxes where they exist
def append_galex_data(table):
	coords = SkyCoord(table['RA_wise'] * u.deg, table['DEC_wise'] * u.deg)
	table['f_ap_FUV'], table['ferr_ap_FUV'], table['f_ap_NUV'], table['ferr_ap_NUV'] = -9999000. * \
	                                                            np.ones(len(table)), \
	                                                            -9999000. * np.ones(len(table)), \
	                                                            -9999000. * np.ones(len(table)), \
	                                                            -9999000. * np.ones(len(table))
	galextable = Table.read('catalogs/matches/r75pm_galex.fits')
	galexcoords = SkyCoord(galextable['RA'] * u.deg, galextable['DEC'] * u.deg)
	galexidx, d2d, d3d = coords.match_to_catalog_sky(galexcoords)
	galextable = galextable[galexidx]
	matchidx = np.where(d2d < 1 * u.arcsec)
	table['f_ap_FUV'][matchidx], table['ferr_ap_FUV'][matchidx] = galextable['Fflux'][matchidx], \
	                                                        galextable['e_Fflux'][matchidx]
	table['f_ap_NUV'][matchidx], table['ferr_ap_NUV'][matchidx] = galextable['Nflux'][matchidx], \
	                                                        galextable['e_Nflux'][matchidx]
	table['f_ap_FUV'][np.where(np.logical_not(np.isfinite(table['f_ap_FUV'])))] = -9999000.
	table['ferr_ap_FUV'][np.where(np.logical_not(np.isfinite(table['ferr_ap_FUV'])))] = -9999000.
	table['f_ap_NUV'][np.where(np.logical_not(np.isfinite(table['f_ap_NUV'])))] = -9999000.
	table['ferr_ap_NUV'][np.where(np.logical_not(np.isfinite(table['ferr_ap_NUV'])))] = -9999000.

	return table


# take a photometry table and append galex fluxes where they exist
def append_chandra_data(table):
	import convert_Fx
	coords = SkyCoord(table['RA_wise'] * u.deg, table['DEC_wise'] * u.deg)
	table['f_ap_Hx'], table['ferr_ap_Hx'], table['f_ap_Fx'], table['ferr_ap_Fx'], table['f_ap_Sx'], \
		table['ferr_ap_Sx'] = -9999000. * np.ones(len(table)), -9999000. * np.ones(len(table)), \
	    -9999000. * np.ones(len(table)), -9999000. * np.ones(len(table)), -9999000. * np.ones(len(table)), \
	    -9999000. * np.ones(len(table))

	xraytable = Table.read('../data/xray_surveys/masini_bootes.fits')
	# use only sources with a match in Spizter IRAC 1
	xraytable = xraytable[np.where(xraytable['SDWFS_RAJ2000'] > 0)]
	xraycoords = SkyCoord(xraytable['SDWFS_RAJ2000'] * u.deg, xraytable['SDWFS_DEJ2000'] * u.deg)
	xrayidx, d2d, d3d = coords.match_to_catalog_sky(xraycoords)
	xraytable = xraytable[xrayidx]
	matchidx = np.where(d2d < 2.5 * u.arcsec)

	hardflux, hardfluxerr = convert_Fx.convt_Fx_to_Fnu(xraytable['FLUX_H'][matchidx] * (u.mW / (u.m**2)).to(
		u.erg/u.s/(u.cm**2)), xraytable['E_FLUX_H_N'][matchidx] * (u.mW / (u.m**2)).to(
		u.erg/u.s/(u.cm**2)), 2, 7)

	fullflux, fullfluxerr = convert_Fx.convt_Fx_to_Fnu(xraytable['FLUX_F'][matchidx] * (u.mW / (u.m**2)).to(
		u.erg/u.s/(u.cm**2)), xraytable['E_FLUX_F_N'][matchidx] * (u.mW / (u.m**2)).to(
		u.erg/u.s/(u.cm**2)), 0.5, 7)
	softflux, softfluxerr = convert_Fx.convt_Fx_to_Fnu(xraytable['FLUX_S'][matchidx] * (u.mW / (u.m**2)).to(
		u.erg/u.s/(u.cm**2)), xraytable['E_FLUX_S_N'][matchidx] * (u.mW / (u.m**2)).to(
		u.erg/u.s/(u.cm**2)), 0.5, 2)

	hardflux[np.where(hardfluxerr < 0)] = -9999.
	fullflux[np.where(fullfluxerr < 0)] = -9999.
	softflux[np.where(softfluxerr < 0)] = -9999.


	table['f_ap_Hx'][matchidx], table['ferr_ap_Hx'][matchidx] = 1000*hardflux, 1000*hardfluxerr
	table['f_ap_Fx'][matchidx], table['ferr_ap_Fx'][matchidx] = 1000*fullflux, 1000*fullfluxerr
	table['f_ap_Sx'][matchidx], table['ferr_ap_Sx'][matchidx] = 1000*softflux, 1000*softfluxerr


	return table

def append_lofar_data(table):
	coords = SkyCoord(table['RA_wise'] * u.deg, table['DEC_wise'] * u.deg)
	table['f_ap_Hx'], table['ferr_ap_Hx'], table['f_ap_Fx'], table['ferr_ap_Fx'], table['f_ap_Sx'], \
	table['ferr_ap_Sx'] = -9999000. * np.ones(len(table)), -9999000. * np.ones(len(table)), \
	                      -9999000. * np.ones(len(table)), -9999000. * np.ones(len(table)), -9999000. * np.ones(
		len(table)), \
	                      -9999000. * np.ones(len(table))

	xraytable = Table.read('../data/radio_cats/lofar_deep')
	# use only sources with a match in Spizter IRAC 1
	xraytable = xraytable[np.where(xraytable['SDWFS_RAJ2000'] > 0)]
	xraycoords = SkyCoord(xraytable['SDWFS_RAJ2000'] * u.deg, xraytable['SDWFS_DEJ2000'] * u.deg)
	xrayidx, d2d, d3d = coords.match_to_catalog_sky(xraycoords)
	xraytable = xraytable[xrayidx]
	matchidx = np.where(d2d < 2.5 * u.arcsec)

	hardflux, hardfluxerr = convert_Fx.convt_Fx_to_Fnu(xraytable['FLUX_H'][matchidx] * (u.mW / (u.m ** 2)).to(
		u.erg / u.s / (u.cm ** 2)), xraytable['E_FLUX_H_N'][matchidx] * (u.mW / (u.m ** 2)).to(
		u.erg / u.s / (u.cm ** 2)), 2, 7)

	fullflux, fullfluxerr = convert_Fx.convt_Fx_to_Fnu(xraytable['FLUX_F'][matchidx] * (u.mW / (u.m ** 2)).to(
		u.erg / u.s / (u.cm ** 2)), xraytable['E_FLUX_F_N'][matchidx] * (u.mW / (u.m ** 2)).to(
		u.erg / u.s / (u.cm ** 2)), 0.5, 7)
	softflux, softfluxerr = convert_Fx.convt_Fx_to_Fnu(xraytable['FLUX_S'][matchidx] * (u.mW / (u.m ** 2)).to(
		u.erg / u.s / (u.cm ** 2)), xraytable['E_FLUX_S_N'][matchidx] * (u.mW / (u.m ** 2)).to(
		u.erg / u.s / (u.cm ** 2)), 0.5, 2)

	hardflux[np.where(hardfluxerr < 0)] = -9999.
	fullflux[np.where(fullfluxerr < 0)] = -9999.
	softflux[np.where(softfluxerr < 0)] = -9999.

	table['f_ap_Hx'][matchidx], table['ferr_ap_Hx'][matchidx] = 1000 * hardflux, 1000 * hardfluxerr
	table['f_ap_Fx'][matchidx], table['ferr_ap_Fx'][matchidx] = 1000 * fullflux, 1000 * fullfluxerr
	table['f_ap_Sx'][matchidx], table['ferr_ap_Sx'][matchidx] = 1000 * softflux, 1000 * softfluxerr

	return table

# convert band names from HELP catalog to corresponding CIGALE filter names
def convert_filternames(filterset, fkey='f_', ferr_key='ferr_'):
	cigale_names = [help_to_cigale_filters[filtername] for filtername in filterset]
	cigale_err_names = ['%s_err' % name for name in cigale_names]
	aperture_list = ['ap_' if k < len(filterset) - 6 else '' for k in range(len(filterset))]
	fullfilters = [fkey + aperture_list[j] + filterset[j] for j in range(len(filterset))]
	fullerrfilters = [ferr_key + aperture_list[j] + filterset[j] for j in range(len(filterset))]
	return cigale_names, cigale_err_names, fullfilters, fullerrfilters


def append_data(table, filterset):
	if ('FUV' in filterset) or ('NUV' in filterset):
		table = append_galex_data(table)
	if ('Sx' in filterset) or ('Fx' in filterset) or ('Hx' in filterset):
		table = append_chandra_data(table)
	return table


def write_table_for_cigale(table, filterset, photoz=False):
	table = append_data(table, filterset)


	cigale_names, cigale_err_names, fullfilters, fullerrfilters = convert_filternames(filterset)
	table.rename_columns(fullfilters, cigale_names)
	table.rename_columns(fullerrfilters, cigale_err_names)

	if photoz:
		table['redshift'] = -1 * np.ones(len(table))

	table = table[np.where((table['zspec'] > 0) & (table['zspec'] < 10))]
	table.add_column(np.arange(len(table)), index=0, name='id')
	fluxanderrnames = cigale_names + cigale_err_names
	final_column_names = ['id'] + ['zspec'] + cigale_names + cigale_err_names
	for name in fluxanderrnames:
		table[name] *= 1.e-3
	table = table[final_column_names].filled(-9999.)
	table.rename_column('zspec', 'redshift')
	table.write('cigale_files/cigale_input.txt', format='ascii', overwrite=True)

# for given set of filters, determine an effective wavelenght in micron by finding maximum of response curve
def peak_wavelengths_from_filters(filterset):
	import pandas as pd
	cigalefilternames = [help_to_cigale_filters[filtername] for filtername in filterset]
	path_to_cigale = '/Users/graysonpetter/Desktop/Dartmouth/cigale/database_builder/default_filters/'
	peaklambdas = []
	for name in cigalefilternames:
		filterfile = pd.read_csv(path_to_cigale + name + '.dat', comment='#', names=['l', 'r'], delim_whitespace=True)

		peaklambdas.append(filterfile['l'][np.argmax(filterfile['r'])] * (u.Angstrom).to(u.micron))
	return peaklambdas



# plot every flux vs wavelength plot for a catalog
def plot_each_observed_sed(table, filternames):
	import plotting
	oldplots = glob.glob('plots/individual_seds/*')
	for oldplot in oldplots:
		os.remove(oldplot)
	table = append_data(table, filternames)
	wavelengths = peak_wavelengths_from_filters(filternames)

	#cigale_names, cigale_err_names, fullfilters, fullerrfilters = convert_filternames(filterset)
	fullfilternames = ['f_best_%s' % filternames[k] if k < len(filternames) - 10 else 'f_%s' % filternames[k] for k in
	                   range(len(filternames))]
	fullerrnames = ['ferr_best_%s' % filternames[k] if k < len(filternames) - 10 else 'ferr_%s' % filternames[k]
	                for k in range(len(filternames))]

	fluxtable = table[fullfilternames]

	fluxerrtable = table[fullerrnames]
	# redshifts =
	plotting.plot_every_observed_sed(fluxtable=fluxtable.as_array(), fluxerrtable=fluxerrtable.as_array(),
	                                 eff_wavelengths=wavelengths, zs=table['Z_best'])

# check where objects lie within a given HELP field with its corresponding MOC file
def in_help_moc(ras, decs, helpfield):
	fieldmoc = MOC.from_fits('../data/HELP/%s/moc.fits' % helpfield)
	goodidxs = fieldmoc.contains(ras * u.deg, decs * u.deg)
	return goodidxs

#
def match_help_catwise(catwiseras, catwisedecs, binnumbers, helpfield, seplimit=2.5 * u.arcsec):
	catwisecoords = SkyCoord(catwiseras * u.deg, catwisedecs * u.deg)


	helpcat = Table.read('../data/HELP/%s/%s_masked.fits' % (helpfield, helpfield))
	helpcoords = SkyCoord(helpcat['ra'], helpcat['dec'])

	helpidx, d2d, d3d = catwisecoords.match_to_catalog_sky(helpcoords)
	helpcat = helpcat[helpidx]
	helpcat = helpcat[d2d < seplimit]
	binnumbers = binnumbers[d2d < seplimit]
	helpcat['bin'] = binnumbers
	helpcat.write('../data/HELP/%s/%s_matched.fits' % (helpfield, helpfield), format='fits', overwrite=True)


# take entire HELP catalog for a field and write a smaller catalog containing only RA, Dec, IRAC2 mag
def write_sparse_help_catalog(helpfield):
	helptab = Table.read('../data/HELP/%s/%s_full.fits' % (helpfield, helpfield))
	try:
		helptab['ra', 'dec', 'm_ap_irac_i2'].write('../data/HELP/%s/%s_sparse.fits' % (helpfield, helpfield),
		                                           format='fits', overwrite=True)
	except:
		print(helpfield + ' has no irac aperture mags, using fit mags')
		helptab['ra', 'dec', 'm_irac_i2'].write('../data/HELP/%s/%s_sparse.fits' % (helpfield, helpfield),
		                                           format='fits', overwrite=True)

# read in MOC files representing masks around bright stars in HELP fields, return indices of rows of table outside
# these masks
def mask_star_regions(ras, decs, helpfield):
	maskfiles = glob.glob('../data/HELP/%s/holes/*.fits' % helpfield)
	if len(maskfiles) > 0:

		outsidemask_lists = []
		for maskfile in maskfiles:
			thismoc = MOC.from_fits(maskfile)
			outsidemask_lists.append(np.array(thismoc.contains(ras, decs)).astype(np.int32))

		idxs = np.prod(outsidemask_lists, axis=0).astype(bool)

		return idxs
	else:
		return np.arange(len(ras))


def get_objects_in_spitzer_footprint(ras, decs, helpfield, w2_vega_limit, mips=True, cade_masks=False):

	f_limit_mujy = 1e6 * 171.787 * (10 ** (-1 * w2_vega_limit / 2.5))

	if cade_masks:
		from source import coord_transforms
		ls, bs = coord_transforms.sky_transform(ras, decs, trans=['C', 'G'])
		swire_ch2_depth_map = Table.read('../data/coverage_maps/spitzer/SWIRE_IRAC2_depth_CADE.fits')
		swire_covered_idxs = np.where(swire_ch2_depth_map['SIGNAL'] < 0.002)



		sdwfs_ch2_map = Table.read('../data/coverage_maps/spitzer/SDWFS_IRAC2_map_CADE.fits')
		sdwfs_covered_idxs = np.where(sdwfs_ch2_map['SIGNAL'] < 5)


		scosmos_ch2_depth_map = Table.read('../data/coverage_maps/spitzer/SCOSMOS_IRAC2_depth_CADE.fits')
		mapls, mapbs = hp.pix2ang(hp.order2nside(14), np.array(scosmos_ch2_depth_map['PIXEL']), nest=True,
		                             lonlat=True)
		mapras, mapdecs = coord_transforms.sky_transform(mapls, mapbs, trans=['G', 'C'])

		scosmos_covered_idxs = np.where((mapras > 149.411) &
		                                (mapras < 150.827) &
		                                (mapdecs > 1.498) &
		                                (mapdecs < 2.913) & (scosmos_ch2_depth_map['SIGNAL'] < 0.001))


		all_covered_pix = list(sdwfs_ch2_map[sdwfs_covered_idxs]['PIXEL']) + \
		                  list(swire_ch2_depth_map[swire_covered_idxs]['PIXEL']) + \
		                  list(scosmos_ch2_depth_map[scosmos_covered_idxs]['PIXEL'])
		hppix14 = hp.ang2pix(nside=hp.order2nside(14), theta=ls, phi=bs, lonlat=True, nest=True)

		if mips:
			mips_depth_map = hp.read_map('../data/HELP/%s/mips_depth.fits' % helpfield)
			hppix12 = hp.ang2pix(nside=hp.order2nside(12), theta=ras, phi=decs, lonlat=True)
			idxs = np.where((mips_depth_map[hppix12] > 0) & (np.in1d(hppix14, all_covered_pix)))
		else:
			idxs = np.where(np.in1d(hppix14, all_covered_pix))
	else:
		depth_map = Table.read('../data/HELP/%s/depths.fits' % helpfield)
		hppix13 = hp.ang2pix(nside=hp.order2nside(13), theta=ras, phi=decs, lonlat=True, nest=True)
		covered_idxs = np.where(depth_map['ferr_ap_irac_i2_mean'] < f_limit_mujy / 5.)
		if mips:
			mips_depth_map = hp.read_map('../data/HELP/%s/mips_depth.fits' % helpfield)
			hppix12 = hp.ang2pix(nside=hp.order2nside(12), theta=ras, phi=decs, lonlat=True)
			idxs = np.where((mips_depth_map[hppix12] > 0) & (np.in1d(hppix13, depth_map[covered_idxs]['hp_idx_O_13'])))
		else:
			idxs = np.where(np.in1d(hppix13, depth_map[covered_idxs]['hp_idx_O_13']))


	return idxs


def get_objects_with_k_imaging(ras, decs, k_mujy_limit, helpfield):
	depth_map = Table.read('../data/HELP/%s/depths.fits' % helpfield)
	try:
		ks_covered = depth_map['ferr_ap_ks_mean'] < k_mujy_limit
	except:
		try:
			ks_covered = depth_map['ferr_ks_mean'] < k_mujy_limit
		except:
			print(helpfield + " contains no Ks data. using K band")

	try:
		k_covered = depth_map['ferr_ap_k_mean'] < k_mujy_limit
	except:
		try:
			k_covered = depth_map['ferr_k_mean'] < k_mujy_limit
		except:
			print(helpfield + " contains no K data. using Ks band")

	try:
		k_or_ks_covered_idxs = np.logical_or(ks_covered, k_covered)
	except:
		try:
			k_or_ks_covered_idxs = ks_covered
		except:
			k_or_ks_covered_idxs = k_covered


	hppix = hp.ang2pix(nside=hp.order2nside(13), theta=ras, phi=decs, lonlat=True, nest=True)

	idxs = np.where(np.in1d(hppix, depth_map[k_or_ks_covered_idxs]['hp_idx_O_13']))

	return idxs


def append_zbest(t, zqual_cut=3):
	t['Z_best'] = t['redshift']

	goodspecz = np.where((t['zspec'] > 0) & (t['zspec'] < 10) & (t['zspec_qual'] >= zqual_cut))
	t['Z_best'][goodspecz] = t['zspec'][goodspecz]
	return t


#
def complete_masking(helpfield):
	print('masking ' + helpfield)

	# read lightweight HELP table which only has RAs, Dec, IRAC2 mags
	helpcat = Table.read('../data/HELP/%s/%s_sparse.fits' % (helpfield, helpfield))
	# mask bright stars
	helpcat = helpcat[mask_star_regions(helpcat['ra'], helpcat['dec'], helpfield)]
	# limit to sky with spitzer coverage
	helpcat = helpcat[get_objects_in_spitzer_footprint(helpcat['ra'].value, helpcat['dec'].value, helpfield,
	                                                   w2_vega_limit=17,
	                                                   cade_masks=True)]
	# limit to sky with deep K or Ks imaging
	helpcat = helpcat[get_objects_with_k_imaging(helpcat['ra'], helpcat['dec'], 20, helpfield)]
	# limit to detections in IRAC2, as WISE AGN should be detected
	try:
		helpcat = helpcat[np.where(np.isfinite(helpcat['m_ap_irac_i2']))]
	except:
		helpcat = helpcat[np.where(np.isfinite(helpcat['m_irac_i2']))]

	# write table of HELP IRAC2 detections in good regions
	helpcat.write('../data/HELP/%s/%s_masked.fits' % (helpfield, helpfield), format='fits', overwrite=True)

	# read WISE AGN catalog
	agncat = Table.read('catalogs/derived/catwise_binned.fits')
	# constrain to within the footprint of the HELP field
	agncat = agncat[in_help_moc(agncat['RA'], agncat['DEC'], helpfield)]

	# apply same masking to WISE AGN
	agncat = agncat[mask_star_regions(agncat['RA'] * u.deg, agncat['DEC'] * u.deg, helpfield)]

	agncat = agncat[np.where(agncat['W2mag'] > 13)]
	agncat = agncat[get_objects_in_spitzer_footprint(agncat['RA'], agncat['DEC'], helpfield, w2_vega_limit=17,
	                                                 cade_masks=True)]

	agncat = agncat[get_objects_with_k_imaging(agncat['RA'], agncat['DEC'], 20, helpfield)]

	agncat.write('../data/HELP/%s/wise_agn.fits' % helpfield, format='fits', overwrite=True)

	# match between WISE AGN and HELP objects
	match_help_catwise(agncat['RA'], agncat['DEC'], agncat['bin'], helpfield)



def tap_help_a_list(helpfield):
	print('matching %s to A-list' % helpfield)
	import pyvo as vo
	service = vo.dal.TAPService("https://herschel-vos.phys.sussex.ac.uk/__system__/tap/run/tap")
	masked_help_catalog = Table.read('../data/HELP/%s/%s_matched.fits' % (helpfield, helpfield))
	adql = """SELECT
			   TOP 20000
			   *
			   FROM help_a_list.main AS db
			   JOIN TAP_UPLOAD.helptab AS tc
			   ON 1=CONTAINS(POINT('ICRS', db.ra, db.dec),
			                 CIRCLE('ICRS', tc.ra, tc.dec, 0.1/3600.))"""
	alistcat = service.search(adql, uploads={'helptab': masked_help_catalog})
	alistcat = alistcat.to_table()
	alistcat.remove_columns(['field', 'help_id', 'stellarity_origin', 'ra_', 'dec_'])
	Table(alistcat).write('../data/HELP/%s/%s_a_list.fits' % (helpfield, helpfield), format='fits', overwrite=True)

def make_master_catalog(fields):
	t = Table.read('../data/HELP/%s/%s_a_list.fits' % (fields[0], fields[0]))

	t['250_nongauss'] = np.int32((t['f_spire_250'] < spire_250_gauss_cut[fields[0]]))
	t['350_nongauss'] = np.int32((t['f_spire_350'] < spire_350_gauss_cut[fields[0]]))
	t['500_nongauss'] = np.int32((t['f_spire_500'] < spire_500_gauss_cut[fields[0]]))
	for j in range(len(fields)):
		newt = Table.read('../data/HELP/%s/%s_a_list.fits' % (fields[j], fields[j]))
		newt['250_nongauss'] = np.int32((newt['f_spire_250'] < spire_250_gauss_cut[fields[j]]))
		newt['350_nongauss'] = np.int32((newt['f_spire_350'] < spire_350_gauss_cut[fields[j]]))
		newt['500_nongauss'] = np.int32((newt['f_spire_500'] < spire_500_gauss_cut[fields[j]]))
		t = vstack((t, newt))
	t = append_zbest(t)
	#t = t[np.where((t['Z_best'] > 0) & (t['ferr_spire_250'] > 0))]
	t.write('../data/HELP/master_agn_catalog.fits', format='fits', overwrite=True)


def prepare_help_data(fields):
	for field in fields:
		complete_masking(field)
		tap_help_a_list(field)
	make_master_catalog(fields)









def prep_cigale_init(xray=False, radio=False, vary_inclination=False, vary_tau=False, vary_galaxy_extinction=False,
	vary_burst_age=False, vary_stellar_age=False, vary_opening_angle=False, disktype='1'):

	from fileinput import FileInput
	import sys

	curdir = os.getcwd()

	# if old ini file present, delete it
	if len(glob.glob('cigale_files/pcigale.ini')) > 0:
		print('removing old ini file')
		os.remove('cigale_files/pcigale.ini')
	if len(glob.glob('cigale_files/pcigale.ini.spec')) > 0:
		os.remove('cigale_files/pcigale.ini.spec')

	from pcigale.session.configuration import Configuration
	import pcigale
	os.chdir('cigale_files')
	# generate new ini file
	pcigale.init(Configuration())


	if xray:
		xkey = 'xray, '
	else:
		xkey = ''
	if radio:
		radkey = 'radio, '
	else:
		radkey = ''

	# edit ini file to contain correct modules
	with FileInput(files=['pcigale.ini'], inplace=True) as f:
		for line in f:
			if line.startswith('data_file'):
				line = 'data_file = cigale_input.txt\n'
			if line.startswith('sed_modules'):
				line = 'sed_modules = sfhdelayed, bc03, nebular, dustatt_calzleit, dale2014, skirtor2016, %s%s ' \
				       'redshifting\n' % (xkey, radkey)
			if line.startswith('analysis_method'):
				line = 'analysis_method = pdf_analysis\n'
			if line.startswith('cores'):
				line = 'cores = 6\n'
			sys.stdout.write(line)

	# have cigale process initial config file
	pcigale.genconf(Configuration())

	# now modify config file again to set parameter

	with FileInput(files=['pcigale.ini'], inplace=True) as f:
		for line in f:
			stripline = line.strip()
			if stripline.startswith('age_main ='):
				if vary_stellar_age:
					line = '    age_main = 1000, 2000, 5000\n'
			if stripline.startswith('age_burst ='):
				if vary_burst_age:
					line = '    age_burst = 20, 50\n'

			if stripline.startswith('E_BVs_young ='):
				if vary_galaxy_extinction:
					line = '    E_BVs_young = 0.01, 0.3, 0.7, 1.5\n'

			if stripline.startswith('t = '):
				if vary_tau:
					line = '    t = 3, 5, 7, 9, 11\n'

			if stripline.startswith('i ='):
				if vary_inclination:
					line = '    i = 0, 10, 20, 30, 40, 50, 60, 70, 80, 90\n'
				else:
					line = '    i = 0\n'
			if stripline.startswith('fracAGN = 0.1'):
				line = '    fracAGN = 0.1, 0.9, 0.99\n'
			if stripline.startswith('EBV'):
				if vary_inclination:
					line = '    EBV = 0.01\n'
				else:
					line = '    EBV = 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1., 2., 5., 10., 20.\n'
			if stripline.startswith('lambda_fracAGN'):
				line = '    lambda_fracAGN = 2/20\n'
			if stripline.startswith('save_best_sed'):
				line = '  save_best_sed = True\n'
			if stripline.startswith('disk_type'):
				line = '  disk_type = %s\n' % disktype
			#if stripline.startswith('temperature = '):
			#	line = '  temperature = 200.0\n'
			if stripline.startswith('oa'):
				if vary_opening_angle:
					line = '    oa=10, 20, 30, 40, 50, 60\n'
				else:
					line = '    oa=30\n'
			sys.stdout.write(line)

	os.chdir(curdir)

def tap_donley_agn(helpfield):
	import agn_criteria
	import pyvo as vo
	service = vo.dal.TAPService("https://herschel-vos.phys.sussex.ac.uk/__system__/tap/run/tap")
	adql = """SELECT *
	                    FROM help_a_list.main AS db
	                        WHERE db.field = '%s' AND db.m_irac_i4 IS NOT NULL""" % fieldtranslations[helpfield]
	alistcat = service.search(adql)
	alistcat = alistcat.to_table()
	alistcat.remove_columns(['field', 'help_id', 'stellarity_origin'])

	spitzer_detection_idxs = np.where((alistcat['f_irac_i1'] > 0) &
	                                  (alistcat['f_irac_i2'] > 0) &
	                                  (alistcat['f_irac_i3'] / alistcat['ferr_irac_i3'] > 3) &
	                                  (alistcat['f_irac_i4'] / alistcat['ferr_irac_i4'] > 3))
	alistcat = alistcat[spitzer_detection_idxs]
	donleycat = alistcat[agn_criteria.donley_selection_flux(alistcat['f_irac_i1'], alistcat['f_irac_i2'],
	                                                     alistcat['f_irac_i3'], alistcat['f_irac_i4'])]
	donleycat = append_zbest(donleycat)
	donleycat.write('../data/HELP/%s/%s_donley.fits' % (helpfield, helpfield), format='fits', overwrite=True)





def redshift_completeness(ids, help_tab):
	help_tab = help_tab[np.where(np.isin(help_tab['id'], ids))]
	totzfraction = len(np.where(help_tab['Z_best'] > 0)[0]) / len(help_tab)
	speczfraction = len(np.where(help_tab['zspec_qual'] >= 2)[0]) / len(help_tab)
	photz_fraction = totzfraction - speczfraction
	return totzfraction, photz_fraction, speczfraction


# take observed fluxes and shift to rest frame nu F_nu
def rest_frame_seds(filternames, catalog):
	fullfilternames = ['f_best_%s' % filternames[k] if k < len(filternames) - 10 else 'f_%s' % filternames[k] for k in
	                   range(len(filternames))]
	fullerrnames = ['ferr_best_%s' % filternames[k] if k < len(filternames) - 10 else 'ferr_%s' % filternames[k]
	                for k in range(len(filternames))]
	#highzcatalog = catalog[np.where(catalog['Z_best'] > 3)]
	#plot_each_observed_sed(highzcatalog, filternames)
	catalog = catalog[np.where(catalog['Z_best'] < 3)]

	zero_nongauss = True
	mask_bad_bright_sources = True
	zs = catalog['Z_best']
	if zero_nongauss:
		catalog['f_spire_250'][np.where(catalog['250_nongauss'])] = 0
		catalog['f_spire_350'][np.where(catalog['350_nongauss'])] = 0
		catalog['f_spire_500'][np.where(catalog['500_nongauss'])] = 0
	# mask sources which aren't flagged because of non-detection, but because of something else
	if mask_bad_bright_sources:
		bad250idxs = np.where((np.logical_not(catalog['250_nongauss'])) & (catalog['flag_spire_250']))
		catalog['f_spire_250'][bad250idxs] = np.nan
		catalog['f_spire_350'][np.where((np.logical_not(catalog['350_nongauss'])) & (catalog['flag_spire_350']))] = \
			np.nan
		catalog['f_spire_500'][np.where((np.logical_not(catalog['500_nongauss'])) & (catalog['flag_spire_500']))] = \
			np.nan


	#cigale_names, cigale_err_names, fullfilters, fullerrfilters = convert_filternames(filternames, fkey='f_best_')
	# calculate rest frame nu F_nu
	obs_wavelengths = peak_wavelengths_from_filters(filternames)
	rest_wavelengths = np.outer(1 / (1 + zs), obs_wavelengths)
	obs_freqs = (con.c / (np.array(obs_wavelengths) * u.micron)).to('Hz').value

	fluxes = catalog[fullfilternames]
	fluxerrs = catalog[fullerrnames]

	newfluxes, newfluxerrs = [], []
	for j in range(len(fluxes)):
		newfluxes.append(np.array(list(fluxes[j])).astype(np.float64))
		newfluxerrs.append(np.array(list(fluxerrs[j])).astype(np.float64))

	fluxes, fluxerrs = np.array(newfluxes), np.array(newfluxerrs)

	obs_nu_f_nu = obs_freqs * fluxes
	obs_nu_ferr_nu = obs_freqs * fluxerrs

	return rest_wavelengths, obs_nu_f_nu, obs_nu_ferr_nu


def normalize_seds(secure_redshift_idx, rest_wavelengths, obs_nu_f_nu, obs_nu_ferr_nu, rest_lambda):
	remove_outliers = False
	interp_lum_list = []
	for j in range(len(obs_nu_f_nu)):
		# should i try to figure out IRAC CH3, CH4 upper limits for objects undetected in those bands to interpolate?
		good_filters = np.where(np.isfinite(obs_nu_f_nu[j]))
		interp_lum_list.append(interpolate_tools.log_interp1d(rest_wavelengths[j][good_filters],
		                                                obs_nu_f_nu[j][good_filters])(rest_lambda))

	interp_lum_list = np.array(interp_lum_list)
	median_lum = np.median(interp_lum_list[secure_redshift_idx])
	lum_ratios = np.array(median_lum / interp_lum_list)

	if remove_outliers:
		non_outliers = np.where(lum_ratios < 5 * np.std(lum_ratios))
		print(non_outliers)
		lum_ratios = lum_ratios[non_outliers]
		obs_nu_f_nu = obs_nu_f_nu[non_outliers]
		obs_nu_ferr_nu = obs_nu_ferr_nu[non_outliers]
		rest_wavelengths = rest_wavelengths[non_outliers]

	obs_nu_f_nu = np.transpose(np.transpose(obs_nu_f_nu) * lum_ratios)
	obs_nu_ferr_nu = np.transpose(np.transpose(obs_nu_ferr_nu) * lum_ratios)


	#plotting.plot_each_sed(1, 1, rest_wavelengths, obs_nu_f_nu, obs_nu_ferr_nu)

	return rest_wavelengths, obs_nu_f_nu, obs_nu_ferr_nu, median_lum, lum_ratios


# composite SED
def construct_composite(rest_wavelengths, nu_f_nu, nu_f_nu_errs, rest_wavelength_bins, nondetection_sigma=1):
	import matplotlib.pyplot as plt
	from source import survival
	import importlib
	importlib.reload(survival)

	survival_analysis = True
	# keep track of which rest wavelength bin each flux falls into for each object
	wavelength_bin_idxs = []
	# for each source with redshift, find rest wavelength bin for each measured flux
	for j in range(len(nu_f_nu)):
		wavelength_bin_idxs.append(np.digitize(rest_wavelengths[j], rest_wavelength_bins))
	wavelength_bin_idxs = np.array(wavelength_bin_idxs)

	nsources = len(nu_f_nu)


	binned_nu_f_nu, binned_lower_errs, binned_upper_errs = [], [], []
	# for each rest frame wavelength bin
	for j in range(1, len(rest_wavelength_bins)):
		# find all fluxes falling in that bin
		inbinidxs = np.where(wavelength_bin_idxs == j)
		nfnu_in_bin = nu_f_nu[inbinidxs]
		nfnu_err_in_bin = nu_f_nu_errs[inbinidxs]
		# remove nans because they weren't observed in that band
		observed_idxs = np.where(np.isfinite(nfnu_in_bin))
		nfnu_in_bin = nfnu_in_bin[observed_idxs]
		nfnu_err_in_bin = nfnu_err_in_bin[observed_idxs]

		# don't take median if < 20% of sources were observed in this bin
		if len(nfnu_in_bin) / nsources < 0.2:
			binned_nu_f_nu.append(np.nan), binned_lower_errs.append(np.nan), binned_upper_errs.append(np.nan)

		else:
			if survival_analysis:

				# separate non-detections
				nondetected_flag = (nfnu_in_bin / nfnu_err_in_bin < nondetection_sigma)

				print('Detection fraction is %s \n' % (len(np.where(nondetected_flag == False)[0]) / len(
					nfnu_in_bin)))

				"""plt.close('all')
				plt.figure(figsize=(8,7))
				nongausslums = nfnu_in_bin[np.where(nondetected_flag)]
				gausslums = nfnu_in_bin[np.where(np.logical_not(nondetected_flag))]
				nondetectedlums = nondetectedlums[np.where(nondetectedlums > 0)]
				plt.hist(np.log10(nondetectedlums), color='k', histtype='step', label='Non-detections')
				plt.hist(np.log10(detectedlums), color='g', histtype='step', label='Detections')
				plt.xlabel(r'Normalized $\nu F_{\nu}$', fontsize=20)
				plt.legend(fontsize=20)
				plt.yscale('log')

				plt.savefig('plots/seds/dists/%s.pdf' % rest_wavelength_bins[j])
				plt.close('all')"""



				survival_median, lowerbound, upperbound = survival.km_median(nfnu_in_bin, nondetected_flag,
				                                                          censorship='upper', return_errors=True)
				if survival_median > 0:
					binned_nu_f_nu.append(survival_median), binned_lower_errs.append(lowerbound)
					binned_upper_errs.append(upperbound)

				else:
					upperlimit = 3*np.median(nfnu_err_in_bin)
					binned_nu_f_nu.append(upperlimit)
					binned_lower_errs.append(np.inf)
					binned_upper_errs.append(0)
			else:
				realizations = []
				binned_nu_f_nu.append(np.nanmedian(nfnu_in_bin))
				for k in range(50):
					bootidxs = np.random.choice(len(nu_f_nu), len(nu_f_nu))
					bootnufnu = nu_f_nu[bootidxs]
					bootwavelength_bin_idxs = wavelength_bin_idxs[bootidxs]
					bootnufnu_in_bin = bootnufnu[np.where(bootwavelength_bin_idxs == j)]
					realizations.append(np.nanmedian(bootnufnu_in_bin))
				err = np.std(realizations)
				binned_lower_errs.append(err), binned_upper_errs.append(err)



	return binned_nu_f_nu, binned_lower_errs, binned_upper_errs


def measure_composites(wavelength_bins):


	mastercat = Table.read('../data/HELP/master_agn_catalog.fits')

	nbins = np.int(np.max(mastercat['bin']))
	binned_fluxs, fluxerrs, binnedseds, lowerrs, uperrs = [], [], [], [], []


	custom_pivot_wavelengths = interpolate_tools.bin_centers(wavelength_bins, method='geo_mean')
	restfreqs = (con.c / (custom_pivot_wavelengths * u.micron)).to('Hz').value


	# keep track of normalization factors
	all_lum_ratios = []
	constmedlum, inverse_square_factor = 0, 0
	# for each sample of quasars
	for j in range(nbins):
		binlumratios = []
		binnedcat = mastercat[np.where(mastercat['bin'] == j+1)]
		restlam, obsnu, obsnuerr = rest_frame_seds(best_filterset, binnedcat)

		# determine source with a secure spectroscopic redshift at z=1 to use as luminosity normalizer
		secure_redshift_idxs = np.where((binnedcat['zspec_qual'] >= 3))

		secure_typical_redshift_idx = np.abs(binnedcat[secure_redshift_idxs]['zspec'] - 1.).argmin()
		secure_typical_redshift = binnedcat[secure_redshift_idxs]['zspec'][secure_typical_redshift_idx]
		secure_typical_redshift_idx = np.where(binnedcat['zspec'] == secure_typical_redshift)

		# normalize SEDs at 6 micron
		restlam, obsf, obserr, medlum, lum_ratios = normalize_seds(secure_typical_redshift_idx, restlam, obsnu,
		                                                           obsnuerr, 6.)
		# if considering the unobscured bin, take the source closest to z=1 as the luminosity normalizer
		if j == 0:
			constmedlum = medlum
			inverse_square_factor = 4 * np.pi * apcosmo.luminosity_distance(secure_typical_redshift).to('m').value ** 2

		# if considering the obscsured sample, use the previously determined luminosity normalizer
		else:
			norm_between_bins = constmedlum / medlum
			obsf *= norm_between_bins
			obserr *= norm_between_bins
			lum_ratios *= norm_between_bins
		binlumratios.append(lum_ratios)
		all_lum_ratios.append(binlumratios)

		sed, loerr, hierr = construct_composite(restlam, obsf, obserr, wavelength_bins)
		sed_flux = np.array(sed) / restfreqs / 1000.

		binned_fluxs.append(sed_flux), binnedseds.append(sed), lowerrs.append(loerr), uperrs.append(hierr)
		fluxerrs.append(sed_flux * np.array(loerr) / np.array(sed))

	binnedseds = np.array(binnedseds)
	eff_wavelengths = interpolate_tools.bin_centers(wavelength_bins, method='geo_mean')

	import plotting
	unit_conversion = (u.uJy * u.Hz * (u.m) ** 2).to('erg/s')
	plotting.plot_luminosity_distributions(constmedlum * inverse_square_factor * unit_conversion, all_lum_ratios)
	plotting.plot_composite_sed(nbins, eff_wavelengths, binnedseds, lowerrs, uperrs)

	np.array(wavelength_bins).dump('composite_seds/wavelength_bins.npy')
	np.array(binnedseds).dump('composite_seds/binned_seds.npy')
	np.array(binned_fluxs).dump('composite_seds/binned_fluxes.npy')
	np.array(fluxerrs).dump('composite_seds/flux_errs.npy')
	np.array([lowerrs, uperrs]).dump('composite_seds/sed_errs.npy')


def fit_composites(xray=False, radio=False, vary_inclination=False, vary_tau=True, vary_galaxy_extinction=True,
	vary_burst_age=True, vary_stellar_age=True, vary_opening_angle=True, disktype='0'):
	wavelength_bins = np.load('composite_seds/wavelength_bins.npy', allow_pickle=True)
	binnedseds = np.load('composite_seds/binned_fluxes.npy', allow_pickle=True)

	binnedseds[np.where(np.logical_not(np.isfinite(binnedseds)))] = -9999.

	errs = np.load('composite_seds/flux_errs.npy', allow_pickle=True)
	errs[np.where(np.logical_not(np.isfinite(errs)))] = -9999.
	dl = apcosmo.luminosity_distance(1.).value

	for k in range(len(binnedseds)):

		with open('cigale_files/cigale_input_%s.txt' % (k + 1), 'w') as f:
			collist = ['id ', 'redshift ', 'distance ']
			collist += ['cust_%s ' % j for j in range(len(wavelength_bins)-1)]
			collist += ['cust_%s_err ' % j for j in range(len(wavelength_bins)-1)]
			collist += ['\n']
			f.writelines(collist)

			datalist = ['%s ' % (k+1), '0. ', '%s ' % dl]
			#print(['%s ' % j for j in lowerrs[k]])
			datalist += ['%s ' % j for j in binnedseds[k]]
			datalist += ['%s ' % j for j in errs[k]]
			datalist += ['\n']
			f.writelines(list(datalist))

	from fileinput import FileInput
	import sys

	curdir = os.getcwd()
	os.chdir('cigale_files')

	for k in range(len(binnedseds)):

		# if old ini file present, delete it
		if len(glob.glob('pcigale.ini')) > 0:
			print('removing old ini file')
			os.remove('pcigale.ini')
		if len(glob.glob('pcigale.ini.spec')) > 0:
			os.remove('pcigale.ini.spec')

		from pcigale.session.configuration import Configuration
		import pcigale

		# generate new ini file
		pcigale.init(Configuration())

		if xray:
			xkey = 'xray, '
		else:
			xkey = ''
		if radio:
			radkey = 'radio, '
		else:
			radkey = ''


		# edit ini file to contain correct modules
		with FileInput(files=['pcigale.ini'], inplace=True) as f:
			for line in f:
				if line.startswith('data_file'):
					line = 'data_file = cigale_input_%s.txt\n' % (k+1)
				if line.startswith('sed_modules'):
					line = 'sed_modules = sfhdelayed, bc03, nebular, dustatt_calzleit, dale2014, skirtor2016, %s%s ' \
					       'redshifting\n' % (xkey, radkey)
				if line.startswith('analysis_method'):
					line = 'analysis_method = pdf_analysis\n'
				if line.startswith('cores'):
					line = 'cores = 1\n'
				sys.stdout.write(line)

		# have cigale process initial config file
		pcigale.genconf(Configuration())

		# now modify config file again to set parameter

		with FileInput(files=['pcigale.ini'], inplace=True) as f:
			for line in f:
				stripline = line.strip()
				if stripline.startswith('age_main ='):
					if vary_stellar_age:
						line = '    age_main = 1000, 2000, 5000\n'
				if stripline.startswith('age_burst ='):
					if vary_burst_age:
						line = '    age_burst = 20, 50\n'

				if stripline.startswith('E_BVs_young ='):
					if vary_galaxy_extinction:
						line = '    E_BVs_young = 0.01, 0.3, 0.7, 1.5\n'

				if stripline.startswith('t = '):
					if vary_tau:
						line = '    t = 3, 5, 7, 9, 11\n'

				if stripline.startswith('i ='):
					if k == 0:
						line = '    i = 0, 10, 20\n'
					else:
						line = '    i = 60, 70, 80, 90\n'
				if stripline.startswith('fracAGN = 0.1'):
					line = '    fracAGN = 0.99\n'
				if stripline.startswith('EBV'):
					line = '    EBV = 0.01\n'
				if stripline.startswith('lambda_fracAGN'):
					line = '    lambda_fracAGN = 2/20\n'
				if stripline.startswith('save_best_sed'):
					line = '  save_best_sed = True\n'
				if stripline.startswith('disk_type'):
					line = '  disk_type = %s\n' % disktype
				# if stripline.startswith('temperature = '):
				#	line = '  temperature = 200.0\n'
				if stripline.startswith('oa'):
					if vary_opening_angle:
						line = '    oa=10, 20, 30, 40, 50, 60\n'
					else:
						line = '    oa=30\n'
				sys.stdout.write(line)

		pcigale.run(Configuration())

		os.system('pcigale-plots sed --type=lum')


	os.chdir(curdir)



custom_wavelengthbins = np.array(list(np.logspace(-1.1, 0.3, 30)) + \
                        list(np.logspace(0.3, 1, 5))[1:] + \
                        list(np.logspace(1.6, 2.5, 10)))

def fake_filters(fake_bins):
	import glob
	import os
	oldfilters = glob.glob('../cigale/database_builder/filters/cust*')
	for filt in oldfilters:
		os.remove(filt)

	fake_bins = fake_bins * 10000.
	for j in range(len(fake_bins)-1):
		fakewavelengths = np.linspace(fake_bins[j], fake_bins[j+1], 10)
		with open('../cigale/database_builder/filters/cust_%s.dat' % j, 'w') as f:
			f.write('# cust_%s\n' % j)
			f.write('# energy\n')
			f.write('# cust_%s\n' % j)
			for j in range(len(fakewavelengths)):
				f.write(' %s   %s\n' % (round(fakewavelengths[j], 2), 1.))


#fake_filters(custom_wavelengthbins)

if __name__ == "__main__":
	prepare_help_data(good_fields)
	measure_composites(custom_wavelengthbins)
	#tap_donley_agn('bootes')
	#fit_composites()