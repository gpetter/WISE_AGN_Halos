import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt


def get_fir_cutouts(field, ploteach=False, cutout_size=60, bin=None):
	#agncat = Table.read('catalogs/derived/catwise_binned.fits')
	agncat = Table.read('../data/HELP/master_agn_catalog.fits')
	if bin is not None:
		agncat = agncat[np.where(agncat['bin'] == bin)]
	agncat = agncat[np.where(agncat['Z_best'] < 1.5)]
	agncat.rename_columns(['ra', 'dec'], ['RA', 'DEC'])

	if field == 'UDS':
		goodidxs = np.where((agncat['RA'] < 36) & (agncat['RA'] > 33) & (agncat['DEC'] > -6) & (agncat['DEC'] < -4))
		lotss_file = fits.open('../data/FIR_surveys/maps/SCUBA2_CLS/matched_filtered/S2CLS_UDS_MF_FLUX_DR1.FITS')
	elif field == 'NEPSC2':
		goodidxs = np.where((agncat['RA'] < 272) & (agncat['RA'] > 266.5) & (agncat['DEC'] > 65.5) & (agncat['DEC'] <
		                                                                                             68))
		lotss_file = fits.open('../data/FIR_surveys/maps/NEPSC2_SCUBA/NEPSC2_mos_mf.fits')
	elif field == 'COSMOS':
		goodidxs = np.where((agncat['RA'] < 151) & (agncat['RA'] > 149) & (agncat['DEC'] > 1.4) & (agncat['DEC'] <
		                                                                                             3.1))
		lotss_file = fits.open('../data/FIR_surveys/maps/SCUBA2_CLS/matched_filtered/S2CLS_COSMOS_MF_FLUX_DR1.FITS')
	elif field == 'Lockman':
		goodidxs = np.where((agncat['RA'] < 162.1) & (agncat['RA'] > 160.9) & (agncat['DEC'] > 58.7) & (agncat['DEC'] <
		                                                                                             59.3))

		lotss_file = fits.open('../data/FIR_surveys/maps/SCUBA2_CLS/matched_filtered/S2CLS_LockmanHoleNorth_MF_FLUX_DR1.FITS')

	elif field == 'SSA':
		goodidxs = np.where((agncat['RA'] < 334.7) & (agncat['RA'] > 334) & (agncat['DEC'] > 0) & (agncat['DEC'] <
		                                                                                             .6))

		lotss_file = fits.open('../data/FIR_surveys/maps/SCUBA2_CLS/matched_filtered/S2CLS_SSA22_MF_FLUX_DR1.FITS')


	agncat = agncat[goodidxs]
	agncoords = SkyCoord(agncat['RA'] * u.deg, agncat['DEC'] * u.deg)

	lotss_img = lotss_file[0].data
	pixsize = round(lotss_file[0].header['CDELT2'] * 3600, 1)     # arcsec / pix

	halfwidth = int(cutout_size / pixsize / 2.)
	w = WCS(lotss_file[0].header, naxis=2)
	cutouts = []

	nsources = 0

	for j in range(len(agncat)):
		try:
			sourcpix = w.world_to_pixel(agncoords[j])
			imgbounds = int(sourcpix[0]+1 + halfwidth), \
			            int(sourcpix[0]+1 - halfwidth), \
			            int(sourcpix[1]+1 + halfwidth), \
			            int(sourcpix[1]+1 - halfwidth)

			cutout = lotss_img[imgbounds[3]:imgbounds[2], imgbounds[1]:imgbounds[0]]

			if (cutout.shape[0] != halfwidth*2) or (cutout.shape[1] != halfwidth*2):
				continue

			if ploteach:
				plt.close('all')
				plt.figure(figsize=(8,7))
				plt.imshow(cutout,  extent=[-pixsize * cutout.shape[1]/2.,
				                            pixsize * cutout.shape[1]/2.,
				                            -pixsize * cutout.shape[0]/2.,
				                            pixsize * cutout.shape[0]/2.])

				plt.xlabel(r'$\theta$ [arcsec]', fontsize=20)
				plt.ylabel(r'$\theta$ [arcsec]', fontsize=20)
				plt.savefig('plots/lotss_deep_cutouts/%s.pdf' % j)

				plt.close('all')
			cutouts.append(cutout)
			nsources += 1
		except Exception as e:
			print(e)
			continue
	print('Number sources in field: ', nsources)
	return cutouts, pixsize


def stack_cutouts(bin, fields, mode='median'):
	cutouts = []
	for field in fields:
		cutoutsfield, pixsize = get_fir_cutouts(field, ploteach=False, cutout_size=120, bin=bin)
		print(np.shape(cutoutsfield))
		cutouts += cutoutsfield

	if mode == 'median':
		stack = np.nanmedian(np.array(cutouts), axis=0)
	elif mode == 'mean':
		stack = np.nanmean(np.array(cutouts), axis=0)

	print(np.max(stack))

	plt.close('all')
	plt.figure(figsize=(8, 7))
	plt.imshow(stack, extent=[-pixsize * stack.shape[1] / 2.,
	                           pixsize * stack.shape[1] / 2.,
	                           -pixsize * stack.shape[0] / 2.,
	                           pixsize * stack.shape[0] / 2.])
	plt.xlabel(r'$\theta$ [arcsec]', fontsize=20)
	plt.ylabel(r'$\theta$ [arcsec]', fontsize=20)
	plt.savefig('plots/fir_stacks/stack_%s.pdf' % bin)

	plt.close('all')

stack_cutouts(2, ['UDS'], mode='median')