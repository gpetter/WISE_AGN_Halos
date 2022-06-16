import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt


def get_lotss_deep_cutouts(field, ploteach=False, cutout_size=60, bin=None):
	agncat = Table.read('catalogs/derived/catwise_binned.fits')
	if bin is not None:
		agncat = agncat[np.where(agncat['bin'] == bin)]

	if field == 'bootes':
		goodidxs = np.where((agncat['RA'] < 222) & (agncat['RA'] > 214) & (agncat['DEC'] > 31) & (agncat['DEC'] < 38))
		lotss_file = fits.open('../data/radio_cats/lofar_deep/bootes_radio_image.fits')

	elif field == 'elais_n1':
		goodidxs = np.where((agncat['RA'] > 237) & (agncat['RA'] < 248) & (agncat['DEC'] > 52) & (agncat['DEC'] < 58))
		lotss_file = fits.open('../data/radio_cats/lofar_deep/en1_radio_image.fits')
	elif field == 'lockman':
		goodidxs = np.where((agncat['RA'] > 156) & (agncat['RA'] < 168) & (agncat['DEC'] > 55) & (agncat['DEC'] < 61))
		lotss_file = fits.open('../data/radio_cats/lofar_deep/lockman_radio_image.fits')

	agncat = agncat[goodidxs]
	agncoords = SkyCoord(agncat['RA'] * u.deg, agncat['DEC'] * u.deg)

	lotss_img = lotss_file[0].data[0][0]
	pixsize = lotss_file[0].header['CDELT2'] * 3600     # arcsec / pix
	halfwidth = int(cutout_size / pixsize / 2.)
	w = WCS(lotss_file[0].header, naxis=2)
	cutouts = []
	for j in range(100):
		try:
			sourcpix = w.world_to_pixel(agncoords[j])
			imgbounds = int(sourcpix[0]+1 + halfwidth), \
			            int(sourcpix[0]+1 - halfwidth), \
			            int(sourcpix[1]+1 + halfwidth), \
			            int(sourcpix[1]+1 - halfwidth)

			cutout = lotss_img[imgbounds[3]:imgbounds[2], imgbounds[1]:imgbounds[0]]
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
		except:
			return
	return cutouts, pixsize


def stack_cutouts(field, bin, mode='median'):
	cutouts, pixsize = get_lotss_deep_cutouts(field, ploteach=False, cutout_size=120, bin=bin)

	if mode == 'median':
		stack = np.median(np.array(cutouts), axis=0)
	elif mode == 'mean':
		stack = np.mean(np.array(cutouts), axis=0)

	print(np.max(stack))

	plt.close('all')
	plt.figure(figsize=(8, 7))
	plt.imshow(stack, extent=[-pixsize * stack.shape[1] / 2.,
	                           pixsize * stack.shape[1] / 2.,
	                           -pixsize * stack.shape[0] / 2.,
	                           pixsize * stack.shape[0] / 2.])
	plt.xlabel(r'$\theta$ [arcsec]', fontsize=20)
	plt.ylabel(r'$\theta$ [arcsec]', fontsize=20)
	plt.savefig('plots/lotss_deep_cutouts/stack_%s.pdf' % bin)

	plt.close('all')

stack_cutouts('lockman', 1, mode='median')