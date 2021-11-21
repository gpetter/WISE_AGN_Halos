
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle
import healpy as hp
import numpy as np
import healpixhelper
import importlib
importlib.reload(healpixhelper)

def downgrade_mask(mask, newnside):
	mask_lowres_proper = hp.ud_grade(mask.astype(float), nside_out=newnside).astype(float)
	mask_lowres_proper = np.where(mask_lowres_proper == 1., True, False).astype(bool)
	return mask_lowres_proper

# read in tons of randoms, make healpix maps where each pixel is average R depth, E(B-V), or WISE bitmask values
def ls_depth_mask(nside, galactic=False):

	# read first random catalog
	tab = Table.read('catalogs/randoms/ls_randoms/ls_randoms_1.fits')
	tab = tab['RA', 'DEC', 'PSFDEPTH_R', 'EBV', 'WISEMASK_W1', 'WISEMASK_W2']

	# append other random catalogs to first one
	for j in range(2, 5):
		newtab = Table.read('catalogs/randoms/ls_randoms/ls_randoms_%s.fits' % j)
		newtab = newtab['RA', 'DEC', 'PSFDEPTH_R', 'EBV', 'WISEMASK_W1', 'WISEMASK_W2']
		tab = vstack((tab, newtab))

	if galactic:
		lons, lats = healpixhelper.equatorial_to_galactic(tab['RA'], tab['DEC'])
	else:
		lons, lats = tab['RA'], tab['DEC']

	# calculate average depth field from randoms
	avg_depth = healpixhelper.healpix_average_in_pixels(lons, lats, nside, tab['PSFDEPTH_R'])
	# and average E(B-V) values
	ebvs = healpixhelper.healpix_average_in_pixels(lons, lats, nside, tab['EBV'])
	w1mask = healpixhelper.healpix_average_in_pixels(lons, lats, 1024, tab['WISEMASK_W1'])
	w2mask = healpixhelper.healpix_average_in_pixels(lons, lats, 1024, tab['WISEMASK_W2'])
	wisemask = hp.ud_grade(w1mask + w2mask, nside)


	hp.write_map('masks/ls_depth.fits', avg_depth, overwrite=True)
	hp.write_map('masks/ebv.fits', ebvs, overwrite=True)
	hp.write_map('masks/wisemask.fits', wisemask, overwrite=True)


def mask_bad_data(nside):
	ls, bs = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
	ras, decs = healpixhelper.galactic_to_equatorial(ls, bs)

	badidxlist = []
	with open('masks/diffraction_spikes.txt', 'r') as f:
		lines = f.readlines()

		for line in lines:
			line = line.split("\n")[0]
			pars = np.array(line.split(' ')).astype('float')

			idxs = np.where((ras > pars[0]) & (ras < pars[1]) & (decs > (ras * pars[3] + pars[2] - pars[4])) & (decs
			        < (ras * pars[3] + pars[2] + pars[4])))
			badidxlist += list(idxs[0])
	with open('masks/circular_masks.txt', 'r') as f:
		lines = f.readlines()

		for line in lines:
			line = line.split("\n")[0]
			pars = np.array(line.split(' ')).astype('float')

			idxs = np.where((ras > pars[0] - pars[2]) & (ras < pars[0] + pars[2]) & (decs > pars[1] - pars[2]) & (
				decs < pars[1] + pars[2]))
			badidxlist += list(idxs[0])

	mask = np.ones(hp.nside2npix(nside))
	mask[badidxlist] = 0
	return mask


def mask_near_sources(nside, sources):
	src_tab = Table.read('masks/assef_cats/%s.fits' % sources)
	print(sources)
	hpxmask = np.ones(hp.nside2npix(nside))

	lons, lats = healpixhelper.equatorial_to_galactic(src_tab['RA'], src_tab['DEC'])
	vecs = hp.ang2vec(lons, lats, lonlat=True)
	for j in range(len(src_tab)):
		hpxmask[hp.query_disc(nside, vecs[j], src_tab['mask_radius'][j] * np.pi / 180., inclusive=True)] = 0

	return hpxmask






def assef_mask(nside):
	#tabcoord = SkyCoord(tab['RA'] * u.deg, tab['DEC'] * u.deg)
	#ls, bs = tabcoord.galactic.l, tabcoord.galactic.b
	#center_dists = SkyCoord(0 * u.deg, 0 * u.deg, frame='galactic').separation(tabcoord)

	#tab = tab[np.where((center_dists.deg > 30.) & (np.abs(bs) > 10.*u.deg))]
	#newcoord = SkyCoord(tab['RA'] * u.deg, tab['DEC'] * u.deg)

	mask = np.ones(hp.nside2npix(nside))
	hpxlons, hpxlats = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
	center_dists = SkyCoord(0 * u.deg, 0 * u.deg).separation(SkyCoord(hpxlons * u.deg, hpxlats * u.deg))

	mask[np.where((np.abs(hpxlats) < 10) | (center_dists.value < 30))] = 0


	passesallmasks = (mask * mask_near_sources(nside, 'pne') * mask_near_sources(nside, 'xsc') * mask_near_sources(
		nside, 'HII') * mask_near_sources(nside, 'lvg') * mask_near_sources(nside, 'LDN') * \
	                 mask_near_sources(nside, 'LBN')).astype(np.bool)


	hp.write_map('masks/assef.fits', passesallmasks, overwrite=True)


"""def planck_mask(tab):
	lons, lats = healpixhelper.equatorial_to_galactic(tab['RA'], tab['DEC'])
	planckmask = hp.read_map('lensing_maps/planck/mask.fits')
	pix = hp.ang2pix(2048, lons, lats, lonlat=True)
	tab = tab[np.where(planckmask[pix] == 1)]
	return tab
"""
def write_masks(nside):
	ls_depth_mask(nside, galactic=True)
	assef_mask(nside)

def total_mask(depth_cut, assef, unwise, planck):

	mask = hp.read_map('masks/ls_depth.fits', dtype=np.float64)
	mask[np.where(np.logical_not(np.isfinite(mask)))] = 0

	mask = (mask > depth_cut).astype(np.int)

	if assef:
		assefmask = hp.read_map('masks/assef.fits', dtype=np.float64)
		assefmask[np.where(np.logical_not(np.isfinite(assefmask)))] = 0
		newmask = (assefmask == 1).astype(np.int)
		mask = mask * newmask

	if unwise:
		wisemask = hp.read_map('masks/wisemask.fits')
		wisemask[np.where(np.logical_not(np.isfinite(wisemask)))] = 0
		newmask = (wisemask < 1).astype(np.int)
		mask = mask * newmask

	if planck:
		lensmask = hp.read_map('lensing_maps/planck/mask.fits')
		if len(lensmask) != len(mask):
			lensmask = downgrade_mask(lensmask, hp.npix2nside(len(mask)))
		mask = mask * lensmask

	hp.write_map('masks/union.fits', mask, overwrite=True)


def mask_tab(tab):
	totmask = hp.read_map('masks/union.fits')

	lons, lats = healpixhelper.equatorial_to_galactic(tab['RA'], tab['DEC'])
	return tab[np.where(totmask[hp.ang2pix(hp.npix2nside(len(totmask)), lons, lats, lonlat=True)])]

