
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle
import healpy as hp
import numpy as np
import healpixhelper
import importlib
from source import coord_transforms
importlib.reload(healpixhelper)

# take a binary mask and properly downgrade it to a lower resolution, expanding the mask to cover all pixels which
# touch bad pixels in the high res map
def downgrade_mask(mask, newnside):
	mask_lowres_proper = hp.ud_grade(mask.astype(float), nside_out=newnside).astype(float)
	mask_lowres_proper = np.where(mask_lowres_proper == 1., True, False).astype(bool)
	return mask_lowres_proper

# read in tons of randoms, make healpix maps where each pixel is average R depth, E(B-V), or WISE bitmask values
def ls_depth_mask(nside, galactic=False):

	# read first random catalog
	tab = Table.read('catalogs/randoms/ls_randoms/ls_randoms_1.fits')
	tab = tab['RA', 'DEC', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'EBV']


	if galactic:
		#lons, lats = healpixhelper.equatorial_to_galactic(tab['RA'], tab['DEC'])
		lons, lats = tab['l'], tab['b']
	else:
		lons, lats = tab['RA'], tab['DEC']

	# calculate average depth field from randoms
	avg_depth = healpixhelper.healpix_average_in_pixels(lons, lats, nside, tab['PSFDEPTH_R'])
	lons_zero_depth, lats_zero_depth = list(lons[np.where(tab['PSFDEPTH_R'] == 0)]), \
	                                   list(lats[np.where(tab['PSFDEPTH_R'] == 0)])
	# and average E(B-V) values
	ebvs = healpixhelper.healpix_average_in_pixels(lons, lats, nside, tab['EBV'])
	hp.write_map('masks/ls_depth.fits', avg_depth, overwrite=True)
	hp.write_map('masks/ebv.fits', ebvs, overwrite=True)

	#lsmask = healpixhelper.healpix_average_in_pixels(lons, lats, 1024, tab['MASKBITS'])
	del tab


	# loop through many random catalogs, keep running total of pixels with masked randoms inside
	# need to use many millions of randoms to properly sample bad pixels
	"""for j in range(2, 5):
		newtab = Table.read('catalogs/randoms/ls_randoms/ls_randoms_%s.fits' % j)
		newtab = newtab['RA', 'DEC', 'PSFDEPTH_R']

		if galactic:
			lons, lats = healpixhelper.equatorial_to_galactic(newtab['RA'], newtab['DEC'])
		else:
			lons, lats = newtab['RA'], newtab['DEC']

		#lsmask += healpixhelper.healpix_average_in_pixels(lons, lats, 1024, newtab['MASKBITS'])
		lons_zero_depth += list(lons[np.where(newtab['PSFDEPTH_R'] == 0)])
		lats_zero_depth += list(lats[np.where(newtab['PSFDEPTH_R'] == 0)])


	zero_depth_map = healpixhelper.healpix_density_map(lons_zero_depth, lats_zero_depth, 1024)

	lsmask = hp.ud_grade(lsmask, nside)
	


	#hp.write_map('masks/zerodepth_mask.fits', zero_depth_map, overwrite=True)
	#hp.write_map('masks/ls_badmask.fits', lsmask, overwrite=True)"""




# deprecated: masking circular regions near objects chosen by hand
def mask_near_sources(nside, sources):
	src_tab = Table.read('masks/assef_cats/%s.fits' % sources)
	print(sources)
	hpxmask = np.ones(hp.nside2npix(nside))

	lons, lats = healpixhelper.equatorial_to_galactic(src_tab['RA'], src_tab['DEC'])
	vecs = hp.ang2vec(lons, lats, lonlat=True)
	for j in range(len(src_tab)):
		hpxmask[hp.query_disc(nside, vecs[j], src_tab['mask_radius'][j] * np.pi / 180., inclusive=True)] = 0

	return hpxmask





# replicate masking procedure used in Assef+18 to remove regions of likely contamination by red objects
# including planetary nebulae, nearby galaxies, HII regions
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


	"""passesallmasks = (mask * mask_near_sources(nside, 'pne') * mask_near_sources(nside, 'xsc') * mask_near_sources(
		nside, 'HII') * mask_near_sources(nside, 'lvg') * mask_near_sources(nside, 'LDN') * \
	                 mask_near_sources(nside, 'LBN')).astype(np.bool)"""
	passesallmasks = (mask_near_sources(nside, 'LDN') * mask_near_sources(nside, 'LBN')).astype(np.bool)


	hp.write_map('masks/assef.fits', passesallmasks, overwrite=True)


# mask regions around bright infrared stars
def mask_bright_stars(nside):
	bright_w3_cat = Table.read('catalogs/bright_sources/bright_w3.fits')
	very_bright_w3_cat = bright_w3_cat[np.where(bright_w3_cat['w3mpro'] < -1.5)]
	brightlons, brightlats = healpixhelper.equatorial_to_galactic(very_bright_w3_cat['RA'], very_bright_w3_cat['DEC'])
	vecs = hp.ang2vec(brightlons, brightlats, lonlat=True)
	mask = np.ones(hp.nside2npix(nside))
	for j in range(len(very_bright_w3_cat)):
		mask[hp.query_disc(nside, vecs[j], 0.5 * np.abs(very_bright_w3_cat['w3mpro'][j] * np.pi / 180.),
		                   inclusive=True)]	= 0

	hp.write_map('masks/bright_mask.fits', mask, overwrite=True)


def mask_ecliptic_caps(nside):
	mask = np.ones(hp.nside2npix(nside))
	l, b = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
	lon, lat = coord_transforms.sky_transform(lons=l, lats=b, trans=['G', 'E'])
	bad_pix = np.where(np.abs(lat) > 80)
	mask[bad_pix] = 0
	hp.write_map('masks/ecliptic_caps.fits', mask, overwrite=True)

def mask_saa_stripes(nside):
	mask = np.ones(hp.nside2npix(nside))
	l, b = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
	lon, lat = coord_transforms.sky_transform(lons=l, lats=b, trans=['G', 'E'])
	bad_pix = np.where((lon > 196) & (lon < 199) & (lat < 3) & (lat > -3))
	mask[bad_pix] = 0
	bad_pix = np.where((lon > 188) & (lon < 193) & (lat < 76) & (lat > 55))
	mask[bad_pix] = 0
	hp.write_map('masks/saa.fits', mask, overwrite=True)

def write_masks(nside):
	#ls_depth_mask(nside, galactic=True)
	assef_mask(nside)
	mask_bright_stars(nside)
	mask_ecliptic_caps(nside)
	mask_saa_stripes(nside)


def decode_bitmask(bitmasks):
	remainder = bitmasks
	for j in range(0, 14):
		largest_bit = np.floor(np.log2(bitmasks))
		remainder -= largest_bit


def total_mask(depth_cut, assef, unwise, planck, bright, ebv, ls_mask, zero_depth_mask, gal_lat_cut=0,
               area_lost_thresh=None, capmask=True, mask_saa=True):

	mask = hp.read_map('masks/ls_depth.fits', dtype=np.float64)
	mask[np.where(np.logical_not(np.isfinite(mask)))] = 0

	mask = (mask > depth_cut).astype(np.int)

	if assef:
		assefmask = hp.read_map('masks/assef.fits', dtype=np.float64)
		assefmask[np.where(np.logical_not(np.isfinite(assefmask)))] = 0
		newmask = (assefmask == 1).astype(np.int)
		mask = mask * newmask

	if unwise:
		print('masking unwise')
		wisemask = hp.read_map('masks/wisemask.fits')
		wisemask[np.where(np.logical_not(np.isfinite(wisemask)))] = 0
		newmask = (wisemask < 1).astype(np.int)
		mask = mask * newmask

	if planck:
		lensmask = hp.read_map('lensing_maps/planck/mask.fits')
		if len(lensmask) != len(mask):
			lensmask = downgrade_mask(lensmask, hp.npix2nside(len(mask)))
		mask = mask * lensmask

	if bright:
		brightmask = hp.read_map('masks/bright_mask.fits')
		mask[np.where(brightmask == 0)] = 0

	if ebv > 0:
		ebvmask = hp.read_map('masks/ebv.fits')
		mask[np.where(ebvmask > ebv)] = 0

	if ls_mask:
		lsbadmask = hp.read_map('masks/ls_badmask.fits')
		mask[np.where(lsbadmask > 0)] = 0

	if zero_depth_mask:
		zerodepth = hp.read_map('masks/zerodepth_mask.fits')
		mask[np.where(zerodepth > 0)] = 0

	if area_lost_thresh is not None:
		arealostmask = hp.read_map('masks/area_lost_combined.fits')
		mask[np.where(arealostmask > area_lost_thresh)] = 0

	if capmask:
		maskcap = hp.read_map('masks/ecliptic_caps.fits')
		newmask = (maskcap == 1).astype(np.int)
		mask = mask * newmask

	if mask_saa:
		maskcap = hp.read_map('masks/saa.fits')
		newmask = (maskcap == 1).astype(np.int)
		mask = mask * newmask

	if gal_lat_cut > 0:
		gallon, gallat = hp.pix2ang(hp.npix2nside(len(mask)), np.arange(len(mask)), lonlat=True)
		mask[np.where(np.abs(gallat) > gal_lat_cut)] = 0


	hp.write_map('masks/union.fits', mask, overwrite=True)


def mask_tab(tab):
	totmask = hp.read_map('masks/union.fits')

	#lons, lats = healpixhelper.equatorial_to_galactic(tab['RA'], tab['DEC'])
	lons, lats = tab['l'], tab['b']
	return tab[np.where(totmask[hp.ang2pix(hp.npix2nside(len(totmask)), lons, lats, lonlat=True)])]

