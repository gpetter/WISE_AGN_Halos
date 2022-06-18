from mocpy import MOC
import mhealpy as mhp
import healpy as hp
import numpy as np
import glob
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import healpixhelper


def make_unwise_bitmask(max_order, low_nside):
	from os.path import expanduser
	home = expanduser("~")
	bitmaskdir = home + '/Dartmouth/data/unwise_bitmask/'


	low_pix_area = hp.nside2pixarea(low_nside)      # steradians
	high_pix_area = (2.75 ** 2) / (206265 ** 2)     # converted unwise-bitmask pixel area to steradian

	area_lost_map = np.zeros(hp.nside2npix(low_nside))

	area_fraction = high_pix_area / low_pix_area

	# loop over parts of sky binned in ra by 100 degrees, otherwise takes more than ~5 GB RAM
	for i in range(4):
		formatted_ra = "{0:01d}".format(i)
		maskfiles = glob.glob(bitmaskdir+'unwise-'+formatted_ra+'*')
		#maskfiles = glob.glob(bitmaskdir+'unwise-'+'00'+'*')
		maskfiles = sorted(maskfiles)

		uniqs = []
		for maskfile in maskfiles:
			print(maskfile)
			# get sky positions of pixels with flags
			thisfile = fits.open(maskfile)
			thisheader = thisfile[0].header
			thiswcs = WCS(thisheader)
			thismap = thisfile[0].data
			badpix = np.where((thismap > 0) & (thismap != 64))
			maskcoords = thiswcs.pixel_to_world(badpix[1], badpix[0])
			# get rid of things in the galactic plane
			ls, bs = maskcoords.galactic.l.value, maskcoords.galactic.b.value
			nongalidxs = np.where(np.abs(bs) > 9.5)
			maskcoords, ls, bs = maskcoords[nongalidxs], ls[nongalidxs], bs[nongalidxs]
			# create a MOC from flagged positions, convert to UNIQ format, list of UNIQ pixels, append to running total
			moc_for_file = MOC.from_skycoords(maskcoords, max_norder=max_order)
			uniqs += list(moc_for_file._uniq_format())

			# masking these small regions will cause sub-pixel masking in a low-res map, like you need when
			# correlating with CMB lensing
			# create a fraction of area lost per low resolution pixel map
			low_gal_pix = hp.ang2pix(low_nside, ls, bs, lonlat=True)
			pixcount = np.bincount(low_gal_pix, minlength=hp.nside2npix(low_nside))
			partial_mask_idxs = np.where(pixcount > 0)
			area_lost_map[partial_mask_idxs] = pixcount[partial_mask_idxs] * area_fraction


		# convert list of UNIQ pixels back to native mocpy format, write to file
		uniqs = np.array(np.unique(uniqs), dtype=np.int64)
		completemoc = MOC.from_valued_healpix_cells(uniqs, np.ones(len(uniqs)), cumul_to=np.inf)
		completemoc.write('masks/mocs/moc_%s.fits' % i, overwrite=True)

	# write the sub-pixel masking map
	hp.write_map('masks/area_lost.fits', area_lost_map, overwrite=True)



def assef_moc_mask():
	import healpixhelper
	maxorder = 15
	sources = 'HII'
	mEq = mhp.HealpixBase(order=maxorder)
	src_tab = Table.read('masks/assef_cats/%s.fits' % sources)

	lons, lats = healpixhelper.equatorial_to_galactic(src_tab['RA'], src_tab['DEC'])
	nongalidx = np.where(np.abs(lats) > 9.5)
	src_tab, lons, lats = src_tab[nongalidx], lons[nongalidx], lats[nongalidx]
	vecs = hp.ang2vec(lons, lats, lonlat=True)
	badpix = []
	for j in range(len(lons)):
		print(j)
		badpix += list(mEq.query_disc(vecs[j], src_tab['mask_radius'][j] * np.pi / 180., inclusive=True))

	np.array(badpix).dump('masks/assef_masks/%s.fits' % sources)
	#m = mhp.HealpixMap.moc_from_pixels(mEq.nside, badpix)
	#m.write_map('masks/assef_masks/%s_moc.fits' % sources, overwrite=True)



def overlap_w_bitmask(contam, order):

	badpix = np.load('masks/assef_masks/%s.fits' % contam, allow_pickle=True)

	badls, badbs = hp.pix2ang(hp.order2nside(order), badpix, lonlat=True)
	badras, baddecs = healpixhelper.galactic_to_equatorial(badls, badbs)
	badras, baddecs = badras * u.deg, baddecs * u.deg

	idxs_in_bitmask = []

	for j in range(4):
		moc = MOC.from_fits('masks/mocs/moc_%s.fits' % j)
		if j == 0:
			partialidxs = np.where((badras.value >= 350) | (badras.value < 120))
		elif j == 1:
			partialidxs = np.where((badras.value >= 90) | (badras.value < 210))
		elif j == 2:
			partialidxs = np.where((badras.value >= 190) | (badras.value < 310))
		else:
			partialidxs = np.where((badras.value >= 290))
		partbadras, partbaddecs = badras[partialidxs], baddecs[partialidxs]
		idxs_in_bitmask += list(np.where(moc.contains(partbadras, partbaddecs))[0])
	outside_bitmask_idxs = np.where(np.logical_not(np.in1d(np.arange(len(badls)), np.unique(idxs_in_bitmask))))
	return badls[outside_bitmask_idxs], badbs[outside_bitmask_idxs]





def combined_area_lost_mask():
	contamlist = ['xsc', 'pne', 'HII', 'lvg']
	orders = [14, 15, 15, 14]

	lost_area = hp.read_map('masks/area_lost.fits')
	nativenside = hp.npix2nside(len(lost_area))

	low_pix_area = hp.nside2pixarea(nativenside)  # steradians


	for j, contam in enumerate(contamlist):
		high_pix_area = hp.nside2pixarea(hp.order2nside(orders[j]))

		area_fraction = high_pix_area / low_pix_area

		badls, badbs = overlap_w_bitmask(contam, orders[j])
		newbadpix = hp.ang2pix(nativenside, badls, badbs, lonlat=True)

		pixcount = np.bincount(newbadpix, minlength=hp.nside2npix(nativenside))
		partial_mask_idxs = np.where(pixcount > 0)
		lost_area[partial_mask_idxs] += (pixcount[partial_mask_idxs] * area_fraction)

	lost_area[np.where(lost_area > 1.)] = 1.
	hp.write_map('masks/area_lost_combined.fits', lost_area, overwrite=True)



# mask data or random catalog with high resolution MOCs from flagged WISE pixels and regions of contamination (Assef
# 2018)
def use_highres_masks(data=True):

	if data:
		catalog = Table.read('catalogs/catwise_r75_ls.fits')
		writename = 'catalogs/catwise_r75_ls_subpixmasked.fits'
	else:
		catalog = Table.read('catalogs/randoms/ls_randoms/ls_randoms_1.fits')
		writename = 'catalogs/ls_randoms_subpixmasked.fits'

	catalog['id'] = np.arange(len(catalog))


	badids = []
	for j in range(4):

		mocfile = MOC.from_fits('masks/mocs/moc_%s.fits' % j)
		if j == 0:
			partialcat = catalog[np.where((catalog['RA'] >= 350) | (catalog['RA'] < 120))]
		elif j == 1:
			partialcat = catalog[np.where((catalog['RA'] > 90) & (catalog['RA'] < 210))]
		elif j == 2:
			partialcat = catalog[np.where((catalog['RA'] > 190) & (catalog['RA'] < 310))]
		else:
			partialcat = catalog[np.where((catalog['RA'] > 290))]

		badids += list(partialcat['id'][mocfile.contains(partialcat['RA']*u.deg, partialcat['DEC']*u.deg)])

		del partialcat
	del mocfile

	catalog = catalog[np.where(np.logical_not(np.in1d(catalog['id'], badids)))]

	contamlist = ['xsc', 'pne', 'HII', 'lvg']
	orders = [14, 15, 15, 14]

	badidxs = []
	for j, contam in enumerate(contamlist):
		contambadpix = np.load('masks/assef_masks/%s.fits' % contam, allow_pickle=True)
		ls, bs = healpixhelper.equatorial_to_galactic(catalog['RA'], catalog['DEC'])
		pix = hp.ang2pix(hp.order2nside(orders[j]), ls, bs, lonlat=True)
		badidxs += list(np.where(np.in1d(pix, contambadpix))[0])

	catalog = catalog[np.where(np.logical_not(np.in1d(np.arange(len(catalog)), np.unique(badidxs))))]


	catalog.write(writename, format='fits', overwrite=True)


