import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import healpy as hp


# confine a large set of coordinates to roughly the same area as a smaller set of coordinates
def crude_cutout_of_catalog(input_lons, input_lats, lons_region, lats_region, nside=16):
	hpidxs = np.unique(hp.ang2pix(nside=nside, theta=lons_region, phi=lats_region, lonlat=True))
	input_idxs = hp.ang2pix(nside=nside, theta=input_lons, phi=input_lats, lonlat=True)

	idxsinregion = np.in1d(input_idxs, hpidxs)
	return idxsinregion

