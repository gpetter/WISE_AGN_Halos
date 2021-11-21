from astropy.coordinates import SkyCoord
import healpy as hp
import astropy.units as u
import numpy as np


def sky_transform(lons, lats, trans):
	transdict = {'G': 'galactic', 'E': 'geocentricmeanecliptic', 'C': 'icrs'}

	incoords = SkyCoord(lons * u.deg, lats * u.deg, frame=transdict[trans[0]])
	outcoords = incoords.transform_to(frame=transdict[trans[1]])

	if trans[1] == 'G':
		return outcoords.l.value, outcoords.b.value
	elif trans[1] == 'E':
		return outcoords.lon.value, outcoords.lat.value
	else:
		return outcoords.ra.value, outcoords.dec.value
