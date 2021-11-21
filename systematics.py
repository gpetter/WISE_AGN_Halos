import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import importlib
import plotting
from source import coord_transforms
importlib.reload(coord_transforms)
importlib.reload(plotting)



def density_by_coord(coords, randcoords, coordframe, lonlat, nbins):
	if coordframe == 'C':
		lon, lat = coords.ra, coords.dec
		randlon, randlat = randcoords.ra, randcoords.dec
	elif coordframe == 'G':
		lon, lat = coords.galactic.l, coords.galactic.b
		randlon, randlat = randcoords.galactic.l, randcoords.galactic.b
	elif coordframe == 'E':
		lon, lat = coords.geocentricmeanecliptic.lon, coords.geocentricmeanecliptic.lat
		randlon, randlat = randcoords.geocentricmeanecliptic.lon, randcoords.geocentricmeanecliptic.lat
	else:
		return

	norm = float(len(randcoords) ) / len(coords)

	if lonlat == 'lon':
		vals, randvals = lon.value, randlon.value
	elif lonlat == 'lat':
		vals, randvals = lat.value, randlat.value
	else:
		return

	maxval, minval = np.max(randvals), np.min(randvals)

	bins = np.linspace(minval-0.001, maxval +0.001, nbins)

	idxs, randidx = np.digitize(vals, bins), np.digitize(randvals, bins)

	ratios = []

	for j in range(1, nbins):
		try:
			ratios.append(len(np.where(idxs == j)[0]) / len(np.where(randidx == j)[0]) * norm)

		except:
			ratios.append(np.nan)

	plotting.density_vs_coord(ratios, bins, coordframe, lonlat)


t = Table.read('catalogs/derived/catwise_r90_filtered.fits')
rand = Table.read('catalogs/derived/ls_randoms_1_masked.fits')

tcoord = SkyCoord(t['RA'] * u.deg, t['DEC'] * u.deg)
rcoord = SkyCoord(rand['RA'] * u.deg, rand['DEC'] * u.deg)

density_by_coord(tcoord, rcoord, 'E', 'lat', 50)

