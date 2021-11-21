from astropy.table import Table
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

def reliability_test():
	chandra_cosmos_cat = Table.read('catalogs/c-cosmos_marchesi16.fits')
	xraycoords = SkyCoord(ra=chandra_cosmos_cat['RA'], dec=chandra_cosmos_cat['DEC'])
	wise_cosmos_cat = Table.read('catalogs/catwise_cosmos_chandra_foot.fits')
	#wise_cosmos_cat = wise_cosmos_cat[np.where((wise_cosmos_cat['w2sigmpro'] < 0.2) & (wise_cosmos_cat['w1mpro'] <
	#                                                                                   17))]

	wisecoords = SkyCoord(ra=wise_cosmos_cat['ra'], dec=wise_cosmos_cat['dec'])

	wiseidx, xrayidx, d2d, d3d = xraycoords.search_around_sky(wisecoords, 5*u.arcsec)

	wise_xray_detected_cat = wise_cosmos_cat[wiseidx]

	#wise_nondetected_cat = wise_cosmos_cat[np.logical_not(np.isin(np.arange(len(wise_cosmos_cat)), wiseidx))]


	brightw2cut, faintw2cut = 14, 18
	w2_cuts = np.linspace(brightw2cut, faintw2cut, 5)
	w2_cuts = [17]

	reliability_matrix = []

	ncolorbins = 10

	for j in range(len(w2_cuts)):
		w2_limited_xray_detected = wise_xray_detected_cat[np.where(wise_xray_detected_cat['w2mpro'] < w2_cuts[j])]
		#w2hist_xray_detected = np.histogram(w2_limited_xray_detected['w1mpro'], bins=10, range=(10, w2_cuts[j]+0.1),
		#                                density=True)

		w2limited_all = wise_cosmos_cat[np.where(wise_cosmos_cat['w2mpro'] < w2_cuts[j])]

		non_detected_probs = w2hist_xray_detected[0][np.digitize(w2limited_all['w2mpro'],
		                                bins=w2hist_xray_detected[1])-1]

		nondetected_w2_matched = w2limited_all[np.random.choice(len(w2limited_all),
		                                size=len(w2limited_all),
		                                p=non_detected_probs/np.sum(non_detected_probs))]

		nondetected_w2_matched = w2limited_all


		w1w2hist_nondetect = np.histogram(nondetected_w2_matched['w1mpro'] - nondetected_w2_matched['w2mpro'],
		                                bins=ncolorbins, range=(0., 1.5))


		w1w2hist_detect = np.histogram(w2_limited_xray_detected['w1mpro'] - w2_limited_xray_detected['w2mpro'],
		                               bins=ncolorbins, range=(0., 1.5))



		dist_sum = w1w2hist_detect[0] + w1w2hist_nondetect[0]

		reliability = w1w2hist_detect[0]/w1w2hist_nondetect[0]

		reliability_matrix.append(reliability)


	plt.figure(figsize=(8, 6))
	if len(reliability_matrix) > 1:
		plt.imshow(np.rot90(reliability_matrix), aspect='auto', extent=(brightw2cut, faintw2cut, 0., 1.5))
		plt.xlabel('W2')
		plt.ylabel('W1-W2')
		plt.colorbar()
	else:
		plt.scatter(np.linspace(0., 1.5, ncolorbins), reliability_matrix[0])
		plt.ylabel('Reliability')
		plt.xlabel('W1-W2')
		plt.ylim(0, 1)
	plt.savefig('plots/reliability.pdf')
	plt.close('all')



reliability_test()





