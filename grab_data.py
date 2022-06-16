import pyvo
import numpy as np
from astropy.table import Table
import time

def get_wise_mask():
	gallatbins = [(-91, -50), (-50, -25), (-25, -9), (9, 25), (25, 50), (50, 91)]
	service = pyvo.dal.TAPService("https://datalab.noirlab.edu/tap")

	for binpair in gallatbins:
		adql_unwise = """SELECT nest4096, w1ab_map, w2ab_map FROM catwise2020.main
						WHERE (w1ab_map > 0 OR w2ab_map > 0) AND (glat >= %s AND glat < %s)""" % (binpair[0],
		                                                                                          binpair[1])

		adql_allwise = """SELECT DISTINCT nest4096, w1cc_map, w2cc_map FROM catwise2020.main
								WHERE (w1cc_map > 0 OR w2cc_map > 0) AND (glat >= %s AND glat < %s)""" % (binpair[0],
		                                                                                          binpair[1])
		unwisequery = service.submit_job(adql_unwise)
		unwisequery.run()
		while unwisequery.phase == 'EXECUTING':
			time.sleep(30)
		unwise_result = unwisequery.fetch_result()


def download_meisner_unwise_bitmasks(dir):
	import os
	for j in range(359):
		os.system("""wget -r -l1 -nd --no-parent --reject="index.html*" -w 7 --random-wait -timeout 30 -c -nc -erobots=off https://faun.rc.fas.harvard.edu/unwise/release/unwise_catalog_dr1_bitmasks/"""  +
		          "{0:03d}".format(j))


def make_unwise_bitmask():
	from pixell import enmap
	from astropy.coordinates import SkyCoord
	from astropy import units as u
	import glob
	from mocpy import MOC



	maskfiles = glob.glob('bitmasks/*.fits')
	maskras, maskdecs = [], []
	for maskfile in maskfiles:
		thismap = enmap.read_fits(maskfile)
		if len(thismap) != 2048:
			print('error')
		maskcoords = enmap.pix2sky([2048, 2048], wcs=thismap.wcs, pix=np.where(thismap > 64))
		maskras += list(maskcoords[1])
		maskdecs += list(maskcoords[0])
	badcoords = SkyCoord(maskras * u.deg, maskdecs * u.deg)
	badmoc = MOC.from_skycoords(badcoords, max_norder=15)




make_unwise_bitmask()