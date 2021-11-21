import numpy as np
import pandas as pd
from astropy import units as u
import astropy.constants as con
#from dust_extinction.averages import G03_SMCBar
import glob
from pyphot import unit
import matplotlib.pyplot as plt
from astropy.table import Table
import pyphot
lib = pyphot.get_library()
import matplotlib.cm as cm
import matplotlib as mpl


#ext = G03_SMCBar()



# convert spectrum in flux/wavelength to flux/frequency units
def f_lambda_to_f_nu(lambdas, f_lambda):
	f_lambda = f_lambda * u.ST
	nus = con.c / lambdas
	f_nu = lambdas * f_lambda / nus
	return f_nu.to('Jy')

# convert spectrum in flux/frequency to flux/wavelength units
def f_nu_to_f_lambda(lambdas, f_nu):
	f_nu = f_nu * u.Jy
	nus = con.c / lambdas
	f_lambda = nus * f_nu / lambdas
	return f_lambda.to(u.erg/u.s/u.Angstrom/(u.cm ** 2))


# pyphot wants spectra as function of wavelength (not frequency) and fluxes as f_lambda: energy/area/time/wavelength
# convert to these standards if necessary
def format_sed(x, y, x_unit=u.Angstrom, f_def='f_lambda'):
	if x_unit != u.Angstrom:
		x = (x * x_unit).to(u.Angstrom)
	else:
		x = x * x_unit

	if f_def != 'f_lambda':
		y = f_nu_to_f_lambda(x, y)

	return x.value, y.value

# read template in txt or fits format
def process_template(filename, format='txt', skiprows=None, colnames=['wavelength', 'flux', 'e_flux']):
	if format == 'txt':
		template_table = pd.read_csv(filename, skiprows=skiprows, header=None, names=colnames,
		                             delim_whitespace=True)
	elif format == 'fits':
		template_table = Table.read(filename)
	return template_table

# generate synthetic color from an sed at a given redshift
# blueband and redband are strings to indicate filters,
# get them from: https://mfouesneau.github.io/docs/pyphot/libcontent.html
def syn_color(rest_wavelengths, rest_sed, redshift, blueband, redband, vega=False):
	bluefilter, redfilter = lib[blueband], lib[redband]
	if vega:
		ab_2_vega = (redfilter.Vega_zero_mag - redfilter.AB_zero_mag) - \
					(bluefilter.Vega_zero_mag - bluefilter.AB_zero_mag)
	else:
		ab_2_vega = 0
	abcorr = redfilter.AB_zero_mag - bluefilter.AB_zero_mag

	# apply redshift
	shifted_wavelength = rest_wavelengths * (1 + redshift) * unit['AA']

	sed = rest_sed * unit['erg/s/cm**2/AA']

	blueflux, redflux = bluefilter.get_flux(shifted_wavelength, sed).value, redfilter.get_flux(shifted_wavelength,
																							   sed).value

	return -2.5 * np.log10(blueflux / redflux) + ab_2_vega + abcorr


# read in templates, calculate colors for given bands, at given range of redshifts
def get_colors(minz, maxz, nzs, blueband, redband, vega=False):
	template_list = sorted(glob.glob('../Templates/*.txt'))
	zlist = np.linspace(minz, maxz, nzs)

	template_colors = []
	# for each template
	for template in template_list:

		colors_for_zs = []
		# read data from file
		template_table = process_template(template, skiprows=3)
		# convert to right units
		temp_wavelengths, temp_fluxes = np.array(template_table['wavelength']), np.array(template_table['flux'])
		rest_wavelengths, rest_fluxes = format_sed(temp_wavelengths, temp_fluxes, u.um, f_def='f_nu')

		# get color at each redshift
		for z in zlist:
			colors_for_zs.append(syn_color(rest_wavelengths, rest_fluxes, z, blueband, redband, vega))

		template_colors.append(colors_for_zs)

	return template_colors


# W1-W2 vs W2-W3 color space
def jarrett_color_space(minz, maxz, nz):

	w1w2colors = get_colors(minz, maxz, nz, 'WISE_RSR_W1', 'WISE_RSR_W2', vega=True)
	w2w3colors = get_colors(minz, maxz, nz, 'WISE_RSR_W2', 'WISE_RSR_W3', vega=True)

	#symbols = ['*', 's', 'o']
	#labels = ['$f_{AGN}$ = 0.0', '$f_{AGN} = 0.5$', '$f_{AGN} = 1.0$']
	#rainbowmap = cm.rainbow(np.linspace(minz, maxz, nz))
	rainbowmap = cm.turbo(np.linspace(0, 1, len(w1w2colors)))


	plt.figure(figsize=(12,10))

	#for j in range(len(w2w3colors)):
	#	sc = plt.scatter(w2w3colors[j], w1w2colors[j], c=rainbowmap[j], s=100)
	sc = plt.scatter(w2w3colors, w1w2colors, c=rainbowmap, s=200)

	cbar = plt.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cm.turbo))
	#cbar = plt.colorbar(sc)
	cbar.set_label('$f_{AGN}$', size=20)

	plt.xlabel('W2 - W3 (Vega)', fontsize=30)
	plt.tick_params('both', labelsize=20)
	plt.ylabel('W1 - W2 (Vega)', fontsize=30)
	plt.savefig('../plots/jarrett.pdf')
	plt.close('all')

jarrett_color_space(0,0,1)


