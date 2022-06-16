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
from source import interpolate_tools
import importlib
importlib.reload(interpolate_tools)
import matplotlib.cm as cm
import matplotlib as mpl


#ext = G03_SMCBar()


filter_dict = {'u': 'SDSS_u', 'g': 'SDSS_g', 'r':'SDSS_r', 'i':'SDSS_i', 'z':'SDSS_z', 'y':'PS1_y', 'J':'2MASS_J',
               'H':'2MASS_H', 'Ks':'2MASS_Ks', 'W1':'WISE_RSR_W1', 'W2':'WISE_RSR_W2', 'W3':'WISE_RSR_W3',
               'W4':'WISE_RSR_W4'}

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
	return f_lambda.to(u.erg/u.s/u.Angstrom/(u.cm ** 2)).value


# pyphot wants spectra as function of wavelength (not frequency) and fluxes as f_lambda: energy/area/time/wavelength
# convert to these standards if necessary
def format_sed(x, y, x_unit=u.Angstrom, f_def='f_lambda'):
	x, y = np.array(x), np.array(y)
	if x_unit != u.Angstrom:
		x = (x * x_unit).to(u.Angstrom)
	else:
		x = x * x_unit



	if f_def != 'f_lambda':
		y = f_nu_to_f_lambda(x, y)


	return x.value, y

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
def syn_color(rest_wavelengths, rest_sed, redshift, blueband, redband, vega=False, grid_factor=None):
	bluefilter, redfilter = lib[blueband], lib[redband]
	if vega:
		ab_2_vega = (redfilter.Vega_zero_mag - redfilter.AB_zero_mag) - \
					(bluefilter.Vega_zero_mag - bluefilter.AB_zero_mag)
	else:
		ab_2_vega = 0
	abcorr = redfilter.AB_zero_mag - bluefilter.AB_zero_mag

	# if grid_factor > 1, create wavelength grid with finer sampling and interpolate SED onto that grid
	if grid_factor is not None:
		minwave, maxwave = np.min(rest_wavelengths), np.max(rest_wavelengths)
		newgrid = np.logspace(np.log10(minwave), np.log10(maxwave), int(len(rest_wavelengths)*grid_factor))
		interpltr = interpolate_tools.log_interp1d(rest_wavelengths, rest_sed)
		rest_wavelengths, rest_sed = newgrid, interpltr(newgrid)


	# apply redshift
	shifted_wavelength = rest_wavelengths * (1 + redshift) * unit['AA']

	# get sed into units that pyphot likes
	sed = rest_sed * unit['erg/s/cm**2/AA']

	blueflux, redflux = bluefilter.get_flux(shifted_wavelength, sed).value, redfilter.get_flux(shifted_wavelength,
																							   sed).value

	return -2.5 * np.log10(blueflux / redflux) + ab_2_vega + abcorr



def process_hickox_17():
	hickoxtable = Table.read('Templates/hickox_17_agn_sed.fits')
	wavelengths = 10 ** (hickoxtable['logWave'])
	type1fluxes = 10 ** (hickoxtable['logSED1'])
	type1flux_err = 10 ** (hickoxtable['e_logSED1'])
	type2fluxes = 10 ** (hickoxtable['logSED2'])
	type2flux_err = 10 ** (hickoxtable['e_logSED2'])

	foo, type1fluxes = format_sed(wavelengths, type1fluxes, x_unit=u.um, f_def='f_nu')
	wavelengths, type2fluxes = format_sed(wavelengths, type2fluxes, x_unit=u.um, f_def='f_nu')


	return wavelengths, type1fluxes, type1flux_err, type2fluxes, type2flux_err


def process_swire(template_name):
	swire_df = pd.read_csv('../data/templates/swire_library/%s_template_norm.sed' % template_name,
	                delim_whitespace=True, header=None, names=['lambda', 'f'])
	wavelengths, fluxes = format_sed(swire_df['lambda'], swire_df['f'], x_unit=u.Angstrom,
	                                 f_def='f_lambda')
	return wavelengths, fluxes

def get_colors_for_zs(wavelengths, sed, zspace, blueband, redband, vega=False, grid_factor=None):
	colors = []
	for z in zspace:
		colors.append(syn_color(wavelengths, sed, z, blueband, redband, vega=vega, grid_factor=grid_factor))
	return colors

def template_colors_by_z(zspace, filter1='W1', filter2='W3', grid_factor=None):
	wavelengths, sed1, e_sed1, sed2, e_sed2 = process_hickox_17()
	m82_wavelengths, m82_sed = process_swire('M82')
	#arp220_wavelengths, arp220_sed = process_swire('Arp220')
	ell5_wavelengths, ell5_sed = process_swire('Ell5')




	type1colors = get_colors_for_zs(wavelengths, sed1, zspace, filter_dict[filter1], filter_dict[filter2],
	                                vega=True, grid_factor=grid_factor)
	type2colors = get_colors_for_zs(wavelengths, sed2, zspace, filter_dict[filter1], filter_dict[filter2],
	                                vega=True, grid_factor=grid_factor)
	m82_colors = get_colors_for_zs(m82_wavelengths, m82_sed, zspace, filter_dict[filter1], filter_dict[filter2], vega=True)
	ell5_colors = get_colors_for_zs(ell5_wavelengths, ell5_sed, zspace, filter_dict[filter1], filter_dict[filter2], vega=True)

	plt.figure(figsize=(8,7))
	plt.plot(zspace, type1colors, c='b', label='H+17 QSO1')
	plt.plot(zspace, type2colors, c='r', label='H+17 QSO2')
	plt.plot(zspace, m82_colors, c='k', label='M82')
	plt.plot(zspace, ell5_colors, c='k', ls='--', label='Ell')
	plt.xlabel('Redshift $(z)$', fontsize=20)
	plt.ylabel('$%s - %s$ [Vega]' % (filter1, filter2), fontsize=20)
	plt.legend(fontsize=20)
	plt.savefig('plots/Template_color_vs_z.pdf')
	plt.close('all')
	return type1colors, type2colors


def two_color_space(x1filter='W2', x2filter='W3', y1filter='W1', y2filter='W2', vega=False):
	wavelengths, sed1, e_sed1, sed2, e_sed2 = process_hickox_17()
	m82_wavelengths, m82_sed = process_swire('M82')
	ell_wavelengths, ell_sed = process_swire('Ell5')
	qso_zspace = np.linspace(0, 3, 20)
	m82_zspace = np.linspace(0, 0.7, 10)
	highz_m82space = np.linspace(1, 3, 10)
	ell_zspace = np.linspace(1, 2, 10)




	xtype1 = get_colors_for_zs(wavelengths, sed1, qso_zspace, filter_dict[x1filter], filter_dict[x2filter], vega=vega,
	                           grid_factor=5)
	ytype1 = get_colors_for_zs(wavelengths, sed1, qso_zspace, filter_dict[y1filter], filter_dict[y2filter],
	                           vega=vega, grid_factor=5)

	xtype2 = get_colors_for_zs(wavelengths, sed2, qso_zspace, filter_dict[x1filter], filter_dict[x2filter],
	                           vega=vega, grid_factor=5)
	ytype2 = get_colors_for_zs(wavelengths, sed2, qso_zspace, filter_dict[y1filter], filter_dict[y2filter],
	                           vega=vega, grid_factor=5)

	x_m82 = get_colors_for_zs(m82_wavelengths, m82_sed, m82_zspace, filter_dict[x1filter],
	                          filter_dict[x2filter], vega=vega)
	y_m82 = get_colors_for_zs(m82_wavelengths, m82_sed, m82_zspace, filter_dict[y1filter],
	                          filter_dict[y2filter], vega=vega)

	highz_x_m82 = get_colors_for_zs(m82_wavelengths, m82_sed, highz_m82space, filter_dict[x1filter],
	                                filter_dict[x2filter], vega=vega)
	highz_y_m82 = get_colors_for_zs(m82_wavelengths, m82_sed, highz_m82space, filter_dict[y1filter],
	                                filter_dict[y2filter],vega=vega)

	ell_x = get_colors_for_zs(ell_wavelengths, ell_sed, ell_zspace, filter_dict[x1filter],
	                                filter_dict[x2filter], vega=vega)
	ell_y = get_colors_for_zs(ell_wavelengths, ell_sed, ell_zspace, filter_dict[y1filter],
	                                filter_dict[y2filter], vega=vega)

	plt.figure(figsize=(8, 7))
	plt.scatter(x_m82, y_m82, marker='*', c='forestgreen', label='Low-z M82', s=60)
	plt.scatter(highz_x_m82, highz_y_m82, marker='*', c='darkgreen', label='High-z M82', s=60)
	plt.scatter(ell_x, ell_y, marker='s', c='orangered', label='Ell')
	plt.scatter(xtype1, ytype1, marker='o', c='royalblue', label='H+17 QSO1')
	plt.scatter(xtype2, ytype2, marker='o', c='firebrick', label='H+17 QSO2')


	if vega:
		unitlabel = 'Vega'
	else:
		unitlabel = 'AB'

	plt.xlabel('$%s - %s$ [%s]' % (x1filter, x2filter, unitlabel), fontsize=20)
	plt.ylabel('$%s - %s$ [%s]' % (y1filter, y2filter, unitlabel), fontsize=20)
	plt.legend(fontsize=20)
	plt.savefig('plots/lowz_selection.pdf')
	plt.close('all')




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




