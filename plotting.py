import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import healpy as hp
import lensingModel
import importlib
from astropy.table import Table
importlib.reload(lensingModel)
#import glob
import mpl_scatter_density
plt.style.use('science')


def return_colorscheme(nbins):
	if nbins == 3:
		return ['royalblue', 'darkgreen', 'firebrick']
	else:
		return cm.rainbow(np.linspace(0, 1, nbins))



def plot_assef_cut(table):
	fig = plt.figure(figsize=(8,7))
	ax = fig.add_subplot(1,1,1, projection='scatter_density')
	ax.scatter_density(table['W2mag'], table['W1mag'] - table['W2mag'], cmap='Greys')
	ax.axvline(17.5, c='k')
	ax.plot(np.linspace(13, 18, 100), -1 * np.linspace(13, 18, 100) + 17.7, c='k')
	ax.text(17.55, 1.5, 'W2 90$\%$ completeness', rotation='vertical', fontsize=20)
	ax.set_xlim(13, 19)
	ax.set_ylim(0, 5)
	plt.xlabel('W2 [mag]', fontsize=20)
	plt.ylabel('W1 - W2 [mag]', fontsize=20)
	plt.savefig('plots/color_cut.pdf')
	plt.close('all')


def plot_color_dists(table, nondet_colors, optband):
	nbins = int(np.max(table['bin']))
	colorscheme = return_colorscheme(nbins)
	#firstcolor, secondcolor = colorkey.split('-')[0], colorkey.split('-')[1]
	binedges = np.histogram(table[np.where(table['detect'])]['color'], bins=300)[1]
	plt.figure(figsize=(8,6))
	detect_table = table[np.where(table['detect'])]



	for j in range(nbins):
		colortab = detect_table[np.where(detect_table['bin'] == j+1)]
		#colordist = colortab['dered_mag_%s' % firstcolor] - colortab['dered_mag_%s' % secondcolor]
		colordist = colortab['color']
		plt.hist(colordist, bins=binedges, color=colorscheme[j])

	nbins_nondet = 7

	"""plt.hist(nondet_colors, color=colorscheme[len(colorscheme)-1], hatch='/', bins=nbins_nondet, fill=False,
	         weights=np.ones(len(nondet_colors)) *
	                 float(len(table) - len(detect_table))/(nbins_nondet * len(nondet_colors)))"""

	print(np.sum(np.ones(len(nondet_colors)) *
	                 float(len(table) - len(detect_table))/(nbins_nondet * len(nondet_colors))))
	print(len(table) - len(detect_table))

	plt.xlim(-2, 11)

	plt.ylabel('Number', fontsize=20)
	plt.xlabel('$%s-W2$' % optband, fontsize=20)

	plt.savefig('plots/color_dists.pdf')
	plt.close('all')


def plot_peakkappa_vs_bin(bin_values, kappas, kappa_errs):
	plt.figure(figsize=(8,6))

	plt.scatter(bin_values, kappas)
	plt.errorbar(bin_values, kappas, yerr=kappa_errs, fmt='none')
	plt.ylabel(r'$\langle \kappa \rangle$', fontsize=20)

	plt.savefig('plots/kappa_v_bin.pdf')
	plt.close('all')

def w3_w4_det_fraction(medcolors, w3_dets, w4_dets):

	plt.close('all')
	plt.figure(figsize=(8,6))
	plt.scatter(medcolors, w3_dets, c='g', label='W3')
	plt.scatter(medcolors, w4_dets, c='r', label='W4')
	plt.ylabel('Detection Fraction', fontsize=20)
	plt.xlabel(r'$\langle r - W2 \rangle$', fontsize=20)


	plt.legend()

	plt.savefig('plots/w3_w4_det_fraction.pdf')
	plt.close('all')

def mips_fraction(medcolors, fracs):
	plt.close('all')
	plt.figure(figsize=(8, 6))
	plt.scatter(medcolors, fracs, c='k')

	plt.ylabel('MIPS Detection Fraction', fontsize=20)
	plt.xlabel(r'$\langle r - W2 \rangle$', fontsize=20)

	plt.savefig('plots/mips_det_fraction.pdf')
	plt.close('all')


def w1_w2_dists(tab):
	plt.figure(figsize=(8,6))

	total_w1_bins = np.histogram(tab['W1mag'], bins=50)[1]
	total_w2_bins = np.histogram(tab['W2mag'], bins=50)[1]

	cs = return_colorscheme(int(np.max(tab['bin'])))

	for j in range(int(np.max(tab['bin']))):
		binnedw1tab = tab[np.where(tab['bin'] == j + 1)]
		plt.hist(binnedw1tab['W1mag'], bins=total_w1_bins, color=cs[j], histtype='step', linestyle='solid',
		         label='W1', density=True)
		binnedw2tab = tab[np.where(tab['bin'] == j + 1)]
		plt.hist(binnedw2tab['W2mag'], bins=total_w2_bins, color=cs[j], histtype='step', linestyle='dashed',
		         label='W2', density=True)

	plt.legend()

	plt.ylabel('Normalized Frequency', fontsize=20)
	plt.xlabel('WISE Magnitude', fontsize=20)
	plt.savefig('plots/w1_w2_dists.pdf')
	plt.close('all')

def w3_w4_dists(tab):
	plt.figure(figsize=(8,6))
	w3tab = tab[np.where(tab['e_W3mag'] > 0)]
	w4tab = tab[np.where(tab['e_W4mag'] > 0)]
	total_w3_bins = np.histogram(w3tab['W3mag'], bins=50)[1]
	total_w4_bins = np.histogram(w4tab['W4mag'], bins=50)[1]

	cs = return_colorscheme(int(np.max(tab['bin'])))

	for j in range(int(np.max(tab['bin']))):
		binnedw3tab = w3tab[np.where(w3tab['bin'] == j + 1)]
		plt.hist(binnedw3tab['W3mag'], bins=total_w3_bins, color=cs[j], histtype='step', linestyle='solid',
		         label='W3', density=True)
		binnedw4tab = w4tab[np.where(w4tab['bin'] == j + 1)]
		plt.hist(binnedw4tab['W4mag'], bins=total_w4_bins, color=cs[j], histtype='step', linestyle='dashed',
		         label='W4', density=True)

	plt.legend()

	plt.ylabel('Normalized Frequency', fontsize=20)
	plt.xlabel('WISE Magnitude', fontsize=20)
	plt.savefig('plots/w3_w4_dists.pdf')
	plt.close('all')

def mips_dists(mags):
	cs = return_colorscheme(len(mags))
	plt.figure(figsize=(8, 6))
	bins = np.linspace(12, 20, 30)

	for j in range(len(mags)):
		plt.hist(mags[j], bins=bins, color=cs[j], histtype='step', linestyle='solid', density=True)


	plt.ylabel('Normalized Frequency', fontsize=20)
	plt.xlabel('[24]$_{\mathrm{AB}}$', fontsize=20)
	plt.savefig('plots/mips_dists.pdf')
	plt.close('all')

def fraction_with_redshifts(medcolors, fractions):
	plt.figure(figsize=(7,7))
	cs = return_colorscheme(len(fractions))


	plt.scatter(medcolors, fractions, c=cs)
	plt.ylabel('Redshift Completeness', fontsize=20)
	plt.xlabel(r'$\langle r - W2 \rangle$', fontsize=20)
	plt.savefig('plots/z_fractions.pdf')
	plt.close('all')

def redshift_dists(zs_by_bin):
	separate_panels = True
	if separate_panels:
		fig, axs = plt.subplots(nrows=len(zs_by_bin), ncols=1, sharex=True, figsize=(8, 6*len(zs_by_bin)))
	else:
		plt.figure(figsize=(8,7))


	cs = return_colorscheme(len(zs_by_bin))

	#bin_edges = np.linspace(0., 4., 20)
	zrange = np.linspace(0, 5, 100)
	kern = lensingModel.dx_dz_lensing_kernel(zrange)
	maxkern = np.max(kern)

	for j in range(len(zs_by_bin)):
		if separate_panels:
			axs[j].hist(zs_by_bin[j], bins=int(len(zs_by_bin[j]) / 5), color=cs[j], histtype='step', linestyle='solid',
			            density=True)
			axs[j].plot(zrange, 1 / maxkern * kern, c='k', ls='--', label=r'Lensing Kernel $\frac{d \chi}{dz} W^{'
			                                                            r'\kappa}$')
		else:
			plt.hist(zs_by_bin[j], bins=bin_edges, color=cs[j], histtype='step', linestyle='solid', density=True)
			if j == 0:
				plt.plot(zrange, 1 / maxkern * kern, c='k', ls='--',
				         label=r'Lensing Kernel $\frac{d \chi}{dz} W^{\kappa}$')


	plt.legend(fontsize=15)

	plt.xlim(0, 4)

	plt.xlabel('Redshift ($z$)', fontsize=20)
	plt.ylabel('Redshift Distribution $(dn/dz)$', fontsize=20)
	plt.savefig('plots/z_dists.pdf')
	plt.close('all')


def plot_ang_autocorrs(scales, wthetas):
	plt.close('all')
	plt.figure(figsize=(8,7))
	cs = return_colorscheme(len(wthetas))

	plt.xlabel(r'$\theta$', fontsize=20)
	plt.ylabel(r'$w_{\theta}$', fontsize=20)

	print(scales)
	print(wthetas[0])

	for j in range(len(wthetas)):
		plt.scatter(scales, wthetas[j], color=cs[j])
	plt.yscale('log')
	plt.xscale('log')

	plt.savefig('plots/ang_autocorr.pdf')
	plt.close('all')


def plot_each_cf_fit(bin, scales, cf, cferr, cf_mod, dm_mod=None):
	plt.figure(figsize=(8,7))

	plt.scatter(scales, cf, c='k')
	plt.errorbar(scales, cf, yerr=cferr, c='k')
	plt.plot(scales, cf_mod)
	if dm_mod is not None:
		plt.plot(scales, dm_mod, c='k', ls='dotted')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$\theta$ [deg]', fontsize=20)
	plt.ylabel(r'$w(\theta)$', fontsize=20)

	plt.savefig('plots/cf_fits/%s.pdf' % bin)
	plt.close('all')

def plot_each_lensing_fit(bin, scales, power, power_err, cf_mod):
	plt.figure(figsize=(8,7))

	plt.scatter(scales, power, c='k')
	plt.errorbar(scales, power, yerr=power_err, c='k')
	plt.plot(scales, cf_mod)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$l$', fontsize=20)
	plt.ylabel(r'$C_l$', fontsize=20)

	plt.savefig('plots/lens_fits/%s.pdf' % bin)
	plt.close('all')


def bias_v_color(medcolors, bias, biaserrs):
	plt.figure(figsize=(8, 7))
	cs = return_colorscheme(len(medcolors))
	plt.scatter(medcolors, bias, color=cs)
	plt.errorbar(medcolors, bias, yerr=biaserrs, ecolor=cs, fmt='none')

	plt.xlabel(r'$\langle r - W2 \rangle$', fontsize=20)
	plt.ylabel(r'$b_{q}$', fontsize=20)

	plt.savefig('plots/bias_v_color.pdf')
	plt.close('all')

def mass_v_color(medcolors, mass, massloerrs, massuperrs):
	plt.figure(figsize=(8, 7))
	cs = return_colorscheme(len(medcolors))
	plt.scatter(medcolors, mass, color=cs)
	plt.errorbar(medcolors, mass, yerr=[massloerrs, massuperrs], ecolor=cs, fmt='none')

	plt.xlabel(r'$\langle r - W2 \rangle$', fontsize=20)
	plt.ylabel(r'$\mathrm{log}_{10}(M_h / h^{-1} M_{\odot})$', fontsize=20)

	plt.savefig('plots/mass_v_color.pdf')
	plt.close('all')

def plot_hpx_map(map, name):
	plt.figure()
	hp.mollview(map)
	plt.savefig('plots/maps/%s.pdf' % name)
	plt.close('all')

def lensing_xcorrs():

	plt.close('all')
	plt.figure(figsize=(8, 7))
	cs = return_colorscheme(5)

	plt.xlabel(r'$l$', fontsize=20)
	plt.ylabel(r'$C_{l}$', fontsize=20)

	scales = np.load('lensing_xcorrs/scales.npy', allow_pickle=True)


	for j in range(1):

		clarr = np.load('lensing_xcorrs/catwise_r90_%s.npy' % (j + 1), allow_pickle=True)
		cl = clarr[0][0]
		print(cl)

		plt.scatter(np.arange(len(cl)), cl, color=cs[j])
	plt.yscale('log')
	plt.xscale('log')

	plt.savefig('plots/lensing_xcorrs.pdf')
	plt.close('all')

def visualize_xcorr(densproj, lensproj):
	plt.figure(figsize=(10, 8))
	plt.imshow(lensproj, cmap='jet')
	#plt.contour(Z=densproj)
	plt.savefig('xcorr_visual.pdf')

def depth_v_density(depths, densities, binno):

	magdepths = -2.5 * (np.log10(5 / np.sqrt(depths)) - 9)
	plt.figure(figsize=(8,7))
	plt.scatter(magdepths, densities)
	plt.xlabel(r'$5 \sigma$ r PSF depth [AB mags]', fontsize=20)
	plt.ylabel('Weight', fontsize=20)
	#plt.ylim(5, 15)
	plt.savefig('plots/depth_v_density/%s.pdf' % binno)
	plt.close('all')

def density_vs_coord(ratios, bins, coordframe, lonlat):
	bincenters = (bins[1:] + bins[:-1])/2
	plt.figure(figsize=(8, 7))
	coordname = lonlat
	if coordframe == 'G':
		coordframename = 'Galactic'
	elif coordframe == 'E':
		coordframename = 'Ecliptic'
	else:
		coordframename = ""
		if lonlat == 'lon':
			coordname = 'RA'
		else:
			coordname = 'DEC'


	plt.scatter(bincenters, ratios)
	plt.xlabel('%s %s [deg]' % (coordframename, coordname), fontsize=20)
	plt.ylabel('$N_{data}/N_{rand}$', fontsize=20)

	plt.savefig('plots/rand_ratio.pdf')
	plt.close('all')

def mateos_plot(binnedtab, longband):
	nondetcat = binnedtab[np.where(np.isnan(binnedtab['e_W%smag' % longband]))]
	# stack w3 nondetections with LS forced photometry
	stacked_w3mag = -2.5 * np.log10(np.nanmean(nondetcat['flux_W%s' % longband]))
	avg_nondet_w2 = np.mean(nondetcat['W2mag'])
	stacked_w2w3 = avg_nondet_w2 - stacked_w3mag
	avg_nondet_w1w2 = np.mean(nondetcat['W1mag'] - nondetcat['W2mag'])

	binnedtab = binnedtab[np.where(np.logical_not(np.isnan(binnedtab['e_W%smag' % longband])))]
	binnum = int(binnedtab['bin'][0])
	plt.close('all')
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	ax.set_xlabel('W2 - W%s [Vega mags]' % longband, fontsize=20)
	ax.set_ylabel('W1 - W2 [Vega mags]', fontsize=20)

	ax.scatter_density(binnedtab['W2mag'] - binnedtab['W%smag' % longband], binnedtab['W1mag'] - binnedtab['W2mag'],
	                   color=return_colorscheme(5)[binnum-1])
	if longband == '3':
		ax.scatter_density(nondetcat['W2mag'] - 10, nondetcat['W1mag'] - nondetcat['W2mag'],
		                   color='k', alpha=0.5)

	all_sources = Table.read('plots/mateos/cosmos_catwise_allwise.fits')
	all_sources = all_sources[np.where(np.logical_not(np.isnan(all_sources['e_W%smag' % longband])))]
	ax.scatter_density(all_sources['w2mpro'] - all_sources['W%smag' % longband], all_sources['w1mpro'] - all_sources[
		'w2mpro'],
	                   color='k')
	ax.scatter(stacked_w2w3, avg_nondet_w1w2, c=return_colorscheme(5)[binnum-1], label='Stacked W%s Non-detections' %
	                                                                                   longband)

	if longband == '3':
		ax.plot(np.linspace(1.958, 8, 50), 0.315 * np.linspace(2.186, 8, 50) + 0.796, ls='--', c='k', label='Mateos+12')
		ax.plot(np.linspace(2.25, 8, 50), 0.315 * np.linspace(2.186, 8, 50) - 0.222, ls='--', c='k')
		ax.plot(np.linspace(1.958, 2.25, 50), -3.172 * np.linspace(1.958, 2.25, 50) + 7.624, ls='--', c='k')
		ax.arrow(5.25, 1, -.5, 0, width=0.01, head_width=.05, facecolor='k')
	ax.text(0, -0.5, 'CatWISE sources in COSMOS', fontsize=15)
	ax.legend(fontsize=15)
	if longband == '3':
		ax.set_xlim(-1, 7)
	else:
		ax.set_xlim(-1, 12)
	ax.set_ylim(-1, 3)

	plt.savefig('plots/mateos/%s_W%s.pdf' % (binnum, longband))
	plt.close('all')


def radio_detection_fraction(colors, fracs, survey):
	plt.close('all')
	plt.figure(figsize=(8, 7))
	plt.scatter(colors, fracs, c=return_colorscheme(len(fracs)))
	plt.xlabel(r'$r - W2$', fontsize=20)
	#plt.ylabel('%s Detection Fraction' % survey, fontsize=20)
	plt.savefig('plots/radio/%s.pdf' % survey)
	plt.close('all')



def donley_plot(matchedtab, fulltab, binnum):
	plt.close('all')
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	ax.set_xlabel('[3.6] - [5.8]$_{AB}$', fontsize=20)
	ax.set_ylabel('[4.5] - [8.0]$_{AB}$', fontsize=20)

	ax.scatter(matchedtab['ch1_4'] + 2.79 - matchedtab['ch3_4'] - 3.73, matchedtab['ch2_4'] + 3.26 -
	           matchedtab['ch4_4'] - 4.40, color=return_colorscheme(5)[binnum - 1])
	ax.scatter_density(fulltab['ch1_4'] + 2.79 - fulltab['ch3_4'] - 3.73,
	                   fulltab['ch2_4'] + 3.26 - fulltab['ch4_4'] - 4.40, color='k', alpha=1, dpi=10)
	ax.plot(np.linspace(0, 10, 50) / 2.5, (1.21 * np.linspace(0, 10, 50) / 2.5 - 0.27), c='k', ls='--',
	        label='Donley+12')
	ax.plot(np.linspace(0, 10, 50) / 2.5, (1.21 * np.linspace(0, 10, 50) / 2.5 + 0.27), c='k', ls='--')
	ax.plot(np.linspace(-0.099, 0.347, 50), 0.375 * np.ones(50), c='k', ls='--')
	ax.plot(np.linspace(-0.25, 10, 50), -0.5 * np.ones(50), c='k')
	ax.vlines(-0.25, ymin=-0.5, ymax=0, colors='k')
	ax.plot(np.linspace(-0.25, 10, 50), 2 * np.linspace(-0.25, 10, 50) + 0.5, c='k')
	
	plt.legend(fontsize=20)
	ax.set_ylim(-2, 3)
	ax.set_xlim(-1.5, 2)
	plt.savefig('plots/donley/%s.pdf' % binnum)
	plt.close('all')


def stern05_plot(matchedtab, fulltab, binnum):
	plt.close('all')
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	ax.set_xlabel('[5.8] - [8.0]$_{Vega}$', fontsize=20)
	ax.set_ylabel('[3.6] - [4.5]$_{Vega}$', fontsize=20)

	ax.scatter(matchedtab['ch3_4'] - matchedtab['ch4_4'] , matchedtab['ch1_4'] -
	           matchedtab['ch2_4'], color=return_colorscheme(5)[binnum - 1])
	ax.scatter_density(fulltab['ch3_4'] - fulltab['ch4_4'],
	                   fulltab['ch1_4'] - fulltab['ch2_4'], color='k', alpha=1, dpi=10)
	ax.vlines(0.6, ymin=0.3, ymax=3, colors='k', ls='--')
	ax.plot(np.linspace(0.6, 1.6, 10), 0.2 * np.linspace(0.6, 1.6, 10) + 0.18, c='k', ls='--')
	ax.plot(np.linspace(1.6, 3, 10), 2.5 * np.linspace(1.6, 3, 10) - 3.5, c='k', ls='--')

	plt.legend(fontsize=20)
	ax.set_ylim(-0.5, 2)
	ax.set_xlim(-1, 3)
	plt.savefig('plots/stern/%s.pdf' % binnum)
	plt.close('all')
