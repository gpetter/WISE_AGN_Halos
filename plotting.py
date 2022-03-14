import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import healpy as hp
import lensingModel
import importlib
from astropy.table import Table
import glob

import plotting_computations

importlib.reload(lensingModel)
#import glob
import mpl_scatter_density
import corner
plt.style.use('science')


def return_colorscheme(nbins):
	colors = ['royalblue', 'darkgreen', 'firebrick']
	if nbins <= 3:
		return colors[:nbins]
	else:
		return cm.rainbow(np.linspace(0, 1, nbins))



def plot_assef_cut(table):
	fig = plt.figure(figsize=(8,7))
	catwise_cosmos = Table.read('catalogs/cosmos/catwise_cosmos.fits')
	ax = fig.add_subplot(1,1,1, projection='scatter_density')
	ax.scatter_density(catwise_cosmos['w2mpro'], catwise_cosmos['w1mpro'] - catwise_cosmos['w2mpro'], cmap='Greys')
	ax.scatter_density(table['W2mag'], table['W1mag'] - table['W2mag'], color='red')
	ax.axvline(17.5, c='k')
	ax.plot(np.linspace(13, 20, 100), -1 * np.linspace(13, 20, 100) + 17.7, c='k')
	ax.plot(np.linspace(13.86, 16.2, 100), 0.65 * np.exp(0.153 * np.square(np.linspace(13.86, 16.2, 100) - 13.86)),
	        c='k', ls='--', label='A+18 R90')
	ax.plot(np.linspace(13, 13.86, 3), 0.65*np.ones(3), c='k', ls='--')

	ax.plot(np.linspace(13.07, 16.37, 100), 0.486 * np.exp(0.092 * np.square(np.linspace(13.07, 16.37, 100) - 13.07)),
	        c='k', ls='dotted', label='A+18 R75')
	ax.plot(np.linspace(13, 13.07, 3), 0.486 * np.ones(3), c='k', ls='dotted')
	ax.text(17.55, 1.5, 'W2 90$\%$ completeness', rotation='vertical', fontsize=20)
	ax.text(14, 2., 'W1 90$\%$ completeness', rotation=-37, fontsize=20)
	ax.set_xlim(13, 19)
	ax.set_ylim(-2, 5)
	ax.text(15, -0.75, 'All CatWISE Sources', c='grey')
	plt.xlabel('W2 [mag]', fontsize=20)
	plt.ylabel('W1 - W2 [mag]', fontsize=20)
	plt.legend(fontsize=15)
	plt.savefig('plots/color_cut.pdf')
	plt.close('all')


def plot_color_dists(table, nondet_colors, band1, band2):
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



	plt.xlim(-2, 11)

	plt.ylabel('Number', fontsize=20)
	plt.xlabel('$%s-%s$' % (band1, band2), fontsize=20)

	plt.savefig('plots/color_dists.pdf')
	plt.close('all')


def plot_peakkappa_vs_bin(bin_values, kappas, kappa_errs):
	plt.figure(figsize=(8,6))
	scheme = return_colorscheme(len(kappas))
	plt.scatter(bin_values, kappas, c=scheme)
	plt.errorbar(bin_values, kappas, yerr=kappa_errs, fmt='none', ecolor=scheme)
	plt.ylabel(r'$\langle \kappa \rangle$', fontsize=20)
	plt.xlabel('Color', fontsize=20)

	plt.savefig('plots/lensing/kappa_v_bin.pdf')
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
			if len(zs_by_bin) > 1:
				thisax = axs[j]
			else:
				thisax = axs
			thisax.hist(zs_by_bin[j], bins=plotting_computations.freedman_diaconis(zs_by_bin[j]), color=cs[j],
			            histtype='step',
			            linestyle='solid',
			            density=True)
			thisax.plot(zrange, 1 / maxkern * kern, c='k', ls='--', label=r'Lensing Kernel $\frac{d \chi}{dz} W^{'
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

def plot_hods(mass_grid, hods):
	plt.figure(figsize=(8,7))
	for j in range(len(hods)):
		plt.plot(mass_grid, hods[j], c=return_colorscheme(len(hods))[j])
	plt.ylabel(r'$\langle N \rangle$', fontsize=20)
	plt.xlabel(r'log$_{10}(M/h^{-1} M_{\odot}$)', fontsize=20)
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(1e-3, 1e3)
	plt.savefig('plots/HODs.pdf')
	plt.close('all')



def plot_ang_autocorrs(samplename):
	scales = np.load('results/clustering/scales.npy', allow_pickle=True)[1:]
	wthetas, werrs, poissonerrs = [], [], []
	clusterfiles = sorted(glob.glob('results/clustering/%s_*' % samplename))
	for file in clusterfiles:
		wfile = np.load(file, allow_pickle=True)
		wthetas.append(wfile[0])
		poissonerrs.append(wfile[1])
		werrs.append(np.std(wfile[2:], axis=0))
	plt.close('all')
	plt.figure(figsize=(8,7))
	cs = return_colorscheme(len(wthetas))

	plt.xlabel(r'$\theta$', fontsize=20)
	plt.ylabel(r'$w_{\theta}$', fontsize=20)


	for j in range(len(wthetas)):
		plt.scatter(scales, wthetas[j], color=cs[j])
		plt.errorbar(scales, wthetas[j], yerr=poissonerrs[j], fmt='none', ecolor='k', alpha=0.5)
		plt.errorbar(scales, wthetas[j], yerr=werrs[j], ecolor=cs[j], fmt='none')

	plt.yscale('log')
	plt.xscale('log')

	plt.savefig('plots/ang_autocorr.pdf')
	plt.close('all')

def cf_err_comparison(samplename):
	scales = np.load('results/clustering/scales.npy', allow_pickle=True)[1:]
	ratios = []
	clusterfiles = sorted(glob.glob('results/clustering/%s_*' % samplename))
	for file in clusterfiles:
		wfile = np.load(file, allow_pickle=True)
		ratios.append(wfile[1] / np.std(wfile[2:], axis=0))
	plt.close('all')
	plt.figure(figsize=(8, 7))
	cs = return_colorscheme(len(clusterfiles))

	plt.xlabel(r'$\theta$', fontsize=20)
	plt.ylabel(r'$\sigma(Poisson) / \sigma(f2f)$', fontsize=20)

	for j in range(len(clusterfiles)):
		plt.scatter((1.2 ** j) * scales, ratios[j], color=cs[j])

	plt.ylim(0, 5)
	plt.xscale('log')

	plt.savefig('plots/error_comparison.pdf')
	plt.close('all')

def plot_each_sed(binnum, nbins, wavelengths, nufnu, nufnu_err):
	plt.figure(figsize=(8,7))
	colors = return_colorscheme(nbins)
	#plt.scatter(wavelengths, nufnu, c='k', alpha=0.5)
	#plt.errorbar(wavelengths, nufnu, yerr=nufnu_err, ecolor='k', alpha=0.5, fmt='none')
	for j in range(len(wavelengths)):
		plt.scatter(wavelengths[j], nufnu[j], color=colors[binnum-1], alpha=0.5)
		plt.errorbar(wavelengths[j], nufnu[j], yerr=nufnu_err[j], ecolor=colors[binnum-1], alpha=0.5, fmt='none')
	plt.yscale('log')
	plt.xscale('log')
	plt.ylabel(r'$\nu F_{\nu}$ [arbitrary]', fontsize=20)
	plt.xlabel(r'$\lambda_{\mathrm{rest}} [\mu \mathrm{m}]$', fontsize=20)
	plt.savefig('plots/seds/%s.pdf' % binnum)
	plt.close('all')

def plot_composite_sed(nbins, wavelength_bins, binned_nu_f_nu, lowerrs, uperrs):
	plt.figure(figsize=(8, 7))
	colors = return_colorscheme(nbins)
	for j in range(nbins):

		plt.plot(wavelength_bins, binned_nu_f_nu[j], color=colors[j], alpha=0.5)
		plt.errorbar(wavelength_bins, binned_nu_f_nu[j], yerr=[lowerrs[j], uperrs[j]], ecolor=colors[j], fmt='none')
	#plt.errorbar(wavelength_bins, binned_nu_f_nu, yerr=nufnu_err, ecolor='k', alpha=0.5, fmt='none')

	plt.yscale('log')
	plt.xscale('log')
	plt.ylabel(r'$\nu F_{\nu}$ [arbitrary]', fontsize=20)
	plt.xlabel(r'$\lambda_{\mathrm{rest}} [\mu \mathrm{m}]$', fontsize=20)
	plt.savefig('plots/seds/composites.pdf')
	plt.close('all')



def plot_each_cf_fit(bin, nbins, scales, cf, cferr, cf_mod_one, cf_mod_two, cf_mod_both, dm_mod=None):
	plt.figure(figsize=(8,7))

	colors = return_colorscheme(nbins)
	plt.scatter(scales, cf, c=colors[bin - 1])
	plt.errorbar(scales, cf, yerr=cferr, fmt='none', ecolor=colors[bin - 1])
	plt.plot(scales, cf_mod_one, c='k', ls='dashdot', label='HOD 1-Halo', alpha=0.5)
	plt.plot(scales, cf_mod_two, c='k', ls='dotted', label='HOD 2-Halo', alpha=0.5)
	plt.plot(scales, cf_mod_both, ls='dashed', c=colors[bin - 1], label='HOD Total')
	if dm_mod is not None:
		plt.plot(scales, dm_mod, c='k', ls='dashed', label='2-Halo Dark Matter')
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(1e-4, 5)
	plt.xlabel(r'$\theta$ [deg]', fontsize=20)
	plt.ylabel(r'$w(\theta)$', fontsize=20)
	plt.legend(fontsize=15)
	plt.savefig('plots/cf_fits/%s.pdf' % bin)
	plt.close('all')

def plot_each_lensing_fit(bin, nbins, scales, power, power_err, cf_mod, unbiasedmod):
	plt.figure(figsize=(8,7))
	colors = return_colorscheme(nbins)

	plt.scatter(scales, power, c=colors[bin - 1])
	plt.errorbar(scales, power, yerr=power_err, fmt='none', ecolor=colors[bin - 1])
	plt.plot(scales, cf_mod, ls='dashed', c=colors[bin - 1])
	plt.plot(scales, unbiasedmod, ls='dotted', c='k', alpha=0.5, label='Dark Matter')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.xlabel(r'$l$', fontsize=20)
	plt.ylabel(r'$C_l$', fontsize=20)

	plt.savefig('plots/lens_fits/%s.pdf' % bin)
	plt.close('all')


def bias_v_color(samplename):
	offset = 0.05

	lens_results = np.load('results/lensing_xcorrs/bias/%s.npy' % samplename, allow_pickle=True)

	clustering_results = np.load('results/clustering/bias/%s.npy' % samplename, allow_pickle=True)

	medcolors, lensbias, lensbias_err = lens_results[0], lens_results[1], lens_results[2]
	medcolors, cfbias, cfbias_err = clustering_results[0], clustering_results[1], clustering_results[2]

	plt.figure(figsize=(8, 7))
	cs = return_colorscheme(len(medcolors))
	plt.scatter(medcolors - offset, lensbias, edgecolors=cs, facecolors='none', label='CMB Lensing Result')
	plt.errorbar(medcolors - offset, lensbias, yerr=lensbias_err, ecolor=cs, fmt='none')

	plt.scatter(medcolors + offset, cfbias, color=cs, label='Angular Clustering Result')
	plt.errorbar(medcolors + offset, cfbias, yerr=cfbias_err, fmt='none', ecolor=cs)

	plt.xlabel(r'$\langle r - W2 \rangle$', fontsize=20)
	plt.ylabel(r'$b_{q}$', fontsize=20)
	plt.legend(fontsize=20)

	plt.savefig('plots/bias_v_color.pdf')
	plt.close('all')


def mass_v_color(samplename):
	plt.close('all')
	plt.figure(figsize=(8, 7))
	offset = 0.05

	obsqsos = Table.read('catalogs/dipomp/obscured.fits')
	unobsqsos = Table.read('catalogs/dipomp/unobscured.fits')

	def smooth(y, box_pts):
		box = np.ones(box_pts) / box_pts
		y_smooth = np.convolve(y, box, mode='same')
		return y_smooth

	ndipompbins = 1000

	unobs_hist = np.histogram(unobsqsos['RMAG'] - unobsqsos['W2'] - 3.339, ndipompbins, range=(0, 6 - 3.339))
	unobs_alphas = unobs_hist[0] / np.max(unobs_hist[0])
	smooth_unob_alphas = smooth(unobs_alphas, 300)

	det_obs_cat = obsqsos[np.where(obsqsos['RMAG'] > 0)]

	obs_hist = np.histogram(det_obs_cat['RMAG'] - det_obs_cat['W2'] - 3.339, ndipompbins, range=(6 - 3.339, 6))
	obs_alphas = obs_hist[0] / np.max(unobs_hist[0])
	smooth_obs_alphas = smooth(obs_alphas, 300)


	plt.figure(figsize=(8, 7))

	im1 = plt.imshow(np.outer(np.ones(ndipompbins), smooth_unob_alphas), cmap=cm.Blues,
	                 extent=[0, 6 - 3.339, 12.49 - 0.1, 12.49 + 0.1],
	                 interpolation="bicubic", alpha=.4, aspect="auto")

	im2 = plt.imshow(np.outer(np.ones(ndipompbins), smooth_obs_alphas), cmap=cm.Reds,
	                 extent=[6 - 3.339, 6, 12.94 - 0.08, 12.94 + 0.08],
	                 interpolation="bicubic", alpha=.4, aspect="auto", label='D+17 Obscured')


	#plt.text(0.8, 12.94, 'D+17 Obscured', fontsize=20, c='red', alpha=0.5)
	#plt.text(3, 12.49, 'D+17 Unobscured', fontsize=20, c='blue', alpha=0.4)



	lens_results = np.load('results/lensing_xcorrs/mass/%s.npy' % samplename, allow_pickle=True)
	cs = return_colorscheme(len(lens_results[0]))
	clustering_results = np.load('results/clustering/mass/%s.npy' % samplename, allow_pickle=True)

	medcolors, lensmass, lensmasserrs = lens_results[0], lens_results[1], lens_results[2]
	medcolors, cfmass, cfmassloerrs, cfmasshierrs = clustering_results[0], clustering_results[1], \
	                                                clustering_results[2], clustering_results[3]

	plt.scatter(medcolors - offset, lensmass, edgecolors=cs, facecolors='none', label='CMB Lensing Result')
	plt.errorbar(medcolors - offset, lensmass, yerr=lensmasserrs, ecolor=cs, fmt='none')

	plt.scatter(medcolors + offset, cfmass, color=cs, label='Angular Clustering Result')
	plt.errorbar(medcolors + offset, cfmass, yerr=[cfmassloerrs, cfmasshierrs], fmt='none', ecolor=cs)

	plt.xlabel(r'$\langle r - W2 \rangle$', fontsize=20)
	plt.ylabel(r'$\mathrm{log}_{10}(M_h / h^{-1} M_{\odot})$', fontsize=20)
	plt.legend(fontsize=20)

	plt.savefig('plots/mass_v_color.pdf')
	plt.close('all')

def plot_hpx_map(map, name):
	plt.figure()
	hp.mollview(map)
	plt.savefig('plots/maps/%s.pdf' % name)
	plt.close('all')

def lensing_xcorrs(samplename):

	plt.close('all')
	plt.figure(figsize=(8, 7))


	plt.xlabel(r'$l$', fontsize=20)
	plt.ylabel(r'$C_{l}$', fontsize=20)

	scales = np.load('results/lensing_xcorrs/scales.npy', allow_pickle=True)

	clfiles = sorted(glob.glob('results/lensing_xcorrs/%s_*' % samplename))
	colors = return_colorscheme(len(clfiles))


	for j in range(len(clfiles)):

		clarr = np.load(clfiles[j], allow_pickle=True)
		cl = clarr[0]
		clerr = np.std(clarr[1:], axis=0)

		plt.scatter(scales, cl, color=colors[j])
		plt.errorbar(scales, cl, yerr=clerr, ecolor=colors[j], fmt='none')
	plt.yscale('log')
	plt.xscale('log')

	plt.savefig('plots/lensing_xcorrs.pdf')
	plt.close('all')

def visualize_xcorr(densproj, lensproj):
	plt.figure(figsize=(10, 8))
	plt.imshow(lensproj, cmap='jet')
	#plt.contour(Z=densproj)
	plt.savefig('xcorr_visual.pdf')

def depth_v_density(depths, densities, errors):

	magdepths = -2.5 * (np.log10(5 / np.sqrt(depths)) - 9)
	plt.figure(figsize=(8,7))
	for j in range(len(densities)):
		plt.scatter(magdepths + 0.05*j, densities[j], color=return_colorscheme(len(densities))[j])
		plt.errorbar(magdepths + 0.05 * j, densities[j], yerr=errors[j], fmt='none',
		             color=return_colorscheme(len(densities))[j])
	plt.xlabel(r'$5 \sigma$ r PSF depth [AB mags]', fontsize=20)
	plt.ylabel('Weight', fontsize=20)
	#plt.ylim(5, 15)
	plt.savefig('plots/depth_v_density/ratio.pdf')
	plt.close('all')

def data_vs_random_density(ratiomap, binno):
	plt.close('all')
	plt.figure(figsize=(9, 7))
	hp.mollview(ratiomap, coord=['G', 'C'])
	plt.savefig('plots/depth_v_density/%s_map.pdf' % binno)
	plt.close('all')

def low_z_cut_plot(colorname, unresolved_xcolors, unresolved_ycolors, resolved_xcolors, resolved_ycolors, m1, b1, m2,
                   b2):

	intersection_x = (b2 - b1) / (m1 - m2)

	plt.close('all')
	fig = plt.figure(figsize=(8, 7))

	# remove nans
	goodunixs = (np.isfinite(unresolved_xcolors) & np.isfinite(unresolved_ycolors))
	goodreixs = (np.isfinite(resolved_xcolors) & np.isfinite(resolved_ycolors))
	unresolved_xcolors, unresolved_ycolors = unresolved_xcolors[goodunixs], unresolved_ycolors[goodunixs]
	resolved_xcolors, resolved_ycolors = resolved_xcolors[goodreixs], resolved_ycolors[goodreixs]

	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	ax.scatter_density(unresolved_xcolors, unresolved_ycolors, color='k', alpha=1)
	ax.scatter_density(resolved_xcolors, resolved_ycolors, color='b', alpha=1)
	ax.plot(np.linspace(-10, intersection_x, 3), m1*np.linspace(-10, intersection_x, 3) + b1, c='k', ls='--')
	ax.plot(np.linspace(intersection_x, 10, 3), m2 * np.linspace(intersection_x, 10, 3) + b2, c='k', ls='--')

	ax.set_xlabel('%s - W2' % colorname, fontsize=20)
	ax.set_ylabel('z - W2', fontsize=20)
	ax.set_xlim(2, 10)
	ax.set_ylim(2, 10)
	plt.savefig('plots/lowz_cut.pdf')
	plt.close('all')



def density_vs_coord(binnum, ratios, errors, bincenters, coordframe, lonlat):

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


	plt.scatter(bincenters, ratios, c='k')
	plt.errorbar(bincenters, ratios, yerr=errors, c='k', fmt='none')
	plt.xlabel('%s %s [deg]' % (coordframename, coordname), fontsize=20)
	plt.ylabel('$N_{data}/N_{rand}$', fontsize=20)

	plt.savefig('plots/density_vs_coord/%s_%s_%s.pdf' % (coordframe, lonlat, binnum))
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


	ax.plot(np.linspace(0.347 * 2.5, 10, 3), 2.5 * (1.21 * np.linspace(0.347 * 2.5, 10, 3) / 2.5 - 0.27), c='k',
	        ls='--', label='Donley+12')
	ax.plot(np.linspace(0.2, 10, 3), 2.5 * (1.21 * np.linspace(0.2, 10, 3) / 2.5 + 0.27), c='k',
	        ls='--')
	ax.plot(np.linspace(0.2, 0.347 * 2.5, 3), 0.375 * np.ones(3), c='k', ls='--')
	ax.vlines(0.2, ymin=0.375, ymax=0.917, colors='k', ls='--')

	# Lacy
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

def plot_kim_diagrams(binnum, ki2, i24, i2mips):
	plt.close('all')
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	ax.set_xlabel('[4.5] - [8.0]$_{AB}$', fontsize=20)
	ax.set_ylabel('K - [4.5]$_{AB}$', fontsize=20)

	detections = np.where((ki2 > -10) & (i24 > -10))


	ax.scatter(i24[detections], ki2[detections], color=return_colorscheme(5)[binnum - 1])
	#ax.scatter_density(fulltab['ch3_4'] - fulltab['ch4_4'],
	#                   fulltab['ch1_4'] - fulltab['ch2_4'], color='k', alpha=1, dpi=10)
	ax.vlines(0, ymin=0, ymax=10, colors='k', ls='--', label='Messias+12')
	ax.hlines(0, xmin=0, xmax=10, colors='k', ls='--')

	plt.legend(fontsize=20)
	ax.set_xlim(-1.5, 4)
	ax.set_ylim(-1.5, 6)

	plt.savefig('plots/messias/ki_%s.pdf' % binnum)
	plt.close('all')

	plt.close('all')
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	ax.set_xlabel('[4.5] - [8.0]$_{AB}$', fontsize=20)
	ax.set_ylabel('[4.5] - [24]$_{AB}$', fontsize=20)
	detections = np.where((i2mips > -10) & (i24 > -10))

	ax.scatter(i24[detections], i2mips[detections], color=return_colorscheme(5)[binnum - 1])
	#ax.scatter_density(fulltab['ch3_4'] - fulltab['ch4_4'],
	#                   fulltab['ch1_4'] - fulltab['ch2_4'], color='k', alpha=1, dpi=10)
	ax.hlines(0.5, xmin=0.793, xmax=10, colors='k', ls='--', label='Messias+12')
	ax.plot(np.linspace(-2, 0.793, 5), -2.9 * np.linspace(-2, 0.793, 5) + 2.8, c='k', ls='--')
	#ax.plot(np.linspace(0.6, 1.6, 10), 0.2 * np.linspace(0.6, 1.6, 10) + 0.18, c='k', ls='--')
	#ax.plot(np.linspace(1.6, 3, 10), 2.5 * np.linspace(1.6, 3, 10) - 3.5, c='k', ls='--')

	plt.legend(fontsize=20)
	ax.set_xlim(-1.5, 4)
	ax.set_ylim(-2, 6)



	plt.savefig('plots/messias/im_%s.pdf' % binnum)
	plt.close('all')


def hod_corner(flatchain, ndim, binnum, nbins):
	plt.close('all')
	param_names = [r'log $M_{\rm min}$', r'$\alpha$', 'log $M1$']
	derived_names = [r'$f_{\rm sat}$', r'$b_{\mathrm{eff}}$',
		        r'log $M_{\mathrm{eff}}$']
	labelnames = param_names[:ndim] + derived_names

	corner.corner(
		flatchain,
		labels=labelnames,
		quantiles=(0.16, 0.84),
		show_titles=True,
		# range=lim,
		levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4)),
		plot_datapoints=False,
		plot_density=False,
		fill_contours=True,
		color=return_colorscheme(nbins)[binnum - 1],
		hist_kwargs={"color": "black"},
		smooth=0.5,
		smooth1d=0.5,
	)

	plt.savefig("plots/cf_fits/hod_params_%s.pdf" % binnum)
	plt.close('all')

def plot_every_observed_sed(fluxtable, fluxerrtable, eff_wavelengths, zs, ids=None):
	obs_nus = 2.998e14 / np.array(eff_wavelengths)

	for j in range(len(fluxtable)):
		plt.close('all')
		plt.figure(figsize=(8,7))
		thisflux, thisfluxerr = np.array(list(fluxtable[j])).astype(np.float64), \
		                        np.array(list(fluxerrtable[j])).astype(np.float64)

		thisflux[np.where(thisflux < 0)] = np.nan
		thisfluxerr[np.where(thisfluxerr < 0)] = np.nan
		thisflux = obs_nus * thisflux
		thisfluxerr = obs_nus * thisfluxerr
		plt.scatter(eff_wavelengths, thisflux, c='k')
		plt.errorbar(eff_wavelengths, thisflux, yerr=thisfluxerr, ecolor='k', fmt='none')
		plt.xlabel(r'$\lambda_{obs} [\mu m]$', fontsize=20)
		plt.ylabel(r'$\nu_{obs} F_{\nu}$', fontsize=20)
		plt.xscale('log')
		plt.yscale('log')
		plt.title(r'$z = %s $' % zs[j], fontsize=20)
		plt.xlim(1e-1, 5e2)
		plt.savefig('plots/individual_seds/%s.pdf' % j)
		plt.close('all')

def plot_luminosity_distributions(medlum, lumratios):
	colors = return_colorscheme(len(lumratios))

	plt.close('all')
	plt.figure(figsize=(8, 7))
	for j in range(len(lumratios)):
		lums = np.log10(medlum * np.array(lumratios[j]))[0]
		lums = lums[np.where(np.isfinite(lums))]
		plt.hist(lums, bins=30, histtype='step', color=colors[j])
	plt.xlabel(r'log$_{10}(\nu L_{\nu}(6 \mu m))$', fontsize=20)
	plt.ylabel('Frequency', fontsize=20)
	plt.savefig('plots/lum_distributions.pdf')
	plt.close('all')