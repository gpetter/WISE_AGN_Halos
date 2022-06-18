import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from astropy.table import Table
import glob
import matplotlib as mpl
from source import organization
organizer = organization.Organizer()

import plotting_computations
from matplotlib.ticker import ScalarFormatter


#import glob
import mpl_scatter_density

plt.style.use('science')
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15


def return_colorscheme(nbins):
	colors = ['royalblue', 'darkgreen', 'firebrick']
	if nbins <= 3:
		if nbins == 2:
			return ['royalblue', 'firebrick']
		return colors[:nbins]
	else:
		return cm.rainbow(np.linspace(0, 1, nbins))


def split_markers(ax, marker, legendlist, markersize=7, alpha=1.):
	m1, = ax.plot([], [], c='firebrick', marker=marker, markersize=markersize,
	               fillstyle='right', linestyle='none', markeredgecolor='none', alpha=alpha)
	m2, = ax.plot([], [], c='royalblue', marker=marker, markersize=markersize,
	               fillstyle='left', linestyle='none', markeredgecolor='none', alpha=alpha)


	legendlist.append((m2, m1))



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
	plt.savefig(organizer.plotdir + 'color_cut.pdf')
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
	plt.xlabel('$%s-%s$ [AB]' % (band1, band2), fontsize=20)

	plt.savefig(organizer.plotdir + 'color_dists.pdf')
	plt.close('all')


def plot_peakkappa_vs_bin(bin_values, kappas, kappa_errs):
	plt.figure(figsize=(8,6))
	scheme = return_colorscheme(len(kappas))
	plt.scatter(bin_values, kappas, c=scheme)
	plt.errorbar(bin_values, kappas, yerr=kappa_errs, fmt='none', ecolor=scheme)
	plt.ylabel(r'$\langle \kappa \rangle$', fontsize=20)
	plt.xlabel('Color', fontsize=20)

	plt.savefig(organizer.plotdir + 'lensing/kappa_v_bin.pdf')
	plt.close('all')

def w3_w4_det_fraction(medcolors, w3_dets, w4_dets):

	plt.close('all')
	plt.figure(figsize=(8,6))
	plt.scatter(medcolors, w3_dets, c='g', label='W3')
	plt.scatter(medcolors, w4_dets, c='r', label='W4')
	plt.ylabel('Detection Fraction', fontsize=20)
	plt.xlabel(r'$\langle r - W2 \rangle$', fontsize=20)


	plt.legend()

	plt.savefig(organizer.plotdir + 'w3_w4_det_fraction.pdf')
	plt.close('all')

def mips_fraction(medcolors, fracs):
	plt.close('all')
	plt.figure(figsize=(8, 6))
	plt.scatter(medcolors, fracs, c='k')

	plt.ylabel('MIPS Detection Fraction', fontsize=20)
	plt.xlabel(r'$\langle r - W2 \rangle$', fontsize=20)

	plt.savefig(organizer.plotdir + 'mips_det_fraction.pdf')
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
	plt.savefig(organizer.plotdir + 'w1_w2_dists.pdf')
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
	plt.savefig(organizer.plotdir + 'w3_w4_dists.pdf')
	plt.close('all')

def mips_dists(mags):
	cs = return_colorscheme(len(mags))
	plt.figure(figsize=(8, 6))
	bins = np.linspace(12, 20, 30)

	for j in range(len(mags)):
		plt.hist(mags[j], bins=bins, color=cs[j], histtype='step', linestyle='solid', density=True)


	plt.ylabel('Normalized Frequency', fontsize=20)
	plt.xlabel('[24]$_{\mathrm{AB}}$', fontsize=20)
	plt.savefig(organizer.plotdir + 'mips_dists.pdf')
	plt.close('all')

def fraction_with_redshifts(medcolors, fractions):
	plt.figure(figsize=(7,7))
	cs = return_colorscheme(len(fractions))


	plt.scatter(medcolors, fractions, c=cs)
	plt.ylabel('Redshift Completeness', fontsize=20)
	plt.xlabel(r'$\langle r - W2 \rangle$', fontsize=20)
	plt.savefig(organizer.plotdir + 'z_fractions.pdf')
	plt.close('all')

def redshift_dists(zs_by_bin, zspecs, zphots, clustering_redshifts=False):
	separate_panels = True
	if separate_panels:
		fig, axs = plt.subplots(nrows=len(zs_by_bin), ncols=1, sharex=True, figsize=(8, 6*len(zs_by_bin)))
	else:
		fig, ax = plt.subplots(figsize=(8,7))


	cs = return_colorscheme(len(zs_by_bin))

	#bin_edges = np.linspace(0., 4., 20)
	zrange = np.linspace(0, 5, 100)
	#import lensingModel
	#kern = lensingModel.dx_dz_lensing_kernel(zrange)
	#maxkern = np.max(kern)


	for j in range(len(zs_by_bin)):
		if separate_panels:
			if len(zs_by_bin) > 1:
				thisax = axs[j]
			else:
				thisax = axs
			fd_bins = plotting_computations.freedman_diaconis(zs_by_bin[j])


			zhist, bin_edges = np.histogram(zs_by_bin[j], bins=fd_bins, range=(0, 4))
			normedzhist, bin_edges = np.histogram(zs_by_bin[j], bins=fd_bins, range=(0, 4), density=True)
			normratio = (normedzhist / zhist)[0]


			thisax.hist(zs_by_bin[j], bins=bin_edges, color=cs[j],
			            histtype='step',
			            linestyle='solid',
			            weights=normratio*np.ones(len(zs_by_bin[j])))
			thisax.hist(zspecs[j], bins=bin_edges, color=cs[j],
			            histtype='step',
			            linestyle='dashed',
			            weights=normratio*np.ones(len(zspecs[j])), label='Spectroscopic')
			thisax.hist(zphots[j], bins=bin_edges, color=cs[j],
			            histtype='step',
			            linestyle='dotted',
			            weights=normratio*np.ones(len(zphots[j])), label='Photometric')
			#thisax.plot(zrange, 1 / maxkern * kern, c='k', ls='--', label=r'Lensing Kernel $\frac{d \chi}{dz} W^{'
			#                                                            r'\kappa}$')
			if clustering_redshifts:
				clusterredshiftfile = np.load('redshifts/clustering/%s.npy' % (j+1), allow_pickle=True)
				redshifts, dndzs = clusterredshiftfile[0], clusterredshiftfile[1]
				thisax.scatter(redshifts, dndzs, c=cs[j])
			thisax.legend(fontsize=20)

		else:
			plt.hist(zs_by_bin[j], bins=bin_edges, color=cs[j], histtype='step', linestyle='solid', density=True)
			if j == 0:
				plt.plot(zrange, 1 / maxkern * kern, c='k', ls='--',
				         label=r'Lensing Kernel $\frac{d \chi}{dz} W^{\kappa}$')


	#plt.legend(fontsize=15)

	plt.xlim(0, 4)
	fig.supylabel('Redshift Distribution $(dn/dz)$', fontsize=20)

	plt.xlabel('Redshift ($z$)', fontsize=20)
	plt.subplots_adjust(hspace=0.1)
	plt.savefig(organizer.plotdir + 'z_dists.pdf')
	plt.close('all')

def plot_hods(mass_grid, hods):
	plt.figure(figsize=(8,7))
	for j in range(len(hods)):
		plt.plot(np.log10(mass_grid), hods[j], c=return_colorscheme(len(hods))[j])
	plt.ylabel(r'$\langle N \rangle$', fontsize=20)
	plt.xlabel(r'log$(M_h/h^{-1} M_{\odot})$', fontsize=20)

	plt.yscale('log')
	plt.ylim(1e-3, 1e3)
	plt.xlim(10, 15)
	plt.savefig(organizer.plotdir + 'HODs.pdf')
	plt.close('all')


def plot_hod_variations(mass_grid, nbins, ndim, modeltype, chainlength=50):

	colors = return_colorscheme(nbins)
	import hod_model
	plt.close('all')
	plt.figure(figsize=(8,7))
	for j in range(nbins):

		chains = np.load('results/chains/%s.npy' % (j+1), allow_pickle=True)
		hodchainparams = chains[:, :ndim][(len(chains) - chainlength):]

		for k in range(len(hodchainparams)):

			plt.plot(np.log10(mass_grid), hod_model.hod_total(hodchainparams[k], modeltype=modeltype), c=colors[j],
			         alpha=0.05, rasterized=True)

	plt.ylabel(r'$\langle N \rangle$', fontsize=20)
	plt.xlabel(r'log$(M_h/h^{-1} M_{\odot})$', fontsize=20)

	plt.yscale('log')
	plt.ylim(1e-3, 1e3)
	plt.xlim(10, 15)
	plt.savefig(organizer.plotdir + 'HOD_scatter.pdf')
	plt.close('all')










def plot_ang_autocorrs(samplename, dipomp=False):
	plt.close('all')
	plt.figure(figsize=(8, 7))

	scales = np.load('results/clustering/scales.npy', allow_pickle=True)[1:]
	wthetas, werrs, poissonerrs = [], [], []
	clusterfiles = sorted(glob.glob('results/clustering/%s_*' % samplename))
	if dipomp:
		import pandas as pd
		unob = pd.read_csv('results/dipomp/unobscured_dipomp_17.csv', names=['theta', 'wtheta'])
		ob = pd.read_csv('results/dipomp/obscured_dipomp_17.csv', names=['theta', 'wtheta'])
		plt.scatter(unob['theta'], unob['wtheta'], c='b', alpha=0.1)
		plt.scatter(ob['theta'], ob['wtheta'], c='r', alpha=0.1)

	labels=['Unobscured', 'Obscured']

	for file in clusterfiles:
		wfile = np.load(file, allow_pickle=True)
		wthetas.append(wfile[0])
		poissonerrs.append(wfile[1])
		#werrs.append(np.std(wfile[2:], axis=0))


	cs = return_colorscheme(len(wthetas))

	plt.xlabel(r'Separation $\theta$ [deg]', fontsize=20)
	plt.ylabel(r'Angular Correlation Function $w(\theta)$', fontsize=20)


	for j in range(len(wthetas)):
		plt.scatter(scales, wthetas[j], color=cs[j])
		plt.errorbar(scales, wthetas[j], yerr=poissonerrs[j], fmt='none', ecolor='k', alpha=0.5)
		#plt.errorbar(scales, wthetas[j], yerr=werrs[j], ecolor=cs[j], fmt='none', label=labels[j])

	plt.yscale('log')
	plt.xscale('log')
	plt.legend(fontsize=20)
	plt.savefig(organizer.plotdir + 'ang_autocorr.pdf')
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

	plt.savefig(organizer.plotdir + "error_comparison.pdf")
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
	plt.savefig(organizer.plotdir + 'seds/%s.pdf' % binnum)
	plt.close('all')

def plot_composite_sed(nbins, wavelengths, binned_nu_f_nu, lowerrs, uperrs):
	plt.figure(figsize=(8, 7))
	colors = return_colorscheme(nbins)
	for j in range(nbins):

		plt.plot(wavelengths, binned_nu_f_nu[j], color=colors[j], alpha=0.5)
		plt.errorbar(wavelengths, binned_nu_f_nu[j], yerr=[lowerrs[j], uperrs[j]], ecolor=colors[j], fmt='none')
	#plt.errorbar(wavelength_bins, binned_nu_f_nu, yerr=nufnu_err, ecolor='k', alpha=0.5, fmt='none')

	plt.yscale('log')
	plt.xscale('log')
	plt.ylabel(r'$\nu F_{\nu}$ [arbitrary]', fontsize=20)
	plt.xlabel(r'$\lambda_{\mathrm{rest}} [\mu \mathrm{m}]$', fontsize=20)
	plt.savefig(organizer.plotdir + 'seds/composites.pdf')
	plt.close('all')



def plot_each_cf_fit(bin, nbins, data_scales, cf, cferr, smoothscales, cf_mod_one, cf_mod_two, cf_mod_both,
                     dm_mod=None):
	plt.figure(figsize=(8,7))

	colors = return_colorscheme(nbins)
	plt.scatter(data_scales, cf, c=colors[bin - 1])
	plt.errorbar(data_scales, cf, yerr=cferr, fmt='none', ecolor=colors[bin - 1])
	plt.plot(smoothscales, cf_mod_one, c='k', ls='dashdot', label='HOD 1-Halo', alpha=0.5)
	plt.plot(smoothscales, cf_mod_two, c='k', ls='dotted', label='HOD 2-Halo', alpha=0.5)
	plt.plot(smoothscales, cf_mod_both, ls='dashed', c=colors[bin - 1], label='HOD Total')
	if dm_mod is not None:
		plt.plot(smoothscales, dm_mod, c='k', ls='dashed', label='Linear Dark Matter')
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(1e-3, 5)
	plt.xlim(1e-3, 1)
	plt.xlabel(r'Separation $\theta$ [deg]', fontsize=20)
	plt.ylabel(r'Angular Correlation Function $w(\theta)$', fontsize=20)
	plt.legend(fontsize=15)
	plt.savefig(organizer.plotdir + 'cf_fits/%s.pdf' % bin)
	plt.close('all')

def plot_all_cf_fits(results):
	cs = return_colorscheme(len(results))
	fig, ax = plt.subplots(figsize=(8, 7))
	for j in range(len(results)):
		plot_scales, w, werr, smooth_thetas, onemodcf, twomodcf, bothmodcf, dmmod = results[j]

		transition_index = np.max(np.where((onemodcf > twomodcf) & (smooth_thetas < 0.5))[0])

		plt.scatter(plot_scales, w, c=cs[j])
		plt.errorbar(plot_scales, w, yerr=werr, fmt='none', ecolor=cs[j])

		plt.plot(smooth_thetas[:(transition_index+1)], bothmodcf[:(transition_index+1)], ls='dotted', c=cs[j])
		plt.plot(smooth_thetas[transition_index:], bothmodcf[transition_index:], ls='dashed', c=cs[j])
		#plt.plot(smooth_thetas, bothmodcf, ls='dashed', c=cs[j])

		plt.plot(smooth_thetas, dmmod, c=cs[j], ls='solid', alpha=0.5)

	plt.text(1e-2, 2e-3, 'Linear Dark Matter', rotation=-7, fontsize=15)

	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(1e-3, 3)
	plt.xlim(2e-3, 5e-1)
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	plt.xlabel(r'Separation $\theta$ [deg]', fontsize=20)
	plt.ylabel(r'Angular Correlation Function $w(\theta)$', fontsize=20)

	plt.savefig(organizer.plotdir + 'all_cf_fits.pdf')
	plt.close('all')


def plot_each_lensing_fit(bin, nbins, scales, power, power_err, cf_mod, unbiasedmod, spt=False, lcl=True):
	fig, ax = plt.subplots(figsize=(8,7))
	colors = return_colorscheme(nbins)

	def ls_to_thetas(x):
		return 180./x
	def thetas_to_ls(x):
		return 180./x

	model_ells = 1+np.arange(len(cf_mod))
	nobias_ells = 1+np.arange(len(unbiasedmod))

	if lcl:
		p = power * scales
		perr = power_err * scales
		cmod = cf_mod * model_ells
		nobiasmod = unbiasedmod * nobias_ells
	else:
		p, perr, cmod, nobiasmod = power, power_err, cf_mod, unbiasedmod


	ax.scatter(scales, p, c=colors[bin - 1])
	ax.errorbar(scales, p, yerr=perr, fmt='none', ecolor=colors[bin - 1])
	ax.plot(model_ells, cmod, ls='dashed', c=colors[bin - 1])
	ax.plot(nobias_ells, nobiasmod, ls='dotted', c='k', alpha=0.5, label='Dark Matter')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(10, 2500)
	ax.xaxis.set_tick_params(which='both', top=False)

	if spt:
		sptscales = np.load('results/lensing_xcorrs/SPT_scales.npy', allow_pickle=True)
		xpowerspt = np.load('results/lensing_xcorrs/catwise_%s_%s.npy' % ('SPT', (bin)), allow_pickle=True)
		power_spt = xpowerspt[0]
		power_err_spt = np.std(xpowerspt[1:], axis=0)
		ax.scatter(sptscales, power_spt, edgecolors=colors[bin - 1], facecolors='none')
		ax.errorbar(sptscales, power_spt, yerr=power_err_spt, fmt='none', ecolor=colors[bin-1])

	secax = ax.secondary_xaxis('top', functions=(ls_to_thetas, thetas_to_ls))
	secax.set_xlabel(r'Angular Scale [deg]', fontsize=20)
	secax.get_xaxis().set_major_formatter(ScalarFormatter())

	ax.legend()
	ax.set_xlabel(r'Multipole Moment $\ell$', fontsize=20)
	if lcl:
		ax.set_ylabel(r'Cross-power $\ell C_{\ell}^{\kappa q}$', fontsize=20)
	else:
		ax.set_ylabel(r'Cross-power $C_{\ell}^{\kappa q}$', fontsize=20)

	plt.ylim(1e-6, 1e-4)

	ax.get_xaxis().set_major_formatter(ScalarFormatter())

	plt.savefig(organizer.plotdir + 'lens_fits/%s.pdf' % bin)
	plt.close('all')



def plot_all_lens_fits(nbins, lensname, modcfs, unbiasedcfs, spt=False, lcl=True):
	fig, ax = plt.subplots(figsize=(8,7))
	colors = return_colorscheme(nbins)
	planckscales = np.load('results/lensing_xcorrs/planck_scales.npy', allow_pickle=True)

	plotshift = 1.03

	ls = 1 + np.arange(len(modcfs[0]))

	def ls_to_thetas(x):
		return 180./x
	def thetas_to_ls(x):
		return 180./x



	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(30, 2500)

	for j in range(nbins):
		if j == 1:
			planckscales *= plotshift

		xpower = np.load('results/lensing_xcorrs/catwise_%s_%s.npy' % (lensname, (j+1)), allow_pickle=True)
		power = xpower[0]
		power_err = np.std(xpower[1:], axis=0)

		model_ells = 1 + np.arange(len(modcfs[j]))
		nobias_ells = 1 + np.arange(len(unbiasedcfs[j]))

		if lcl:
			power *= planckscales
			power_err *= planckscales
			cf_mod = modcfs[j] * model_ells
			unbiasedmod = unbiasedcfs[j] * nobias_ells
		else:
			cf_mod = modcfs[j]
			unbiasedmod = unbiasedcfs[j]

		ax.scatter(planckscales, power, c=colors[j])
		ax.errorbar(planckscales, power, yerr=power_err, fmt='none', ecolor=colors[j])

		if spt:
			sptscales = np.load('results/lensing_xcorrs/SPT_scales.npy', allow_pickle=True)
			xpowerspt = np.load('results/lensing_xcorrs/catwise_%s_%s.npy' % ('SPT', (j + 1)), allow_pickle=True)
			power_spt = xpowerspt[0]
			power_err_spt = np.std(xpowerspt[1:], axis=0)
			ax.scatter(sptscales, power_spt, edgecolors=colors[j], facecolors='none')
			ax.errorbar(sptscales, power_spt, yerr=power_err_spt, fmt='none', ecolor=colors[j])
		ax.plot(ls, cf_mod, ls='dashed', c=colors[j])
		ax.plot(ls, unbiasedmod, ls='dotted', c=colors[j], alpha=0.5)

	#ax.plot(ls, np.zeros(len(ls)), c='k', ls='dotted')
	ax.xaxis.set_tick_params(which='both', top=False)

	secax = ax.secondary_xaxis('top', functions=(ls_to_thetas, thetas_to_ls))
	secax.set_xlabel(r'Angular Scale [deg]', fontsize=20)
	secax.get_xaxis().set_major_formatter(ScalarFormatter())

	ax.text(70, 1e-5, 'Linear Dark Matter', fontsize=15)

	ax.set_xlabel(r'Multipole Moment $\ell$', fontsize=20)

	if lcl:
		ax.set_ylabel(r'Cross-power $\ell C_{\ell}^{\kappa q}$', fontsize=20)
	else:
		ax.set_ylabel(r'Cross-power $C_{\ell}^{\kappa q}$', fontsize=20)
	ax.get_xaxis().set_major_formatter(ScalarFormatter())


	plt.ylim(1e-6, 1e-4)

	plt.savefig(organizer.plotdir + 'lens_fits/all_lens_fits.pdf')
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

	plt.savefig(organizer.plotdir + 'bias_v_color.pdf')
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
	clustering_results = np.load('results/clustering/mass_2h/%s.npy' % samplename, allow_pickle=True)

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

	plt.savefig(organizer.plotdir + 'mass_v_color.pdf')
	plt.close('all')

def plot_hpx_map(map, name):
	import healpy as hp
	plt.figure()

	hp.mollview(map)
	plt.savefig(organizer.plotdir + '%s.pdf' % name)
	plt.close('all')

def lensing_xcorrs(samplename, lensnames=['planck'], lcl=False):

	plt.close('all')
	fig, ax = plt.subplots(figsize=(8, 7))

	planck_scales = np.load('results/lensing_xcorrs/planck_scales.npy', allow_pickle=True)


	planckclfiles = sorted(glob.glob('results/lensing_xcorrs/%s_planck_*' % samplename))
	colors = return_colorscheme(len(planckclfiles))

	markers = ['o', 's', 'D', '^']



	def ls_to_thetas(x):
		return 180./x
	def thetas_to_ls(x):
		return 180./x

	labels = ['Unobscured', 'Obscured']

	for j in range(len(planckclfiles)):

		for k, name in enumerate(lensnames):

			scales = np.load('results/lensing_xcorrs/%s_scales.npy' % name, allow_pickle=True)
			clfile = sorted(glob.glob('results/lensing_xcorrs/%s_%s_%s*' % (samplename, name, (j+1))))


			clarr = np.load(clfile[0], allow_pickle=True)
			if name == 'planck+SPT':
				cl = clarr[0]
				clerr = clarr[1]
			else:
				cl = clarr[0]
				clerr = np.std(clarr[1:], axis=0)

			if lcl:
				cl *= scales
				clerr *= scales
			if k>0:
				alpha=0.3
				scales *= 1.1*k
			else:
				alpha=1.

			ax.scatter(scales, cl, color=colors[j], alpha=alpha)
			ax.errorbar(scales, cl, yerr=clerr, ecolor=colors[j], fmt='none', label=labels[j], marker=markers[j],
			            alpha=alpha)


	ax.xaxis.set_tick_params(which='both', top=False)

	secax = ax.secondary_xaxis('top', functions=(ls_to_thetas, thetas_to_ls))
	secax.set_xlabel(r'Angular Scale [deg]', fontsize=20)

	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.legend(fontsize=20)
	ax.set_xlabel(r'Multipole Moment $\ell$', fontsize=20)
	if lcl:
		ax.set_ylabel(r'Cross-power $\ell C_{\ell}^{\kappa q}$', fontsize=20)
	else:
		ax.set_ylabel(r'Cross-power $C_{\ell}^{\kappa q}$', fontsize=20)

	plt.savefig(organizer.plotdir + 'lensing_xcorrs.pdf')
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
	plt.savefig(organizer.plotdir + 'depth_v_density/ratio.pdf')
	plt.close('all')

def data_vs_random_density(ratiomap, binno):
	import healpy as hp
	plt.close('all')
	plt.figure(figsize=(9, 7))
	hp.mollview(ratiomap)
	plt.savefig(organizer.plotdir + 'depth_v_density/%s_map.pdf' % binno)
	plt.close('all')

def low_z_cut_plot(colorname, unresolved_xcolors, unresolved_ycolors, resolved_xcolors, resolved_ycolors, m1, b1, m2,
                   b2, highzcut):

	intersection_x = (b2 - b1) / (m1 - m2)

	plt.close('all')
	fig = plt.figure(figsize=(8, 7))

	if highzcut:
		xintersect1 = (4.5 - b1) / m1
	else:
		xintersect1 = -10

	# remove nans
	goodunixs = (np.isfinite(unresolved_xcolors) & np.isfinite(unresolved_ycolors))
	goodreixs = (np.isfinite(resolved_xcolors) & np.isfinite(resolved_ycolors))
	unresolved_xcolors, unresolved_ycolors = unresolved_xcolors[goodunixs], unresolved_ycolors[goodunixs]
	resolved_xcolors, resolved_ycolors = resolved_xcolors[goodreixs], resolved_ycolors[goodreixs]

	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	ax.scatter_density(unresolved_xcolors, unresolved_ycolors, color='k', alpha=1)
	ax.scatter_density(resolved_xcolors, resolved_ycolors, color='g', alpha=1)
	ax.plot(np.linspace(xintersect1, intersection_x, 3),
	        m1*np.linspace(xintersect1, intersection_x, 3) + b1, c='k', ls='--')
	ax.plot(np.linspace(intersection_x, 10, 3), m2 * np.linspace(intersection_x, 10, 3) + b2, c='k', ls='--')
	if highzcut:
		ax.hlines(4.5, color='k', ls='--', xmin=-10, xmax=xintersect1)

	ax.set_xlabel('%s - W2' % colorname, fontsize=20)
	ax.set_ylabel('z - W2', fontsize=20)
	ax.set_xlim(2, 10)
	ax.set_ylim(2, 10)
	plt.savefig(organizer.plotdir + 'lowz_cut.pdf')
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

	plt.savefig(organizer.plotdir + 'density_vs_coord/%s_%s_%s.pdf' % (coordframe, lonlat, binnum))
	plt.close('all')

def mateos_plot(nbins, binnedtab, longband):
	nondetcat = binnedtab[np.where(np.isnan(binnedtab['e_W%smag' % longband]))]
	# stack w3 nondetections with LS forced photometry
	stacked_w3mag = -2.5 * np.log10(np.nanmean(nondetcat['flux_W%s' % longband]))
	avg_nondet_w2 = np.mean(nondetcat['W2mag'])
	stacked_w2w3 = avg_nondet_w2 - stacked_w3mag
	avg_nondet_w1w2 = np.mean(nondetcat['W1mag'] - nondetcat['W2mag'])
	colors = return_colorscheme(nbins)

	binnedtab = binnedtab[np.where(np.logical_not(np.isnan(binnedtab['e_W%smag' % longband])))]
	binnum = int(binnedtab['bin'][0])
	plt.close('all')
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	ax.set_xlabel('W2 - W%s [Vega mags]' % longband, fontsize=20)
	ax.set_ylabel('W1 - W2 [Vega mags]', fontsize=20)

	ax.scatter_density(binnedtab['W2mag'] - binnedtab['W%smag' % longband], binnedtab['W1mag'] - binnedtab['W2mag'],
	                   color=colors[binnum-1])
	if longband == '3':
		ax.scatter_density(nondetcat['W2mag'] - 10, nondetcat['W1mag'] - nondetcat['W2mag'],
		                   color='k', alpha=0.5)

	all_sources = Table.read(organizer.plotdir + 'mateos/cosmos_catwise_allwise.fits')
	all_sources = all_sources[np.where(np.logical_not(np.isnan(all_sources['e_W%smag' % longband])))]
	ax.scatter_density(all_sources['w2mpro'] - all_sources['W%smag' % longband], all_sources['w1mpro'] - all_sources[
		'w2mpro'],
	                   color='k')
	ax.scatter(stacked_w2w3, avg_nondet_w1w2, c=colors[binnum-1], label='Stacked W%s Non-detections' %
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

	plt.savefig(organizer.plotdir + 'mateos/%s_W%s.pdf' % (binnum, longband))
	plt.close('all')


def radio_detection_fraction(colors, fracs, survey):
	plt.close('all')
	plt.figure(figsize=(8, 7))
	plt.scatter(colors, fracs, c=return_colorscheme(len(fracs)))
	plt.xlabel(r'$r - W2$', fontsize=20)
	plt.ylabel(r'Detection Fraction', fontsize=20)
	plt.savefig(organizer.plotdir + 'radio/%s.pdf' % survey)
	plt.close('all')



def donley_plot(nbins, matchedtab, fulltab, binnum):
	plt.close('all')
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	ax.set_xlabel('[3.6] - [5.8]$_{AB}$', fontsize=20)
	ax.set_ylabel('[4.5] - [8.0]$_{AB}$', fontsize=20)
	colors = return_colorscheme(nbins)

	ax.scatter(matchedtab['ch1_4'] + 2.79 - matchedtab['ch3_4'] - 3.73, matchedtab['ch2_4'] + 3.26 -
	           matchedtab['ch4_4'] - 4.40, color=colors[binnum - 1])
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
	plt.savefig(organizer.plotdir + 'donley/%s.pdf' % binnum)
	plt.close('all')

def all_clustering_systematics(nbins, agnbin, bincenters, corbincenters, ratios, corratios, errors, corerrs, systnames):
	separate_panels = True
	if separate_panels:
		fig, axs = plt.subplots(nrows=len(bincenters), ncols=1, sharex=False, figsize=(8, 6 * len(bincenters)))
	else:
		plt.figure(figsize=(8, 7))

	colors = return_colorscheme(nbins)


	for j in range(len(bincenters)):
		if separate_panels:
			if len(bincenters) > 1:
				thisax = axs[j]
			else:
				thisax = axs
			thisax.scatter(bincenters[j], ratios[j], c='k')
			thisax.errorbar(bincenters[j], ratios[j], yerr=errors[j], ecolor='k', fmt='none')
			thisax.scatter(corbincenters[j], corratios[j], marker='s', edgecolors=colors[agnbin-1],
			               facecolors='none')
			thisax.errorbar(corbincenters[j], corratios[j], yerr=corerrs[j], ecolor=colors[agnbin-1], fmt='none')
			thisax.set_xlabel(systnames[j], fontsize=20)

			thisax.axhline(1, ls='--', c='k')

		else:
			print('fix me')

	#plt.legend(fontsize=15)
	plt.ylabel('$N_{\mathrm{data}} / N_{\mathrm{random}}$', fontsize=25)

	plt.savefig(organizer.plotdir + 'systematics/all_%s.pdf' % agnbin)
	plt.close('all')


def stern05_plot(nbins, matchedtab, fulltab, binnum):
	plt.close('all')
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	ax.set_xlabel('[5.8] - [8.0]$_{Vega}$', fontsize=20)
	ax.set_ylabel('[3.6] - [4.5]$_{Vega}$', fontsize=20)
	colors = return_colorscheme(nbins)

	ax.scatter(matchedtab['ch3_4'] - matchedtab['ch4_4'] , matchedtab['ch1_4'] -
	           matchedtab['ch2_4'], color=colors[binnum - 1])
	ax.scatter_density(fulltab['ch3_4'] - fulltab['ch4_4'],
	                   fulltab['ch1_4'] - fulltab['ch2_4'], color='k', alpha=1, dpi=10)
	ax.vlines(0.6, ymin=0.3, ymax=3, colors='k', ls='--')
	ax.plot(np.linspace(0.6, 1.6, 10), 0.2 * np.linspace(0.6, 1.6, 10) + 0.18, c='k', ls='--')
	ax.plot(np.linspace(1.6, 3, 10), 2.5 * np.linspace(1.6, 3, 10) - 3.5, c='k', ls='--')

	plt.legend(fontsize=20)
	ax.set_ylim(-0.5, 2)
	ax.set_xlim(-1, 3)
	plt.savefig(organizer.plotdir + 'stern/%s.pdf' % binnum)
	plt.close('all')

def plot_kim_diagrams(nbins, binnum, ki2, i24, i2mips):
	plt.close('all')
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	ax.set_xlabel('[4.5] - [8.0]$_{AB}$', fontsize=20)
	ax.set_ylabel('K - [4.5]$_{AB}$', fontsize=20)
	colors = return_colorscheme(nbins)

	detections = np.where((ki2 > -10) & (i24 > -10))


	ax.scatter(i24[detections], ki2[detections], color=colors[binnum - 1])
	#ax.scatter_density(fulltab['ch3_4'] - fulltab['ch4_4'],
	#                   fulltab['ch1_4'] - fulltab['ch2_4'], color='k', alpha=1, dpi=10)
	ax.vlines(0, ymin=0, ymax=10, colors='k', ls='--', label='Messias+12')
	ax.hlines(0, xmin=0, xmax=10, colors='k', ls='--')

	plt.legend(fontsize=20)
	ax.set_xlim(-1.5, 4)
	ax.set_ylim(-1.5, 6)

	plt.savefig(organizer.plotdir + 'messias/ki_%s.pdf' % binnum)
	plt.close('all')

	plt.close('all')
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
	ax.set_xlabel('[4.5] - [8.0]$_{AB}$', fontsize=20)
	ax.set_ylabel('[4.5] - [24]$_{AB}$', fontsize=20)
	detections = np.where((i2mips > -10) & (i24 > -10))

	ax.scatter(i24[detections], i2mips[detections], color=colors[binnum - 1])
	#ax.scatter_density(fulltab['ch3_4'] - fulltab['ch4_4'],
	#                   fulltab['ch1_4'] - fulltab['ch2_4'], color='k', alpha=1, dpi=10)
	ax.hlines(0.5, xmin=0.793, xmax=10, colors='k', ls='--', label='Messias+12')
	ax.plot(np.linspace(-2, 0.793, 5), -2.9 * np.linspace(-2, 0.793, 5) + 2.8, c='k', ls='--')
	#ax.plot(np.linspace(0.6, 1.6, 10), 0.2 * np.linspace(0.6, 1.6, 10) + 0.18, c='k', ls='--')
	#ax.plot(np.linspace(1.6, 3, 10), 2.5 * np.linspace(1.6, 3, 10) - 3.5, c='k', ls='--')

	plt.legend(fontsize=20)
	ax.set_xlim(-1.5, 4)
	ax.set_ylim(-2, 6)



	plt.savefig(organizer.plotdir + 'messias/im_%s.pdf' % binnum)
	plt.close('all')

def plot_stacks(return_ax=False):
	bluestack = np.load('lens_stacks/catwise_stack0.npy', allow_pickle=True)*1000
	redstack = np.load('lens_stacks/catwise_stack1.npy', allow_pickle=True)*1000
	bluestd, redstd = np.std(bluestack), np.std(redstack)
	cmap = 'inferno'

	minval = np.min(np.array([bluestack.flatten(), redstack.flatten()]))
	maxval = np.max(np.array([bluestack.flatten(), redstack.flatten()]))

	halfwidth = 50

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 7), dpi=300)
	im = ax1.imshow(bluestack, vmin=minval, vmax=maxval, cmap=cmap, extent=[-halfwidth, halfwidth, -halfwidth,
	                                                                       halfwidth])
	ax1.get_xaxis().set_visible(False)
	ax1.set_ylabel(r'$\theta$ (arcminutes)', fontsize=20)

	ax1.tick_params(which='both', right=False, labelsize=15)

	ax1.tick_params('both', which='major', length=8)
	ax1.tick_params('both', which='minor', length=3)


	ax1.text(-halfwidth/1.5, halfwidth*1.1, 'Unobscured', c='royalblue', fontsize=20)


	ax2.imshow(redstack, vmin=minval, vmax=maxval, cmap=cmap, extent=[-halfwidth, halfwidth, -halfwidth, halfwidth])

	#ax3.axis('off')
	ax2.tick_params(which='both', length=0, labelbottom=False, labelleft=False)

	ax2.text(-halfwidth/1.3, halfwidth*1.1, 'Obscured', c='firebrick', fontsize=20)

	fig.subplots_adjust(bottom=0.1)
	cbar_ax = fig.add_axes([0.2, 0.16, 0.6, 0.06])
	cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
	cbar.set_label('$10^{3} \ \kappa$', fontsize=20)
	cbar_ax.tick_params(labelsize=15)
	#fig.colorbar(im)
	plt.subplots_adjust(wspace=0.03)
	if return_ax:
		return plt.gca()
	plt.savefig(organizer.plotdir + 'lens_stacks.pdf', bbox_inches='tight')
	plt.close('all')




def hod_corner(analysis, flatchain, ndim, binnum, nbins):
	import corner
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

	plt.savefig(organizer.plotdir + "%s_hod_params_%s.pdf" % (analysis, binnum))
	plt.close('all')


def overlapping_corners(analysis, flatchains, ndim, nbins):
	import corner
	plt.close('all')
	param_names = [r'log $M_{\rm min}$', r'$\alpha$', 'log $M1$']
	derived_names = [r'$f_{\rm sat}$', r'$b_{\mathrm{eff}}$',
	                 r'log $M_{\mathrm{eff}}$']
	labelnames = param_names[:ndim] + derived_names
	colors = return_colorscheme(nbins)

	fig = corner.corner(
		flatchains[0],
		labels=labelnames,
		show_titles=False,
		# range=lim,
		levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4)),
		plot_datapoints=False,
		plot_density=False,
		fill_contours=True,
		color=colors[0],
		hist_kwargs={"color": "%s" % colors[0]},
		label_kwargs={'fontsize': 20},
		smooth=0.5,
		smooth1d=0.5,
	)

	for j in range(1, nbins):
		corner.corner(
			flatchains[j],
			labels=labelnames,
			show_titles=False,
			# range=lim,
			levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4)),
			plot_datapoints=False,
			plot_density=False,
			fill_contours=True,
			color=colors[j],
			hist_kwargs={"color": "%s" % colors[j]},
			label_kwargs={'fontsize': 20},
			smooth=0.5,
			smooth1d=0.5, fig=fig,
		)

	plt.savefig(organizer.plotdir + "%s_hod_params.pdf" % analysis)
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
		plt.savefig(organizer.plotdir + 'individual_seds/%s.pdf' % j)
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
	plt.savefig(organizer.plotdir + 'lum_distributions.pdf')
	plt.close('all')


def plot_bias_vs_z():
	from source import bias_tools
	import redshift_dists

	b, berr = bias_tools.combine_biases()
	blens1 = np.load('results/lensing_xcorrs/bias/catwise_1.npy', allow_pickle=True)
	blens2 = np.load('results/lensing_xcorrs/bias/catwise_2.npy', allow_pickle=True)
	colorsche = return_colorscheme(len(b))


	plottingshift = 0.008
	zrange = np.linspace(0., 4., 50)
	laurent_b, laurent_b_err = bias_tools.qso_bias_for_z('laurent', zs=zrange)

	laurent_mass, laurent_upmass, laurent_lowmass = [], [], []
	for j in range(len(zrange)):
		laurent_mass.append(bias_tools.bias_to_mass(inputbias=laurent_b[j], z=zrange[j]))
		laurent_upmass.append(bias_tools.bias_to_mass(inputbias=laurent_b[j]+laurent_b_err[j], z=zrange[j]))
		laurent_lowmass.append(bias_tools.bias_to_mass(inputbias=laurent_b[j]-laurent_b_err[j], z=zrange[j]))

	plt.close('all')
	#fig, ax = plt.subplots(figsize=(7,6))
	fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(7, 6 * 2))
	laurentart = ax.fill_between(zrange, laurent_b-laurent_b_err, laurent_b+laurent_b_err, color='k',
	                             alpha=0.1)

	laurentmassart = ax2.fill_between(zrange, laurent_lowmass, laurent_upmass, color='k',
	                             alpha=0.1)
	legendlist = []
	ax.text(1.6, 2.2, 'Optical Type-1 QSOs (Laurent+17)', rotation=30, alpha=0.5)
	#ax2.text(1.6, 12.35, 'Optical Type-1 QSOs (Laurent+17)', rotation=0, alpha=0.5)

	ax.scatter([1.27], [2.2], c='royalblue', marker='s', alpha=0.2)
	ax.scatter([1.26], [3.05], c='firebrick', marker='s', alpha=0.2)
	ax.errorbar([1.27], [2.2], yerr=[0.45], fmt='none', ecolor='royalblue', alpha=0.2)
	ax.errorbar([1.26], [3.05], yerr=[0.7], fmt='none', ecolor='firebrick', alpha=0.2)
	split_markers(ax, marker='s', legendlist=legendlist, alpha=0.2)


	ax.scatter([1.05], [1.7], c='royalblue', marker='X', alpha=0.2)
	ax.scatter([0.98], [2.15], c='firebrick', marker='X', alpha=0.2)
	ax.errorbar([1.05], [1.7], yerr=[0.1], fmt='none', ecolor='royalblue', alpha=0.2)
	ax.errorbar([0.98], [2.15], yerr=[0.11], fmt='none', ecolor='firebrick', alpha=0.2)
	split_markers(ax, marker='X', legendlist=legendlist, alpha=0.2)

	ax.scatter([0.7], [0.9], c='royalblue', marker='<', alpha=0.2)
	ax.scatter([0.77], [0.7], c='firebrick', marker='<', alpha=0.2)
	ax.errorbar([0.7], [0.9], yerr=[0.2], fmt='none', ecolor='royalblue', alpha=0.2)
	ax.errorbar([0.77], [0.7], yerr=[0.2], fmt='none', ecolor='firebrick', alpha=0.2)
	split_markers(ax, marker='<', legendlist=legendlist, alpha=0.2)


	cross_names = ['SDWFS (Hickox+11)', 'WISE+SDSS (Dipompeo+17)', 'PRIMUS+DEEP2 (Mendez+16)', 'This Work (Lensing + '
	                                                                                  'Clustering)']


	ax.tick_params('both', labelsize=15, which='major', length=8)
	ax.tick_params('both', which='minor', length=3)

	ax2.tick_params('both', labelsize=15, which='major', length=8)
	ax2.tick_params('both', which='minor', length=3)




	for j in range(len(b)):
		f, zs, szs, pzs = redshift_dists.get_redshifts(bin=(j+1))
		medz = np.median(zs)
		ax.scatter(medz, b[j], c=colorsche[j])
		ax.errorbar(medz, b[j], yerr=berr[j], fmt='none', ecolor=colorsche[j])

	split_markers(ax, marker='o', legendlist=legendlist)




	ax.set_ylabel('$b_q$', fontsize=25, rotation=0, labelpad=15)
	ax2.set_ylabel(r'log$_{10}(M_h/h^{-1} M_{\odot})$', fontsize=25)

	ax.set_xlim(0.5, 2.5)
	ax.set_ylim(0.5, 5)
	ax2.set_xlabel('$\mathrm{Redshift} \ (z)$', fontsize=20)
	ax2.set_ylim(11.5, 13.5)

	ax.legend(legendlist, cross_names, fontsize=12, loc=2)
	plt.subplots_adjust(hspace=0.1)
	plt.savefig(organizer.plotdir + 'bias_v_z.pdf')
	plt.close('all')