import importlib
import numpy as np
import redshift_dists
import sample
#import stacking
import random_catalogs
#import convergence_map
import astropy.units as u
import masking
import systematics
import convergence_map
import stacking

from source import catalog_tools

from source import template_tools
#template_tools.template_colors_by_z(np.linspace(0, 4, 200), filter1='z', filter2='W1', grid_factor=5)
#template_tools.two_color_space('r', 'z', 'r', 'W2', vega=False)

wisename = 'catwise'
opticalname = 'ls'
samplename = wisename + '_' + opticalname
nside = 1024
band1 = 'g'

#catalog_tools.prep_cat_for_opt_match()


# generate Planck lensing map and noise realizations from raw k_lm's
#convergence_map.write_planck_maps(real=True, noise=False, width=15.*u.arcmin, nsides=nside, lmin=0)
#convergence_map.weak_lensing_map(tomo_bin='full', reconstruction='glimpse')


# generate mask for both data + randoms. mask Galactic contaminants, regions with high E(B-V), low imaging depth etc.
#masking.write_masks(nside)


# clean up WISE AGN catalog, select magnitude measurements, perform cuts and masking
sample.filter_table(criterion='r90', w1cut=17.5, pmsncut=5, lowzcut=1, highzcut=0, bands=['r','W2'])
#redshift_dists.match_to_spec_surveys(wisename, 2.5*u.arcsec)


# bin sample by r-W2 color
sample.bin_sample(wisename, 'color', band1='r', band2='W2', nbins=5, combinebins=3)


# ensure randoms match data
#systematics.correct_randoms()


# match against many spectroscopic surveys for redshift distribution

#sample.redshifts(wisename)


# make plots of multiwavelength properties of sample
#sample.long_wavelength_properties(wisename, ['w1w2', 'w3w4', 'mips_sep', 'mateos', 'donley', 'radio', 'kim'])

#stacking.stack_suite('catwise_r90', False, False, mode='cutout')
#stacking.stack_without_projection()
#stacking.quick_stack_suite(wisename, 'planck', nsides=nside)
