import importlib
import numpy as np
import redshift_dists
import sample
#import stacking
#import random_catalogs
#import convergence_map
import astropy.units as u
import masking
import systematics

#import moc_tools
#from source import catalog_tools

#from source import template_tools
#template_tools.template_colors_by_z(np.linspace(0, 5, 200), filter1='W1', filter2='W3', grid_factor=5)
#template_tools.two_color_space('r', 'W2', 'z', 'W2', vega=False)



#moc_tools.make_unwise_bitmask(15, nside)
#moc_tools.combined_area_lost_mask()
#moc_tools.use_highres_masks(data=True)
#moc_tools.use_highres_masks(data=False)


#catalog_tools.prep_cat_for_opt_match()

wisename = 'catwise'
opticalname = 'ls'
samplename = wisename + '_' + opticalname
nside = 1024
band1 = 'r'

#import convergence_map
# generate Planck lensing map and noise realizations from raw k_lm's
#convergence_map.write_planck_maps(real=0, noise=0, width=15., nsides=nside, lmin=30, lmax=3000)
#convergence_map.write_spt_maps(5, nsides=nside, lmin=30, lmax=3000, noise=True)
#convergence_map.ACT_map(nside=nside, lmin=30, lmax=3000)
#convergence_map.sptpol_map(0)
#convergence_map.weak_lensing_map(tomo_bin='full', reconstruction='glimpse')


# generate mask for both data + randoms. mask Galactic contaminants, regions with high E(B-V), low imaging depth etc.
# masking.write_masks(nside)

#moc_tools.use_highres_masks(data=False)
# clean up WISE AGN catalog, select magnitude measurements, perform cuts and masking
#sample.filter_table(criterion='r90', w2faintcut=16.5, pmcut=0.25, lowzcut=1, highzcut=0, bands=['r','W2'], sepcut=2.,
#					magnification_bias_test=False, remake_mask=True, premasked=True)



# bin sample by r-W2 color
#sample.bin_sample(wisename, 'color', band1='r', band2='W2', nbins=2, combinebins=None)
#redshift_dists.match_to_spec_surveys(wisename, 2.*u.arcsec)
#sample.redshifts(wisename)
systematics.elat_and_depth_weights(10)



# make plots of multiwavelength properties of sample
#sample.long_wavelength_properties(wisename, ['w1w2', 'w3w4', 'mips_sep', 'mateos', 'donley', 'radio'])

#stacking.stack_suite('catwise_r90', False, False, mode='cutout')
#stacking.stack_without_projection()
#stacking.quick_stack_suite(wisename, 'planck', nsides=nside)
