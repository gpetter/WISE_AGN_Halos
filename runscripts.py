import importlib
import sample
#import stacking
#import random_catalogs
#import convergence_map
import astropy.units as u
import masking
importlib.reload(masking)
#importlib.reload(convergence_map)
#importlib.reload(random_catalogs)
#importlib.reload(stacking)
importlib.reload(sample)

wisename = 'catwise_r90'
opticalname = 'ls'
samplename = wisename + '_' + opticalname
nside = 1024

#convergence_map.write_planck_maps(real=True, noise=True, width=0*u.arcmin, nsides=nside, lmin=0)
#masking.write_masks(nside)
#masking.total_mask(depth_cut=150, assef=True, unwise=True, planck=True)


# clean up WISE AGN catalog, select magnitude measurements, perform cuts and masking
#sample.filter_table('mpro', planckmask=True, bitmask=True, custom_mask=True, w1cut=17.5, pmsncut=5)
# bin sample by r-W2 color
#sample.bin_sample(wisename, 'color', 1)
sample.redshifts(wisename, z_sample='cosmos')
#random_catalogs.correct_for_depth()
#stacking.stack_suite('catwise_r90', False, False, mode='cutout')
#stacking.stack_without_projection()
#stacking.quick_stack_suite(wisename, 'planck', nsides=1024)

#sample.long_wavelength_properties(samplename, 'mips_sep')