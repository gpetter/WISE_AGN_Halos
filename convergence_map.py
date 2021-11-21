import numpy as np
import healpy as hp
import pandas as pd
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
#from pixell import enmap, reproject, utils
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy import wcs
import importlib
import healpixhelper
import masking
import glob
importlib.reload(masking)
importlib.reload(healpixhelper)

# number of sides to each healpix pixel
#nsides = 2048


# take in healpix map which defaults to using the UNSEEN value to denote masked pixels and return
# a masked map with NaNs instead
def set_unseen_to_nan(map):
    map[np.where(np.logical_or(map == hp.UNSEEN, np.logical_and(map < -1e30, map > -1e31)))] = np.nan
    return map


# convert a NaN scheme masked map back to the UNSEEN scheme for healpix manipulation
def set_nan_to_unseen(map):
    map[np.isnan(map)] = hp.UNSEEN
    return map


# convert UNSEEN scheme masked map to native numpy masked scheme
def set_unseen_to_mask(map):
    x = np.ma.masked_where(map == hp.UNSEEN, map)
    x.fill_value = hp.UNSEEN
    return x


# zeroes out alm amplitudes for less than a maximum l cutoff
def zero_modes(almarr, lmin_cut, lmax_cut):
    lmax = hp.Alm.getlmax(len(almarr))
    l, m = hp.Alm.getlm(lmax=lmax)
    almarr[np.where(l < lmin_cut)] = 0.0j
    almarr[np.where(l > lmax_cut)] = 0.0j
    return almarr


def wiener_filter(almarr):
    lmax = hp.Alm.getlmax(len(almarr))
    l, m = hp.Alm.getlm(lmax=lmax)

    noise_table = pd.read_csv('maps/nlkk.dat', delim_whitespace=True, header=None)
    cl_plus_nl = np.array(noise_table[2])
    nl = np.array(noise_table[1])
    cl = cl_plus_nl - nl

    wien_factor = cl/cl_plus_nl

    almarr = hp.smoothalm(almarr, beam_window=wien_factor)
    return almarr


def masked_smoothing(inmap, rad=5.0):
    inmap[np.where(inmap == hp.UNSEEN)] = np.nan
    copymap = inmap.copy()
    copymap[inmap != inmap] = 0
    smooth = hp.smoothing(copymap, fwhm=np.radians(rad))
    mask = 0 * inmap.copy() + 1
    mask[inmap != inmap] = 0
    smoothmask = hp.smoothing(mask, fwhm=np.radians(rad))
    final = smooth / smoothmask
    final[np.where(np.isnan(final))] = hp.UNSEEN
    return final

# read in a klm fits lensing convergence map, zero l modes desired, write out map
def klm_2_map(klmname, mapname, nsides):
    # read in planck alm convergence data
    planck_lensing_alm = hp.read_alm(klmname)
    filtered_alm = zero_modes(planck_lensing_alm, 100)
    # generate map from alm data
    planck_lensing_map = hp.sphtfunc.alm2map(filtered_alm, nsides, lmax=4096)
    hp.write_map(mapname, planck_lensing_map, overwrite=True)


# smooth map with gaussian of fwhm = width arcminutes
def smooth_map(mapname, width, outname):
    map = hp.read_map(mapname)
    fwhm = width/60.*np.pi/180.
    smoothed_map = hp.sphtfunc.smoothing(map, fwhm=fwhm)

    hp.write_map(outname, smoothed_map, overwrite=True)


# mask map and remove the mean field if desired
def mask_map(map, mask, outmap):
    # read in map and mask
    importmap = hp.read_map(map)
    importmask = hp.read_map(mask).astype(np.bool)
    # set mask, invert
    masked_map = hp.ma(importmap)
    masked_map.mask = np.logical_not(importmask)
    masked_map = masked_map.filled()

    hp.write_map(outmap, masked_map, overwrite=True)


# input klm file and output final smoothed, masked map for analysis
def klm_2_product(klmname, width, maskname, nsides, lmin, coord=None, subtract_mf=False, writename=None):

    # read in planck alm convergence data
    planck_lensing_alm = hp.read_alm(klmname)

    lmax_cut = 2 * nsides

    # if you're going to transform coordinates, usually from equatorial to galactic
    if coord is not None:
        r = hp.Rotator(coord=[coord, 'G'])
        planck_lensing_alm = r.rotate_alm(planck_lensing_alm)

    lmax_fixed = hp.Alm.getlmax(len(planck_lensing_alm))

    if subtract_mf:
        mf_alm = hp.read_alm('maps/mf_klm.fits')
        planck_lensing_alm = planck_lensing_alm - mf_alm

    # if you want to smooth with a gaussian
    if width > 0:
        # transform a gaussian of FWHM=width in real space to harmonic space
        k_space_gauss_beam = hp.gauss_beam(fwhm=width.to('radian').value, lmax=lmax_fixed)
        # if truncating small l modes
        if lmin > 0:
            # zero out small l modes in k-space filter
            k_space_gauss_beam[:lmin] = 0

        # smooth in harmonic space
        filtered_alm = hp.smoothalm(planck_lensing_alm, beam_window=k_space_gauss_beam)
    else:
        # if not smoothing with gaussian, just remove small l modes
        filtered_alm = zero_modes(planck_lensing_alm, lmin, lmax_cut)

    planck_lensing_map = hp.sphtfunc.alm2map(filtered_alm, nsides, lmax=lmax_fixed)

    # mask map
    importmask = hp.read_map(maskname)
    mask_nside = hp.npix2nside(len(importmask))
    if nsides != mask_nside:
        """mask_proper = hp.ud_grade(importmask.astype(float), nside_out=nsides).astype(float)

        if nsides < mask_nside:
            finalmask = np.where(mask_proper == 1., True, False).astype(bool)
        else:
            finalmask = np.where(mask_proper > 0, True, False).astype(bool)"""
        finalmask = masking.downgrade_mask(importmask, nsides)
    else:
        finalmask = importmask.astype(np.bool)


    if coord is not None:
        r = hp.Rotator(coord=[coord, 'G'])
        finalmask = r.rotate_map_pixel(finalmask)
    # set mask, invert
    smoothed_masked_map = hp.ma(planck_lensing_map)
    smoothed_masked_map.mask = np.logical_not(finalmask)


    if writename:
        hp.write_map('%s.fits' % writename, smoothed_masked_map.filled(), overwrite=True, dtype=np.single)

    return smoothed_masked_map.filled()


def write_planck_maps(real, noise, width, nsides, lmin):
    if real:
        klm_2_product(klmname='lensing_maps/planck/dat_klm.fits', width=width, maskname='lensing_maps/planck/mask.fits',
                      nsides=nsides, lmin=lmin, writename='lensing_maps/planck/smoothed_masked')
    if noise:
        realsnames = glob.glob('lensing_maps/planck/noise/klms/sim*')
        for j in range(len(realsnames)):
            klm_2_product(realsnames[j], width=width, maskname='lensing_maps/planck/mask.fits', nsides=nsides,
                          lmin=lmin, writename='lensing_maps/planck/noise/maps/%s' % j)




def weak_lensing_map(width):
    k_map = Table.read('lensing_maps/desy1/y1a1_spt_im3shape_0.9_1.3_kE.fits')['kE']
    mask = Table.read('lensing_maps/desy1/y1a1_spt_im3shape_0.9_1.3_mask.fits')['mask']

    smoothed = hp.smoothing(k_map, width.to('radian').value)
    smoothed_masked_map = hp.ma(smoothed)
    smoothed_masked_map.mask = np.logical_not(mask)

    hp.write_map('lensing_maps/desy1/smoothed_masked.fits', smoothed_masked_map.filled(), overwrite=True,
                 dtype=np.single)




"""def ACT_map(nside, lmax, smoothfwhm):
    bnlensing = enmap.read_map('ACTlensing/act_planck_dr4.01_s14s15_BN_lensing_kappa_baseline.fits')
    #bnlensing = enmap.read_map('ACTlensing/act_dr4.01_s14s15_BN_lensing_kappa.fits')
    bnmask = enmap.read_map('ACTlensing/act_dr4.01_s14s15_BN_lensing_mask.fits')
    wc_bn_mean = np.mean(np.array(bnmask) ** 2)
    bnlensing = bnlensing * wc_bn_mean

    bnlensing_hp = reproject.healpix_from_enmap(bnlensing, lmax=lmax, nside=nside)
    bnlensing_alm = hp.map2alm(bnlensing_hp, lmax=lmax)
    bnlensing_alm = zero_modes(bnlensing_alm, 100)
    bnlensing_hp = hp.alm2map(bnlensing_alm, nside, lmax)



    wc_bn = reproject.healpix_from_enmap(bnmask, lmax=lmax, nside=nside)


    #wc_bn_mean = np.mean(wc_bn**2)
    #bnlensing_hp = bnlensing_hp * wc_bn_mean



    smoothbn = hp.smoothing(bnlensing_hp, fwhm=(smoothfwhm * u.arcmin.to('rad')))


    smoothbn = healpixhelper.change_coord(smoothbn, ['C', 'G'])
    wc_bn = healpixhelper.change_coord(wc_bn, ['C', 'G'])
    smoothbn[np.where(wc_bn < 0.8)] = hp.UNSEEN

    hp.write_map('maps/BN.fits', smoothbn, overwrite=True)


    dlensing = enmap.read_map('ACTlensing/act_planck_dr4.01_s14s15_D56_lensing_kappa_baseline.fits')
    dmask = enmap.read_map('ACTlensing/act_dr4.01_s14s15_D56_lensing_mask.fits')
    wc_d_mean = np.mean(np.array(dmask) ** 2)
    dlensing = dlensing * wc_d_mean
    #dlensing = enmap.read_map('ACTlensing/act_dr4.01_s14s15_D56_lensing_kappa.fits')
    dlensing_hp = reproject.healpix_from_enmap(dlensing, lmax=lmax, nside=nside)
    dlensing_alm = hp.map2alm(dlensing_hp, lmax=lmax)
    dlensing_alm = zero_modes(dlensing_alm, 100)
    dlensing_hp = hp.alm2map(dlensing_alm, nside, lmax)




    wc_d = reproject.healpix_from_enmap(dmask, lmax=lmax, nside=nside)

    #wc_d_mean = np.mean(wc_d**2)

    #dlensing_hp = dlensing_hp * wc_d_mean
    smoothd = hp.smoothing(dlensing_hp, fwhm=smoothfwhm*u.arcmin.to('rad'))


    smoothd = healpixhelper.change_coord(smoothd, ['C', 'G'])
    wc_d = healpixhelper.change_coord(wc_d, ['C', 'G'])
    smoothd[np.where(wc_d < 0.8)] = hp.UNSEEN

    hp.write_map('maps/D56.fits', smoothd, overwrite=True)


    ddataidxs = np.where(wc_d > 0.8)
    combinedmask = wc_bn + wc_d

    smoothbn[ddataidxs] = smoothd[ddataidxs]
    smoothbn[np.where(combinedmask < 0.8)] = hp.UNSEEN

    hp.write_map('maps/both_ACT.fits', smoothbn, overwrite=True)


    planckmap = hp.read_map('maps/smoothed_masked_planck.fits')
    planckmap[np.where(combinedmask > 0.8)] = smoothbn[np.where(combinedmask > 0.8)]

    hp.write_map('maps/Planck_plus_ACT.fits', planckmap, overwrite=True)


    #return smoothbn


def sptpol_map(smoothfwhm):
    mvarr = np.load('lensing_maps/SPTpol/mv_map.npy')
    w = wcs.WCS(naxis=2)
    w.wcs.crval = [0, -59]
    w.wcs.crpix = [1260.5, 660.5]
    w.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
    w.wcs.cdelt = np.array([-0.0166667, 0.0166667])
    header = w.to_header()
    flippedarr = np.fliplr(np.flipud(mvarr))
    hdu = fits.PrimaryHDU(flippedarr, header=header)
    hdu.writeto('mv_tmp.fits', overwrite=True)
    imap = enmap.read_map('mv_tmp.fits')
    hpmap = imap.to_healpix(nside=4096)
    smoothmap = hp.smoothing(hpmap, fwhm=(smoothfwhm * u.arcmin.to('rad')))
    pixra, pixdec = hp.pix2ang(4096, np.arange(hp.nside2npix(4096)), lonlat=True)
    smoothmap[np.where(((pixra > 30) & (pixra < 330)) | (pixdec > -50) | (pixdec < -65))] = hp.UNSEEN
    hp.write_map('lensing_maps/SPTpol/smoothed_masked.fits', smoothmap, overwrite=True)
"""


