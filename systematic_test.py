import numpy as np
import astropy.units as u
import astropy.constants as con
import matplotlib.pyplot as plt


def planck_law(nu, t):
	return (2 * con.h * (nu ** 3) / (con.c) ** 2) * 1 / (np.exp(con.h * nu / (con.k_B * t)) - 1)


def beam_area(beam_fwhm):
	return (beam_fwhm.to('radian').value ** 2) * np.pi / (4 * np.log(2))


def cmb_spectrum(t, nus):
	return (planck_law(nus, t * u.K) * beam_area(15 * u.arcmin)).to('mJy')


def powerlaw_spec(refnu, refflux, newnu, alpha):
	return refflux * (newnu / refnu) ** alpha

avgt = 2.725
sigma = 0.00008

nusi = np.logspace(10, 12, 100) * u.Hz

plt.figure(figsize=(8,6))
hotspec = cmb_spectrum(avgt + sigma, nusi) / cmb_spectrum(2.725, nusi)
coldspec = cmb_spectrum(avgt - sigma, nusi) / cmb_spectrum(2.725, nusi)

plt.plot(nusi, hotspec, label=r'$\Delta T = + 80 \mu K$', c='b')
plt.plot(nusi, coldspec, label=r'$\Delta T = - 80 \mu K$', c='r')
#plt.plot(nusi, powerlaw_spec(1.4e9 * u.Hz, 1000, nusi, -0.5) / cmb_spectrum(2.725, nusi), c='k')
planck_bands = np.array([1e11, 1.43e11, 2.17e11, 3.53e11, 5.45e11, 8.57e11]) * u.Hz
#plt.scatter(planck_bands, cmb_spectrum(avgt, planck_bands), label='Planck Filters', c='b')

plt.yscale('log')
plt.xscale('log')
plt.legend(fontsize=20)
plt.ylabel(r'$S_{\nu}(T) / \langle S_{\nu} \rangle$', fontsize=20)
plt.xlabel(r'$\nu [\mathrm{Hz}]$', fontsize=20)
plt.savefig('plots/cmb_spec.pdf')
plt.close('all')
