import numpy as np
import warnings
import camb
from camb import model

defaultConstants = {'TCMB': 2.7255,
                    'G_CGS': 6.67259e-08,
                    'MSUN_CGS': 1.98900e+33,
                    'MPC2CM': 3.085678e+24,
                    'ERRTOL': 1e-12,
                    'K_CGS': 1.3806488e-16,
                    'H_CGS': 6.62608e-27,
                    'C': 2.99792e+10}

# Planck TT,TE,EE+lowP 2015 cosmology but with updated tau and minimal neutrino mass
defaultCosmology = {'omch2': 0.1198,
                    'ombh2': 0.02225,
                    'H0': 67.3,
                    'ns': 0.9645,
                    'As': 2.2e-9,
                    'mnu': 0.06,
                    'w0': -1.0,
                    'tau': 0.06,
                    'nnu': 3.046}


class Cosmology:
    '''
    Parts from orphics https://github.com/msyriac/orphics M
    A wrapper around CAMB that tries to pre-calculate as much as possible
    Intended to be inherited by other classes

    Many member functions were copied/adapted from Cosmicpy:
    http://cosmicpy.github.io/
    '''

    def __init__(
            self,
            paramDict=defaultCosmology,
            constDict=defaultConstants,
            kmax=10.,
            skip_growth=True,
            nonlinear=True,
            low_acc=False
    ):
        cosmo = paramDict
        self.paramDict = paramDict
        c = constDict
        self.c = c
        self.cosmo = paramDict

        self.c['TCMBmuK'] = self.c['TCMB'] * 1.0e6

        try:
            self.nnu = cosmo['nnu']
        except KeyError:
            self.nnu = defaultCosmology['nnu']

        self.omch2 = cosmo['omch2']
        self.ombh2 = cosmo['ombh2']

        self.rho_crit0H100 = 3. / (8. * np.pi) * (100 * 1.e5)**2. / c['G_CGS'] * c['MPC2CM'] / c['MSUN_CGS']

        try:
            self.tau = cosmo['tau']
        except KeyError:
            self.tau = defaultCosmology['tau']

        try:
            self.mnu = cosmo['mnu']
        except KeyError:
            self.mnu = defaultCosmology['mnu']
            warnings.warn("No mnu specified; assuming default of "+str(self.mnu))

        try:
            self.w0 = cosmo['w0']
        except KeyError:
            self.w0 = -1

        try:
            self.wa = cosmo['wa']
        except KeyError:
            self.wa = 0.

        self.pars = camb.CAMBparams()
        self.pars.Reion.Reionization = 0

        try:
            self.pars.set_dark_energy(w=self.w0, wa=self.wa, dark_energy_model='ppf')
        except:
            if np.abs(self.wa) >= 1e-3:
                # I am not sure this is actually true any more, as the master branch now has PPF by default - DC
                raise ValueError("Non-zero wa requires PPF, which requires devel version of pycamb to be installed.")

            warnings.warn("Could not use PPF dark energy model with pycamb. \
                           Falling back to non-PPF. Please install the devel branch of pycamb.")
            self.pars.set_dark_energy(w=self.w0)

        try:
            theta = cosmo['theta100']/100.
            H0 = None
            warnings.warn("Using theta100 parameterization. H0 ignored.")
        except KeyError:
            H0 = cosmo['H0']
            theta = None

        self.pars.set_cosmology(H0=H0, cosmomc_theta=theta, ombh2=self.ombh2, omch2=self.omch2,
                                mnu=self.mnu, tau=self.tau, nnu=self.nnu, num_massive_neutrinos=3)
        self.pars.Reion.Reionization = 0

        self.pars.InitPower.set_params(ns=cosmo['ns'], As=cosmo['As'])

        self.ns = cosmo['ns']
        self.As = cosmo['As']

        self.nonlinear = nonlinear
        if nonlinear:
            self.pars.NonLinear = model.NonLinear_both
        else:
            self.pars.NonLinear = model.NonLinear_none

        self.results = camb.get_background(self.pars)

        self.H0 = self.results.hubble_parameter(0.)
        assert self.H0 > 40. and self.H0 < 100.
        self.h = self.H0/100.

        self.om = (self.omch2+self.ombh2)/self.h**2.

        self.ob = (self.ombh2)/self.h**2.
        self.pars.set_matter_power(redshifts=[0.], kmax=5.0)

        # self.pars.NonLinear = model.NonLinear_none
        results = camb.get_results(self.pars)
        kh, z, pk = results.get_matter_power_spectrum(minkh=2e-4, maxkh=1, npoints=200)

        self.s8 = results.get_sigma8()

        self.omnuh2 = self.pars.omnuh2
