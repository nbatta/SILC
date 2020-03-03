import numpy as np
import warnings
import camb
from camb import model
from scipy.interpolate import interp1d

try:
    import cPickle as pickle
except:
    import pickle

import time, re, os


defaultConstants = {'TCMB': 2.7255,
                    'G_CGS': 6.67259e-08,
                    'MSUN_CGS': 1.98900e+33,
                    'MPC2CM': 3.085678e+24,
                    'ERRTOL': 1e-12,
                    'K_CGS': 1.3806488e-16,
                    'H_CGS': 6.62608e-27,
                    'C': 2.99792e+10
                    ,'A_ps': 3.1
                    ,'A_g': 0.9
                    ,'nu0': 150.
                    ,'n_g': -0.7
                    ,'al_g': 3.8
                    ,'al_ps': -0.5
                    ,'Td': 9.7
                    ,'al_cib': 2.2
                    ,'A_cibp': 6.9
                    ,'A_cibc': 4.9
                    ,'n_cib': 1.2
                    ,'A_tsz': 5.6
                    ,'ell0sec': 3000.
                    ,'zeta':0.1
}

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
            lmax=8000,
            clTTFixFile=None,
            skipCls=False,
            skip_growth=True,
            verbose=True,
            nonlinear=True,
            low_acc=False,
            fill_zero=True,
            dimensionless=True,
            pickling=False

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

        if (clTTFixFile is not None) and not(skipCls):
            ells,cltts = np.loadtxt(clTTFixFile,unpack=True)
            from scipy.interpolate import interp1d
            self.clttfunc = interp1d(ells,cltts,bounds_error=False,fill_value=0.)

        if not(low_acc):
            self.pars.set_accuracy(AccuracyBoost=2.0, lSampleBoost=4.0, lAccuracyBoost=4.0)
        else:
            self.pars.set_accuracy(AccuracyBoost=1.0, lSampleBoost=1.0, lAccuracyBoost=1.0)

        if nonlinear:
            self.pars.NonLinear = model.NonLinear_both
        else:
            self.pars.NonLinear = model.NonLinear_none
        if not(skipCls) and (clTTFixFile is None):
            if verbose: print("Generating theory Cls...")
            if not(low_acc):
                self.pars.set_for_lmax(lmax=(lmax+500), lens_potential_accuracy=3 if nonlinear else 0, max_eta_k=2*(lmax+500))
            else:
                self.pars.set_for_lmax(lmax=(lmax+500), lens_potential_accuracy=1 if nonlinear else 0, max_eta_k=2*(lmax+500))
            if nonlinear:
                self.pars.NonLinear = model.NonLinear_both
            else:
                self.pars.NonLinear = model.NonLinear_none
            theory = loadTheorySpectraFromPycambResults(self.results,self.pars,lmax,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=lmax,pickling=pickling,fill_zero=fill_zero,get_dimensionless=dimensionless,verbose=verbose,prefix="_low_acc_"+str(low_acc))
            self.clttfunc = lambda ell: theory.lCl('TT',ell)
            self.cleefunc = lambda ell: theory.lCl('EE',ell)
            self.theory = theory


def loadTheorySpectraFromPycambResults(results,pars,kellmax,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,pickling=False,fill_zero=False,get_dimensionless=True,verbose=True,prefix=""):

    if get_dimensionless:
        tmul = 1.
    else:
        tmul = TCMB**2.

    if useTotal:
        uSuffix = "unlensed_total"
        lSuffix = "total"
    else:
        uSuffix = "unlensed_scalar"
        lSuffix = "lensed_scalar"

    try:
        assert pickling
        clfile = "output/clsAll"+prefix+"_"+str(kellmax)+"_"+time.strftime('%Y%m%d') +".pkl"
        cmbmat = pickle.load(open(clfile,'rb'))
        if verbose: print("Loaded cached Cls from ", clfile)
    except:
        cmbmat = results.get_cmb_power_spectra(pars)
        if pickling:
            import os
            directory = "output/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            pickle.dump(cmbmat,open("output/clsAll"+prefix+"_"+str(kellmax)+"_"+time.strftime('%Y%m%d') +".pkl",'wb'))

    theory = TheorySpectra()
    for i,pol in enumerate(['TT','EE','BB','TE']):
        cls =cmbmat[lSuffix][2:,i]

        ells = np.arange(2,len(cls)+2,1)
        cls *= 2.*np.pi/ells/(ells+1.)*tmul
        theory.loadCls(ells,cls,pol,lensed=True,interporder="linear",lpad=lpad,fill_zero=fill_zero)

        if unlensedEqualsLensed:
            theory.loadCls(ells,cls,pol,lensed=False,interporder="linear",lpad=lpad,fill_zero=fill_zero)
        else:
            cls = cmbmat[uSuffix][2:,i]
            ells = np.arange(2,len(cls)+2,1)
            cls *= 2.*np.pi/ells/(ells+1.)*tmul
            theory.loadCls(ells,cls,pol,lensed=False,interporder="linear",lpad=lpad,fill_zero=fill_zero)

    try:
        assert pickling
        clfile = "output/clphi"+prefix+"_"+str(kellmax)+"_"+time.strftime('%Y%m%d') +".txt"
        clphi = np.loadtxt(clfile)
        if verbose: print("Loaded cached Cls from ", clfile)
    except:
        lensArr = results.get_lens_potential_cls(lmax=kellmax)
        clphi = lensArr[2:,0]
        if pickling:
            import os
            directory = "output/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.savetxt("output/clphi"+prefix+"_"+str(kellmax)+"_"+time.strftime('%Y%m%d') +".txt",clphi)

    clkk = clphi* (2.*np.pi/4.)
    ells = np.arange(2,len(clkk)+2,1)
    theory.loadGenericCls(ells,clkk,"kk",lpad=lpad,fill_zero=fill_zero)


    theory.dimensionless = get_dimensionless
    return theory


class TheorySpectra:

    '''
    Essentially just an interpolator that takes a CAMB-like
    set of discrete Cls and provides lensed and unlensed Cl functions
    for use in integrals
    '''


    def __init__(self):

        self.always_unlensed = False
        self.always_lensed = True
        self._uCl={}
        self._lCl={}
        self._gCl = {}


    def loadGenericCls(self,ells,Cls,keyName,lpad=9000,fill_zero=True):
        if not(fill_zero):
            fillval = Cls[ells<lpad][-1]
            self._gCl[keyName] = lambda x: np.piecewise(x, [x<=lpad,x>lpad], [lambda y: interp1d(ells[ells<lpad],Cls[ells<lpad],bounds_error=False,fill_value=0.)(y),lambda y: fillval*(lpad/y)**4.])

        else:
            fillval = 0.
            self._gCl[keyName] = interp1d(ells[ells<lpad],Cls[ells<lpad],bounds_error=False,fill_value=fillval)




    def gCl(self,keyName,ell):

        if len(keyName)==3:
            ultype = keyName[0].lower()
            if ultype=="u":
                return self.uCl(keyName[1:],ell)
            elif ultype=="l":
                return self.lCl(keyName[1:],ell)
            else:
                raise ValueError

        try:
            return self._gCl[keyName](ell)
        except:
            return self._gCl[keyName[::-1]](ell)

    def loadCls(self,ell,Cl,XYType="TT",lensed=False,interporder="linear",lpad=9000,fill_zero=True):

        mapXYType = XYType.upper()
        validateMapType(mapXYType)


        if not(fill_zero):
            fillval = Cl[ell<lpad][-1]
            f = lambda x: np.piecewise(x, [x<=lpad,x>lpad], [lambda y: interp1d(ell[ell<lpad],Cl[ell<lpad],bounds_error=False,fill_value=0.)(y),lambda y: fillval*(lpad/y)**4.])

        else:
            fillval = 0.
            f = interp1d(ell[ell<lpad],Cl[ell<lpad],bounds_error=False,fill_value=fillval)

        if lensed:
            self._lCl[XYType]=f
        else:
            self._uCl[XYType]=f


    def _Cl(self,XYType,ell,lensed=False):

        mapXYType = XYType.upper()
        validateMapType(mapXYType)

        if mapXYType=="ET": mapXYType="TE"
        ell = np.array(ell)

        try:
            if lensed:
                retlist = np.array(self._lCl[mapXYType](ell))
                return retlist
            else:
                retlist = np.array(self._uCl[mapXYType](ell))
                return retlist

        except:
            zspecs = ['EB','TB']
            if (XYType in zspecs) or (XYType[::-1] in zspecs):
                return ell*0.
            else:
                raise

    def uCl(self,XYType,ell):
        if self.always_lensed:
            assert not(self.always_unlensed)
            return self.lCl(XYType,ell)
        return self._Cl(XYType,ell,lensed=False)

    def lCl(self,XYType,ell):
        if self.always_unlensed:
            assert not(self.always_lensed)
            return self.uCl(XYType,ell)
        return self._Cl(XYType,ell,lensed=True)

def validateMapType(mapXYType):
    assert not(re.search('[^TEB]', mapXYType)) and (len(mapXYType)==2), \
      bcolors.FAIL+"\""+mapXYType+"\" is an invalid map type. XY must be a two" + \
      " letter combination of T, E and B. e.g TT or TE."+bcolors.ENDC
