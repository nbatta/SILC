from builtins import str
from builtins import range
from builtins import object
import numpy as np
from silc.foregrounds import fgNoises
from scipy.interpolate import interp1d
#from silc.cosmology import Cosmology
import silc.cosmology as cosmo
from orphics.cosmology import noise_func
from orphics.cosmology import LensForecast
from silc.foregrounds import f_nu

import sys,os
from configparser import SafeConfigParser 
from orphics.io import list_from_config


def weightcalculator(f,N):
    """
    Return single component weight
    W=f^TN/f^TNf
    """
    N_i=np.linalg.inv(N)
    C=np.matmul(f,np.matmul(N_i,np.transpose([f])))
    W=(1/C)*np.matmul(f,N_i)
    return W

def constweightcalculator(f_1,f_2,N):
    """
    Return weight such that upweight f_2,downweight f_1
    W=(f_1^TNf_1f_2^TN-f_2^TNf_1f_1^TN)/(f_1^TNf_1f_2^TNf_2-(f_2^TNf_1)^2)
    """
    C=np.matmul(np.transpose(f_1),np.matmul(N,f_1))*np.matmul(np.transpose(f_2),np.matmul(N,f_2))-(np.matmul(np.transpose(f_2),np.matmul(N,f_1)))**2
    M=np.matmul(np.transpose(f_1),np.matmul(N,f_1))*np.matmul(np.transpose(f_2),N)-np.matmul(np.transpose(f_2),np.matmul(N,f_1))*np.matmul(np.transpose(f_1),N)
    W=M/C
    return W

def combineexpnoise(A1,A2):
    '''
    Add together noise matrices of uncorrelated experiements
    '''
    #assert the shape of A,B 
    assert A1.shape[0] == A1.shape[1], "Matrix is not square"
    assert A2.shape[0] == A2.shape[1], "Matrix is not square"
    #add together matrices with uncorrelated experiments
    ans = np.block([[A1,np.zeros((len(A1), len(A2)))],[np.zeros((len(A2), len(A1))),A2]])
    return ans


class ILC_simple:
    def __init__(self, Cosmology, fgs ,fwhms=[1.5], rms_noises=[1.], freqs=[150.], lmax=8000, lknee=0., alpha=1., dell=1., v3mode=-1, fsky=None, noatm=False, name1='rsx', name2='None', add='None'):
        '''
        Inputs
        clusterCosmology is a class that contains cosmological parameters and power spectra.
        fgs is a class that contains the functional forms for the foregrounds and constants

        Options

        '''
        #initial set up for ILC
        self.cc = Cosmology

        #initializing the frequency matrices
        self.fgs = fgs
    
        self.dell = dell
        #set-up ells to evaluate up to lmax
        self.evalells = np.arange(2,lmax,self.dell)
        #Only for SO forecasts, including the SO atmosphere modeling
        #vmode selecting for observation source classification
        #vmode=-1 for everything other than SO and CCATp

        self.fsky = fsky

        #Set up the noise (this is overly complex)
        if v3mode==-1:
            freqs=np.array(freqs)

        elif v3mode>-1:
            print("V3 flag enabled.")
            import silc.V3_calc_public as v3
            import silc.so_noise_lat_v3_1_CAND as v3_1
            
            if v3mode <= 2:
                lat = v3_1.SOLatV3point1(v3mode,el=50.)
                vfreqs = lat.get_bands()
                print("Simons Obs")
                print("Replacing ",freqs,  " with ", vfreqs)
                N_bands = len(vfreqs)
                freqs = vfreqs
                vbeams = lat.get_beams()
                print("Replacing ",fwhms,  " with ", vbeams)
                fwhms = vbeams

                v3lmax = self.evalells.max()
                v3dell = np.diff(self.evalells)[0]
                print("Using ",fsky," for fsky")

                v3ell,N_ell_T_LA_full, N_ell_P_LA = lat.get_noise_curves(fsky, v3lmax+v3dell, v3dell, full_covar=True, deconv_beam=True)

                N_ell_T_LA = np.diagonal(N_ell_T_LA_full).T
                Map_white_noise_levels = lat.get_white_noise(fsky)**.5

            elif v3mode == 3:
                vfreqs = v3.AdvACT_bands()
                freqs = vfreqs
                vbeams = v3.AdvACT_beams()
                fwhms = vbeams

                v3lmax = self.evalells.max()
                v3dell = np.diff(self.evalells)[0]
                v3ell, N_ell_T_LA, N_ell_P_LA, Map_white_noise_levels = v3.AdvACT_noise(f_sky=fsky,ell_max=v3lmax+v3dell,delta_ell=v3dell)

            elif v3mode == 5:
                import silc.ccat_noise_200306 as ccatp
                lat = ccatp.CcatLatv2('baseline',el=50.)
                vfreqs = lat.get_bands()
                print("CCATP")
                print("Replacing ",freqs,  " with ", vfreqs)
                N_bands = len(vfreqs)
                freqs = vfreqs
                vbeams = lat.get_beams()
                print("Replacing ",fwhms,  " with ", vbeams)
                fwhms = vbeams

                v3lmax = self.evalells.max()
                v3dell = np.diff(self.evalells)[0]
                print("Using ",fsky," for fsky")

                v3ell,N_ell_T_LA_full, N_ell_P_LA = lat.get_noise_curves(fsky, v3lmax+v3dell, v3dell, full_covar=True, deconv_beam=True)
                #_noatm
                N_ell_T_LA = np.diagonal(N_ell_T_LA_full).T
                Map_white_noise_levels = lat.get_white_noise(fsky)**.5

            elif v3mode == 6:
                import silc.lat_noise_190819_w350ds4 as ccatp
                lat = ccatp.CcatLatv2(v3mode,el=50.,survey_years=4000/24./365.24,survey_efficiency=1.0)
                vfreqs = lat.get_bands()# v3.Simons_Observatory_V3_LA_bands()
                print("CCATP + SO goal")
                print("Replacing ",freqs,  " with ", vfreqs)
                N_bands = len(vfreqs)
                freqs = vfreqs
                vbeams = lat.get_beams()#v3.Simons_Observatory_V3_LA_beams() 
                print("Replacing ",fwhms,  " with ", vbeams)
                fwhms = vbeams

                v3lmax = self.evalells.max()
                v3dell = np.diff(self.evalells)[0]
                print("Using ",fsky," for fsky")

                v3ell,N_ell_T_LA_full, N_ell_P_LA = lat.get_noise_curves(fsky, v3lmax+v3dell, v3dell, full_covar=True, deconv_beam=True)
                #_noatm
                N_ell_T_LA = np.diagonal(N_ell_T_LA_full).T
                Map_white_noise_levels = lat.get_white_noise(fsky)**.5

            if add!='None':
                iniFile = "input/exp_config.ini"
                Config = SafeConfigParser()
                Config.optionxform=str
                Config.read(iniFile)

                experimentName1 = 'PlanckHFI'

                beams1 = list_from_config(Config,experimentName1,'beams')
                noises1 = list_from_config(Config,experimentName1,'noises')
                freqs1 = list_from_config(Config,experimentName1,'freqs')
                lmax1 = int(Config.getfloat(experimentName1,'lmax'))
                lknee1 = list_from_config(Config,experimentName1,'lknee')[0]
                alpha1 = list_from_config(Config,experimentName1,'alpha')[0]
                fsky1 = 0.6

                #Add Planck
                freqs = np.append(freqs,freqs1)

        #if (len(freqs) > 1):
        #    fq_mat   = np.matlib.repmat(freqs,len(freqs),1)
        #    fq_mat_t = np.transpose(np.matlib.repmat(freqs,len(freqs),1))
        #else:
        #    fq_mat   = freqs
        #    fq_mat_t = freqs 
        
        #initialize components to be saved
        #cmb_ell = np.array([])
        #cmb_ell_pol = np.array([])
        nell = np.array([])
        nell_pol = np.array([])
        totfg = np.array([])
        totfgrs = np.array([])
        totfg_cib = np.array([])
        fgrspol=np.array([])
        #ksz = np.array([])
        tsz = np.array([])
        cib = np.array([])
   
        for ii in range(len(self.evalells)):
            if v3mode < 0:
                inst_noise = (noise_func(self.evalells[ii],np.array(fwhms),np.array(rms_noises),lknee,alpha,dimensionless=False) /  self.cc.c['TCMBmuK']**2.)
                nells = abs(np.diag(inst_noise))
                nells_pol=nells*np.sqrt(2.)
            elif v3mode<=2:
                nells = N_ell_T_LA_full[:,:,ii]/ self.cc.c['TCMBmuK']**2.
                nells_pol = N_ell_P_LA[:,:,ii]/ self.cc.c['TCMBmuK']**2.

            elif v3mode==3:
                ndiags = []
                ndiags_pol = []
                for ff in range(len(freqs)):
                    inst_noise = N_ell_T_LA[ff,ii] / self.cc.c['TCMBmuK']**2.
                    ndiags.append(inst_noise)
                    inst_noise_pol = N_ell_P_LA[ff,ii] / self.cc.c['TCMBmuK']**2.
                    ndiags_pol.append(inst_noise_pol)
           
                nells = np.diag(np.array(ndiags))
                nells_pol = np.diag(np.array(ndiags_pol)) 

            elif v3mode>=5:
                nells = N_ell_T_LA_full[:,:,ii]/ self.cc.c['TCMBmuK']**2.
                nells_pol = N_ell_P_LA[:,:,ii]/ self.cc.c['TCMBmuK']**2.
            
                #evalells = np.arange(2,lmax1,dell1)
            if add!='None':
                inst_noise1 = noise_func(self.evalells[ii],np.array(beams1),np.array(noises1),lknee1,alpha1,dimensionless=False)/self.cc.c['TCMBmuK']**2.
                inst_noise1_pol = np.sqrt(2.)* inst_noise1
                
                nells_planck = np.diag(inst_noise1)
                nells_planck_pol = np.diag(inst_noise1_pol)
                nells=combineexpnoise(nells,nells_planck)
                nells_pol = combineexpnoise(nells_pol,nells_planck_pol)

            #cmb_ell = np.append(cmb_ell, self.cc.clttfunc(self.evalells[ii]))
            #cmb_ell_pol = np.append(cmb_ell_pol,self.cc.cleefunc(self.evalells[ii]))

            nell = np.append(nell,nells) #np.diagonal(nells* self.cc.c['TCMBmuK']**2.))
            nell_pol = np.append(nell_pol,nells_pol) #np.diagonal(nells_pol* self.cc.c['TCMBmuK']**2.))

            totfg = np.append(totfg, 
                              (self.fgs.rad_ps(self.evalells[ii],freqs[None,:],freqs[:,None]) + self.fgs.cib_p(self.evalells[ii],freqs[None,:],freqs[:,None]) +
                               self.fgs.cib_c(self.evalells[ii],freqs[None,:],freqs[:,None]) + self.fgs.tSZ_CIB(self.evalells[ii],freqs[None,:],freqs[:,None])) \
                                  / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi)

            totfgrs = np.append(totfgrs,
                                (self.fgs.rad_ps(self.evalells[ii],freqs[None,:],freqs[:,None]) + self.fgs.cib_p(self.evalells[ii],freqs[None,:],freqs[:,None]) +
                                 self.fgs.cib_c(self.evalells[ii],freqs[None,:],freqs[:,None]) + self.fgs.rs_auto(self.evalells[ii],freqs[None,:],freqs[:,None]) + \
                                     self.fgs.tSZ_CIB(self.evalells[ii],freqs[None,:],freqs[:,None])) \
                                    / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii] ) * 2.* np.pi)

                                #totfg+
                                #self.fgs.rs_auto(self.evalells[ii],freqs[None,:],freqs[:,None]) \
                                #    / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi)


            totfg_cib = np.append(totfg_cib,
                                  (self.fgs.rad_ps(self.evalells[ii],freqs[None,:],freqs[:,None]) + self.fgs.tSZ_CIB(self.evalells[ii],freqs[None,:],freqs[:,None])) \
                                      / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi)

            fgrspol=np.append(fgrspol,
                              (self.fgs.rs_autoEE(self.evalells[ii],freqs[None,:],freqs[:,None]))/ self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi)
                              
            #ksz = np.append(ksz,
            #                fq_mat*0.0 + self.fgs.ksz_temp(self.evalells[ii]) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi)

            tsz = np.append(tsz,
                            self.fgs.tSZ(self.evalells[ii],freqs[None,:],freqs[:,None]) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi)
                            
            cib = np.append(cib,
                            (self.fgs.cib_p(self.evalells[ii],freqs[None,:],freqs[:,None]) + self.fgs.cib_c(self.evalells[ii],freqs[None,:],freqs[:,None])) \
                                / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi)

        #self.nell_ll = np.reshape(nell_ll,[self.evalells,self.freq,self.freq])
        #self.nell_ll_pol = np.reshape(nell_ll_pol,[self.evalells,self.freq])

        self.freqs = freqs
        self.cmb_ell = self.cc.clttfunc(self.evalells) #np.reshape(cmb_ell,[self.evalells,self.freq,self.freq])
        self.cmb_ell_pol = self.cc.cleefunc(self.evalells)
        self.ksz = self.fgs.ksz_temp(self.evalells) / self.cc.c['TCMBmuK']**2. / ((self.evalells+1.)*self.evalells) * 2.* np.pi

        self.totfg = np.reshape(totfg,[len(self.evalells),len(self.freqs),len(self.freqs)])
        self.totfgrs = np.reshape(totfgrs,[len(self.evalells),len(self.freqs),len(self.freqs)])
        self.totfg_cib = np.reshape(totfg_cib,[len(self.evalells),len(self.freqs),len(self.freqs)])

        self.fgrspol = np.reshape(fgrspol,[len(self.evalells),len(self.freqs),len(self.freqs)])
        self.tsz = np.reshape(tsz,[len(self.evalells),len(self.freqs),len(self.freqs)])
        self.cib = np.reshape(cib,[len(self.evalells),len(self.freqs),len(self.freqs)])

        #np.reshape(cmb_ell_pol,[self.evalells,self.freq,self.freq])
        self.nells = np.reshape(nell,[len(self.evalells),len(self.freqs),len(self.freqs)])
        self.nells_pol =np.reshape(nell_pol,[len(self.evalells),len(self.freqs),len(self.freqs)])

    def gen_ilc(self,f,g,constrained=None,noforegrounds=None):

        Nll_ilc = np.array([])
        Nll_pol_ilc = np.array([])
        Wll_out = []
        Wll_pol_out= []
        
        for ii in range(len(self.evalells)):
            if noforegrounds == None:
                Nll = self.totfgrs[ii,:,:] + self.cmb_ell[ii,None,None] + self.tsz[ii,:,:] + self.ksz[ii,None,None] + self.nells[ii,:,:]
                Nll_pol = self.cmb_ell_pol[ii,None,None] + self.fgrspol[ii,:,:] + self.nells_pol[ii,:,:]

            else:
                Nll = self.nells[ii,:,:]
                Nll_pol = self.nells_pol[ii,:,:]

            Nll_inv=np.linalg.inv(Nll)
            Nll_pol_inv=np.linalg.inv(Nll_pol)

            if constrained == None:
                Wll = weightcalculator(f, Nll)
                Wll_pol = weightcalculator(f, Nll_pol)
            else:
                Wll = constweightcalculator(g,f,Nll_inv)
                Wll_pol = constweightcalculator(g,f,Nll_pol_inv)
                
            Wll_out.append(Wll)
            Wll_pol_out.append(Wll_pol)
            Nll_ilc = np.append(Nll_ilc, np.dot(np.transpose(Wll), np.dot(Nll, Wll)))
            Nll_pol_ilc = np.append(Nll_pol_ilc, np.dot(np.transpose(Wll_pol), np.dot(Nll_pol, Wll_pol)))

        return Nll_ilc, Nll_pol_ilc, Wll_out, Wll_pol_out

    def gen_ilc_nopol(self,f,g,constrained=None,noforegrounds=None):

        Nll_ilc = np.array([])
        Wll_out = []

        for ii in range(len(self.evalells)):
            if noforegrounds == None:
                Nll = self.totfgrs[ii,:,:] + self.cmb_ell[ii,None,None] + self.tsz[ii,:,:] + self.ksz[ii,None,None] + self.nells[ii,:,:]
            else:
                Nll = self.nells[ii,:,:]

            Nll_inv=np.linalg.inv(Nll)

            if constrained == None:
                Wll = weightcalculator(f, Nll)
            else:
                Wll = constweightcalculator(g,f,Nll_inv)

            Wll_out.append(Wll)
            Nll_ilc = np.append(Nll_ilc, np.dot(np.transpose(Wll), np.dot(Nll, Wll)))

        return Nll_ilc, Wll_out

    def cmb_opt(self,constrained='rs',noforegrounds=None,returnW=False):
        f = self.freqs*0.0 + 1. #CMB

        if constrained=='rs':
            g = self.fgs.rs_nu(np.array(self.freqs)) #Rayliegh
        elif constrained=='tsz':
            g = f_nu(self.cc.c,np.array(self.freqs)) #tSZ
        elif constrained=='cib':
            g = self.fgs.f_nu_cib(np.array(self.freqs)) #CIB
        else:
            g = 0

        Nll, Nll_pol, Wll, Wll_pol = self.gen_ilc(f,g,constrained,noforegrounds)

        if (returnW):
            return Nll, Nll_pol, Wll, Wll_pol
        else:
            return Nll, Nll_pol

    def rs_opt(self,constrained='cmb',noforegrounds=None,returnW=False):
        f = self.fgs.rs_nu(np.array(self.freqs)) #Rayliegh

        if constrained=='cmb':
            g = self.freqs*0.0 + 1. #CMB
        else:
            g = 0

        Nll, Nll_pol, Wll, Wll_pol = self.gen_ilc(f,g,constrained,noforegrounds)

        if (returnW):
            return Nll, Nll_pol, Wll, Wll_pol
        else:
            return Nll, Nll_pol

    def tsz_opt(self,constrained=None,noforegrounds=None,returnW=False):
        f = f_nu(self.cc.c,np.array(self.freqs)) #tSZ

        if constrained=='cmb':
            g = self.freqs*0.0 + 1. #CMB
        elif constrained=='cib':
            g = self.fgs.f_nu_cib(np.array(self.freqs)) #CIB
        else:
            g = 0

        Nll, Wll = self.gen_ilc_nopol(f,g,constrained,noforegrounds)

        if (returnW):
            return Nll, Wll
        else:
            return Nll

    '''
    def err_calc(ell,C1,N1):
        
        return s2n, errs
    '''
    def cross_err_calc(self,ell,C1,N1,C2,N2,X,detect=True):
        covs = []
        s2n=[]
        
        for k in range(len(ell)):
            i=int(ell[k])
            ClSum = np.nan_to_num(((C1[k]+N1[k])*(C2[k]+N2[k])+(X[i])**2))
            if (detect):
                s2nper=(2*i+1)*np.nan_to_num((X[i]**2)/((C1[k]+N1[k])*(C2[k]+N2[k])))
            else:
                s2nper=(2*i+1)*np.nan_to_num((X[i]**2)/((C1[k]+N1[k])*(C2[k]+N2[k])+(X)**2))
            var = ClSum/(2.*i+1.)/self.fsky
            covs.append(var)
            s2n.append(s2nper)

        errs=np.sqrt(np.array(covs))
        s2n=self.fsky*sum(s2n)
        s2n=np.sqrt(s2n)
        return s2n,errs

    #def Rayleigh_forecast(self,ellBinEdges,type='tt',detection=True):
    def Rayleigh_forecast(self,ellmax,type='tt',detection=True,noforegrounds=None):

        #ellMids = (ellBinEdges[1:] + ellBinEdges[:-1]) / 2
        #ellWidths = np.diff(ellBinEdges)
        signalfile ="input/CMB_rayleigh_500.dat"
        fsky = self.fsky

        ells = np.arange(2,ellmax,1)

        Nll_cmb, Nll_cmb_pol = self.cmb_opt(noforegrounds=noforegrounds)
        Nll_rs,  Nll_rs_pol  = self.rs_opt(noforegrounds=noforegrounds)
        
        cmb = self.cmb_ell
        cmb_pol = self.cmb_ell_pol
        
        rs = self.fgs.rs_auto(self.evalells,self.fgs.nu_rs,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
            / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        
        rs_pol = self.fgs.rs_autoEE(self.evalells,self.fgs.nu_rs,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
            / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        
        ell_temp=np.loadtxt(signalfile,unpack=True,usecols=[0])

        assert np.min(ell_temp) == np.min(self.evalells) and np.min(ells) == np.min(self.evalells), "Issue with template ells"
        assert np.max(ells) < np.max(self.evalells) and np.max(ells) < np.max(ell_temp), "Issue with template ells"
        
        if type=='tt': #T_CMB T_rs
            clsX=np.loadtxt(signalfile,unpack=True,usecols=[4]) 
            clsX=clsX*(self.fgs.nu_rs/500)**4 / self.cc.c['TCMBmuK']**2./ ((ell_temp+1.)*ell_temp) * 2.* np.pi
            sn,NllX = self.cross_err_calc(ells,cmb,Nll_cmb,rs,Nll_rs,clsX,detection)
        elif type=='te':#T_CMB E_rs
            clsX=np.loadtxt(signalfile,unpack=True,usecols=[5]) 
            clsX=clsX*(self.fgs.nu_rs/500)**4 / self.cc.c['TCMBmuK']**2./ ((ell_temp+1.)*ell_temp) * 2.* np.pi
            sn,NllX = self.cross_err_calc(ells,cmb,Nll_cmb,rs_pol,Nll_rs_pol,clsX,detection)
        elif type=='et': #E_CMB T_rs
            clsX=np.loadtxt(signalfile,unpack=True,usecols=[6]) 
            clsX=clsX*(self.fgs.nu_rs/500)**4 / self.cc.c['TCMBmuK']**2./ ((ell_temp+1.)*ell_temp) * 2.* np.pi
            sn, NllX = self.cross_err_calc(ells,cmb_pol,Nll_cmb_pol,rs,Nll_rs,clsX,detection)
        elif type=='ee': #E_CMB E_rs
            clsX=np.loadtxt(signalfile,unpack=True,usecols=[7]) 
            clsX=clsX*(self.fgs.nu_rs/500)**4 / self.cc.c['TCMBmuK']**2./ ((ell_temp+1.)*ell_temp) * 2.* np.pi
            sn, NllX = self.cross_err_calc(ells,cmb_pol,Nll_cmb_pol,rs_pol,Nll_rs_pol,clsX,detection)
        else:
            print('wrong option')
            
        return clsX, NllX, sn

    '''
    def tsz_forecast(self,ellmax,constrained=None):
        fsky = self.fsky
        ells = np.arange(2,ellmax,1)
        Nll_tsz = self.tsz_opt()
        return cls,nlls,sn
    '''    


#Cross ET            #ells=ells[0:ellmax]
            

            #cls_out=clsout[0:ellmax]
            #cls_out=cls_out/ self.cc.c['TCMBmuK']**2./ ((ells+1.)*ells) * 2.* np.pi

            #sn2=(2.*self.evalells+1.)*np.nan_to_num((cls_out**2)/((clrsee+errrsee)*(cltt+errtt)+(cls_out)**2))                                                                                                        
            #covs = []
            #s2n=[]
            #l=l1
            #for k in range(len(l)):
            #    i=int(l[k])
            #    ClSum = np.nan_to_num(((clrsee[k]+errrsee[k])*(cltt[k]+errtt[k])+(cls_out[i])**2))
            #    s2nper=(2*i+1)*np.nan_to_num((cls_out[i]**2)/((clrsee[k]+errrsee[k])*(cltt[k]+errtt[k])+(cls_out[i])**2))
            #    var = ClSum/(2.*i+1.)/fsky2/400
            #    covs.append(var)
            #    s2n.append(s2nper)
            #errs=np.sqrt(np.array(covs))
            #s2n=fsky2/2.*sum(s2n)
            #s2n=np.sqrt(s2n)
            #return ellMids,cls_out,errs,s2n


        #return cls,nlls,S/N 



    '''
        self.W_ll = np.zeros([len(self.evalells),len(np.array(freqs))])
        self.W_ll_1_c_2  = np.zeros([len(self.evalells),len(np.array(freqs))])
        self.W_ll_2_c_1  = np.zeros([len(self.evalells),len(np.array(freqs))])
        self.freqs=freqs
        if name1=='tsz':
            f = f_nu(self.cc.c,np.array(freqs)) #tSZ
        elif name1=='cmb':
            f = f_nu(self.cc.c,np.array(freqs))
            f = f*0.0 + 1. #CMB
        elif name1=='cmbee':
            f = f_nu(self.cc.c,np.array(freqs))
            f = f*0.0 + 1.
        elif name1=='cib':
            f = self.fgs.f_nu_cib(np.array(freqs)) #CIB
        elif name1=='rsx':
            f = self.fgs.rs_nu(np.array(freqs)) #Rayleigh Cross
        elif name1=='rsxee':
            f = self.fgs.rs_nu(np.array(freqs)) #Rayleigh Cross
        else:
            f = f_nu(self.cc.c,np.array(freqs))
            f = f*0.0
            
        if name2=='tsz':
            g = f_nu(self.cc.c,np.array(freqs)) #tSZ
        elif name2=='cmb':
            g = f_nu(self.cc.c,np.array(freqs))
            g = g*0.0 + 1. #CMB
        elif name2=='cmbee':
            g = f_nu(self.cc.c,np.array(freqs))
            g = f*0.0 + 1.
        elif name2=='cib':
            g = self.fgs.f_nu_cib(np.array(freqs)) #CIB
        elif name2=='rsx':
            g = self.fgs.rs_nu(np.array(freqs)) #Rayleigh Cross
        elif name2=='rsxee':
            g = self.fgs.rs_nu(np.array(freqs)) #Rayleigh Cross
        else:
            g = f_nu(self.cc.c,np.array(freqs))
            g = g*0.0
        nell_ll=[]
        nell_ee_ll=[]
        #noise2=[]
        #f_1=self.fgs.rs_nu(np.array(self.freqs))
        #f_nu2=f_1*0.0+1.


            
            #covariance matrices, which include information from all signals
            #### NB THIS IS NOT RIGHT
            if name1!='rsxee' and name2!='cmbee':
                N_ll= totfgrs + cmb_els + tsz + ksz + nells#*1000
            else:
                N_ll= cmbee + fgrspol + nells
        
            #print (N_ll)
    
            #self.cov=N_ll
            N_ll_NoFG=nells#*1000
            #N_ll_pol=nellsee+cmbee+fgrspol
            #N_ll_pol_NoFG=nellsee
            
            N_ll_inv=np.linalg.inv(N_ll)
            self.N=N_ll
            self.Ninv=N_ll_inv
            N_ll_NoFG_inv=np.linalg.inv(N_ll_NoFG)
            #N_ll_pol_inv=np.linalg.inv(N_ll_pol)
            #N_ll_pol_NoFG_inv=np.linalg.inv(N_ll_pol_NoFG)
            self.W_ll[ii,:]=weightcalculator(f,N_ll)
            self.N_ll[ii] = np.dot(np.transpose(self.W_ll[ii,:]),np.dot(N_ll,self.W_ll[ii,:]))
            self.W_ll_1_c_2[ii,:]=constweightcalculator(g,f,self.Ninv)
            self.W_ll_2_c_1[ii,:]=constweightcalculator(f,g,self.Ninv)
        
            self.N_ll_1_c_2 [ii] = np.dot(np.transpose(self.W_ll_1_c_2[ii,:]) ,np.dot(self.N, self.W_ll_1_c_2[ii,:]))
            self.N_ll_2_c_1 [ii] = np.dot(np.transpose(self.W_ll_2_c_1[ii,:]) ,np.dot(self.N, self.W_ll_2_c_1[ii,:]))
    '''

    def GeneralClCalc(self,ellBinEdges,fsky,name1='None',name2='None',constraint='None'):
        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1])/2
        if name1=='tsz':
            cls = self.fgs.tSZ(self.evalells,self.freqs[0],self.freqs[0]) / self.cc.c['TCMBmuK']**2. \
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi

        elif name1=='cmb':
            cls = self.cc.clttfunc(self.evalells)
        elif name1=='cmbee':
            cls = self.cc.cleefunc(self.evalells)
        elif name1=='cib':
            pass#f_nu_cib = self.fgs.f_nu_cib(np.array(freqs)) #CIB
        elif name1=='rsx':
            cls = self.fgs.rs_cross(self.evalells,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        elif name1=='rsxee':
            cls = self.fgs.rs_autoEE(self.evalells,self.fgs.nu_rs,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        else:
            return 'wrong input'

        LF = LensForecast()
        if (constraint=='None'):
            LF.loadGenericCls("tt",self.evalells,cls,self.evalells,self.N_ll)
        else:
            return "Wrong option"

        sn,errs = LF.sn(ellBinEdges,fsky,"tt") # not squared

        cls_out = np.interp(ellMids,self.evalells,cls)

        return ellMids,cls_out,errs,sn
    
    def GeneralClCalcrsx(self,ellBinEdges,fsky,name1='rsx',name2='None'):
        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1])/ 2
        
        if name1=='tsz':
            cls1 = self.fgs.tSZ(self.evalells,self.freqs[0],self.freqs[0]) / self.cc.c['TCMBmuK']**2. \
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi

        elif name1=='cmb':
            cls1 = self.cc.clttfunc(self.evalells)
        elif name1=='cmbee':
            cls1 = self.cc.cleefunc(self.evalells)
        elif name1=='cib':
            pass#f_nu_cib = self.fgs.f_nu_cib(np.array(freqs)) #CIB
        elif name1=='rsx':
            cls1 = self.fgs.rs_cross(self.evalells,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        elif name1=='rsxee':
            cls1 = self.fgs.rs_crossEE(self.evalells,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        else:
            return 'wrong input'
        if name2=='tsz':
            cls2 = self.fgs.tSZ(self.evalells,self.freqs[0],self.freqs[0]) / self.cc.c['TCMBmuK']**2. \
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi

        elif name2=='cmb':
            cls2 = self.cc.clttfunc(self.evalells)
        elif name2=='cmbee':
            cls2 = self.cc.cleefunc(self.evalells)
        elif name2=='cib':
            pass#f_nu_cib = self.fgs.f_nu_cib(np.array(freqs)) #CIB
        elif name2=='rsx':
            cls2 = self.fgs.rs_cross(self.evalells,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        elif name2=='rsxee':
            cls2 = self.fgs.rs_crossEE(self.evalells,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        else:
            return 'wrong input'
        
        if name1=='rsx':
            cls_rs = self.fgs.rs_auto(self.evalells,self.fgs.nu_rs,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        elif name1=='rsxee':
            cls_rs = self.fgs.rs_autoEE(self.evalells,self.fgs.nu_rs,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        else:
            return 'wrong input'
                
        LF = LensForecast()

        LF.loadGenericCls("rr",self.evalells,cls_rs,self.evalells,self.N_ll_1_c_2)#self.N_ll_rsx)
        LF.loadGenericCls("xx",self.evalells,cls1,self.evalells,self.N_ll_1_c_2*0.0)
        LF.loadGenericCls("tt",self.evalells,cls2,self.evalells,self.N_ll_2_c_1)#self.N_ll_cmb)
        Nellrs = self.N_ll_1_c_2 #self.N_ll_rsx
        #print(Nellrs)
        Nell2 = self.N_ll_2_c_1 #self.N_ll_cmb
        
        sn2=(2.*self.evalells+1.)*np.nan_to_num((cls1**2)/((cls_rs*0.0+Nellrs)*(cls2+Nell2)+(cls1)**2))
        snsq=fsky/2.*sum(sn2)
        sn=np.sqrt(snsq)
        cls_out = np.interp(ellMids,self.evalells,cls1)

        #errs = cls_out * 0.0 + 1.
        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2
        ellWidths = np.diff(ellBinEdges)

        covs = []
        er=[]
        for ell_left,ell_right in zip(ellBinEdges[:-1],ellBinEdges[1:]):
            ClSum = LF._bin_cls("rr",ell_left,ell_right)*LF._bin_cls("tt",ell_left,ell_right)+(LF._bin_cls("xx",ell_left,ell_right))**2
            er.append(ClSum)
            ellMid = (ell_right+ell_left)/2.
            ellWidth = ell_right-ell_left
            var = ClSum/(2.*ellMid+1.)/ellWidth/fsky
            covs.append(var)
        errs=np.sqrt(np.array(covs))
        #print(er)

        return ellMids,cls_out,errs,sn
    
    def GeneralClCalccmb(self,ellBinEdges,fsky,name1='rsx',name2='cmb'):
        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1])/ 2
        
        if name1=='tsz':
            cls1 = self.fgs.tSZ(self.evalells,self.freqs[0],self.freqs[0]) / self.cc.c['TCMBmuK']**2. \
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi

        elif name1=='cmb':
            cls1 = self.cc.clttfunc(self.evalells)
        elif name1=='cmbee':
            cls1 = self.cc.cleefunc(self.evalells)
        elif name1=='cib':
            pass#f_nu_cib = self.fgs.f_nu_cib(np.array(freqs)) #CIB
        elif name1=='rsx':
            cls1 = self.fgs.rs_cross(self.evalells,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        elif name1=='rsxee':
            cls1 = self.fgs.rs_crossEE(self.evalells,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        else:
            return 'wrong input'
        if name2=='tsz':
            cls2 = self.fgs.tSZ(self.evalells,self.freqs[0],self.freqs[0]) / self.cc.c['TCMBmuK']**2. \
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi

        elif name2=='cmb':
            cls2 = self.cc.clttfunc(self.evalells)
        elif name2=='cmbee':
            cls2 = self.cc.cleefunc(self.evalells)
        elif name2=='cib':
            pass#f_nu_cib = self.fgs.f_nu_cib(np.array(freqs)) #CIB
        elif name2=='rsx':
            cls2 = self.fgs.rs_cross(self.evalells,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        elif name2=='rsxee':
            cls2 = self.fgs.rs_crossEE(self.evalells,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        else:
            return 'wrong input'
        
        if name1=='rsx':
            cls_rs = self.fgs.rs_auto(self.evalells,self.fgs.nu_rs,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        elif name1=='rsxee':
            cls_rs = self.fgs.rs_autoEE(self.evalells,self.fgs.nu_rs,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
        else:
            return 'wrong input'
 
        LF = LensForecast()
        LF.loadGenericCls("rr",self.evalells,cls_rs,self.evalells,self.N_ll_2_c_1)#self.N_ll_rsx)
        LF.loadGenericCls("xx",self.evalells,cls1,self.evalells,self.N_ll_2_c_1*0.0)
        LF.loadGenericCls("tt",self.evalells,cls2,self.evalells,self.N_ll_1_c_2)#self.N_ll_cmb)
        Nellrs = self.N_ll_2_c_1 #self.N_ll_rsx
        #print(Nellrs)
        Nell2 = self.N_ll_1_c_2 #self.N_ll_cmb
        
        sn2=(2.*self.evalells+1.)*np.nan_to_num((cls1**2)/((cls_rs*0.0+Nellrs)*(cls2+Nell2)+(cls1)**2))
        snsq=fsky/2.*sum(sn2)
        sn=np.sqrt(snsq)
        cls_out = np.interp(ellMids,self.evalells,cls1)

        #errs = cls_out * 0.0 + 1.
        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2
        ellWidths = np.diff(ellBinEdges)

        covs = []
        er=[]
        for ell_left,ell_right in zip(ellBinEdges[:-1],ellBinEdges[1:]):
            ClSum = LF._bin_cls("rr",ell_left,ell_right)*LF._bin_cls("tt",ell_left,ell_right)+(LF._bin_cls("xx",ell_left,ell_right))**2
            er.append(ClSum)
            ellMid = (ell_right+ell_left)/2.
            ellWidth = ell_right-ell_left
            var = ClSum/(2.*ellMid+1.)/ellWidth/fsky
            covs.append(var)
        errs=np.sqrt(np.array(covs))
        return ellMids,cls_out,errs,sn
    

    def Noise_ellrsx(self,option='None'):
        if (option=='None'):
            return self.evalells,self.N_ll
        elif (option=='NoILC'):
            return self.evalells,self.N_ll_1_c_2
        else:
            return "Wrong option"
        
    def Forecast_CellrsxEEPlanck(self,ellBinEdges,fsky,option='None', add='None',ellmax=3000):
        '''
        RS E cross CMB T
        '''
        if add=='None':
            pass
        else:
            fsky1 = 0.6
        
          
            fsky2 = fsky
            
            l1,cltt,errtt,sn1=self.GeneralClCalccmb(ellBinEdges,fsky=0.6,name1='rsxee',name2='cmb')#,constraint='None')
            #print(cltt)
            l2,clrsee,errrsee,sn2=self.GeneralClCalcrsx(ellBinEdges,fsky=fsky2,name1='rsxee',name2='cmb')##,constraint='None')
            cls_rs = self.fgs.rs_autoEE(self.evalells,self.fgs.nu_rs,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
            #clsout=cls_rs*cltt
            ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2
            ellWidths = np.diff(ellBinEdges)
            signal="input/CMB_rayleigh_500.dat"
            ells,clsout=np.loadtxt(signal,unpack=True,usecols=[0,5]) #Cross ET
            ells=ells[0:ellmax]
            clsout=clsout*(self.fgs.nu_rs/500)**4
            
            cls_out=clsout[0:ellmax]
            cls_out=cls_out/ self.cc.c['TCMBmuK']**2./ ((ells+1.)*ells) * 2.* np.pi
        
            #sn2=(2.*self.evalells+1.)*np.nan_to_num((cls_out**2)/((clrsee+errrsee)*(cltt+errtt)+(cls_out)**2))
            covs = []
            s2n=[]
            l=l1
            for k in range(len(l)):
                i=int(l[k])
                ClSum = np.nan_to_num(((clrsee[k]+errrsee[k])*(cltt[k]+errtt[k])+(cls_out[i])**2))
                s2nper=(2*i+1)*np.nan_to_num((cls_out[i]**2)/((clrsee[k]+errrsee[k])*(cltt[k]+errtt[k])+(cls_out[i])**2))
                var = ClSum/(2.*i+1.)/fsky2/400
                covs.append(var)
                s2n.append(s2nper)
            errs=np.sqrt(np.array(covs))
            s2n=fsky2/2.*sum(s2n)
            s2n=np.sqrt(s2n)
            return ellMids,cls_out,errs,s2n
        
        
    def Forecast_CellrsxPlanck(self,ellBinEdges,fsky,option='None', add='None',ellmax=3000):
        '''
        RS T cross CMB E
        '''
        if add=='None':
            pass
        else:
            fsky1 = 0.6
    
            fsky2 = fsky
      
            l1,cltt,errtt,sn1=self.GeneralClCalccmb(ellBinEdges,fsky=0.6,name1='rsx', name2='cmbee')#,constraint='None')
            #print(cltt)
            l2,clrsee,errrsee,sn2=self.GeneralClCalcrsx(ellBinEdges,fsky=fsky2,name1='rsx',name2='cmbee')##,constraint='None')
            cls_rs = self.fgs.rs_auto(self.evalells,self.fgs.nu_rs,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
            #clsout=cls_rs*cltt
            ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2
            ellWidths = np.diff(ellBinEdges)
            signal="input/CMB_rayleigh_500.dat"
            ells,clsout=np.loadtxt(signal,unpack=True,usecols=[0,4]) #Cross TE
            ells=ells[0:ellmax]
            clsout=clsout*(self.fgs.nu_rs/500)**4
            
            cls_out=clsout[0:ellmax]
            cls_out=cls_out/ self.cc.c['TCMBmuK']**2./ ((ells+1.)*ells) * 2.* np.pi
        
            #sn2=(2.*self.evalells+1.)*np.nan_to_num((cls_out**2)/((clrsee+errrsee)*(cltt+errtt)+(cls_out)**2))
            covs = []
            s2n=[]
            l=l1
            for k in range(len(l)):
                i=int(l[k])
                ClSum = np.nan_to_num(((clrsee[k]+errrsee[k])*(cltt[k]+errtt[k])+(cls_out[i])**2))
                s2nper=(2*i+1)*np.nan_to_num((cls_out[i]**2)/((clrsee[k]+errrsee[k])*(cltt[k]+errtt[k])+(cls_out[i])**2))
                var = ClSum/(2.*i+1.)/fsky2/400
                covs.append(var)
                s2n.append(s2nper)
            errs=np.nan_to_num(np.sqrt(np.array(covs)))
            s2n=fsky2/2.*sum(s2n)
            s2n=np.nan_to_num(np.sqrt(s2n))
            return ellMids,cls_out,errs,s2n
        
    def Forecast_CellrsxTTPlanck(self,ellBinEdges,fsky,option='None', add='None',ellmax=3000):
        '''
        RS T cross CMB T
        '''
        if add=='None':
            pass
        else:

            fsky1 = 0.6
    
            fsky2 = fsky
      
            l1,cltt,errtt,sn1=self.GeneralClCalccmb(ellBinEdges,fsky=0.6,name1='rsx', name2='cmb')#,constraint='None')
            #print(cltt)
            l2,clrsee,errrsee,sn2=self.GeneralClCalcrsx(ellBinEdges,fsky=fsky2,name1='rsx',name2='cmb')##,constraint='None')
            cls_rs = self.fgs.rs_auto(self.evalells,self.fgs.nu_rs,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
            #clsout=cls_rs*cltt
            ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2
            ellWidths = np.diff(ellBinEdges)
            signal="input/CMB_rayleigh_500.dat"
            ells,clsout=np.loadtxt(signal,unpack=True,usecols=[0,3])#need check
            ells=ells[0:ellmax]
            clsout=clsout*(self.fgs.nu_rs/500)**4
            
            cls_out=clsout[0:ellmax]
            cls_out=cls_out/ self.cc.c['TCMBmuK']**2./ ((ells+1.)*ells) * 2.* np.pi
        
            #sn2=(2.*self.evalells+1.)*np.nan_to_num((cls_out**2)/((clrsee+errrsee)*(cltt+errtt)+(cls_out)**2))
            covs = []
            s2n=[]
            l=l1
            for k in range(len(l)):
                i=int(l[k])
                ClSum = np.nan_to_num(((clrsee[k]+errrsee[k])*(cltt[k]+errtt[k])+(cls_out[i])**2))
                s2nper=(2*i+1)*np.nan_to_num((cls_out[i]**2)/((clrsee[k]+errrsee[k])*(cltt[k]+errtt[k])+(cls_out[i])**2))
                var = ClSum/(2.*i+1.)/fsky2/400
                covs.append(var)
                s2n.append(s2nper)
            errs=np.nan_to_num(np.sqrt(np.array(covs)))
            s2n=fsky2/2.*sum(s2n)
            s2n=np.nan_to_num(np.sqrt(s2n))
            return ellMids,cls_out,errs,s2n

    def Forecast_CellrsxPPPlanck(self,ellBinEdges,fsky,option='None', add='None',ellmax=3000):
        '''
        RS E cross CMB E
        '''
        if add=='None':
            pass
        else:
            fsky1 = 0.6
    
            fsky2 = fsky
      
            l1,cltt,errtt,sn1=self.GeneralClCalccmb(ellBinEdges,fsky=0.6,name1='rsxee', name2='cmbee')#,constraint='None')
            #print(cltt)
            l2,clrsee,errrsee,sn2=self.GeneralClCalcrsx(ellBinEdges,fsky=fsky2,name1='rsxee',name2='cmbee')##,constraint='None')
            cls_rs = self.fgs.rs_autoEE(self.evalells,self.fgs.nu_rs,self.fgs.nu_rs) / self.cc.c['TCMBmuK']**2.\
                / ((self.evalells+1.)*self.evalells) * 2.* np.pi
            #clsout=cls_rs*cltt
            ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2
            ellWidths = np.diff(ellBinEdges)
            signal="input/CMB_rayleigh_500.dat"
            ells,clsout=np.loadtxt(signal,unpack=True,usecols=[0,6])#need check
            ells=ells[0:ellmax]
            clsout=clsout*(self.fgs.nu_rs/500)**4
            
            cls_out=clsout[0:ellmax]
            cls_out=cls_out/ self.cc.c['TCMBmuK']**2./ ((ells+1.)*ells) * 2.* np.pi
        
            #sn2=(2.*self.evalells+1.)*np.nan_to_num((cls_out**2)/((clrsee+errrsee)*(cltt+errtt)+(cls_out)**2))
            covs = []
            s2n=[]
            l=l1
            for k in range(len(l)):
                i=int(l[k])
                ClSum = np.nan_to_num(((clrsee[k]+errrsee[k])*(cltt[k]+errtt[k])+(cls_out[i])**2))
                s2nper=(2*i+1)*np.nan_to_num((cls_out[i]**2)/((clrsee[k]+errrsee[k])*(cltt[k]+errtt[k])+(cls_out[i])**2))
                var = ClSum/(2.*i+1.)/fsky2/400
                covs.append(var)
                s2n.append(s2nper)
            errs=np.nan_to_num(np.sqrt(np.array(covs)))
            s2n=fsky2/2.*sum(s2n)
            s2n=np.nan_to_num(np.sqrt(s2n))
            return ellMids,cls_out,errs,s2n

