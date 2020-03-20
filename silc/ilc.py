from builtins import str
from builtins import range
from builtins import object
import numpy as np
from silc.foregrounds import fgNoises
from scipy.interpolate import interp1d
from silc.cosmology import Cosmology
import silc.cosmology as cosmo
from orphics.cosmology import noise_func
from orphics.cosmology import LensForecast
from silc.foregrounds import f_nu

import numpy.matlib
from scipy.special import j1
import sys,os
from configparser import SafeConfigParser 
import pickle as pickle
from orphics.io import dict_from_section, list_from_config


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

def doubleweightcalculator(f_1,f_2,N):
    """
    Return weight that upweight both f_1 and f_2
    W=((f_1^TNf_1-f_2^TNf_1)f_2^TN+(f_2^TNf_2-f_2^TNf_1)f_1^TN)/(f_1^TNf_1f_2^TNf_2-(f_2^TNf_1)^2)
    """
    C=np.matmul(np.transpose(f_1),np.matmul(N,f_1))*np.matmul(np.transpose(f_2),np.matmul(N,f_2))-(np.matmul(np.transpose(f_2),np.matmul(N,f_1)))**2
    M=(np.matmul(np.transpose(f_1),np.matmul(N,f_1)) - np.matmul(np.transpose(f_2),np.matmul(N,f_1))) *np.matmul(np.transpose(f_2),N) \
        + (np.matmul(np.transpose(f_2),np.matmul(N,f_2)) - np.matmul( np.transpose(f_2),np.matmul(N,f_1)))*np.matmul(np.transpose(f_1),N)
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
        #Cosmology.__init__(self, paramDict, constDict)

        #Inputs
        #clusterCosmology is a class that contains cosmological parameters and power spectra.
        #fgs is a class that contains the functional forms for the foregrounds and constants

        #Options

        #initial set up for ILC
        self.cc = Cosmology

        #initializing the frequency matrices

        self.fgs = fgs

    
        self.dell = dell
        #set-up ells to evaluate up to lmax
        self.evalells = np.arange(2,lmax,self.dell)
        self.N_ll = self.evalells*0.0
        self.N_ll_1_c_2  = self.evalells*0.0
        self.N_ll_2_c_1  = self.evalells*0.0
        #Only for SO forecasts, including the SO atmosphere modeling
        #vmode selecting for observation source classification
        #vmode=-1 for everything other than SO and CCATp
        if v3mode>-1:
            print("V3 flag enabled.")
            import silc.V3_calc_public as v3
            import silc.so_noise_lat_v3_1_CAND as v3_1

            if v3mode <= 2:
                lat = v3_1.SOLatV3point1(v3mode,el=50.)
                vfreqs = lat.get_bands()# v3.Simons_Observatory_V3_LA_bands()                                                               
                print("Simons Obs")
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

                N_ell_T_LA = np.diagonal(N_ell_T_LA_full).T
                Map_white_noise_levels = lat.get_white_noise(fsky)**.5

            #if v3mode <= 2:
            #    vfreqs = v3.Simons_Observatory_V3_LA_bands()
            #    freqs = vfreqs
            #    vbeams = v3.Simons_Observatory_V3_LA_beams()
            #    fwhms = vbeams

            #    v3lmax = self.evalells.max()
            #    v3dell = np.diff(self.evalells)[0]

            #    v3ell, N_ell_T_LA, N_ell_P_LA, Map_white_noise_levels = v3.Simons_Observatory_V3_LA_noise(sensitivity_mode=v3mode,f_sky=fsky,ell_max=v3lmax+v3dell,delta_ell=v3dell)
            elif v3mode == 3:
                vfreqs = v3.AdvACT_bands()
                freqs = vfreqs
                vbeams = v3.AdvACT_beams()
                fwhms = vbeams

                v3lmax = self.evalells.max()
                v3dell = np.diff(self.evalells)[0]
                v3ell, N_ell_T_LA, N_ell_P_LA, Map_white_noise_levels = v3.AdvACT_noise(f_sky=fsky,ell_max=v3lmax+v3dell,delta_ell=\
v3dell)
            elif v3mode == 5:
                #import silc.lat_noise_190819_w350ds4 as ccatp
                import silc.ccat_noise_200306 as ccatp
                #tubes = (0,0,0,2,2,1)
                lat = ccatp.CcatLatv2(el=50.)#v3mode,el=50.,survey_years=4000/24./365.24,survey_efficiency=1.0,N_tubes=tubes)
                vfreqs = lat.get_bands()# v3.Simons_Observatory_V3_LA_bands()
                print("CCATP")
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

            elif v3mode == 6:
                import silc.lat_noise_190819_w350ds4 as ccatp
                #tubes = (0,0,0,2,2,1)
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
            freqs=np.array([ 222., 280., 348., 405., 850.,100.,143.,217.,353.])
                
        if (len(freqs) > 1):
            fq_mat   = np.matlib.repmat(freqs,len(freqs),1) 
            fq_mat_t = np.transpose(np.matlib.repmat(freqs,len(freqs),1))
        else:
            fq_mat   = freqs
            fq_mat_t = freqs
                
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
        noise2=[]
        #f_1=self.fgs.rs_nu(np.array(self.freqs))
        #f_nu2=f_1*0.0+1.
        for ii in range(len(self.evalells)):

            cmb_els = fq_mat*0.0 + self.cc.clttfunc(self.evalells[ii])
            cmbee=fq_mat*0.0 + self.cc.cleefunc(self.evalells[ii])
            if v3mode < 0:
                inst_noise = (noise_func(self.evalells[ii],np.array(fwhms),np.array(rms_noises),lknee,alpha,dimensionless=False) /  self.cc.c['TCMBmuK']**2.)
                nells = abs(np.diag(inst_noise))
                nellsee=nells*np.sqrt(2.)
            elif v3mode<=2:
                nells = N_ell_T_LA_full[:,:,ii]/ self.cc.c['TCMBmuK']**2.
                #nellsee = N_ell_P_LA[:,:,ii]/ self.cc.c['TCMBmuK']**2.

            elif v3mode==3:
                ndiags = []
                for ff in range(len(freqs)):
                    inst_noise = N_ell_T_LA[ff,ii] / self.cc.c['TCMBmuK']**2.
                    ndiags.append(inst_noise)
                    
                nells = np.diag(np.array(ndiags))
                #nellsee = N_ell_P_LA[:,:,ii]/ self.cc.c['TCMBmuK']**2.

            elif v3mode>=5:
                if name1!='rsxee':
                    nells = N_ell_T_LA_full[:,:,ii]/ self.cc.c['TCMBmuK']**2.
                elif name1=='rsxee':
                    nells = N_ell_P_LA[:,:,ii]/ self.cc.c['TCMBmuK']**2.
                else:
                    pass
            
            #nell_ee_ll.append(np.diagonal(nellsee* self.cc.c['TCMBmuK']**2.))

            #self.nell_ee_ll=np.array(nell_ee_ll)
            
            
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

                dell1 = 1

                evalells = np.arange(2,lmax1,dell1)
                if name2=='cmb':
                    inst_noise1 = noise_func(evalells[ii],np.array(beams1),np.array(noises1),lknee1,alpha1,dimensionless=False)/self.cc.c['TCMBmuK']**2.#/(2.725e6)**2#/ self.cc.c['TCMBmuK']**2.))
                elif name2=='cmbee':
                    inst_noise1 = np.sqrt(2.)*noise_func(evalells[ii],np.array(beams1),np.array(noises1),lknee1,alpha1,dimensionless=False)/self.cc.c['TCMBmuK']**2.#/(2.725e6)**2#/ self.cc.c['TCMBmuK']**2.))
                else:
                    print('error')
                
                nellsp = np.diag(inst_noise1)
                noise2.append(np.diagonal(nellsp* self.cc.c['TCMBmuK']**2.))
                nellc=nells[0:5,0:5]#/self.cc.c['TCMBmuK']**2.
                #print(nellc)
                #nellce=nells[0:10,0:10]#/self.cc.c['TCMBmuK']**2.
                nells=combineexpnoise(nellc,nellsp)
                #nellsee=combineexpnoise(nellce,np.sqrt(2)*nellsp)
                #self.noise2=np.array(noise2)

            nell_ll.append(np.diagonal(nells* self.cc.c['TCMBmuK']**2.))
            self.nell_ll=np.array(nell_ll)
            totfg = (self.fgs.rad_ps(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.cib_p(self.evalells[ii],fq_mat,fq_mat_t) +
                      self.fgs.cib_c(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.tSZ_CIB(self.evalells[ii],fq_mat,fq_mat_t)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi 

            totfgrs = (self.fgs.rad_ps(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.cib_p(self.evalells[ii],fq_mat,fq_mat_t) +
                       self.fgs.cib_c(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.rs_auto(self.evalells[ii],fq_mat,fq_mat_t) + \
                       self.fgs.tSZ_CIB(self.evalells[ii],fq_mat,fq_mat_t)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii] ) * 2.* np.pi 

            totfg_cib = (self.fgs.rad_ps(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.tSZ_CIB(self.evalells[ii],fq_mat,fq_mat_t)) \
                      / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi
            fgrspol=(self.fgs.rs_autoEE(self.evalells[ii],fq_mat,fq_mat_t))/ self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi#+self.fgs.totalds(self.evalells[ii],fq_mat,fq_mat_t)+self.fgs.rad_pol_ps(self.evalells[ii],fq_mat,fq_mat_t))/ self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi

            ksz = fq_mat*0.0 + self.fgs.ksz_temp(self.evalells[ii]) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi

            tsz = self.fgs.tSZ(self.evalells[ii],fq_mat,fq_mat_t) / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi            

            cib = (self.fgs.cib_p(self.evalells[ii],fq_mat,fq_mat_t) + self.fgs.cib_c(self.evalells[ii],fq_mat,fq_mat_t)) \
                     / self.cc.c['TCMBmuK']**2. / ((self.evalells[ii]+1.)*self.evalells[ii]) * 2.* np.pi
            
            #covariance matrices, which include information from all signals
            if name1!='rsxee' and name2!='cmbee':
                N_ll=((totfgrs) + (cmb_els) + tsz + (ksz)+(nells))#*1000
            else:
                N_ll=(nells)+(cmbee)+(fgrspol)
            
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
        
    def Forecast_CellrsxEEPlanck(self,ellBinEdges,fsky,option='None', add='None'):
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
            ells,clsout=np.loadtxt(signal,unpack=True,usecols=[0,5])
            ells=ells[0:3000]
            clsout=clsout*(self.fgs.nu_rs/500)**4
            
            cls_out=clsout[0:3000]
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
        
        
    def Forecast_CellrsxPlanck(self,ellBinEdges,fsky,option='None', add='None'):
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
            ells,clsout=np.loadtxt(signal,unpack=True,usecols=[0,6])#need check
            ells=ells[0:3000]
            clsout=clsout*(self.fgs.nu_rs/500)**4
            
            cls_out=clsout[0:3000]
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
        
    def Forecast_CellrsxTTPlanck(self,ellBinEdges,fsky,option='None', add='None'):
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
            ells,clsout=np.loadtxt(signal,unpack=True,usecols=[0,6])#need check
            ells=ells[0:3000]
            clsout=clsout*(self.fgs.nu_rs/500)**4
            
            cls_out=clsout[0:3000]
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

    def Forecast_CellrsxPPPlanck(self,ellBinEdges,fsky,option='None', add='None'):
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
            ells=ells[0:3000]
            clsout=clsout*(self.fgs.nu_rs/500)**4
            
            cls_out=clsout[0:3000]
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

