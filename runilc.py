#import matplotlib
#matplotlib.use('Agg')
import numpy as np
from silc.foregrounds import fgNoises
import silc.ilc as ilc
import silc.cosmology as cos
import sys,os
from configparser import SafeConfigParser 
import pickle as pickle
from orphics.io import dict_from_section, list_from_config
from orphics.cosmology import noise_func

#Read ini file

cosmo = cos.Cosmology()

iniFile = "input/exp_config.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

fgs = fgNoises(cosmo.c,ksz_battaglia_test_csv="data/ksz_template_battaglia.csv",tsz_battaglia_template_csv="data/sz_template_battaglia.csv")

cf = 0
constraint_tag = ['','_constrained']

experimentName = "CCATp-v1-40"#"CCATpSOg-v1-40"

beams = list_from_config(Config,experimentName,'beams')
noises = list_from_config(Config,experimentName,'noises')
freqs = list_from_config(Config,experimentName,'freqs')
lmax = int(Config.getfloat(experimentName,'lmax'))
lknee = list_from_config(Config,experimentName,'lknee')[0]
alpha = list_from_config(Config,experimentName,'alpha')[0]
fsky = Config.getfloat(experimentName,'fsky')
try:
        v3mode = Config.getint(experimentName,'V3mode')
except:
        v3mode = -1

try:
        noatm = Config.getint(experimentName,'noatm')
except:
        noatm = False

print (v3mode,noatm)

fsky = 0.01

#initialize ILC
ILC  = ilc.ILC_simple(cosmo,fgs, rms_noises = noises,fwhms=beams,freqs=freqs,lmax=lmax,lknee=lknee,alpha=alpha,v3mode=v3mode,fsky=fsky,noatm=noatm)

#set ells
lsedges = np.arange(10,4001,50)


el_ilc,cls_ilc,err_il,s2nrs_c_TT = ILC.Forecast_CellrsxTTPlanck(lsedges,fsky)

sys.exit()

#calc ILC
if (cf == 0):
    el_il,  cls_il,  err_il,  s2ny  = ILC.Forecast_Cellyy(lsedges,fsky)
    el_ilc,  cls_ilc,  err_ilc,  s2n  = ILC.Forecast_Cellcmb(lsedges,fsky)
    el_ilr,  cls_ilr,  err_ilr,  s2nr  = ILC.Forecast_Cellrsx(lsedges,fsky)
    el_ilr2,  cls_ilr2,  err_ilr2,  s2nr2  = ILC.Forecast_Cellrsx(lsedges,fsky,option='NoILC')
if (cf == 1):
    el_il,  cls_il,  err_il,  s2ny  = ILC.Forecast_Cellyy(lsedges,fsky,constraint="cmb")
    el_ilc,  cls_ilc,  err_ilc,  s2n  = ILC.Forecast_Cellcmb(lsedges,fsky,constraint="tsz")
print ('S/N y', s2ny)
print ('S/N CMB', s2n)
print ('S/N rs', s2nr)
print ('S/N rs NF', s2nr2)

eln, nell_rsx = ILC.Noise_ellrsx()
eln, nell_rsx2 = ILC.Noise_ellrsx(option='NoILC')

facts = el_ilc*(el_ilc+1) / (2*np.pi) * cc.c['TCMBmuK']**2.
factsn = eln*(eln+1) / (2*np.pi) * cc.c['TCMBmuK']**2.


plt.figure()
plt.loglog(el_ilc,  cls_ilc*facts) 
#plt.errorbar(el_ilc,  cls_ilc*facts,yerr=err_ilc*facts)
#plt.errorbar(el_ilr,  np.abs(cls_ilr*facts),yerr=err_ilr*facts)
#plt.plot(el_ilc,  cls_ilc*facts)
plt.plot(el_ilr,  np.abs(cls_ilr*facts))
plt.plot(eln,ILC.N_ll_rs_c_cmb*factsn,'--')
plt.plot(eln,ILC.N_ll_cmb_c_rs*factsn)
plt.plot(eln,ILC.N_ll_rsx_NoFG*factsn,'--')


#plt.plot(el_ilr,  err_ilr*facts)
#plt.plot(eln, nell_rsx2*factsn)

#plt.errorbar(el_ilr2,  np.abs(cls_ilr2*facts),yerr=err_ilr2*facts)
plt.show()


rxs280 = ILC.fgs.rs_cross(eln,280)

elrsx,rs_cross,rs_crossEE = np.loadtxt('input/fiducial_scalCls_lensed_1_4.txt',unpack=True,usecols=[0,1,2])

'''
plt.figure()
plt.plot(eln,rxs280)
#plt.plot(elrsx,rs_cross)
plt.plot(elrsx,rs_cross/ILC.fgs.rs_nu(93),'--')
#plt.plot(elrsx,rs_cross*(1 + 1/ILC.fgs.rs_nu(145)),'--')
plt.show()
'''




#print (err_ilr2/err_ilr)
#print (cls_ilc/cls_ilr)

#Extra stuff

#print 'S/N' , np.sqrt(np.sum((cls_ilc/err_ilc)**2))

#outDir = "/Users/nab/Desktop/Projects/SO_forecasts/"

#outfile1 = outDir + experimentName + "_y_weights"+constraint_tag[cf]+".png"
#outfile2 = outDir + experimentName + "_cmb_weights"+constraint_tag[cf]+".png"

#ILC.PlotyWeights(outfile1)
#ILC.PlotcmbWeights(outfile2)

#if (cf == 0): 
#    eln,nl = ILC.Noise_ellyy()
#    elnc,nlc = ILC.Noise_ellcmb()

#if (cf == 1): 
#    eln,nl = ILC.Noise_ellyy(constraint='cib')
#    elnc,nlc = ILC.Noise_ellcmb(constraint='tsz')




