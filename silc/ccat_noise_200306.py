"""Modified CMB-S4 LAT Survey Simple Noise Model

Based on v190311 version, incorrectly labeled v191103 in some places.

The CcatLat class adds a 350 GHz channel.

- added 410 GHz and removed 281 -SC
190717: Nl_red updated based on AM code output and dicussions with M Hasse
190718: uK rt(s) updated from 6000 TES per tube to 10500 KID per tube
for each freq, it's 3000 TES per tube and 10500 KID per tube
but we are counting 2 tubes 
190819: using numbers from CCAT-Prime_SZ-FPI_20190805b.xlsx
has transmissions updated, spills, eta(tel), solid angle, detector count updated NETs are per single color tube 
200306: copied from lat_noise_190819_w350_ds4.py to standalone

"""

from __future__ import print_function

import numpy as np

def get_atmosphere_C(freqs, version=1, el=None):
    """
    Returns atmospheric noise power at ell=1000, for an ACTPol optics
    tube.  In units of [uK^2 sec].  This only works for a few special
    frequencies.

    Basic model assumes el=50.  A simple rescaling (proportional to
    csc(el)) is applied for other values of el.

    version=0: This atmospheric model was used in SO V3 forecasts but
    contains an error.

    version=1: This atmospheric model is better than version=0, in
    that at least one error has been corrected.  The relative
    atmospheric powers have been measured in AT model, and then
    calibrated to ACT.  Low frequency results are inflated somewhat
    because ACT sees more power at 90 GHz than predicted by this
    modeling.
    """
    # This el_correction is quite naive; atmosphere may be less
    # structured (i.e. smoothed out) at lower elevation because of the
    # increased thickness.
    if el is None:
        el = 50.
    el_correction = np.sin(50.*np.pi/180) / np.sin(el*np.pi/180)
    ccat_pwv_cor = 0.6/1.0
    if version == 0:
        data_bands = np.array([ 27., 39., 93., 145., 225., 280.])
        data_C = np.array([200., 77., 1800., 12000., 68000., 124000.])
        data = {}
        for b,C in zip(data_bands, data_C):
            data[b] = C
        # Jam in a fake value for 20 GHz.
        data[20.] = 200.
    elif version == 1:
        data_bands = np.array([ 222., 280., 348., 405., 850.])
        data_C = np.array([ 
                             # below factors from am_output/mult_pwv/get_derivative_ccat.py
                             2.31956542e+05,
                             1.61527385e+06,
                             4.03473727e+07,
                             2.51490116e+08,
                             9.10884821e+13
        ])
        # plt.plot(data_bands,data_C)
        # plt.yscale('log')
        # plt.show()
        # exit()
        data = {}
        for b,C in zip(data_bands, data_C):
            data[b] = C
    return np.array([data[f] * (ccat_pwv_cor*el_correction)**2 for f in freqs])

"""See lower for subclasses of SOLatType -- instrument specific
parameters are configured in __init__.  """

class SOLatType:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('You should subclass this.')

    def get_bands(self):
        return self.bands.copy()

    def get_beams(self):
        return self.beams.copy()

    def precompute(self, N_tubes, N_tels=1):

        white_noise_el_rescale = np.array([1.] * len(self.bands))
        if self.el is not None:
            el_data = self.el_noise_params
            el_lims = el_data.get('valid')
            if el_lims[0] == 'only':
                assert(self.el == el_lims[1])  # noise model only valid at one elevation...
            else:
                assert(el_lims[0] <= self.el) and (self.el <= el_lims[1])
                band_idx = np.array([np.argmin(abs(el_data['bands'] - b)) for b in self.bands])
                assert(np.all(abs(np.array(el_data['bands'])[band_idx] - self.bands) < 5))
                coeffs = el_data['coeffs']
                white_noise_el_rescale = np.array(
                    [el_noise_func(coeffs[i], self.el) / el_noise_func(coeffs[i], 50.)
                     for i in band_idx])

        # Accumulate total white noise level and atmospheric
        # covariance matrix for this configuration.
        
        band_weights = np.zeros(self.n_bands)
        for (tube_name, tube_count) in N_tubes:
            # commenting out white noise el rescaling as the white noise computed already has this in there.
            tube_noise = self.tube_configs[tube_name] #* white_noise_el_rescale
            s = (tube_noise != 0)
            band_weights[s] += tube_count * N_tels * tube_noise[s]**-2

        self.band_sens = np.zeros(self.n_bands) + 1e9
        s = (band_weights > 0)
        self.band_sens[s] = band_weights[s]**-0.5

        # Special for atmospheric noise model.
        self.Tatmos_C = get_atmosphere_C(self.bands, self.atm_version,
                                         el=self.el) * self.FOV_mod
        self.Tatmos_ell = 1000. + np.zeros(self.n_bands)
        self.Tatmos_alpha = -3.5 + np.zeros(self.n_bands)

        # Compute covariant weight matrix (atmosphere parameters).
        cov_weight = np.zeros((self.n_bands,self.n_bands))
        pcov_weight = np.zeros((self.n_bands,self.n_bands))
        atm_rho = 0.9
        for (tube_name, tube_count) in N_tubes:
            # Get the list of coupled bands; e.g. [1,2] for MF.
            nonz = self.tube_configs[tube_name].nonzero()[0]
            for i in nonz:
                for j in nonz:
                    w = {True: 1., False: atm_rho}[i==j]
                    assert(cov_weight[i,j] == 0.) # Can't do overlapping
                                                  # tubes without weights.
                    cov_weight[i,j] += tube_count * N_tels / ( w * (
                        self.Tatmos_C[i] * self.Tatmos_C[j])**.5 )
                    pcov_weight[i,j] = w

        # Reciprocate non-zero elements.
        s = (cov_weight!=0)
        self.Tatmos_cov = np.diag([1e9]*self.n_bands)
        self.Tatmos_cov[s] = 1./cov_weight[s]

        # Polarization is simpler...
        self.Patmos_ell = 700. + np.zeros(self.n_bands)
        self.Patmos_alpha = -1.4 + np.zeros(self.n_bands)
        
        self.Patmos_cov = pcov_weight

    def get_survey_time(self):
        t = self.survey_years * 365.25 * 86400.    ## convert years to seconds
        return t * self.survey_efficiency

    def get_survey_spread(self, f_sky, units='arcmin2'):
        # Factor that converts uK^2 sec -> uK^2 arcmin^2.
        A = f_sky * 4*np.pi
        if units == 'arcmin2':
            A *= (60*180/np.pi)**2
        elif units != 'sr':
            raise ValueError("Unknown units '%s'." % units)
        return A / self.get_survey_time()

    def get_white_noise(self, f_sky, units='arcmin2'):
        return self.band_sens**2 * self.get_survey_spread(f_sky, units=units)

    def get_noise_curves(self, f_sky, ell_max, delta_ell, deconv_beam=True,
                         full_covar=False):
        ell = np.arange(2, ell_max, delta_ell)
        W = self.band_sens**2

        # Get full covariance; indices are [band,band,ell]
        ellf = (ell/self.Tatmos_ell[:,None])**(self.Tatmos_alpha[:,None])
        T_noise = self.Tatmos_cov[:,:,None] * (ellf[:,None,:] * ellf[None,:,:])**.5

        # P noise is tied directly to the white noise level.
        P_low_noise = (2*W[:,None]) * (ell / self.Patmos_ell[:,None])**self.Patmos_alpha[:,None]
        P_noise = (self.Patmos_cov[:,:,None] *
                   (P_low_noise[:,None,:] * P_low_noise[None,:,:])**.5)

        # Add in white noise on the diagonal.
        for i in range(len(W)):
            T_noise[i,i] += W[i]
            P_noise[i,i] += W[i] * 2

        # Deconvolve beams.
        if deconv_beam:
            beam_sig_rad = self.get_beams() * np.pi/180/60 / (8.*np.log(2))**0.5
            beams = np.exp(-0.5 * ell*(ell+1) * beam_sig_rad[:,None]**2)
            T_noise /= (beams[:,None,:] * beams[None,:,:])
            P_noise /= (beams[:,None,:] * beams[None,:,:])

        # Diagonal only?
        if not full_covar:
            ii = range(self.n_bands)
            T_noise = T_noise[ii,ii]
            P_noise = P_noise[ii,ii]

        sr_per_arcmin2 = (np.pi/180/60)**2
        return (ell,
                T_noise * self.get_survey_spread(f_sky, units='sr'),
                P_noise * self.get_survey_spread(f_sky, units='sr'))


""" Elevation-dependent noise model is parametrizerd below.  Detector
white noise is determined by a fixed, instrumental component as well
as a contribution from atmospheric loading, which is should scale
roughly with 1/sin(elevation).  The 'coeffs' below give coefficents A,
B for the model

   sens \propto A + B / sin(elevation)

For 27 GHz and higher, results are from the Simons Observatory LAT
noise model provided by Carlos Sierra and Jeff McMahon on March 11,
2019.  For 20 GHz, fits from Ben Racine and Denis Barkats are used
(for a slightly different instrument configuration that nevertheless
gives compatible results across all bands).  """

def el_noise_func(P, el):
    a, b = P
    return a + b / np.sin(el*np.pi/180)

SO_el_noise_func_params = {
    'threshold': {
        'valid': ('only', 50.),
    },
    'baseline': {
        'valid': (25., 70.),
        'bands': [20,27,39,93,145,225,280],
        'coeffs': [
            (178.59719595, 33.72945249),  # From B. Racine & D. Barkats.
            (.87, .09),                   # From Carlos Sierra and J. McMahon vvv
            (.64, .25),
            ((.80+.74)/2, (.14+.19)/2),
            ((.76+.73)/2, (.17+.19)/2),
            (.58, .30),
            (.49, .36),
        ],
    },
    'goal': {
        'valid': (25., 70.),
        'bands': [20,27,39,93,145,222., 280., 348., 405., 850.],
        # 20,350,410 GHz coeffs below got modified by plotting freq vs b 
        # and extrapolating by eye
        # 20 GHz is normalized
        'coeffs': [
            (178.59719595/226.297845113, 33.72945249/226.297845113),  # From B. Racine & D. Barkats.
            (.85, .11),                   # From Carlos Sierra and J. McMahon vvv
            (.65, .25),
            ((.76+.69)/2, (.17+.22)/2),
            ((.73+.69)/2, (.19+.22)/2),
            (.55, .32),
            (.47, .37),
            # (.47, .37),  # 281 &  -- hack for XHF tube "at CCAT site"
            # below from: el_vs_NET_model_190503.py
            (.47/1.04558491989, .37*1.1/1.04558491989),  # 350
            (.47/1.08744564133, .37*1.18/1.08744564133),  # 410
            (.47/1.08744564133, .37*1.18/1.08744564133),  # 860    -- copied from above -SC
        ],
    },
}


class SOLatV3(SOLatType):
    atm_version = 0
    def __init__(self, sensitivity_mode=None, N_tubes=None, survey_years=5.,
                 survey_efficiency = 0.2*0.85, el=None):
        # Define the instrument.
        self.n_bands = 6
        self.bands = np.array([
            27., 39., 93., 145., 225., 280.])
        self.beams = np.array([
            7.4, 5.1, 2.2, 1.4, 1.0, 0.9])

        # Set defaults for survey area, time, efficiency
        self.survey_years = survey_years
        self.survey_efficiency  = survey_efficiency
        
        # Translate integer to string mode; check it.
        sens_modes = {0: 'threshold',
                      1: 'baseline',
                      2: 'goal'}
        if sensitivity_mode is None:
            sensitivity_mode = 'baseline'
        elif sensitivity_mode in sens_modes.keys():
            sensitivity_mode = sens_modes.get(sensitivity_mode)

        assert(sensitivity_mode in sens_modes.values())  # threshold,baseline,goal? 0,1,2?
        self.sensitivity_mode = sensitivity_mode
        
        # Sensitivities of each kind of optics tube, in uK rtsec, by
        # band.  0 represents 0 weight, not 0 noise...
        nar = np.array
        self.tube_configs = {
            'threshold': {
                'LF':  nar([  61.,  30.,    0,    0,    0,    0 ]),
                'MF':  nar([    0,    0,  6.5,  8.1,    0,    0 ])*4**.5,
                'UHF': nar([    0,    0,    0,    0,  17.,  42. ])*2**.5,
            },
            'baseline': {
                'LF':  nar([  48.,  24.,    0,    0,    0,    0 ]),
                'MF':  nar([    0,    0,  5.4,  6.7,    0,    0 ])*4**.5,
                'UHF': nar([    0,    0,    0,    0,  15.,  10. ])*2**.5,
            },
            'goal': {
                'LF':  nar([  35.,  18.,    0,    0,    0,    0 ]),
                'MF':  nar([    0,    0,  3.9,  4.2,    0,    0 ])*4**.5,
                'UHF': nar([    0,    0,    0,    0,  10.,  25. ])*2**.5,
            },
        }[sensitivity_mode]

        self.el_noise_params = SO_el_noise_func_params[sensitivity_mode]
        
        # Save the elevation request.
        self.el = el
        
        # Factor by which to attenuate atmospheric power, given FOV
        # relative to ACT?
        self.FOV_mod = 0.5

        # The reference tube config.
        ref_tubes = [('LF', 1), ('MF', 4), ('UHF', 2)]

        if N_tubes is None:
            N_tubes = ref_tubes
        else:
            N_tubes = [(b,x) for (b,n),x in zip(ref_tubes, N_tubes)]
            
        self.precompute(N_tubes)

    
class SOLatV3point1(SOLatV3):
    atm_version = 1

    
class S4LatV1(SOLatType):
    def __init__(self, sensitivity_mode=None,
                 N_tubes=None, N_tels=None,
                 survey_years=7.,
                 survey_efficiency=0.23,
                 el=None):
        # Define the instrument.
        self.n_bands = 7
        self.bands = np.array([
            20., 27., 39., 93., 145., 225., 280.])
        self.beams = np.array([
            10., 7.4, 5.1, 2.2, 1.4, 1.0, 0.9])

        # Set defaults for survey area, time, efficiency
        self.survey_years = survey_years
        self.survey_efficiency  = survey_efficiency

        # Sensitivities of each kind of optics tube, in uK rtsec, by
        # band.  0 represents 0 weight, not 0 noise...
        nar = np.array
        self.tube_configs = {
            'ULF': nar([ 60.,    0,    0,    0,    0,    0,    0 ]),
            'LF':  nar([   0, 28.6, 16.0,    0,    0,    0,    0 ]),
            'MF':  nar([   0,    0,    0,  6.6,  6.8,    0,    0 ]),
            'UHF': nar([   0,    0,    0,    0,    0, 12.5, 30.0 ]),
        }

        # Save the elevation request.
        self.el = el
        
        self.el_noise_params = SO_el_noise_func_params['goal']

        # Factor by which to attenuate atmospheric power, given FOV
        # relative to ACT?
        self.FOV_mod = 0.5

        # Version of the atmospheric model to use?
        self.atm_version = 1

        # The reference tube config.
        ref_tubes = [('ULF', 1), ('LF', 2), ('MF', 12), ('UHF', 4)]

        if N_tels is None:
            N_tels = 2

        if N_tubes is None:
            N_tubes = ref_tubes
        else:
            N_tubes = [(b,x) for (b,n),x in zip(ref_tubes, N_tubes)]
            
        self.precompute(N_tubes, N_tels)

    
class CcatLat(SOLatType):
    """This special edition S4 LAT is equipped with an additional tube
    class, XHF, containing 280 and 350 GHz detectors and blessed with
    the calm atmosphere available at the CCAT site.  Otherwise it
    should reproduce S4Lat results exactly.  This is a candidate for
    introduction to the main noise model.

    """
    atm_version = 1
    def __init__(self, sensitivity_mode=None,
                 N_tubes=None, N_tels=None,
                 survey_years=7.,
                 survey_efficiency=0.23,
                 el=None):
        # Define the instrument.
        self.bands = np.array([
            20., 27., 39., 93., 145., 225., 280., 281., 350.])
        self.beams = np.array([
            10., 7.4, 5.1, 2.2, 1.4, 1.0, 0.9, 0.9, 0.72])
        self.n_bands = len(self.bands)

        # Set defaults for survey area, time, efficiency
        self.survey_years = survey_years
        self.survey_efficiency  = survey_efficiency

        # Sensitivities of each kind of optics tube, in uK rtsec, by
        # band.  0 represents 0 weight, not 0 noise...
        nar = np.array
        self.tube_configs = {
            'ULF': nar([ 60.,    0,    0,    0,    0,    0,    0,     0,     0 ]),
            'LF':  nar([   0, 28.6, 16.0,    0,    0,    0,    0,     0,     0 ]),
            'MF':  nar([   0,    0,    0,  6.6,  6.8,    0,    0,     0,     0 ]),
            'UHF': nar([   0,    0,    0,    0,    0, 12.5, 30.0,     0,     0 ]),
            'XHF': nar([   0,    0,    0,    0,    0,    0,    0,  30.0, 115.0  ]),
        }

        # Save the elevation request.
        self.el = el
        
        self.el_noise_params = SO_el_noise_func_params['goal']

        # Factor by which to attenuate atmospheric power, given FOV
        # relative to ACT?
        self.FOV_mod = 0.5

        # The reference tube config.
        ref_tubes = [('ULF', 1), ('LF', 2), ('MF', 12), ('UHF', 4), ('XHF', 0)]

        if N_tels is None:
            N_tels = 2

        if N_tubes is None:
            N_tubes = ref_tubes
        else:
            N_tubes = [(b,x) for (b,n),x in zip(ref_tubes, N_tubes)]
            
        self.precompute(N_tubes, N_tels)


class CcatLatv2(SOLatType):
    """This special edition S4 LAT is equipped with an additional tube
    class, XHF, containing 280 and 350 GHz detectors and blessed with
    the calm atmosphere available at the CCAT site.  Otherwise it
    should reproduce S4Lat results exactly.  This is a candidate for
    introduction to the main noise model.

    """
    atm_version = 1
    def __init__(self, sensitivity_mode=None,
                 N_tubes=None, N_tels=None,
                 survey_years=4000/24./365.24,
                 survey_efficiency=1.0,
                 el=None):
        # Define the instrument.
        self.bands = np.array([
            222., 280., 348., 405., 850.])
        # scaled beam for 410 GHz but need to check 350 and 410.
        self.beams = np.array([
            57/60., 45/60., 35/60., 30/60., 14/60.])
        self.n_bands = len(self.bands)

        # Set defaults for survey area, time, efficiency
        self.survey_years = survey_years
        self.survey_efficiency  = survey_efficiency

        # Sensitivities of each kind of optics tube, in uK rtsec, by
        # band.  0 represents 0 weight, not 0 noise...
        nar = np.array
        self.tube_configs = {
            # 'ULF': nar([ 60.,   0,   0,   0,   0,   0,   0,    0,    0, 0]),
            # 'LF':  nar([   0,28.6,16.0,   0,   0,   0,   0,    0,    0, 0]),
            # 'MF':  nar([   0,   0,   0, 6.6, 6.8,   0,   0,    0,    0, 0]),
            # from MFH:
            # 'UHF': nar([   0,   0,   0,   0,   0,12.5,30.0,    0,    0,0]),
            # from gordon's spreadsheet for 1
            'UHF': nar([   7.6,14.2,    0,    0, 0]),
            'XHF': nar([      0,   0,54.1,192.2, 0]),
            'ZHF': nar([      0,   0,    0,  0,296605.4]),
        }

        # Save the elevation request.
        self.el = el
        
        self.el_noise_params = SO_el_noise_func_params['goal']

        # Factor by which to attenuate atmospheric power, given FOV
        # relative to ACT?
        # print("check fov!")
        self.FOV_mod = 0.5

        # The reference tube config.
        ref_tubes = [('UHF', 1), ('XHF', 1),('ZHF', 1)]

        if N_tels is None:
            N_tels = 1

        if N_tubes is None:
            N_tubes = ref_tubes
        else:
            N_tubes = [(b,x) for (b,n),x in zip(ref_tubes, N_tubes)]
            
        self.precompute(N_tubes, N_tels)



# def T_tot_nl((ell,bl),Nwhite,Nred):
#     ell0 = 1000.
#     alpha = -3.5
#     return (Nred*(ell/ell0)**(alpha) + Nwhite)/bl**2

# def P_tot_nl((ell,bl),Nwhite):
#     ell0 = 700.
#     alpha = -1.4
#     return (Nwhite*(ell/ell0)**(alpha) + Nwhite)/bl**2

# if __name__ == '__main__':
#     import matplotlib
#     # matplotlib.use('png')
#     matplotlib.rc('font', family='serif', serif='cm10')
#     matplotlib.rc('text', usetex=True)
#     fontProperties = {'family':'sans-serif',
#                       'weight' : 'normal', 'size' : 16}
#     import matplotlib.pyplot as plt

#     ####################################################################
#     ####################################################################
#     ##                   demonstration of the code
#     ####################################################################

#     target = 'ccat'
#     #tubes = (1,2,12,4,0) # ULF,LF,MF,UHF,XHF
#     tubes = (0,0,0,1,1,1) # ULF,LF,MF,UHF,XHF #JCH mod
#     mode=2
#     suffix='png'
#     ellmax=1e4
#     el = 45.
#     # SO
#     fsky=0.4
#     survey_year=5
#     survey_eff=0.2
#     # ccat
#     fsky=15000./(4*np.pi*(180/np.pi)**2)
#     survey_year=4000/24./365.24

#     # fsky=410./(4*np.pi*(180/np.pi)**2)
#     # survey_year=680/24./365.24

#     survey_eff=1.0
#     print('# tube at UHF (225/280) GHz')
#     print(tubes[-3])
#     print('# tube at XHF (350/410) GHz')
#     print(tubes[-2])
#     print('# tube at ZHF (860) GHz')
#     print(tubes[-1])
#     print('elevation (deg)')
#     print(el)
#     print('fraction sky (%)')
#     print(fsky*100)
#     print('survey_year (years)')
#     print(survey_year)
#     print('survey_eff')
#     print(survey_eff)
#     print('eff survey_year (years)')
#     print(survey_year*survey_eff)

#     colors=['b','r','g','m','k','y','pink','grey','skyblue','green']
#     corr_colors = ['orange', 'fuchsia', 'springgreen','teal']

#     if target == 'S4':
#         dset_label = 'S4\\_2LAT'
#         lat = S4LatV1(mode, N_tels=2, el=el, N_tubes=tubes)
#         corr_pairs = [(1,2),(3,4),(5,6)]
#         ylims = (5e-8,1e-1)
#         colors.insert(0, 'gray')

#     elif target == 'SO':
#         dset_label = 'SO\\_V3'
#         lat = SOLatV3(mode, el=el, N_tubes=tubes)
#         corr_pairs = [(0,1),(2,3),(4,5)]
#         ylims = (5e-7,1e0)

#     elif target == 'SOv3.1':
#         dset_label = 'SO\\_V3.1'
#         lat = SOLatV3point1(mode, el=el, N_tubes=tubes)
#         corr_pairs = [(0,1),(2,3),(4,5)]
#         ylims = (5e-7,1e0)

#     elif target == 'ccat':
#         dset_label = 'CCAT-prime'
#         lat = CcatLatv2(mode, el=el, 
#                  N_tubes=tubes,
#                  N_tels=1,
#                  # survey_years=5,
#                  # survey_years=0.45,
#                  # survey_efficiency=0.2)
#                  # survey_efficiency=1.0)
#                  survey_years=survey_year,
#                  survey_efficiency=survey_eff)
#         corr_pairs = [(5,6),(7,8)]
#         ylims = (1e-5,1000)

#     print(dset_label)
#     bands = lat.get_bands()
#     print("band centers: ", lat.get_bands()[-5:], "[GHz]")
#     print("beam sizes: "  , lat.get_beams()[-5:], "[arcmin]")
#     N_bands = len(bands)

#     ell, N_ell_LA_T_full,N_ell_LA_P_full = lat.get_noise_curves(
#         fsky, ellmax, 1, full_covar=True, deconv_beam=True)

#     WN_levels = lat.get_white_noise(fsky)**.5

#     beam_sig_rad = lat.get_beams() * np.pi/180/60 / (8.*np.log(2))**0.5
#     beams = np.exp(-0.5 * ell*(ell+1) * beam_sig_rad[:,None]**2)

#     N_ell_LA_T  = N_ell_LA_T_full[range(N_bands),range(N_bands)]
#     N_ell_LA_Tx = [N_ell_LA_T_full[i,j] for i,j in corr_pairs]
#     N_ell_LA_P  = N_ell_LA_P_full[range(N_bands),range(N_bands)]
#     N_ell_LA_Px = [N_ell_LA_P_full[i,j] for i,j in corr_pairs]

#     print("white noise levels: "  , WN_levels[-5:], "[uK-arcmin]")

#     ## plot the temperature noise curves
#     plt.clf()

#     import pcl_actpol_utility_v4 as pau
#     binfile = '/Users/stevekchoi/work/projects/actpol/ps/binning/BIN_ACTPOL_50_4_SC_low_ell'
#     lmax_cut = 8000

#     lbands = pau.read_binfile(binfile, lcut=lmax_cut)
#     (ell_bin, band, nbins, lmax, ell3, dl) = pau.get_lbands_all(lbands)

#     bin_flat = pau.get_flat_binning(lbands)
#     bin_TT = np.dot(bin_flat,theory[:lmax+1,1])
#     bin_EE = np.dot(bin_flat,theory[:lmax+1,4])

#     g_fac = np.sqrt(2)*pau.get_analytic_var_cross(fsky, ell_bin, band)
    
#     nl_TT_col = []
#     err_TT_col = []
#     N_T_fit_col = []
#     nl_TT_col.append(ell)
#     freqs = []

#     for i in range(5,N_bands):
#         plt.loglog(ell,N_ell_LA_T[i], label='%i GHz' % (bands[i]),color=colors[i], ls='-', lw=3.)
#         guess = [0.1,1.0]
#         popt = op.curve_fit(T_tot_nl,(ell,beams[i]),
#                             N_ell_LA_T[i],p0=guess)[0]
#         N_T_fit_col.append([bands[i],popt[0],popt[1]])
#         nl_TT_col.append(N_ell_LA_T[i])
#         N_ell_tmp = np.append([0,0],N_ell_LA_T[i])
#         err_TT_col.append(g_fac*(bin_TT+np.dot(bin_flat,N_ell_tmp[:lmax+1])))
#         freqs.append('%i'%bands[i])
#         print(bands[i],'N_white=%E'%popt[0],'N_red=%E'%popt[1])
#     # include correlated atmospheric noise across frequencies
#     # for _c,(i,j) in enumerate(corr_pairs):
#     #     plt.loglog(ell, N_ell_LA_T_full[i,j],
#     #                label=r'$%i \times %i$ GHz atm.' % (bands[i],bands[j]),
#     #                color=corr_colors[_c], lw=1.5)
#     #     freqs.append('%ix%i'%(bands[i],bands[j]))
#     #     nl_TT_col.append(N_ell_LA_T_full[i,j])

#     # plt.loglog(theory[:,0],theory[:,1],alpha=0.9)
#     plt.title(r"$N(\ell$) Temperature", fontsize=20)
#     plt.ylabel(r"$N(\ell$) [$\mu$K${}^2$]", fontsize=20)
#     plt.xlabel(r"$\ell$", fontsize=20)
#     plt.ylim(*ylims)
#     plt.xlim(100,10000)
#     plt.legend(loc='lower left', ncol=2, fontsize=20)
#     plt.grid()
#     fig = mpl.pyplot.gcf()
#     fig.set_size_inches(12,6)
#     plt.subplots_adjust(bottom=0.1,top=0.95,wspace=0.43,hspace=0.47, right=0.98,left=0.08)

#     plt.savefig('%s/%s_mode%i_fsky%.3f_LAT_190819_w350ds4_T.%s' % (outdir,target, mode, fsky, suffix),dpi=300)
#     plt.close()

#     form = '%d'+'\t%1.6e'*len(freqs)
#     np.savetxt('%s/%s_mode%i_fsky%.3f_LAT_190819_w350ds4_T_noise_CMB.txt' % (outdir, target, mode, fsky),np.column_stack(nl_TT_col),delimiter='\t',
#         header='ell\t'+'\t'.join(freqs)+'\nelevation=%i\tsurvey_year=%.4f\tsurvey_eff=%.4f'%(el,survey_year,survey_eff),fmt=form)

#     check_fit = 0
#     if check_fit:
#         for i in xrange(len(N_T_fit_col)):
#             nl_fit = T_tot_nl((ell,beams[i+5]),N_T_fit_col[i][1],N_T_fit_col[i][2])
#             plt.loglog(ell,nl_fit, label='%i GHz' % (bands[i+5]),color=colors[i+5], ls='-', lw=3.)
#         plt.title(r"$N(\ell$) Temperature", fontsize=20)
#         plt.ylabel(r"$N(\ell$) [$\mu$K${}^2$]", fontsize=20)
#         plt.xlabel(r"$\ell$", fontsize=20)
#         plt.ylim(*ylims)
#         plt.xlim(100,10000)
#         plt.legend(loc='lower left', ncol=2, fontsize=20)
#         plt.grid()
#         plt.show()

#     ## plot the polarization noise curves
#     plt.clf()

#     nl_PP_col = []
#     err_PP_col = []
#     N_P_fit_col = []
#     nl_PP_col.append(ell)
#     freqs = []

#     ylims = (1e-5,10)

#     for i in range(5,N_bands):
#         plt.loglog(ell,N_ell_LA_P[i], label='%i GHz' % (bands[i]),color=colors[i], ls='-', lw=3.)
#         guess = 0.1
#         popt = op.curve_fit(P_tot_nl,(ell,beams[i]),
#                             N_ell_LA_P[i],p0=guess)[0]
#         N_P_fit_col.append([bands[i],popt[0]])
#         print(bands[i],'N_white=%E'%popt[0])
#         nl_PP_col.append(N_ell_LA_P[i])
#         freqs.append('%i'%bands[i])
#         N_ell_tmp = np.append([0,0],N_ell_LA_P[i])
#         err_PP_col.append(g_fac*(bin_EE+np.dot(bin_flat,N_ell_tmp[:lmax+1])))

#     # include correlated atmospheric noise across frequencies
#     # for _c,(i,j) in enumerate(corr_pairs):
#     #     plt.loglog(ell, N_ell_LA_P_full[i,j],
#     #                label=r'$%i \times %i$ GHz atm.' % (bands[i],bands[j]),
#     #                color=corr_colors[_c], lw=1.5)
#     #     freqs.append('%ix%i'%(bands[i],bands[j]))
#     #     nl_PP_col.append(N_ell_LA_P_full[i,j])

#     # plt.loglog(theory[:,0],theory[:,4],alpha=0.9)
#     plt.title(r"$N(\ell$) Polarization", fontsize=20)
#     plt.ylabel(r"$N(\ell$) [$\mu$K${}^2$]", fontsize=20)
#     plt.xlabel(r"$\ell$", fontsize=20)
#     plt.ylim(*ylims)
#     plt.xlim(100,10000)
#     plt.legend(loc='upper right', ncol=2, fontsize=20)
#     plt.grid()
#     fig = mpl.pyplot.gcf()
#     fig.set_size_inches(12,6)
#     plt.subplots_adjust(bottom=0.1,top=0.95,wspace=0.43,hspace=0.47, right=0.98,left=0.08)

#     plt.savefig('%s/%s_mode%i_fsky%.3f_LAT_190819_w350ds4_P.%s' % (outdir,target, mode, fsky, suffix),dpi=300)
#     plt.close()

#     if check_fit:
#         for i in xrange(len(N_P_fit_col)):
#             nl_fit = P_tot_nl((ell,beams[i+5]),N_P_fit_col[i][1])
#             plt.loglog(ell,nl_fit, label='%i GHz' % (bands[i+5]),color=colors[i+5], ls='-', lw=3.)

#         plt.title(r"$N(\ell$) Polarization", fontsize=20)
#         plt.ylabel(r"$N(\ell$) [$\mu$K${}^2$]", fontsize=20)
#         plt.xlabel(r"$\ell$", fontsize=20)
#         plt.ylim(*ylims)
#         plt.xlim(100,10000)
#         plt.legend(loc='upper right', ncol=2, fontsize=9)
#         plt.grid()
#         plt.show()

#     form = '%d'+'\t%1.6e'*len(freqs)
#     np.savetxt('%s/%s_mode%i_fsky%.3f_LAT_190819_w350ds4_P_noise_CMB.txt' % (outdir, target, mode, fsky),np.column_stack(nl_PP_col),delimiter='\t',
#         header='ell\t'+'\t'.join(freqs)+'\nelevation=%i\tsurvey_year=%.4f\tsurvey_eff=%.4f'%(el,survey_year,survey_eff),fmt=form)

#     ylims = (5e-6,1e9)

#     nl_TT_col = []
#     nl_TT_col.append(ell)
#     for i in range(5,N_bands):
#         plt.loglog(ell,N_ell_LA_T[i], label='%i GHz' % (bands[i]),
#                    color=colors[i], ls='-', lw=3.)
#         nl_TT_col.append(N_ell_LA_T[i])

#     # include correlated atmospheric noise across frequencies
#     for _c,(i,j) in enumerate(corr_pairs):
#         plt.loglog(ell, N_ell_LA_T_full[i,j],
#                    label=r'$%i \times %i$ GHz atm.' % (bands[i],bands[j]),
#                    color=corr_colors[_c], lw=1.5)

#     # plt.loglog(theory[:,0],theory[:,1],alpha=0.9)
#     plt.title(r"$N(\ell$) Temperature", fontsize=20)
#     plt.ylabel(r"$N(\ell$) [$\mu$K${}^2$]", fontsize=20)
#     plt.xlabel(r"$\ell$", fontsize=20)
#     plt.ylim(*ylims)
#     plt.xlim(100,10000)
#     plt.legend(loc='lower left', ncol=2, fontsize=20)
#     plt.grid()
#     fig = mpl.pyplot.gcf()
#     fig.set_size_inches(12,6)
#     plt.subplots_adjust(bottom=0.1,top=0.95,wspace=0.43,hspace=0.47, right=0.98,left=0.08)

#     plt.savefig('%s/%s_mode%i_fsky%.3f_LAT_190819_w350ds4_T_zoom.%s' % (outdir,target, mode, fsky, suffix),dpi=300)
#     plt.close()
#     ## plot the polarization noise curves
#     plt.clf()

#     # plt.loglog(theory[:,0],theory[:,1],alpha=0.9)
#     dl_bin = ell_bin*(ell_bin+1)/(2*np.pi)
#     plt.plot(theory[:,0],theory[:,1]*dl_theory,
#         color='k',alpha=0.4,linewidth=1.3)
#     plt.errorbar(ell_bin-12,dl_bin*bin_TT,dl_bin*err_TT_col[0],
#         fmt='o',elinewidth=2.5,markersize=4,color='RoyalBlue',
#         label='TT 225 GHz')
#     plt.errorbar(ell_bin+12,dl_bin*bin_TT,dl_bin*err_TT_col[1],
#         fmt='o',elinewidth=2.5,markersize=4,color='Orange',
#         label='TT 280 GHz')
#     plt.plot(theory[:,0],theory[:,4]*dl_theory,
#         color='k',alpha=0.4,linewidth=1.3)
#     plt.errorbar(ell_bin-12,dl_bin*bin_EE,dl_bin*err_PP_col[0],
#         fmt='o',elinewidth=2.5,markersize=4,color='ForestGreen',
#         label='EE 225 GHz')
#     plt.errorbar(ell_bin+12,dl_bin*bin_EE,dl_bin*err_PP_col[1],
#         fmt='o',elinewidth=2.5,markersize=4,color='LightPink',
#         label='EE 280 GHz')
#     # plt.title(r"$N(\ell$) Temperature", fontsize=20)
#     plt.ylabel("$\mathcal{D}_\ell \, [\mu\mathrm{K}^2]$", fontsize=20)
#     plt.xlabel(r"$\ell$", fontsize=20)
#     plt.ylim(1e-1,1e4)
#     plt.xlim(300,4000)
#     plt.legend(loc='lower left', ncol=2, fontsize=20)
#     plt.grid()
#     plt.yscale('log')
#     fig = mpl.pyplot.gcf()
#     fig.set_size_inches(12,6)
#     plt.subplots_adjust(bottom=0.1,top=0.95,wspace=0.43,hspace=0.47, right=0.98,left=0.08)

#     plt.savefig('%s/%s_mode%i_fsky%.3f_LAT_190819_w350ds4_TT_EE_err.%s' % (outdir,target, mode, fsky, suffix),dpi=300)
#     plt.close()
#     ## plot the polarization noise curves
#     plt.clf()






#     ####################################################################
#     ####################################################################

    
