import numpy as np
_ar = np.array

new_lat_beams = _ar([11.0, 7.3, 5.5, 2.3, 1.5, 1.0, 0.8])
#old_lat_beams = _ar([10.0, 7.4, 5.1, 2.2, 1.4, 1.0, 0.9])

old_lat_beams = _ar([10.0, 7.4, 5.1, 2.2, 1.4, 1.0, 0.9])/2.

meta = {
    'hires_deepwide': {
        'freq': _ar([30. , 40. , 95. , 145. , 220. , 270. ]),
        'beam': old_lat_beams[1:],
        'corr_bands': [(0,1,0.9), (2,3,.9), (4,5,.9)],
        'TT': _ar([
            [21.8,12.4,2.0,2.0,6.9,16.7],
            [471,428,2154,4364,7334,7308],
            [-3.5,-3.5,-3.5,-3.5,-3.5,-3.5],
        ]),
        'PP': _ar([
            [30.8,17.6,2.9,2.8,9.8,23.6],
            [700,700,700,700,700,700],
            [-1.4,-1.4,-1.4,-1.4,-1.4,-1.4],
        ]),
        'fsky_depends': ('gal_cut', (0,10,20,30)),
        'fsky_noise': [70.9, 64.8, 57.3, 50.1],
        'fsky_signal': [66.2, 60.6, 53.3, 46.5],
        'ell_min': 2,
    },
    'hires_ultradeep': {
        'freq': _ar([20., 30. , 40. , 95. , 145. , 220. , 270. ]),
        'beam': new_lat_beams,
        'corr_bands': [(1,2,0.9), (3,4,.9), (5,6,.9)],
        'PP': _ar([
            [8.44,5.03,4.51,0.68,0.96,5.72,9.80],
            [200,200,200,200,200,200,200],
            [-2.0,-2.0,-2.0,-2.0,-3.0,-3.0,-3.0],
        ]),
        'ell_min': 30,
        'fsky_noise': .0288,
        'fsky_signal': None,
    },
    'lores_ultradeep': {
        'freq': _ar([30, 40, 85, 95, 145, 155, 220, 270]),
        'beam': _ar([72.8, 72.8, 25.5, 25.5, 22.7, 22.7, 13.0, 13.0]),
        'corr_bands': [(0,1,0.9), (2,4,.9), (3,5,.9), (6,7,.9)],
        'TT': _ar([
            [5.64,7.14,1.41,1.24,2.71,2.90,7.50,12.85],
            [150,150,150,150,230,230,230,230],
            [-4.4,-4.4,-4.4,-4.4,-3.8,-3.8,-3.8,-3.8],
        ]),
        'EE': _ar([
            [3.74,4.73,0.93,0.82,1.25,1.34,3.48,8.08],
            [60,60,60,60,65,65,65,65],
            [-2.2,-2.2,-2.2,-2.2,-3.1,-3.1,-3.1,-3.1],
        ]),
        'BB': _ar([
            [3.53,4.46,0.88,0.78,1.23,1.34,3.48,5.97],
            [60,60,60,60,60,60,60,60],
            [-1.7,-1.7,-1.7,-1.7,-3.0,-3.0,-3.0,-3.0],
        ]),
        'ell_min': 0,
        'fsky_noise': .0288,
        'fsky_signal': None,
    },
}

def get_model(field_name, component, ell, gal_cut=None,
              deconv_beam=True):
    """Returns a tuple with information dictionary and a list of noise
    curves, like:
    
        (info_dict, freqs, Nmatrix)

    The freqs vector is one dimensional.  The noise matrix Nmatrix has
    shape (n_freq,n_freq,n_ell), and units uK^2 arcmin^2.

    """

    # Populate info..
    info = {'field_name': field_name,
            'component': component,
            'deconv_beam': deconv_beam}

    table = meta[field_name]
    if component in ['EE', 'BB']:
        if 'PP' in table:
            component = 'PP'
    comp_data = table[component]  # Looking for TT / EE / BB / PP...

    if table.get('fsky_depends'):
        param, val_list = table['fsky_depends']
        assert(param == 'gal_cut')
        val = gal_cut
        try:
            val_index = val_list.index(val)
        except ValueError as e:
            raise ValueError('This field requires %s in %s; you gave me %s.' % (param, val_list, val))

    else:
        info['fsky_noise'] = table['fsky_noise']
        info['fsky_signal'] = table['fsky_signal']

    ell = np.asarray(ell)
    if np.any(ell < table.get('ell_min', 0)):
        raise ValueError('Minimum ell for this field is %.1f' % table.get('ell_min', 0))

    def get_comps(i):
        return np.transpose(comp_data)[i]

    out_rows = []
    nband = len(table['freq'])

    N = np.zeros((nband, nband, len(ell)))
    for i in range(nband):
        w, knee, gamma = get_comps(i)
        N[i,i,:] = (w*np.pi/180./60.)**2 * (1 + (ell/knee)**gamma)
    # Correlated pairs.
    info['corr_bands'] = [_x for _x in table.get('corr_bands', [])]
    for i,j,rho in info['corr_bands']:
        w1, knee1, gamma1 = get_comps(i)
        w2, knee2, gamma2 = get_comps(j)
        N[i,j,:] = rho * (w1*np.pi/180./60. * (ell/knee1)**(gamma1/2)) * (w2*np.pi/180./60. * (ell/knee2)**(gamma2/2))
        N[j,i,:] = N[i,j,:]
        
    if deconv_beam:
        beam_tf = np.zeros((nband, len(ell)))
        for i in range(nband):
            beam_fwhm_arcmin = table['beam'][i]
            beam_sig_rad = beam_fwhm_arcmin * np.pi/180/60 / (8.*np.log(2))**0.5
            beam_tf[i,:] = np.exp(-0.5 * ell*(ell+1) * beam_sig_rad**2)
        N /= beam_tf[:,None,:] * beam_tf[None,:,:]
    return (info, table['freq'].copy(), N)

   
if __name__ == '__main__':
    #import pylab as pl
    import matplotlib
    matplotlib.use('pdf')
    matplotlib.rc('font', family='serif', serif='cm10')
    matplotlib.rc('text', usetex=True)
    fontProperties = {'family':'sans-serif',
                      'weight' : 'normal', 'size' : 16}
    import matplotlib.pyplot as pl

    ell = np.arange(100,8000,1)
    info, freqs, Nmatrix = get_model('hires_deepwide', 'TT', ell,
                                     gal_cut=10, deconv_beam=True)
    for i,f in enumerate(freqs):
        pl.loglog(ell, Nmatrix[i,i], label='%i GHz' % f, lw=2.)
    for i,j,_ in info['corr_bands']:
        pl.loglog(ell, Nmatrix[i,j], label='%i x %i' % (freqs[i], freqs[j]), lw=2.)

    #pl.title('{field_name} - {component}'.format(**info))
    pl.legend(ncol=2,loc='upper right',fontsize=7)
    pl.xlabel(r'$\ell$',fontsize=17)
    pl.ylabel(r'$N_{\ell}^{{\rm TT,hires-deepwide}} \, [\mu {\rm K}^2]$',fontsize=17)
    pl.grid(alpha=0.5)
    pl.xlim(100,8000)
    pl.ylim(5e-7,1e0)
    #pl.show()
    pl.savefig('hires_deepwide_S4_TT_noise_190604d.pdf')

    pl.clf()
    info, freqs, Nmatrix = get_model('hires_deepwide', 'EE', ell,
                                     gal_cut=10, deconv_beam=True)
    for i,f in enumerate(freqs):
        pl.loglog(ell, Nmatrix[i,i], label='%i GHz' % f, lw=2.)
    for i,j,_ in info['corr_bands']:
        pl.loglog(ell, Nmatrix[i,j], label='%i x %i' % (freqs[i], freqs[j]), lw=2.)

    #pl.title('{field_name} - {component}'.format(**info))
    pl.legend(ncol=2,loc='upper right',fontsize=7)
    pl.xlabel(r'$\ell$',fontsize=17)
    pl.ylabel(r'$N_{\ell}^{{\rm EE,hires-deepwide}} \, [\mu {\rm K}^2]$',fontsize=17)
    pl.grid(alpha=0.5)
    pl.xlim(100,8000)
    pl.ylim(1e-7,1e-1)
    #pl.show()
    pl.savefig('hires_deepwide_S4_EE_noise_190604d.pdf')

    pl.clf()
    ell = np.arange(10,300,10)
    info, freqs, Nmatrix = get_model('lores_ultradeep', 'EE', ell,
                                     deconv_beam=True)
    for i,f in enumerate(freqs):
        pl.semilogy(ell, Nmatrix[i,i], label='%i GHz' % f)
    for i,j,_ in info['corr_bands']:
        pl.semilogy(ell, Nmatrix[i,j], label='%ix%i' % (freqs[i], freqs[j]))
    #pl.title('{field_name} - {component}'.format(**info))
    pl.legend()
    pl.xlabel(r'$\ell$')
    pl.ylabel(r'$N_{\ell}^{EE,lores-ultradeep} \, [\mu {\rm K}^2]$')
    pl.xlim(100,300)
    pl.ylim(5e-8,1e-1)
    #pl.show()
    pl.savefig('lores_ultradeep_S4_EE_noise_190604d.pdf')

