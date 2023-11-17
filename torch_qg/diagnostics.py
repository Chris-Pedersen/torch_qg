import numpy as np
import math
import copy
import torch

# Define dict for variable dimensions
spatial_dims = ('time','lev','y','x')
spectral_dims = ('time','lev','l','k')
dim_database = {
    'q': spatial_dims,
    'u': spatial_dims,
    'v': spatial_dims,
    'ufull': spatial_dims,
    'vfull': spatial_dims, 
    'qh': spectral_dims,
    'uh': spectral_dims,
    'vh': spectral_dims,
    'ph': spectral_dims, 
    'dqhdt': spectral_dims, 
    'Ubg': ('lev'),
    'Qy': ('lev'),
}

# dict for variable dimensions
var_attr_database = {
    'q':     { 'units': 's^-1',      'long_name': 'potential vorticity in real space',},
    'qh':    { 'units': 's^-1',      'long_name': 'potential vorticity in spectral space',},
    'ph':    { 'units': 'm^2 s^-1',  'long_name': 'streamfunction in spectral space',},
    'p':     { 'units': 'm^2 s^-1',  'long_name': 'streamfunction in real space',},
    'dqhdt': { 'units': 's^-2',      'long_name': 'previous partial derivative of potential vorticity wrt. time in spectral space',} , 
    'dqdt':  { 'units': 's^-2',      'long_name': 'previous partial derivative of potential vorticity wrt. time in real space',} , 
}

# dict for coordinate dimensions
coord_database = {
    'time': ('time'),
    'lev': ('lev'),
    'lev_mid': ('lev_mid'),
    'x': ('x'),
    'y': ('y'),
    'l': ('l'),
    'k': ('k'),
}

# list for dataset attributes
attribute_database = [
    'beta',
    'delta',
    'del2',
    'dt',
    'filterfac',
    'L',
    'M',
    'nk',
    'nl',
    'ntd',
    'nx',
    'ny',
    'nz',
    'pmodes',
    'radii',
    'rd',
    'rho',
    'rek',
    'taveint',
    'tavestart',
    'tc',
    'tmax',
    'tsnapint',
    'tsnapstart',
    'twrite',
    'W',
]

class Diagnostics():
    """ Include everything related to spectral diagnostics in this class """
    def _spectral_grid(self):
        """ Set up ispec grid """

        ll_max = np.abs(self.ll).max()
        kk_max = np.abs(self.kk).max()

        kmax = np.minimum(ll_max, kk_max)
        self.dkr = np.sqrt(self.dk**2 + self.dl**2)
        self.k1d=np.arange(0, kmax, self.dkr)

        return

    def _calc_derived_fields(self):
        self.xi=torch.fft.irfftn(-self.kappa2*self.ph,dim=(1,2))
        self.Jptpc = -self._advection(
                    (self.p[0] - self.p[1]),
                    (self.del1*self.u[0] + self.del2*self.u[1]),
                    (self.del1*self.v[0] + self.del2*self.v[1]))
        # fix for delta.neq.1
        self.Jpxi = self._advection(self.xi, self.u, self.v)

        self.Jq = self._advection(self.q, self.u, self.v)

    ## This is not used in the forward model, but used to calculate diagnostic
    ## quantities. Just copying the pyqg code for this for now..
    def _advection(self, q, u=None, v=None):
        """Given real inputs q, u, v, returns the advective tendency for
        q in spectral space. Do everything in numpy here, since we are just
        doing diagnostics """
        if u is None:
            u = self.u.numpy()
        if v is None:
            v = self.v.numpy()
        uq = u*q
        vq = v*q

        ## Hack imported from pyqg, to avoid shaping issues when passing single-layer
        ## tensors to the fft. It's a bit messy but in a rush right now
        is_2d = (uq.ndim==2)
        if is_2d:
            uq = np.tile(uq[np.newaxis,:,:], (2,1,1))
            vq = np.tile(vq[np.newaxis,:,:], (2,1,1))

        tend = self.ik*np.fft.rfftn(uq,axes=(1,2)) + self.il*np.fft.rfftn(vq,axes=(1,2))
        if is_2d:
            return tend[0]
        else:
            return tend


    def get_KE_ispec(self):
        """ From current state variables, calculate isotropically averaged KE spectra
            Do this for both upper and lower layers at once """

        phr = np.zeros((2,self.kr.size()[0]))
        KEspec=(self.kappa2*np.abs(self.ph)**2/self.M**2)
        kespec=copy.copy(KEspec)

        ## Account for complex conjugate
        kespec[:,:,0] /= 2
        kespec[:,:,-1] /= 2

        ## Loop over wavenumbers. Average all modes within a given |k| range
        for i in range(self.kr.size()[0]):
            if i == self.kr.size()[0]-1:
                fkr = (self.kappa>=self.k1d[i]) & (self.kappa<=self.k1d[i]+self.dkr)
            else:
                fkr = (self.kappa>=self.k1d[i]) & (self.kappa<self.k1d[i+1])
            phr[:,i] = kespec[:,fkr].mean(axis=-1) * (self.kr[i]+self.dkr/2) * math.pi / (self.dk * self.dl)

            phr[:,i] *= 2 # include full circle

        return phr

        def to_dataset(self):
            """ Convert current state variables to xarray dataset. Include
                spectral quantities """

            return 
