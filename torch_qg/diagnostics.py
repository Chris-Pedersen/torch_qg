import numpy as np
import math
import copy
import torch
import xarray as xr

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

    def _increment_diagnostics(self):
        """ Add diagnostics of current system state to the self.diagnostics list """

        ## First update derived quantities
        self._calc_derived_fields()

        self.diagnostics["KEspec"].append(self.get_KE_ispec())
        self.diagnostics["SPE"].append(self.get_spectral_energy_transfer())
        self.diagnostics["Ensspec"].append(self.get_enstrophy_ispec())

    def _calc_derived_fields(self):
        """ Taken from pyqg - compupte various quantities that are used
            for diagnostics. These are stored as self objects """

        self.xi=torch.fft.irfftn(-self.kappa2*self.ph,dim=(1,2))
        self.Jptpc = -self._advection(
                    (self.p[0] - self.p[1]),
                    (self.del1*self.u[0] + self.del2*self.u[1]),
                    (self.del1*self.v[0] + self.del2*self.v[1]))
        # fix for delta.neq.1
        self.Jpxi = self._advection(self.xi, self.u, self.v)

        self.Jq = self._advection(self.q, self.u, self.v)

        return

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

    def get_ispec_1(self,field):
        """ For an input [nx,ny] field, calculate the isotropically averaged spectra """

        ## Array to output isotropically averaged wavenumbers
        phr = np.zeros((self.k1d.size))

        ispec=copy.copy(field)

        ## Account for complex conjugate
        ispec[:,0] /= 2
        ispec[:,-1] /= 2

        ## Loop over wavenumbers. Average all modes within a given |k| range
        for i in range(self.k1d.size):
            if i == self.k1d.size-1:
                fkr = (self.kappa>=self.k1d[i]) & (self.kappa<=self.k1d[i]+self.dkr)
            else:
                fkr = (self.kappa>=self.k1d[i]) & (self.kappa<self.k1d[i+1])
            phr[i] = ispec[fkr].mean(axis=-1) * (self.k1d[i]+self.dkr/2) * math.pi / (self.dk * self.dl)

            phr[i] *= 2 # include full circle

        return phr

    def get_ispec_2(self,field):
        """ For an input [2,nx,ny] field, calculate the isotropically averaged spectra """

        ## Array to output isotropically averaged wavenumbers
        phr = np.zeros((2,self.k1d.size))
        ispec=copy.copy(field)

        ## Account for complex conjugate
        ispec[:,:,0] /= 2
        ispec[:,:,-1] /= 2

        ## Loop over wavenumbers. Average all modes within a given |k| range
        for i in range(self.k1d.size):
            if i == self.k1d.size-1:
                fkr = (self.kappa>=self.k1d[i]) & (self.kappa<=self.k1d[i]+self.dkr)
            else:
                fkr = (self.kappa>=self.k1d[i]) & (self.kappa<self.k1d[i+1])
            phr[:,i] = ispec[:,fkr].mean(axis=-1) * (self.k1d[i]+self.dkr/2) * math.pi / (self.dk * self.dl)

            phr[:,i] *= 2 # include full circle

        return phr

    def get_KE_ispec(self):
        """ From current state variables, calculate isotropically averaged KE spectra
            Do this for both upper and lower layers at once """

        #KEspec=(self.kappa2*np.abs(self.ph)**2/self.M**2)
        return self.get_ispec_2(self.kappa2*np.abs(self.ph)**2/self.M**2)

    def get_spectral_energy_transfer(self):
        """ Return spectral energy transfer """

        kef=((np.real(self.del1*self.ph[0]*np.conj(self.Jpxi[0])) + np.real(self.del2*self.ph[1]*np.conj(self.Jpxi[1])))/self.M**2)
        ape=(self.rd**-2*self.del1*self.del2*np.real((self.ph[0]-self.ph[1])*np.conj(self.Jptpc))/self.M**2)

        return self.get_ispec_1(ape+kef)

    def get_enstrophy_ispec(self):
        """ Get enstrophy spectrum """

        return self.get_ispec_2(np.abs(self.qh)**2/self.M**2)

    def get_aved_diagnostics(self,diag):
        """ For a given diagnostic string, average over all saved states """

        ## First we create the self.diagnostic[diag] list of arrays into a single array
        ## Do this by first creating a tensor of shape
        ## [n_layers,number of wavenumber bins,number of saved states]
        ## to store all saved values
        diag_tensor=np.empty(self.diagnostics[diag][0].shape+tuple([len(self.diagnostics[diag])]))

        ## Populate array by looping over list of stored arrays
        for aa in range((len(self.diagnostics[diag]))):
            diag_tensor[...,aa]=self.diagnostics[diag][aa]

        ## Now we can just average over this tensor
        return np.mean(diag_tensor,axis=-1)

    def get_coord_dic(self):
        """ Generate coordinates dictionary for xarray output """

        coordinates = {}
        coordinates["time"] = ("time",np.array([self.dt*self.timestep]),
                        {'long_name': 'model time', 'units': 's'})
        coordinates["lev"] = ("lev",np.array([1,2]),{'long_name': 'vertical levels'})
        coordinates["x"] = ("x",self.x[:,0].numpy(),
                        {'long_name': 'real space grid points in the x direction', 'units': 'grid point',})
        coordinates["y"] = ("y",self.y[0,:],
                        {'long_name': 'real space grid points in the y direction', 'units': 'grid point',})
        coordinates["k1d"] = ("k1d",self.k1d,
                        {'long_name':'1D Fourier wavenumber for isotropically averaged spectra', 'units':'m^-1'})

        return coordinates


    def state_to_dataset(self):
        """ Convert current state variables to xarray dataset. Do not include
            spectral quantities """

        coords=self.get_coord_dic()

        variables={}
        variables["q"]=(('time','lev','y','x'),self.q.unsqueeze(0).numpy().copy(),
                { 'units': 's^-1',      'long_name': 'potential vorticity in real space',})
        variables["p"]=(('time','lev','y','x'),self.p.unsqueeze(0).numpy().copy(),
                { 'units': 'm^2 s^-1',      'long_name': 'streamfunction in real space',})

        ## Add spectral diagnostics if there are any
        if len(self.diagnostics["KEspec"])>0:
            variables["KEspec"]=(('time','lev','k1d'),np.expand_dims(self.get_aved_diagnostics("KEspec"),axis=0),
                            { 'units': 'm^2 s^-2',  'long_name': 'KE spectrum'})
            variables["Enspec"]=(('time','lev','k1d'),np.expand_dims(self.get_aved_diagnostics("Ensspec"),axis=0),
                            { 'units': 's^-2',  'long_name': 'Enstrophy spectrum'})
            variables["SPE"]=(('time','k1d'),np.expand_dims(self.get_aved_diagnostics("SPE"),axis=0),
                            { 'units': 'm^2 s^-3',  'long_name': 'Spectral energy transfer'})

        return xr.Dataset(variables,coords=coords)
