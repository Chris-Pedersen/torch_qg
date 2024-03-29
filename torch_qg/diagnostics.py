import numpy as np
import math
import copy
import torch
import xarray as xr
import torch_qg.util as util

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
    'scheme'  ## Record whether Arakawa or PseudoSpectral
]

class Diagnostics():
    """ Include everything related to spectral diagnostics in this class """

    def _increment_diagnostics(self):
        """ Add diagnostics of current system state to the self.diagnostics list """

        ## First update derived quantities
        self._calc_derived_fields()

        self.diagnostics["KEspec"].append(self.get_KE_ispec())
        self.diagnostics["SPE"].append(self.get_spectral_energy_transfer())
        #self.diagnostics["SPE2"].append(self.get_spectral_energy_transfer2())
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
            u = self.u
        if v is None:
            v = self.v
        uq = u*q
        vq = v*q

        ## Hack imported from pyqg, to avoid shaping issues when passing single-layer
        ## tensors to the fft. It's a bit messy but in a rush right now
        is_2d = (uq.ndim==2)
        if is_2d:
            uq = uq.expand(2,uq.shape[-1],uq.shape[-1])
            vq = vq.expand(2,vq.shape[-1],vq.shape[-1])

        tend = self.ik*torch.fft.rfftn(uq,dim=(1,2)) + self.il*torch.fft.rfftn(vq,dim=(1,2))
        if is_2d:
            return tend[0]
        else:
            return tend

    def get_ispec_1(self,field):
        """ For an input [nx,ny] field, calculate the isotropically averaged spectra """

        ## Array to output isotropically averaged wavenumbers
        phr = np.zeros((self.k1d.size))

        ispec=copy.copy(field.cpu().numpy())

        ## Account for complex conjugate
        ispec[:,0] /= 2
        ispec[:,-1] /= 2

        ## Loop over wavenumbers. Average all modes within a given |k| range
        for i in range(self.k1d.size):
            if i == self.k1d.size-1:
                fkr = (self.kappa_cpu>=self.k1d[i]) & (self.kappa_cpu<=self.k1d[i]+self.dkr)
            else:
                fkr = (self.kappa_cpu>=self.k1d[i]) & (self.kappa_cpu<self.k1d[i+1])
            phr[i] = ispec[fkr].mean(axis=-1) * (self.k1d[i]+self.dkr/2) * math.pi / (self.dk * self.dl)

            phr[i] *= 2 # include full circle

        return phr

    def get_ispec_2(self,field):
        """ For an input [2,nx,ny] field, calculate the isotropically averaged spectra """

        ## Array to output isotropically averaged wavenumbers
        phr = np.zeros((2,self.k1d.size))
        ispec=copy.copy(field.cpu().numpy())

        ## Account for complex conjugate
        ispec[:,:,0] /= 2
        ispec[:,:,-1] /= 2

        ## Loop over wavenumbers. Average all modes within a given |k| range
        for i in range(self.k1d.size):
            if i == self.k1d.size-1:
                fkr = (self.kappa_cpu>=self.k1d[i]) & (self.kappa_cpu<=self.k1d[i]+self.dkr)
            else:
                fkr = (self.kappa_cpu>=self.k1d[i]) & (self.kappa_cpu<self.k1d[i+1])
            phr[:,i] = ispec[:,fkr].mean(axis=-1) * (self.k1d[i]+self.dkr/2) * math.pi / (self.dk * self.dl)

            phr[:,i] *= 2 # include full circle

        return phr

    def get_total_KE(self):
        """ Calculate total kinetic energy in system """

        ke=(self.u**2 + self.v**2) * 0.5
        ke=(self.L*torch.sum(ke))/(self.nx**2)

        return ke.cpu()

    def get_KE_ispec(self):
        """ From current state variables, calculate isotropically averaged KE spectra
            Do this for both upper and lower layers at once """

        #KEspec=(self.kappa2*np.abs(self.ph)**2/self.M**2)
        return self.get_ispec_2(self.kappa2.cpu()*np.abs(self.ph.cpu())**2/self.M**2)

    def get_spectral_energy_transfer(self):
        """ Return spectral energy transfer. Calculations taken from pyqg - individual terms
            due to ke flux and ape are calculated """

        kef=((torch.real(self.del1*self.ph[0]*torch.conj(self.Jpxi[0])) + torch.real(self.del2*self.ph[1]*torch.conj(self.Jpxi[1])))/self.M**2)
        ape=(self.rd**-2*self.del1*self.del2*torch.real((self.ph[0]-self.ph[1])*torch.conj(self.Jptpc))/self.M**2)
        
        ## Sum contributions of ape and kinetic energy flux
        spec_trans=ape+kef
        
        ## If we are using a parameterization, include contribution too
        if self.parameterization:
            paramspec=-torch.real((self.height_ratios * torch.conj(self.ph) * self.dqh).sum(axis=0)) / self.M**2
            spec_trans+=paramspec

        return self.get_ispec_1(spec_trans)

    def get_spectral_energy_transfer2(self):
        """ Return spectral energy transfer:
            calculate using an alternative method, just the cross-spectrum between
            streamfunction and rhs """

        rhsh=self.rhsh(self.q,self.qh,self.ph,self.u,self.v)

        spec_trans=-torch.real((self.height_ratios * torch.conj(self.ph) * rhsh).sum(axis=0)) / self.M**2

        return self.get_ispec_1(spec_trans)

    def get_enstrophy_ispec(self):
        """ Get enstrophy spectrum """

        return self.get_ispec_2(torch.abs(self.qh)**2/self.M**2)

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
        coordinates["x"] = ("x",self.x[:,0].cpu().numpy(),
                        {'long_name': 'real space grid points in the x direction', 'units': 'grid point',})
        coordinates["y"] = ("y",self.y[0,:].cpu(),
                        {'long_name': 'real space grid points in the y direction', 'units': 'grid point',})
        coordinates["k1d"] = ("k1d",self.k1d_plot,
                        {'long_name':'1D Fourier wavenumber for isotropically averaged spectra', 'units':'m^-1'})

        return coordinates

    def state_to_dataset(self):
        """ Convert current state variables to xarray dataset. """

        coords=self.get_coord_dic()

        variables={}
        variables["q"]=(('time','lev','y','x'),self.q.cpu().unsqueeze(0).numpy().copy(),
                { 'units': 's^-1',      'long_name': 'potential vorticity in real space',})
        variables["p"]=(('time','lev','y','x'),self.p.cpu().unsqueeze(0).numpy().copy(),
                { 'units': 'm^2 s^-1',      'long_name': 'streamfunction in real space',})
        variables["u"]=(('time','lev','y','x'),self.u.cpu().unsqueeze(0).numpy().copy(),
                { 'units': 'm s^-1',      'long_name': 'zonal velocity',})
        variables["v"]=(('time','lev','y','x'),self.v.cpu().unsqueeze(0).numpy().copy(),
                { 'units': 'm s^-1',      'long_name': 'meridional velocity',})
        variables["KE"]=(('time'),self.get_total_KE().unsqueeze(0).numpy(),
                { 'units': 'm^2 s^-2',      'long_name': 'total KE',})

        ## Add spectral diagnostics if there are any
        if len(self.diagnostics["KEspec"])>0:
            variables["KEspec"]=(('time','lev','k1d'),np.expand_dims(self.get_aved_diagnostics("KEspec"),axis=0),
                            { 'units': 'm^2 s^-2',  'long_name': 'KE spectrum'})
            variables["Enspec"]=(('time','lev','k1d'),np.expand_dims(self.get_aved_diagnostics("Ensspec"),axis=0),
                            { 'units': 's^-2',  'long_name': 'Enstrophy spectrum'})
            variables["SPE"]=(('time','k1d'),np.expand_dims(self.get_aved_diagnostics("SPE"),axis=0),
                            { 'units': 'm^3 s^-3',  'long_name': 'Spectral energy transfer'})
            #variables["SPE2"]=(('time','k1d'),np.expand_dims(self.get_aved_diagnostics("SPE2"),axis=0),
            #                { 'units': 'm^3 s^-3',  'long_name': 'Spectral energy transfer'})

        global_attrs = {}
        for aname in attribute_database:
            if hasattr(self, aname):
                data = getattr(self, aname)
                global_attrs[f"torchqg:{aname}"] = (data)

        ds=xr.Dataset(variables,coords=coords,attrs=global_attrs)
        ds.attrs['title'] = 'torchqg: 2-layer Quasigeostrophic system evolved in PyTorch'
        ds.attrs['reference'] = 'https://github.com/Chris-Pedersen/torch_qg'

        return ds

    def forcing_dataset(self,lr_model):
        """ Produce a dataset of subgrid forcing quantities. Pass a lower resolution model
            which we will use the filter, rhs and gridpoints from to produce a dataset from """

        ## Set q field in low res model to the downsampled, high res fields
        ## Dervied quantities are calculated in the low res model automatically
        ## when set_q1q2 is called
        lr_model.set_q1q2(torch.fft.irfftn(util.spectral_smoothing(self.qh,self.nx,lr_model),dim=(1,2)))
        ## Calculate rhs for high res model, then downsample
        down_rhsh=util.spectral_smoothing(self.rhsh(self.q,self.qh,self.ph,self.u,self.v),self.nx,lr_model)
        ## Get rhs for low res model
        lowres_rhsh=lr_model.rhsh(lr_model.q,lr_model.qh,lr_model.ph,lr_model.u,lr_model.v)
        ## Subgrid forcing field is the difference between these two rhshs
        S=torch.fft.irfftn(down_rhsh-lowres_rhsh,dim=(1,2))

        ## Store the downwsampled high res q, and the corresponding subgrid forcing field
        ## in an xarray dataset and return
        coords = {}
        coords["time"] = ("time",np.array([self.dt*self.timestep]),
                        {'long_name': 'model time', 'units': 's'})
        coords["lev"] = ("lev",np.array([1,2]),{'long_name': 'vertical levels'})
        coords["x"] = ("x",lr_model.x[:,0].cpu().numpy(),
                        {'long_name': 'real space grid points in the x direction', 'units': 'grid point',})
        coords["y"] = ("y",lr_model.y[0,:].cpu().numpy(),
                        {'long_name': 'real space grid points in the y direction', 'units': 'grid point',})

        variables={}
        variables["q"]=(('time','lev','y','x'),lr_model.q.cpu().unsqueeze(0).numpy(),
                { 'units': 's^-1',      'long_name': 'downsampled, high res potential vorticity field',})
        variables["S"]=(('time','lev','y','x'),S.cpu().unsqueeze(0).numpy(),
                { 'units': 'm^2 s^-1',      'long_name': 'subgrid forcing field',})

        global_attrs = {}
        for aname in attribute_database:
            if hasattr(self, aname):
                data = getattr(self, aname)
                global_attrs[f"torchqg:{aname}"] = (data)

        ds=xr.Dataset(variables,coords=coords,attrs=global_attrs)
        ds.attrs['title'] = 'torchqg: 2-layer Quasigeostrophic system evolved in PyTorch'
        ds.attrs['reference'] = 'https://github.com/Chris-Pedersen/torch_qg'

        return ds
