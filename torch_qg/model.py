import torch
import math
import numpy as np
from tqdm import tqdm
import xarray as xr

import torch_qg.diagnostics as diagnostics


class BaseQGModel():
    def __init__(
        self,
        nx=64,                      # number of gridpoints
        dt=3600.,                   # numerical timestep
        L=1e6,                      # domain size is L [m]
        beta=1.5e-11,               # gradient of coriolis parameter
        rek=5.787e-7,               # linear drag in lower layer
        rd=15000.0,                 # deformation radius
        delta=0.25,                 # layer thickness ratio (H1/H2)
        H1 = 500,                   # depth of layer 1 (H1)
        U1=0.025,                   # upper layer flow
        U2=0.0,                     # lower layer flow
        parameterization=None,      # parameterization
        diagnostics_start=4e4,      # Number of timesteps after which to start sampling diagnostics
        diagnostics_freq=25,        # Frequency at which to sample diagnostics
        silence=False,              # Set to True to disable progress bar (to prevent slurm logs being polluted)
        **kwargs
        ):
        """
        Parameters
        ----------

        beta : number
            Gradient of coriolis parameter. Units: meters :sup:`-1`
            seconds :sup:`-1`
        rek : number
            Linear drag in lower layer. Units: seconds :sup:`-1`
        rd : number
            Deformation radius. Units: meters.
        delta : number
            Layer thickness ratio (H1/H2)
        U1 : number
            Upper layer flow. Units: meters seconds :sup:`-1`
        U2 : number
            Lower layer flow. Units: meters seconds :sup:`-1`
        """

        # physical
        self.beta = beta
        self.rek = rek
        self.rd = rd
        self.delta = delta
        self.Hi = torch.tensor([ H1, H1/delta])
        self.U1 = U1
        self.U2 = U2
        self.nx = nx
        self.ny = nx
        self.L = L
        self.W = L
        self.nl = nx
        self.nk = nx/2 + 1
        self.dt = dt
        self.parameterization = parameterization
        self.silence=silence

        ## Diagnostics config
        self.diagnostics_start=diagnostics_start
        self.diagnostics_freq=diagnostics_freq

        ## Set previous timestep rhs to None at initialisation
        self.rhs_minus_one=None
        self.rhs_minus_two=None

        ## Counter to record simulation timesteps
        self.timestep=0

        self._initialise_background()
        self._initialise_grid()
        self._initialise_q1q2()

    def _initialise_background(self):
        """ Calculate various quantities relevant to model background """
        
        self.F1 = self.rd**-2 / (1.+self.delta)
        self.F2 = self.delta*self.F1
        self.H = self.Hi.sum()
        self.betas=torch.tensor([self.beta+self.F1*(self.U1-self.U2),self.beta-self.F2*(self.U1-self.U2)])

        ## Set up tensor for background velocities, used in rhs calculations
        self.u_mean=torch.ones((2,self.nx,self.ny))
        self.u_mean[0]*=self.U1
        self.u_mean[1]*=self.U2

        ## Layer spacing - used in calculations of APE/KE fluxes
        self.del1 = self.delta/(self.delta+1.)
        self.del2 = (self.delta+1.)**-1
        
    def _initialise_grid(self):
        """ Set up real-space and spectral-space grids """
        
        self.x,self.y = torch.meshgrid(
        torch.arange(0.5,self.nx,1.)/self.nx*self.L,
        torch.arange(0.5,self.ny,1.)/self.ny*self.W )
        ## physical grid spacing
        self.dx = self.L / self.nx
        self.dy = self.W / self.ny
        self.M = self.nx*self.ny

        # Notice: at xi=1 U=beta*rd^2 = c for xi>1 => U>c
        # wavenumber one (equals to dkx/dky)
        self.dk = 2.*math.pi/self.L
        self.dl = 2.*math.pi/self.W

        ## Define wavenumber arrays - nb that fft comes out
        ## with positive frequencies, then negative frequencies,
        ## and that rfft returns only half the plane
        self.ll = self.dl*torch.cat((torch.arange(0.,self.nx/2),
            torch.arange(-self.nx/2,0.)))
        self.kk = self.dk*torch.arange(0.,self.nk)

        ## Get wavenumber grids in complex plane
        self.k, self.l = torch.meshgrid(self.kk, self.ll)
        
        ## Torch meshgrid produces arrays with opposite indices to numpy, so we take the transpose
        self.k=self.k.T ## Zonal
        self.l=self.l.T ## Meridional
        self.ik = 1j*self.k
        self.il = 1j*self.l
        
        ## kappa2 represents the wavenumber squared at each gridpoint
        self.kappa2=(self.l**2+self.k**2)
        self.kappa=torch.sqrt(self.kappa2)
        
        ## Evaluate 2x2 matrix determinant for calculation of streamfunction
        self.determinant=self.kappa2*(self.kappa2+self.F1+self.F2)
        ## Set to false value so matrix inversion doesn't throw warnings
        self.determinant[0,0]=1e-19
        
        
        ## spectral grid for isoptrically averaged spectra (in numpy for now)
        ## as this quantity will be output into xarray, and not included in
        ## any backprop
        ll_max = np.abs(self.ll).max()
        kk_max = np.abs(self.kk).max()

        kmax = np.minimum(ll_max, kk_max)
        self.dkr = np.sqrt(self.dk**2 + self.dl**2)
        self.k1d=np.arange(0, kmax, self.dkr)+self.dkr/2

        ## Diagnostics are kinetic energy spectrum, spectral energy transfer, enstrophy spectrum
        self.diagnostics={"KEspec":[],
                    "SPE":[],
                    "Ensspec":[]}
        
        return
    
    def _initialise_q1q2(self):
        """ Initialise potential vorticity using a randomly generated pattern to instigate baroclinic instability """

        ## Should already be None, but just making sure in case this is called externally
        self.rhs_minus_one=None
        self.rhs_minus_two=None

        self.q=torch.stack((1e-7*torch.rand(self.ny,self.nx,dtype=torch.float64) + 1e-6*(torch.ones((self.ny,1),dtype=torch.float64)
                                    * torch.rand(1,self.nx,dtype=torch.float64) ),torch.zeros(self.nx,self.nx,dtype=torch.float64)))

        ## Update other state variables
        self.qh=torch.fft.rfftn(self.q,dim=(1,2))
        self.ph=self.invert(self.qh)
        self.p=torch.fft.irfftn(self.ph,dim=(1,2))
        ## Get u, v in spectral space, then ifft to real space
        self.u=torch.fft.irfftn(-self.il*self.ph,dim=(1,2))
        self.v=torch.fft.irfftn(self.ik*self.ph,dim=(1,2))
        self.timestep=0 ## Reset timestep counter
        
        return

    def calc_cfl(self):
        """ Calculate CFL for the current state of the system """

        ## Get u, v in spectral space, then ifft to real space
        u=torch.fft.irfftn(-self.il*self.ph,dim=(1,2))+self.u_mean
        v=torch.fft.irfftn(self.ik*self.ph,dim=(1,2))

        ## Stack u and v velocities, compare to gridstep and grid sizes
        return torch.abs(torch.stack((u,v))).max()*self.dt/self.dx

    def set_q1q2(self,q):
        """ Set potential vorticity to some specific configuration. We make sure to remove any previous timesteps stored in
            the rhs for the AB3 scheme. """

        if type(q)==np.ndarray:
            self.q=torch.tensor(q,dtype=torch.float64)
        else:
            ## Ensure we are float64 even if passing a torch tensor
            self.q=q.type(torch.float64)

        ## Remove cached RHS, so we start from Euler
        self.rhs_minus_one=None
        self.rhs_minus_two=None

        ## Update other state variables
        self.qh=torch.fft.rfftn(self.q,dim=(1,2))
        self.ph=self.invert(self.qh)
        self.p=torch.fft.irfftn(self.ph,dim=(1,2))
        ## Get u, v in spectral space, then ifft to real space
        self.u=torch.fft.irfftn(-self.il*self.ph,dim=(1,2))
        self.v=torch.fft.irfftn(self.ik*self.ph,dim=(1,2))
        
        ## Make sure we are float64
        assert self.q.dtype==torch.float64, "Not float64"

        return

    def invert(self,qh):
        """ Invert 2x2 matrix equation to get streamfunction from potential vorticity.
            Takes q in spectral space, returns psi in spectral space """
        
        ph1=(-(self.kappa2+self.F2)*qh[0]-self.F1*qh[1])/self.determinant
        ph2=(-self.F2*qh[0]-(self.kappa2+self.F1)*qh[1])/self.determinant

        ## Fundamental mode is 0
        ph1[0,0]=0
        ph2[0,0]=0
        
        return torch.stack((ph1,ph2))

    def run_sim(self,steps,interval=1000):
        """ Evolve system forward in time by some number of steps. Interval
            sets the interval at which to store snapshots - these will be concat
            into an xarray dataset and returned after the function has finished running. """

        ds=self.state_to_dataset()

        for aa in tqdm(range(steps),disable=self.silence):
            self._step_ab3()

            ## If we hit NaNs, stop the show
            if torch.sum(torch.isnan(self.q))!=0:
                print("NaNs in pv field, stopping sim")
                break

            ## Check CFL every 1k timesteps
            if self.timestep % 1000==0:
                cfl=self.calc_cfl()
                assert cfl<1., "CFL condition violated"

            if self.timestep % interval==0:
                ds=xr.concat((ds,self.state_to_dataset()),dim="time")
                
        return ds

    def _step_ab3(self):
        raise NotImplementedError("Implemented by subclass")


class ArakawaModel(BaseQGModel, diagnostics.Diagnostics):
    def __init__(self,*args,**kwargs):
        super(ArakawaModel,self).__init__(*args,**kwargs)
        self.scheme="Arakawa"

    @staticmethod
    def diffx(x,dx):
        """ Central difference approximation to the spatial derivative in x direction 
            nb that our indices are [layer,y coordinate, x coordinate] so we torch.roll in
            dimension 2 for x derivative """

        return (torch.roll(x,shifts=-1,dims=2)-torch.roll(x,shifts=1,dims=2))/(2*dx)

    @staticmethod
    def diffy(x,dx):
        """ Central difference approximation to the spatial derivative in x direction 
            nb that our indices are [layer,y coordinate, x coordinate] so we torch.roll in
            dimension 1 for y derivative """

        return (torch.roll(x,shifts=-1,dims=1)-torch.roll(x,shifts=1,dims=1))/(2*dx)
    
    def advect(self,q,p):
        """ Arakawa advection scheme of q. Returns Arakawa advection - but does not update
            any state variables """
        
        f1 = self.diffx(p,self.dx)*self.diffy(q,self.dx) - self.diffy(p,self.dx)*self.diffx(q,self.dx)
        f2 = self.diffy(self.diffx(p,self.dx)*q,self.dx) - self.diffx(self.diffy(p,self.dx)*q,self.dx)
        f3 = self.diffx(self.diffy(q,self.dx)*p,self.dx) - self.diffy(self.diffx(q,self.dx)*p,self.dx)
        
        f = - (f1 + f2 + f3) / 3
        return f

    def rhs(self,q,qh,p,ph):
        """ Build a tensor of dq/dt. Does not update any state variables. """
        
        #### Spatial derivatives ####
        ## Spatial derivative of streamfunction using Fourier tensor
        dp=torch.fft.irfftn(self.ik*ph,dim=(1,2))
        d2p=torch.fft.irfftn(-self.kappa2*ph,dim=(1,2))
        dq=torch.fft.irfftn(self.ik*qh,dim=(1,2))
        
        rhs=-1*self.advect(q,p)
        rhs[0]+=(-self.betas[0]*dp[0])
        rhs[1]+=(-self.betas[1]*dp[1])
        
        rhs[0]+=-dq[0]*self.U1
        rhs[1]+=-dq[1]*self.U2
        
        ## Bottom drag
        rhs[1]+=-self.rek*d2p[1]
        
        if self.parameterization is not None:
            rhs+=self.parameterization(q,ph,self.ik,self.il,self.dx)
        
        return rhs

    def clean(self,q):
        """
        Remove frequencies which potentially
        can harm reversibility of rfftn
        """
        Xf = torch.fft.rfftn(q,dim=(1,2))
        n = q[1].shape[0] // 2
        Xf[0,n,0] = 0
        Xf[0,:,n] = 0
        Xf[1,n,0] = 0
        Xf[1,:,n] = 0
        return torch.fft.irfftn(Xf,dim=(1,2))

    def _step_ab3(self):
        """ Step the system state forward by one timestep. NB we assume that all 4 state variables (q,qh,psi,psih)
            are always on the same timestep. So we first use the rhs to update q_i to q_{i+1}, then use this to
            update the remaining 3 quantities to time {i+1} """

        ## First update q -> q_{i+1}
        rhs=self.rhs(self.q,self.qh,self.p,self.ph)
        ## If we have no n_{i-1} timestep, we just do forward Euler
        if self.rhs_minus_one==None:
            self.q=self.q+rhs*self.dt
            ## Store rhs as rhs_{i-1}
            self.rhs_minus_one=rhs
        ## If we have no n_{i-2} timestep, we do AB2
        elif self.rhs_minus_two==None:
            self.q=self.q+(0.5*self.dt)*(3*rhs-self.rhs_minus_one)
            ## Update previous timestep rhs
            self.rhs_minus_two=self.rhs_minus_one
            self.rhs_minus_one=rhs
        else:
            ## If we have two previous timesteps stored, use AB3
            self.q=self.q+(self.dt/12.)*(23*rhs-16*self.rhs_minus_one+5*self.rhs_minus_two)
            ## Update previous timesteps
            self.rhs_minus_two=self.rhs_minus_one
            self.rhs_minus_one=rhs

        ## "clean" q
        self.q=self.clean(self.q)
        ## self.q is now self.q_{i+1}. Now update the spectral quantities, and streamfunction
        ## such that all state variables are on the same timestep
        self.qh=torch.fft.rfftn(self.q,dim=(1,2))
        self.ph=self.invert(self.qh)
        ## Get u, v in spectral space, then ifft to real space
        self.u=torch.fft.irfftn(-self.il*self.ph,dim=(1,2))
        self.v=torch.fft.irfftn(self.ik*self.ph,dim=(1,2))
        self.p=torch.fft.irfftn(self.ph,dim=(1,2))
        self.timestep+=1

        ## Increment diagnostics
        if self.timestep>self.diagnostics_start and (self.timestep % self.diagnostics_freq ==0):
            self._increment_diagnostics()

        return


class PseudoSpectralModel(BaseQGModel, diagnostics.Diagnostics):
    def __init__(self,dealias=False,*args,**kwargs):
        super(PseudoSpectralModel,self).__init__(*args,**kwargs)
        self.filterfac=23.6
        self.scheme="PseudoSpectral"
        self.dealias=dealias
        self._initialize_filter(self.dealias)

    def _initialize_filter(self,dealias=False):
        """Set up frictional filter."""

        if self.dealias==False:
            # this defines the spectral filter (following Arbic and Flierl, 2003)
            cphi=0.65*math.pi
            wvx=np.sqrt((self.k*self.dx)**2.+(self.l*self.dy)**2.)
            filtr = np.exp(-self.filterfac*(wvx-cphi)**4.)
            filtr[wvx<=cphi] = 1.
        else:
            filtr = torch.zeros_like(self.kappa2)
            n = self.nx // 3
            filtr[:n,:n] = 1
            filtr[-n:,:n] = 1
        self.filtr = filtr

        return
    
    def advect(self,q,u,v):
        """ Pseudo-spectral advection. Takes as input q in real, p in
            spectral space. Returns advection tendency in spectral space.
            Does not update any state variables. """

        tend = -self.ik*torch.fft.rfftn(u*q,dim=(1,2)) - self.il*torch.fft.rfftn(v*q,dim=(1,2))

        return tend

    def rhsh(self,q,qh,ph,u,v):
        """ Build a tensor of dq/dt in spectral space.
            Does not update any state variables. """

        ## Advection term
        rhsh=-self.advect(q,u,v)

        ## Beta effect
        rhsh[0]+=-self.ik*self.betas[0]*ph[0]
        rhsh[1]+=-self.ik*self.betas[1]*ph[1]
        
        ## Mean flow
        rhsh[0]+=-self.ik*self.U1*qh[0]
        rhsh[1]+=-self.ik*self.U2*qh[1]

        ## Bottom drag
        rhsh[1]+=self.rek*self.kappa2*ph[1]

        if self.parameterization is not None:
            rhsh+=torch.fft.rfftn(self.parameterization(q,ph,self.ik,self.il,self.dx),dim=(1,2))

        return rhsh

    def _step_ab3(self):
        """ Step the system state forward by one timestep. NB we assume that all 4 state variables (q,qh,p,ph)
            are always on the same timestep. So we first use the rhs to update q_i to q_{i+1}, then use this to
            update the remaining 3 quantities to time {i+1} """

        ## First update qh -> qh_{i+1}
        rhsh=self.rhsh(self.q,self.qh,self.ph,self.u,self.v)
        ## If we have no n_{i-1} timestep, we just do forward Euler
        if self.rhs_minus_one==None:
            self.qh=self.qh+rhsh*self.dt
            ## Store rhs as rhs_{i-1}
            self.rhs_minus_one=rhsh
        ## If we have no n_{i-2} timestep, we do AB2
        elif self.rhs_minus_two==None:
            self.qh=self.qh+(0.5*self.dt)*(3*rhsh-self.rhs_minus_one)
            ## Update previous timestep rhs
            self.rhs_minus_two=self.rhs_minus_one
            self.rhs_minus_one=rhsh
        else:
            ## If we have two previous timesteps stored, use AB3
            self.qh=self.qh+(self.dt/12.)*(23*rhsh-16*self.rhs_minus_one+5*self.rhs_minus_two)
            ## Update previous timesteps
            self.rhs_minus_two=self.rhs_minus_one
            self.rhs_minus_one=rhsh

        ## Apply dissipative filter
        self.qh[0]=self.filtr*self.qh[0]
        self.qh[1]=self.filtr*self.qh[1]

        ## Now we have qh_{i+1}, update 3 remaining states
        self.q=torch.fft.irfftn(self.qh,dim=(1,2))
        self.ph=self.invert(self.qh)
        ## Get u, v in spectral space, then ifft to real space
        self.u=torch.fft.irfftn(-self.il*self.ph,dim=(1,2))
        self.v=torch.fft.irfftn(self.ik*self.ph,dim=(1,2))
        self.p=torch.fft.irfftn(self.ph,dim=(1,2))
        ## Update timestep
        self.timestep+=1

        ## Increment diagnostics
        if self.timestep>self.diagnostics_start and (self.timestep % self.diagnostics_freq ==0):
            self._increment_diagnostics()

        return
