import torch
import math
from tqdm import tqdm

class QG_model():
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
        
        self._initialise_q1q2()
        self._initialise_background()
        self._initialise_grid()
        
    def _initialise_background(self):
        
        self.F1 = self.rd**-2 / (1.+self.delta)
        self.F2 = self.delta*self.F1
        self.betas=torch.tensor([self.beta-self.F1*(self.U1-self.U2),self.beta+self.F1*(self.U1-self.U2)])
        
    def _initialise_grid(self):
        """ Set up real-space and spectral-space grids """
        
        self.x,self.y = torch.meshgrid(
        torch.arange(0.5,self.nx,1.)/self.nx*self.L,
        torch.arange(0.5,self.ny,1.)/self.ny*self.W )
        ## physical grid spacing
        self.dx = self.L / self.nx
        self.dy = self.W / self.ny

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
        self.k=self.k.T
        #self.k2=self.k*self.k
        self.l=self.l.T
        self.ik = 1j*self.k
        self.il = 1j*self.l
        
        ## kappa2 represents the wavenumber squared at each gridpoint
        self.kappa2=(self.l**2+self.k**2)
        
        
        ## Evaluate 2x2 matrix determinant for calculation of streamfunction
        self.determinant=self.kappa2*(self.kappa2+self.F1+self.F2)
        ## Set to false value so matrix inversion doesn't throw warnings
        self.determinant[0,0]=1e-19
        
        return
        
    
    def _initialise_q1q2(self):
        self.q=torch.stack((1e-7*torch.rand(self.ny,self.nx) + 1e-6*(torch.ones((self.ny,1)) * torch.rand(1,self.nx) ),torch.zeros(self.nx,self.nx)))
        
        return
    
    def _invert(self,qh):
        """ Invert 2x2 matrix equation to get streamfunction from potential vorticity.
            Also return fft of streamfunction for use in other calculations """
        
        
        ph1=(-(self.kappa2+self.F2)*qh[0]-self.F1*qh[1])/self.determinant
        ph2=(-self.F2*qh[0]-(self.kappa2+self.F1)*qh[1])/self.determinant

        ## Fundamental mode is 0
        ph1[0,0]=0
        ph2[0,0]=0
        
        return torch.stack((ph1,ph2))
    
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
    
    def _advect(self,q,psi):
        """ Arakawa advection scheme of q """
        
        f1 = self.diffx(psi,dx=self.dx)*self.diffy(q,self.dx) - self.diffy(psi,self.dx)*self.diffx(q,self.dx)
        f2 = self.diffy(self.diffx(psi,self.dx)*q,self.dx) - self.diffx(self.diffy(psi,self.dx)*q,self.dx)
        f3 = self.diffx(self.diffy(q,self.dx)*psi,self.dx) - self.diffy(self.diffx(q,self.dx)*psi,self.dx)
        
        f = - (f1 + f2 + f3) / 3
        return f
        
    def rhs(self,q):
        """ Build a tensor of dq/dt """
        
        ## FFT of potential vorticity
        qh=torch.fft.rfftn(q,dim=(1,2))
        
        ## Invert coupling matrix to get streamfunction in Fourier domain
        ph=self._invert(qh)
        ## Invert fft to get streamfunction in real space
        psi=torch.fft.irfftn(ph,dim=(1,2))
        
        #### Spatial derivatives ####
        ## Spatial derivative of streamfunction using Fourier tensor
        dpsi=torch.fft.irfftn(self.ik*ph,dim=(1,2))
        d2psi=torch.fft.irfftn(-self.kappa2*ph,dim=(1,2))
        dq=torch.fft.irfftn(self.ik*qh,dim=(1,2))
        
        rhs=-1*self._advect(q,psi)
        rhs[0]+=(-self.betas[0]*dpsi[0])
        rhs[1]+=(-self.betas[1]*dpsi[1])
        
        rhs[0]+=-dq[0]*self.U1
        rhs[1]+=-dq[1]*self.U2
        
        ## Bottom drag
        rhs[1]+=-self.rek*d2psi[1]
        
        return rhs
        
    def timestep_euler(self,q):
        """ Advance system forward in time one step using forward Euler """
                
        return q+self.rhs(q)*self.dt
    
    def run_ab(self,q,steps,store_snaps=False):
        """ Advance system forward in time using AB3 """
        rhs_n=None
        rhs_n_minus_one=None
        rhs_n_minus_two=None
        snaps=None
        
        if store_snaps:
            ## Initialise empty tensor to store q of shape
            ## [step_index,layer number, nx, ny]
            snaps=torch.empty((steps,2,self.nx,self.nx))
        
        for aa in tqdm(range(steps)):
            ## If we have no n_{i-1} timestep, we just do forward Euler
            if rhs_n_minus_one==None:
                self.q=q+self.rhs(q)*self.dt
                ## Store rhs as rhs_{i-1}
                rhs_n_minus_one=self.rhs(q)
            ## If we have no n_{i-2} timestep, we do AB2
            elif rhs_n_minus_two==None:
                rhs=self.rhs(self.q)
                self.q=self.q+(0.5*self.dt)*(3*rhs-rhs_n_minus_one)
                ## Update previous timestep rhs
                rhs_n_minus_two=rhs_n_minus_one
                rhs_n_minus_one=rhs
            else:
                ## If we have two previous timesteps stored, use AB3
                rhs=self.rhs(self.q)
                self.q=self.q+(self.dt/12.)*(23*rhs-16*rhs_n_minus_one+5*rhs_n_minus_two)
                ## Update previous timesteps
                rhs_n_minus_two=rhs_n_minus_one
                rhs_n_minus_one=rhs
            if store_snaps:
                snaps[aa]=self.q
            
        return snaps
    
    def run_sim(self,steps):
        for aa in tqdm(range(steps)):
            self.q=self.timestep_euler(self.q)
            
        return
    
    
class QG_model_spectral():
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
        
        # the F parameters
        self.F1 = self.rd**-2 / (1.+self.delta)
        self.F2 = self.delta*self.F1


        # initial conditions: (PV anomalies)
        self._initialise_q1q2()
        self._initialise_grid()
        
    def _initialise_grid(self):
        """ Set up real-space and spectral-space grids """
        
        self.x,self.y = torch.meshgrid(
        torch.arange(0.5,self.nx,1.)/self.nx*self.L,
        torch.arange(0.5,self.ny,1.)/self.ny*self.W )
        ## physical grid spacing
        self.dx = self.L / self.nx
        self.dy = self.W / self.ny

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
        self.k=self.k.T
        self.l=self.l.T
        self.ik = 1j*self.k
        self.il = 1j*self.l
        
        ## kappa2 represents the wavenumber squared at each gridpoint
        self.kappa2=(self.l**2+self.k**2)
        
        
        ## Evaluate 2x2 matrix determinant for calculation of streamfunction
        self.determinant=self.kappa2*(self.kappa2+self.F1+self.F2)
        ## Set to false value so matrix inversion doesn't throw warnings
        self.determinant[0,0]=1e-19
        
        return
        
    
    def _initialise_q1q2(self):
        self.q=torch.stack((1e-7*torch.rand(self.ny,self.nx) + 1e-6*(torch.ones((self.ny,1)) * torch.rand(1,self.nx) ),torch.zeros(self.nx,self.nx)))
        return
    
    
    def _advection(self):
    
    
    
    
    
    
    
        return
    
    
    def _invert(self,qh):
        """ Invert 2x2 matrix equation to get streamfunction from potential vorticity.
            Also return fft of streamfunction for use in other calculations """
        
        
        ph1=(-(self.kappa2+self.F2)*qh[0]-self.F1*qh[1])/self.determinant
        ph2=(-self.F2*qh[0]-(self.kappa2+self.F1)*qh[1])/self.determinant

        ## Fundamental mode is 0
        ph1[0,0]=0
        ph2[0,0]=0
        
        return torch.stack((ph1,ph2))
    
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
    
    def _advect(self,q,psi):
        """ Arakawa advection scheme of q """
        
        f1 = self.diffx(psi,dx=self.dx)*self.diffy(q,self.dx) - self.diffy(psi,self.dx)*self.diffx(q,self.dx)
        f2 = self.diffy(self.diffx(psi,self.dx)*q,self.dx) - self.diffx(self.diffy(psi,self.dx)*q,self.dx)
        f3 = self.diffx(self.diffy(q,self.dx)*psi,self.dx) - self.diffy(self.diffx(q,self.dx)*psi,self.dx)
        
        f = - (f1 + f2 + f3) / 3
        return f
        
    def rhs(self,q):
        """ Build a tensor of dq/dt """
        
        ## FFT of potential vorticity
        qh=torch.fft.rfftn(q,dim=(1,2))
        ## Spatial derivative of q field
        
        ## Invert coupling matrix to get streamfunction in Fourier domain
        ph=self._invert(qh)
        ## Invert fft to get streamfunction in real space
        psi=torch.fft.irfftn(ph,dim=(1,2))
        
        #### Spatial derivatives ####
        ## Spatial derivative of streamfunction using Fourier tensor
        dpsi=torch.fft.irfftn(1j*torch.sqrt(self.kappa2)*ph,dim=(1,2))
        d2psi=torch.fft.irfftn(self.kappa2*ph,dim=(1,2))
        dq=torch.fft.irfftn(1j*torch.sqrt(self.kappa2)*qh,dim=(1,2))
        
        ## Background quantities
        beta_upper=self.beta-self.F1*(self.U1-self.U2)
        beta_lower=self.beta+self.F2*(self.U1-self.U2)
        
        rhs=-1*self._advect(q,psi)
        rhs[0]+=(-beta_upper*dpsi[0])
        rhs[1]+=(-beta_lower*dpsi[1])
        
        rhs[0]+=-dq[0]*self.U1
        rhs[1]+=-dq[1]*self.U2
        
        ## Bottom drag
        rhs[1]+=-self.rek*d2psi[1]
        
        return rhs
        
    def timestep(self,q):
        """ Advance system forward in time one step """
        
        """ 1. Need a function for the RHS
            2. Pass RHS to some numerical solver
            
        """
                
        return q+self.rhs(q)*self.dt