import torch
import torch_qg.util as util

class Smagorinsky():
    r"""Velocity parameterization from `Smagorinsky 1963`_.

    This parameterization assumes that due to subgrid stress, there is an
    effective eddy viscosity

    .. math:: \nu = (C_S \Delta)^2 \sqrt{2(S_{x,x}^2 + S_{y,y}^2 + 2S_{x,y}^2)}

    which leads to updated velocity tendencies :math:`\Pi_{i}, i \in \{1,2\}`
    corresponding to :math:`x` and :math:`y` respectively (equation is the same
    in each layer):

    .. math:: \Pi_{i} = 2 \partial_i(\nu S_{i,i}) + \partial_{2-i}(\nu S_{i,2-i})

    where :math:`C_S` is a tunable Smagorinsky constant, :math:`\Delta` is the
    grid spacing, and

    .. math:: S_{i,j} = \frac{1}{2}(\partial_i \mathbf{u}_j
                                  + \partial_j \mathbf{u}_i)

    .. _Smagorinsky 1963: https://doi.org/10.1175/1520-0493(1963)091%3C0099:GCEWTP%3E2.3.CO;2
    """

    def __init__(self, constant=0.1):
        r"""
        Parameters
        ----------
        constant : number
            Smagorinsky constant :math:`C_S`. Defaults to 0.1.
        """

        self.constant = constant

    def __call__(self, q, ph, ik, il, dx):
        r"""
        Parameters
        ----------
        m : Model
            The model for which we are evaluating the parameterization.
        just_viscosity : bool
            Whether to just return the eddy viscosity (e.g. for use in a
            different parameterization which assumes a Smagorinsky dissipation
            model). Defaults to false.
        """
        ## Get u, v in spectral space
        uh=-il*ph
        vh=ik*ph
        Sxx = torch.fft.irfftn(uh*ik,dim=(1,2))
        Syy = torch.fft.irfftn(vh*il,dim=(1,2))
        Sxy = 0.5 * torch.fft.irfftn(uh * il + vh * ik,dim=(1,2))
        nu = (self.constant * dx)**2 * torch.sqrt(2 * (Sxx**2 + Syy**2 + 2 * Sxy**2))
        nu_Sxxh = torch.fft.rfftn(nu * Sxx,dim=(1,2))
        nu_Sxyh = torch.fft.rfftn(nu * Sxy,dim=(1,2))
        nu_Syyh = torch.fft.rfftn(nu * Syy,dim=(1,2))
        du = 2 * (torch.fft.irfftn(nu_Sxxh * ik,dim=(1,2)) + torch.fft.irfftn(nu_Sxyh * il,dim=(1,2)))
        dv = 2 * (torch.fft.irfftn(nu_Sxyh * ik,dim=(1,2)) + torch.fft.irfftn(nu_Syyh * il,dim=(1,2)))
        ## Take curl to convert u, v forcing to potential vorticity forcing
        dq = -torch.fft.irfftn(il*torch.fft.rfftn(du,dim=(1,2)),dim=(1,2))+torch.fft.irfftn(ik*torch.fft.rfftn(dv,dim=(1,2)),dim=(1,2))
        return dq

class CNNParameterization():
    """ pyqg subgrid parameterisation for the potential vorticity"""
    
    def __init__(self,model,normalise=True,cache_forcing=False):
        """ Initialise with a list of torch models, one for each layer """
        self.model=model
        self.model.eval() ## Ensure we are in eval
        self.type="CNN"
        self.normalise=normalise
        self.cache_forcing=cache_forcing
        self.cached_forcing=None
        self.coeff=1

    def get_cached_forcing(self):
        return self.cached_forcing

    def __call__(self,*args):
        """ 
            Inputs:
                - Currently the model classes pass a set of args:
                        q,ph,self.ik,self.il,self.dx
                  which are needed for the Smagorinsky paramterisation
                  We only need q for the ML parameterisation, so we just
                  take the full list of args, assuming that args[0]=q
            Outputs:
                - forcing: a numpy array of shape (nz, nx, ny) with the subgrid
                           forcing values """
        
        q=args[0].float()
        ## Map from physical to normalised space using the factors used to train the network
        ## Normalise each field individually, then cat arrays back to shape appropriate for a torch model
        x_upper = util.normalise_field(q[0],self.model.config["q_mean_upper"],self.model.config["q_std_upper"])
        x_lower = util.normalise_field(q[1],self.model.config["q_mean_lower"],self.model.config["q_std_lower"])
        q = torch.stack((x_upper,x_lower),dim=0).unsqueeze(0)

        ## Pass the normalised fields through our network
        q = self.model(q)
        q=q.clone().detach()

        ## Map back from normalised space to physical units
        s_upper=util.denormalise_field(q[:,0,:,:],self.model.config["s_mean_upper"],self.model.config["s_std_upper"])
        s_lower=util.denormalise_field(q[:,1,:,:],self.model.config["s_mean_lower"],self.model.config["s_std_lower"])

        ## Reshape to match pyqg dimensions, and cast to numpy array
        s=torch.cat((s_upper,s_lower))

        if self.normalise:
            means=torch.mean(s,axis=(1,2))
            s[0]=s[0]-means[0]
            s[1]=s[1]-means[1]

        ## Rescale by scaling coefficient
        s=s*self.coeff

        if self.cache_forcing:
            self.cached_forcing=s

        return s
