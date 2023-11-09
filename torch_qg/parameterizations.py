import torch


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
        uh=-ik*ph
        vh=-il*ph
        Sxx = torch.irfftn(uh,dim=(1,2))
        Syy = torch.irfftn(vu,dim=(1,2))
        Sxy = 0.5 * m.irfftn(uh * il + vh * ik)
        nu = (self.constant * dx)**2 * torch.sqrt(2 * (Sxx**2 + Syy**2 + 2 * Sxy**2))
        nu_Sxxh = torch.rfftn(nu * Sxx,dim=(1,2))
        nu_Sxyh = torch.rfftn(nu * Sxy,dim=(1,2))
        nu_Syyh = torch.rfftn(nu * Syy,dim=(1,2))
        du = 2 * (torch.irfftn(nu_Sxxh * ik,dim=(1,2)) + torch.irfftn(nu_Sxyh * il,dim=(1,2)))
        dv = 2 * (torch.irfftn(nu_Sxyh * ik,dim=(1,2)) + torch.irfftn(nu_Syyh * il,dim=(1,2)))
        ## Take curl to convert u, v forcing to potential vorticity forcing
        dq = -torch.irfft(il*torch.rfft(du))+torch.irfft(ik*torch.rfft(dv))
        return dq

    
        def __call__(self, m, just_viscosity=False):
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
        uh = m.fft(m.u)
        vh = m.fft(m.v)
        Sxx = m.ifft(uh * m.ik)
        Syy = m.ifft(vh * m.il)
        Sxy = 0.5 * m.ifft(uh * m.il + vh * m.ik)
        nu = (self.constant * m.dx)**2 * np.sqrt(2 * (Sxx**2 + Syy**2 + 2 * Sxy**2))
        if just_viscosity:
            return nu
        nu_Sxxh = m.fft(nu * Sxx)
        nu_Sxyh = m.fft(nu * Sxy)
        nu_Syyh = m.fft(nu * Syy)
        du = 2 * (m.ifft(nu_Sxxh * m.ik) + m.ifft(nu_Sxyh * m.il))
        dv = 2 * (m.ifft(nu_Sxyh * m.ik) + m.ifft(nu_Syyh * m.il))
        return du, dv