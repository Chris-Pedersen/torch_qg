import pytest
import torch
import torch_qg.model as torch_model

def test_advection1():
    """ Ensure that the advected field produces a field with zero integrated vorticity """
    nx=512

    ## Produce random fields
    psi=torch.stack((torch.rand(nx,nx,dtype=torch.float64),torch.rand(nx,nx,dtype=torch.float64)))
    q=torch.stack((torch.rand(nx,nx,dtype=torch.float64),torch.rand(nx,nx,dtype=torch.float64)))

    ## Advect using Arakawa scheme
    qg_model=torch_model.QG_model(nx=nx)
    advected=qg_model._advect(q,psi)

    assert advected.sum().abs() < 1e-10

def test_advection2():
    """ Ensure that the Jacobian produces zeros for trig functions where we know the analytical solution """
    nx=64

    dx=(2*torch.pi)/nx

    ## Produce sin fields
    x=torch.linspace(0,2*torch.pi-dx,nx,dtype=torch.float64)
    y=torch.linspace(0,2*torch.pi-dx,nx,dtype=torch.float64)

    xx,yy=torch.meshgrid(x,y)

    psi=torch.stack((torch.sin(xx),torch.sin(xx)))
    q=torch.stack((-torch.sin(xx),-torch.sin(xx)))

    ## Advect using Arakawa scheme
    qg_model=torch_model.QG_model(nx=nx)
    advected=qg_model._advect(q,psi)

    ## Ensure all values are exactly 0
    assert advected.sum()==0.
    

