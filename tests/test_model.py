import pytest
import torch
import torch_qg.model as torch_model
import torch_qg.parameterizations as torch_param

def test_advection1():
    """ Ensure that the advected field produces a field with zero integrated vorticity """
    nx=512

    ## Produce random fields
    psi=torch.stack((torch.rand(nx,nx,dtype=torch.float64),torch.rand(nx,nx,dtype=torch.float64)))
    q=torch.stack((torch.rand(nx,nx,dtype=torch.float64),torch.rand(nx,nx,dtype=torch.float64)))

    ## Advect using Arakawa scheme
    qg_model=torch_model.ArakawaModel(nx=nx)
    advected=qg_model.advect(q,psi)

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
    qg_model=torch_model.ArakawaModel(nx=nx)
    advected=qg_model.advect(q,psi)

    ## Ensure all values are exactly 0
    assert advected.sum()==0.
    
def test_sim_Arakawa():
    for nx in ([32,64,128,256]):
        qg_model=torch_model.ArakawaModel(nx=nx)
        qg_model.run_sim(1000)

def test_sim_param_Arakawa():
    for nx in ([32,64,128,256]):
        qg_model=torch_model.ArakawaModel(nx=nx,parameterization=torch_param.Smagorinsky())
        qg_model.run_sim(1000)

def test_sim_PseudoSpectral():
    for nx in ([32,64,128,256]):
        qg_model=torch_model.PseudoSpectralModel(nx=nx)
        qg_model.run_sim(1000)