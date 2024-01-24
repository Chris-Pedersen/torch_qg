import torch_qg.model as torch_model
import torch_qg.parameterizations as torch_param
import matplotlib.pyplot as plt
import torch
import numpy as np
import xarray as xr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_to', type=str)
parser.add_argument('--run_number', type=int, default=0)
args, extra = parser.parse_known_args()

## Add run number to save file name
save_file=args.save_to+"torchqg_%d.nc" % args.run_number

print(save_file)

psh=torch_model.PseudoSpectralModel(nx=256,dt=3600,dealias=True,parameterization=torch_param.Smagorinsky())
ps=torch_model.PseudoSpectralModel(nx=64,dt=3600,dealias=True)

def run_forcing(hr_model,lr_model,steps,interval=1000):    
    ds=[]
    for aa in range(steps):
        hr_model._step_ab3()
        ## Check CFL every 1k timesteps
        if aa % interval==0:
            cfl=hr_model.calc_cfl()
            assert cfl<1., "CFL condition violated"
            ds.append(hr_model.forcing_dataset(lr_model))
            
    ds=xr.concat(ds,dim="time")
    return ds

ds=run_forcing(psh,ps,int(8e4))
ds.to_netcdf(save_file)
