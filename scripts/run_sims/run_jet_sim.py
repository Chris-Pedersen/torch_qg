import torch_qg.model as torch_model
import torch_qg.parameterizations as torch_param
import matplotlib.pyplot as plt
import torch
import numpy as np
import xarray as xr
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--save_to', type=str)
parser.add_argument('--run_number', type=int, default=0)
parser.add_argument('--increment', type=int, default=0)
parser.add_argument('--rollout', type=int, default=0)
args, extra = parser.parse_known_args()

## Add run number to save file name
save_file=args.save_to+"torchqg_jet_%d.nc" % args.run_number

print(save_file)

jet_config={'rek': 7e-08, 'delta': 0.1, 'beta': 1e-11}

psh=torch_model.PseudoSpectralModel(nx=256,dt=3600,dealias=True,parameterization=torch_param.Smagorinsky(),**jet_config)
ps=torch_model.PseudoSpectralModel(nx=64,dt=3600,dealias=True,**jet_config)

def run_forcing(hr_model,lr_model,steps,sampling_freq=1000,increment=2,rollout=4):    
    ds=[]
    for aa in range(steps):
        if increment == 0:
            ## If increment == 0, just sample at each sampling freq
            should_sample = (aa % sampling_freq == 0)
        else:
            ## If sampling at increments, identify indices to sample at
            should_sample = (aa % sampling_freq == 0) or ((aa % sampling_freq % increment == 0) and aa % sampling_freq <= rollout*increment)
        ## Don't sample from t=0 (we have no subgrid forcing for very first snapshot)
        if aa<500:
            should_sample=False
            
        hr_model._step_ab3()
        ## Check CFL every 1k timesteps
        if should_sample:
            cfl=hr_model.calc_cfl()
            assert cfl<1., "CFL condition violated"
            ds.append(hr_model.forcing_dataset(lr_model))
            
    ds=xr.concat(ds,dim="time")
    params={}
    params["sampling_freq"]=sampling_freq
    params["increment"]=increment
    params["rollout"]=rollout
    return ds.assign_attrs(rollout_config=json.dumps(params))

ds=run_forcing(psh,ps,int(2e5),increment=args.increment,rollout=args.rollout)
ds.to_netcdf(save_file)
