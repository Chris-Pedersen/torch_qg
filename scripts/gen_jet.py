import matplotlib.pyplot as plt
import xarray as xr
import torch_qg.model as torch_model
import torch_qg.parameterizations as torch_param
import torch_qg.util as util
import torch

import copy
import pyqg

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import cmocean

import math
import numpy as np
import time 


## Generate a load of sims we want to look at
save_path="/scratch/cp3759/pyqg_data/sims/torchqg_tests/grid_sims/jet/"
grid_sizes=[48,64,128,256]
jet_config={'rek': 7e-08, 'delta': 0.1, 'beta': 1e-11}

for grid_size in grid_sizes:
    model=torch_model.PseudoSpectralModel(nx=grid_size,dt=3600,**jet_config)
    ds=model.run_sim(int(8e4))
    save_string=save_path+"ps_exp_"+str(grid_size)+"_jet.nc"
    print("Saving as",save_string)
    ds.to_netcdf(save_string)

    model=torch_model.PseudoSpectralModel(nx=grid_size,dt=3600,parameterization=torch_param.Smagorinsky(),dealias=True,**jet_config)
    ds=model.run_sim(int(8e4))
    save_string=save_path+"ps_deal_s_"+str(grid_size)+"_jet.nc"
    print("Saving as",save_string)
    ds.to_netcdf(save_string)

    model=torch_model.PseudoSpectralModel(nx=grid_size,dt=3600,dealias=True,**jet_config)
    ds=model.run_sim(int(8e4))
    save_string=save_path+"ps_deal_"+str(grid_size)+"_jet.nc"
    print("Saving as",save_string)
    ds.to_netcdf(save_string)

    model=torch_model.ArakawaModel(nx=grid_size,dt=3600,**jet_config)
    ds=model.run_sim(int(8e4))
    save_string=save_path+"arakawa_"+str(grid_size)+"_jet.nc"
    print("Saving as",save_string)
    ds.to_netcdf(save_string)

    model=torch_model.ArakawaModel(nx=grid_size,dt=3600,parameterization=torch_param.Smagorinsky(),**jet_config)
    ds=model.run_sim(int(8e4))
    save_string=save_path+"arakawa_s_"+str(grid_size)+"_jet.nc"
    print("Saving as",save_string)
    ds.to_netcdf(save_string)
