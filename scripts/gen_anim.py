import xarray as xr
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import cmocean
import torch_qg.model as torch_model
import torch_qg.parameterizations as torch_param

snap_index=5000

## Load in saved dataset
#ds=xr.load_dataset("/scratch/cp3759/pyqg_data/sims/rollouts/rollout_BScat_252.nc")
ds_hr=xr.load_dataset("/scratch/cp3759/pyqg_data/sims/animation_sims/highres_1k.nc")

#q=torch.tensor(ds.q[snap_index].to_numpy())
#psi=torch.tensor(ds.p[snap_index].to_numpy())

#dqdt=torch.tensor(ds.dqdt[snap_index].to_numpy())

q_hr=torch.tensor(ds_hr.q[500].to_numpy())
psi_hr=torch.tensor(ds_hr.p[500].to_numpy())

#dqdt_hr=torch.tensor(ds_hr.dqdt[501].to_numpy())


qg_model=torch_model.QG_model(nx=256,parameterization=torch_param.Smagorinsky())
q_snaps=qg_model.run_ab(q_hr,10000,store_snaps=True)

class SimAnimation():
    def __init__(self,q_tensor,fps=10,nSteps=1000,normalise=True):
        self.q=q_tensor
        self.fps = fps
        self.nSteps = nSteps
    
    def animate_func(self,i):
        if i % self.fps == 0:
            print( '.', end ='' )
    
        ## Set image and colorbar for each panel
        image=self.q[i,0]
        self.ax1.set_array(image)
        #self.ax1.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.q[i,1]
        self.ax2.set_array(image)
        #self.ax2.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        return 
    
    def animate(self):
        fig, (axs1,axs2) = plt.subplots(1, 2,figsize=(14,6))
        self.ax1=axs1.imshow(self.q[0,0], cmap=cmocean.cm.balance)
        fig.colorbar(self.ax1, ax=axs1)
        axs1.set_xticks([]); axs1.set_yticks([])
        axs1.set_title("Upper")

        self.ax2=axs2.imshow(self.q[0,1], cmap=cmocean.cm.balance)
        fig.colorbar(self.ax1, ax=axs2)
        axs2.set_xticks([]); axs2.set_yticks([])
        axs2.set_title("Lower")
        
        anim = animation.FuncAnimation(
                                       fig, 
                                       self.animate_func, 
                                       frames = self.nSteps,
                                       interval = 1000 / self.fps, # in ms
                                       )
        # saving to m4 using ffmpeg writer 
        writervideo = animation.FFMpegWriter(fps=60) 
        anim.save('torchqg_smag.mp4', writer=writervideo) 
        plt.close()
        
        return HTML(anim.to_html5_video())
    
simanim=SimAnimation(q_snaps[0],fps=200,nSteps=10000)
simanim.animate()
