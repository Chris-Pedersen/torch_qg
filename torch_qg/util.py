import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import cmocean
import numpy as np
from matplotlib.pyplot import cm


## Second to year conversion for animation time ticker
YEAR = 24*60*60*360.


def plot_many_spectra(ds_list,string_list,suptitle=None,savename=None):
    """ For a list of datasets, plot common spectral quantities. string_list will
        set the label for each line """

    col = cm.rainbow(np.linspace(0, 1, len(ds_list)))
    fig, axs = plt.subplots(2, 2,figsize=(10,5))
    axs[0,0].set_title("Spectral energy transfer")
    axs[0,1].set_title("Enstrophy spectrum")

    axs[1,0].set_title("Kinetic energy spectrum")
    axs[1,1].set_title("Kinetic energy over time")
    
    for aa,ds in enumerate(ds_list):
        ## Spectral energy transfer
        axs[0,0].semilogx(ds.k1d,ds.SPE[-1],color=col[aa])

        ## Enstrophy spectra
        axs[0,1].loglog(ds.k1d,ds.Enspec[-1,0],color=col[aa])
        axs[0,1].loglog(ds.k1d,ds.Enspec[-1,1],color=col[aa],linestyle="dashed")
        axs[0,1].set_ylim(5e-10,6e-6)

        ## Kinetic energy spectra
        axs[1,0].loglog(ds.k1d,ds.KEspec[-1,0],color=col[aa],label=string_list[aa])
        axs[1,0].loglog(ds.k1d,ds.KEspec[-1,1],color=col[aa],linestyle="dashed")
        axs[1,0].legend()
        axs[1,0].set_ylim(1e-4,5e2)
        axs[1,0].set_xlabel("k")

        ## Kinetic energy over time
        axs[1,1].plot(ds.time,ds.KE,color=col[aa])
        axs[1,1].set_xlabel("time (seconds)")
        
    if suptitle is not None:
        fig.suptitle(suptitle)

    plt.tight_layout()
    if savename is not None:
        plt.savefig("%s.png" % savename)

class SimAnimation():
    """ Generate animation of a given dataset. Will just plot the upper and
        lower q fields for every snapshot in the dataset """
    def __init__(self,ds,fps=10,save_string=None):
        self.ds=ds
        self.nSteps=len(self.ds.q)
        self.fps=fps
        self.save_string=save_string
    
    def animate_func(self,i):
        if i % self.fps == 0:
            print( '.', end ='' )
    
        ## Set image and colorbar for each panel
        image=self.ds.q[i,0]
        self.ax1.set_array(image)
        self.ax1.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.ds.q[i,1]
        self.ax2.set_array(image)
        self.ax2.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        self.time_text.set_text("time=%.2f (years)" % (self.ds.time[i].to_numpy()/YEAR))
        
        return 
    
    def animate(self):
        fig, (axs1,axs2) = plt.subplots(1, 2,figsize=(14,6))
        self.ax1=axs1.imshow(self.ds.q[0,0], cmap=cmocean.cm.balance)
        fig.colorbar(self.ax1, ax=axs1)
        axs1.set_xticks([]); axs1.set_yticks([])
        axs1.set_title("Upper")

        self.ax2=axs2.imshow(self.ds.q[0,1], cmap=cmocean.cm.balance)
        fig.colorbar(self.ax1, ax=axs2)
        axs2.set_xticks([]); axs2.set_yticks([])
        axs2.set_title("Lower")
        self.time_text=axs2.text(-20,-20,"")
        
        anim = animation.FuncAnimation(
                                       fig, 
                                       self.animate_func, 
                                       frames = self.nSteps,
                                       interval = 1000 / self.fps, # in ms
                                       )
        # saving to m4 using ffmpeg writer 
        if self.save_string is not None:
            writervideo = animation.FFMpegWriter(fps=60)
            anim.save('%s.mp4' % self.save_string, writer=writervideo) 
        plt.close()
        
        return HTML(anim.to_html5_video())
