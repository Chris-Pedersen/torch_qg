import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import cmocean
import numpy as np
from matplotlib.pyplot import cm
import torch

############################### Transforms ###############################

def normalise_field(field, mean, std):
    """ Map a field in the form of a torch tensor to a normalised space """
    field = field.clone()
    field.sub_(mean).div_(std)
    return field

def denormalise_field(field, mean, std):
    """ Take a normalised field (torch tensor), denormalise it """
    field = field.clone()
    field.mul_(std).add_(mean)
    return field

############################### Smoothing ################################
def spectral_smoothing(hires_field,hr_nx,lr_model):
    """
    hires_field: input field we want to downsample (in spectral space,
                 must be 2 layers
    hr_nx:       nx from high res model
    lr_model:    low res model (has to be pseudospectral for now)
    We are just using "operator 1" from Ross et al for now.
    
    returns downsampled fields
    """
    hires_var=hires_field
    filtr = lr_model.filtr
    keep = lr_model.qh.shape[1]//2
    downed=torch.hstack((
                hires_var[:,:keep,:keep+1],
                hires_var[:,-keep:,:keep+1]
            )) * filtr / (hr_nx / lr_model.nx)**2
    
    return downed

############################## Plot helpers ##############################
## Second to year conversion for animation time ticker
YEAR = 24*60*60*360.

def plot_fields(ds,suptitle=None):
    fig, axs = plt.subplots(2, 4,figsize=(14,6))
    axs[0,0].set_title("potential vorticity")
    axs[0,0].imshow(ds.q[-1,0].to_numpy())
    axs[1,0].imshow(ds.q[-1,1].to_numpy())
    
    axs[0,1].set_title("psi")
    axs[0,1].imshow(ds.p[-1,0].to_numpy())
    axs[1,1].imshow(ds.p[-1,1].to_numpy())
    
    axs[0,2].set_title("u")
    axs[0,2].imshow(ds.u[-1,0].to_numpy())
    axs[1,2].imshow(ds.u[-1,1].to_numpy())
    
    axs[0,3].set_title("v")
    axs[0,3].imshow(ds.v[-1,0].to_numpy())
    axs[1,3].imshow(ds.v[-1,1].to_numpy())
    
def PDF_histogram(x, xmin=None, xmax=None, Nbins=30):
    """
    x is 1D numpy array with data
    How to use:
        first apply without arguments
        Then adjust xmin, xmax, Nbins
    """    
    N = x.shape[0]

    mean = x.mean()
    sigma = x.std()
    
    if xmin is None:
        xmin = mean-4*sigma
    if xmax is None:
        xmax = mean+4*sigma

    bandwidth = (xmax - xmin) / Nbins
    
    hist, bin_edges = np.histogram(x, range=(xmin,xmax), bins = Nbins)

    # hist / N is probability to go into bin
    # probability / bandwidth = probability density
    density = hist / N / bandwidth

    # we assign one value to each bin
    points = (bin_edges[0:-1] + bin_edges[1:]) * 0.5

    #print(f"Number of bins = {Nbins}, over the interval ({xmin},{xmax}), with bandwidth = {bandwidth}")
    #print(f"This interval covers {sum(hist)/N} of total probability")
    
    return points, density

def plot_many_spectra(ds_list,string_list,suptitle=None,savename=None):
    """ For a list of datasets, plot common spectral quantities. string_list will
        set the label for each line """

    col = cm.rainbow(np.linspace(0, 1, len(ds_list)))
    fig, axs = plt.subplots(2, 3,figsize=(10,5))
    axs[0,0].set_title("Spectral energy transfer")
    axs[0,1].set_title("Enstrophy spectrum")

    axs[1,0].set_title("Kinetic energy spectrum")
    axs[1,1].set_title("Kinetic energy over time")
    
    axs[0,2].set_title("u velocity pdf, upper layer")
    axs[1,2].set_title("v velocity pdf, upper layer")
    
    for aa,ds in enumerate(ds_list):
        ## Spectral energy transfer
        axs[0,0].semilogx(ds.k1d,ds.SPE[-1],color=col[aa])

        ## Enstrophy spectra
        axs[0,1].loglog(ds.k1d,ds.Enspec[-1,0],color=col[aa])
        axs[0,1].loglog(ds.k1d,ds.Enspec[-1,1],color=col[aa],linestyle="dashed")
        axs[0,1].set_ylim(5e-10,6e-6)

        ## Kinetic energy spectra
        axs[1,0].loglog(ds.k1d,ds.KEspec[-1,0],color=col[aa])
        axs[1,0].loglog(ds.k1d,ds.KEspec[-1,1],color=col[aa],linestyle="dashed")
        axs[1,0].set_ylim(1e-4,5e2)
        axs[1,0].set_xlabel("k")

        ## Kinetic energy over time
        axs[1,1].plot(ds.time,ds.KE,color=col[aa])
        axs[1,1].set_xlabel("time (seconds)")
        
        ux,uy=PDF_histogram(ds.u[-15:,0,:,:].values.flatten())
        axs[0,2].semilogy(ux,uy,color=col[aa],label=string_list[aa])
        axs[0,2].legend()
        
        vx,vy=PDF_histogram(ds.v[-15:,0,:,:].values.flatten())
        axs[1,2].semilogy(vx,vy,color=col[aa])
        
    if suptitle is not None:
        fig.suptitle(suptitle)

    plt.tight_layout()
    if savename is not None:
        plt.savefig("%s.png" % savename)
        
def KE(ds_test):
    return (ds_test.u**2 + ds_test.v**2) * 0.5
        
def get_ke_time(ds_test):
    ke=KE(ds_test)
    ke_array=[]
    for snaps in ke:
        ke_array.append(ds_test.attrs['pyqg:L']*np.sum(snaps.data)/(ds_test.attrs['pyqg:nx'])**2)
    return ke_array

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
