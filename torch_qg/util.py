import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

class SimAnimation():
    def __init__(self,qinit,kernel,fps=10,nSteps=1000,normalise=True):
        self.q=qinit
        self.kernel=kernel
        self.fps = fps
        self.nSteps = nSteps

    def _push_forward(self):
        """ Update q by one timestep """

        self.q=self.kernel.timestep(self.q)

        return

    def animate_func(self,i):
        if i % self.fps == 0:
            print( '.', end ='' )

        ## Set image and colorbar for each panel
        image=self.q[0]
        self.ax1.set_array(image)
        #self.ax1.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))

        image=self.q[1]
        self.ax2.set_array(image)
        #self.ax2.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        self._push_forward()

        return

    def animate(self):
        fig, (axs1,axs2) = plt.subplots(1, 2,figsize=(14,6))
        self.ax1=axs1.imshow(self.q[0], cmap=cmocean.cm.balance)
        fig.colorbar(self.ax1, ax=axs1)
        axs1.set_xticks([]); axs1.set_yticks([])
        axs1.set_title("Upper")

        self.ax2=axs2.imshow(self.q[1], cmap=cmocean.cm.balance)
        fig.colorbar(self.ax1, ax=axs2)
        axs2.set_xticks([]); axs2.set_yticks([])
        axs2.set_title("Lower")

        anim = animation.FuncAnimation(
                                       fig,
                                       self.animate_func,
                                       frames = self.nSteps,
                                       interval = 1000 / self.fps, # in ms
                                       )
        plt.close()

        return HTML(anim.to_html5_video())
