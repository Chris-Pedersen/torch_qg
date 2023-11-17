import numpy as np
import math
import copy

class Diagnostics():
    """ Include everything related to spectral diagnostics in this class """
    def _spectral_grid(self):
        """ Set up ispec grid """

        ll_max = np.abs(self.ll).max()
        kk_max = np.abs(self.kk).max()

        kmax = np.minimum(ll_max, kk_max)
        self.dkr = np.sqrt(self.dk**2 + self.dl**2)
        self.ispec_k=np.arange(0, kmax, self.dkr)

        return

    def get_KE_ispec(self):
        """ From current state variables, calculate isotropically averaged KE spectra
            Do this for both upper and lower layers at once """

        phr = np.zeros((2,self.kr.size()[0]))
        KEspec=(self.kappa2*np.abs(self.psih)**2/self.M**2)
        kespec=copy.copy(KEspec)

        ## Account for complex conjugate
        kespec[:,:,0] /= 2
        kespec[:,:,-1] /= 2

        ## Loop over wavenumbers. Average all modes within a given |k| range
        for i in range(self.kr.size()[0]):
            if i == self.kr.size()[0]-1:
                fkr = (self.kappa>=self.ispec_k[i]) & (self.kappa<=self.ispec_k[i]+self.dkr)
            else:
                fkr = (self.kappa>=self.ispec_k[i]) & (self.kappa<self.ispec_k[i+1])
            phr[:,i] = kespec[:,fkr].mean(axis=-1) * (self.kr[i]+self.dkr/2) * math.pi / (self.dk * self.dl)

            phr[:,i] *= 2 # include full circle

        return phr
