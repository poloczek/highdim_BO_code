from GPy.kern.src.kern import Kern
from GPy.core.parameterization import Param
import numpy as np
from scipy.spatial import distance_matrix
from kernel_inputs import InputY, InputX, InputPsi
from GPy.util.linalg import tdot
from GPy import util

class CustomMatern52(Kern):
    def __init__(self, input_dim, input_type, variance=1., lengthscale=1., active_dims=None):
        super(CustomMatern52, self).__init__(input_dim, active_dims, 'matern52')
        self.variance = Param('variance', variance)
        self.lengthscale = Param('lengthscale', lengthscale)
        self.link_parameters(self.variance, self.lengthscale)
        assert isinstance(input_type, (InputY, InputX, InputPsi)), "The type of input_object is not supported"
        self.input_type=input_type

    def parameters_changed(self):
        # nothing todo here
        pass

    def scaled_dist(self, X, X2=None):
        X = self.input_type.evaluate(X)
        if X2 is None:
            Xsq = np.sum(np.square(X),1)
            r2 = -2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
            util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)/self.lengthscale
        else:
            #X2, = self._slice_X(X2)
            X2 = self.input_type.evaluate(X2)
            X1sq = np.sum(np.square(X),1)
            X2sq = np.sum(np.square(X2),1)
            r2 = -2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:])
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)/self.lengthscale

    def K(self, X, X2=None):
        r = self.scaled_dist(X, X2)
        return self.variance*(1+np.sqrt(5.)*r+5./3*r**2)*np.exp(-np.sqrt(5.)*r)

    def Kdiag(self, X):
        return self.variance * np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2=None):
        r = self.scaled_dist(X, X2)
        dvar = (1+np.sqrt(5.)*r+5./3*r**2)*np.exp(-np.sqrt(5.)*r)
        dl = self.variance*(10./3*r -5.*r -5.*np.sqrt(5.)/3*r**2)*np.exp(-np.sqrt(5.)*r)*(-r/self.lengthscale)
        self.variance.gradient = np.sum(dvar * dL_dK)
        self.lengthscale.gradient = np.sum(dl * dL_dK)
