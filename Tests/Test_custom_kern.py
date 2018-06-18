import unittest
import numpy as np
from custom_kern import CustomMatern52
from kernel_inputs import InputY, InputX, InputPsi
from GPy.kern.src.stationary import Matern52
from scipy.spatial import distance_matrix
from GPy.util.linalg import tdot
from GPy import util

class CustomMatern52TestCase(unittest.TestCase):
    def dist_from_original_matern(self, X, X2=None, length=1.):
        if X2 is None:
            X2 = X
        dist = distance_matrix(X, X2)
        r = dist / length
        return r

    def test_for_kern_value(self):
        length=1.5
        var=2
        A = np.array([[0.4, 0.7, 0.3],
                      [0.1, 0.1, 0.9]])
        obj=InputX(A)
        org_kern = Matern52(2, lengthscale=length, variance=var)
        cst_kern = CustomMatern52(2,obj, lengthscale=length, variance=var)
        y=np.array([[0.4, 0.9],
                    [-0.6, 0.52],
                    [0.82, -0.67]])
        org_r = self.dist_from_original_matern(obj.evaluate(y), length=length)
        org_k=org_kern.K_of_r(org_r)
        cst_k=cst_kern.K(y)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(org_k[i][j], cst_k[i][j], places=3)

    def test_for_gradients(self):
        length = 1.5
        var = 2
        A = np.array([[0.4, 0.7, 0.3],
                      [0.1, 0.1, 0.9]])
        obj = InputPsi(A)
        cst_kern = CustomMatern52(2, obj, lengthscale=length, variance=var)
        y = np.array([[0.4, 0.9],
                      [-0.6, 0.52],
                      [0.82, -0.67]])
        dl_dk=np.ones((3,3))
        cst_kern.update_gradients_full(dl_dk, y)
        org_dvar=np.array([[1, 0.6978, 0.6057],
                           [0.6978, 1, 0.6544],
                           [0.6057, 0.6544, 1]])
        org_dl=np.array([[0, 0.5963, 0.7038],
                         [0.5963, 0, 0.6514],
                         [0.7038, 0.6514, 0]])

        self.assertAlmostEqual(cst_kern.variance.gradient[0], np.sum(org_dvar * dl_dk), places=3)
        self.assertAlmostEqual(cst_kern.lengthscale.gradient[0], np.sum(org_dl * dl_dk), places=2)

if __name__=='__main__':
    unittest.main()

