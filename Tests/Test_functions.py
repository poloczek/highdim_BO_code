from functions import Rosenbrock, Branin, Hartmann6
import unittest
import numpy as np

class RosenbrockTestCase(unittest.TestCase):
    def test_for_single_point(self):
        """It tests if the out put for [0.5,0.5] is zero"""
        func=Rosenbrock()
        self.assertAlmostEqual(func.evaluate([[0.5, 0.5]]), 0, delta=1e-3)

    def test_for_two_points(self):
        """It tests if the function works for several points"""
        func = Rosenbrock()
        f=func.evaluate([[0.5, 0.5], [1, 1]])
        org_f=[[0],[-401]]
        for i in range(len(f)):
            self.assertAlmostEqual(f[i],org_f[i], delta=1e-3)

class BraninTestCase(unittest.TestCase):
    """It tests if the out put for [9.4247, 2.475] is 0.3978"""
    def test_for_single_point(self):
        func = Branin()
        self.assertAlmostEqual(func.evaluate([[(9.42478- (10 - 5) / 2) * 2 / (10 + 5),
                                                (2.475- (15 - 0) / 2) * 2 / (15 + 0)]])[0,0], -0.3978, places=3)

class Hartmann6TestCase(unittest.TestCase):
    def test_for_single_point(self):
        func = Hartmann6()
        self.assertAlmostEqual(func.evaluate((np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])
                                             -0.5)*2)[0,0], 3.3223, places=3)

if __name__=='__main__':
    unittest.main()
