from REMBO import RunRembo, EI
import numpy as np
import unittest

class EITestCase(unittest.TestCase):
    def test(self):
        f_max=1.5
        mu=np.array([[2], [1.5], [1]])
        var=np.array([[0.25], [0], [4]])
        ei=EI(3,f_max,mu,var)
        org_ei=np.array([[0.5416],[0], [0.5726]])
        for i in range(3):
            self.assertAlmostEqual(ei[i,0], org_ei[i,0], places=3)

if __name__=='__main__':
    unittest.main()
