from projections import SimpleEmbedding, ConvexProjection, WarpingBackProjection
import unittest
import numpy as np

class SimpleEmbeddingTestCase(unittest.TestCase):
    def test(self):
        A=np.array([[0.4, 0.7, 0.3],
                    [0.1, 0.1, 0.9]])
        y=np.array([[2, -3]])
        prj=SimpleEmbedding(A)
        x=prj.evaluate(y)
        org_x=np.array([[0.5, 1.1, -2.1]])
        for i in range(3):
            self.assertAlmostEqual(x[0,i],org_x[0,i])

class ConvexProjectionTestCase(unittest.TestCase):
    def test(self):
        A = np.array([[0.4, 0.7, 0.3],
                      [0.1, 0.1, 0.9]])
        y = np.array([[2, -3]])
        prj=ConvexProjection(A)
        x=prj.evaluate(y)
        org_x= np.array([[0.5, 1, -1]])
        for i in range(3):
            self.assertAlmostEqual(x[0,i],org_x[0,i])

class WarpingBackProjectionTestCase(unittest.TestCase):
    def test_of_back_projection(self):
        A = np.array([[0.4, 0.7, 0.3],
                      [0.1, 0.1, 0.9]])
        y = np.array([[2, -7]])
        x=np.array([[0.1, 0.7, -1]])
        prj = WarpingBackProjection(A)
        z=np.transpose(prj.bp_matrix)@np.transpose(x)
        org_z=np.array([[0.2800],[0.6009],[-1.0090]])
        for i in range(3):
            self.assertAlmostEqual(z[i,0], org_z[i,0], places=3)

    def test_for_final_result(self):
        A = np.array([[0.4, 0.7, 0.3],
                      [0.1, 0.1, 0.9]])
        y = np.array([[2, -7]])
        x = np.array([[0.1, 0.7, -1]])
        prj = WarpingBackProjection(A)
        psi=prj.evaluate(y)
        org_psi=np.array([[0.3264, 0.7005, -1.1761]])
        for i in range(3):
            self.assertAlmostEqual(psi[0,i], org_psi[0,i], places=3)

if __name__=='__main__':
    unittest.main()
