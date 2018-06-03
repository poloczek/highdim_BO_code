import numpy as np
import projections

class Inputs:
    def evaluate(self, A, y):
        pass

class InputY(Inputs):
    def evaluate(self, A, y):
        return y

class InputX(Inputs):
    def evaluate(self, A, y):
        cp=projections.ConvexProjection(A)
        x=cp.evaluate(y)
        return x

class InputPsi(Inputs):
    def evaluate(self, A, y):
        wbp=projections.WarpingBackProjection(A)
        psi=wbp.evaluate(y)
        return psi
