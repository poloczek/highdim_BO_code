import math
import numpy as np

# All function are defined in such a way that have global maximums,
# if a function originally has a minimum, the final objective value is multiplied by -1

class TestFunction:
    def evaluate(self,x):
        pass

class Rosenbrock(TestFunction):
    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        x_copy *= 2.0
        return x_copy

    def evaluate(self,x):
        # Calculating the output
        scaled_x=self.scale_domain(x)
        f = [[0]]
        f[0] = [-(math.pow(1 - i[0], 2) + 100 * math.pow(i[1] - math.pow(i[0], 2), 2)) for i in scaled_x]
        f = np.transpose(f)
        return f

class Branin(TestFunction):
    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        x_copy[:, 0] = x_copy[:, 0] * (10 + 5) / 2 + (10 - 5) / 2
        x_copy[:, 1] = x_copy[:, 1] * (15 + 0) / 2 + (15 - 0) / 2
        return x_copy

    def evaluate(self,x):
        scaled_x=self.scale_domain(x)
        # Calculating the output
        f = [[0]]
        f[0] = [-((i[1] - (5.1 / (4 * math.pi ** 2)) * i[0] ** 2 + i[0] * 5 / math.pi - 6) ** 2 + 10 * (
                1 - 1 / (8 * math.pi)) * np.cos(i[0]) + 10) for i in scaled_x]
        f = np.transpose(f)
        return f

class Hartmann6(TestFunction):
    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        x_copy = x_copy*0.5+0.5
        return x_copy

    def evaluate(self,x):
        # Calculating the output
        #Created on 08.09.2016
        # @author: Stefan Falkner
        alpha = [1.00, 1.20, 3.00, 3.20]
        A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                      [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                      [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                      [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
        P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])
        scaled_x = self.scale_domain(x)
        n=len(scaled_x)
        external_sum = np.zeros((n,1))
        for r in range(n):
            for i in range(4):
                internal_sum = 0
                for j in range(6):
                    internal_sum = internal_sum + A[i, j] * (scaled_x[r,j] - P[i, j]) ** 2
                external_sum[r] = external_sum[r] + alpha[i] * np.exp(-internal_sum)
        return external_sum
