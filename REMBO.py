import GPy
import numpy as np
import math
from pyDOE import lhs
from scipy.stats import norm
import functions
import projection_matrix
import projections
import kernel_inputs


def EI(D_size,f_max,mu,var):
    ei=[0]*D_size
    for i in range(D_size):
        if var[i]!=0:
            z= (mu[i] - f_max) / var[i]
            ei[i]= (mu[i]-f_max) * norm.cdf(z) + var[i] * norm.pdf(z)
    return ei

def RunRembo(effective_dim=2, high_dim=20, initial_n=20, total_itr=100,
             func_type='Branin', matrix_type='simple', kern_inp_type='Y', A_input=None, s=None, f_s=None):

    #Specifying the type of objective function
    if func_type=='Branin':
        test_func = functions.Branin()
    elif func_type=='Rosenbrock':
        test_func = functions.Rosenbrock()
    elif func_type=='Hartmann6':
        test_func = functions.Hartmann6()
    else:
        TypeError('The input for func_type variable is invalid, which is', func_type)
        return

    #Specifying the type of embedding matrix
    if matrix_type=='simple':
        matrix=projection_matrix.SimpleGaussian(effective_dim, high_dim)
    elif matrix_type=='normal':
        matrix= projection_matrix.Normalized(effective_dim, high_dim)
    elif matrix_type=='orthogonal':
        matrix = projection_matrix.Orthogonalized(effective_dim, high_dim)
    else:
        TypeError('The input for matrix_type variable is invalid, which is', matrix_type)
        return

    #Specifying the input type of kernel
    if kern_inp_type=='Y':
        kern_inp=kernel_inputs.InputY()
    elif kern_inp_type=='X':
        kern_inp=kernel_inputs.InputX()
    elif kern_inp_type == 'psi':
        kern_inp = kernel_inputs.InputPsi()
    else:
        TypeError('The input for kern_inp_type variable is invalid, which is', kern_inp_type)
        return

    best_results=np.zeros([1,total_itr])

    # Generating matrix A
    if A_input is not None:
        matrix.A= A_input

    A = matrix.evaluate()
    #Specifying the convex projection
    cnv_prj=projections.ConvexProjection(A)

    # Initiating first sample    # Sample points are in [-d^1/2, d^1/2]
    if s is None:
        s = lhs(effective_dim, initial_n) * 2 * math.sqrt(effective_dim) - math.sqrt(effective_dim)
        f_s = test_func.evaluate(cnv_prj.evaluate(s))

    # Generating GP model
    k = GPy.kern.Matern52(input_dim=effective_dim)
    m = GPy.models.GPRegression(s, f_s, kernel=k)
    m.likelihood.variance = 1e-6
    m.optimize()

    # Main loop of the algorithm
    for i in range(total_itr):
        D = lhs(effective_dim, 1000) * 2 * math.sqrt(effective_dim) - math.sqrt(effective_dim)

        # Updating GP model
        m.set_XY(s,f_s)
        if i % 50 == 0:
            m.optimize()
        mu, var = m.predict(kern_inp.evaluate(A, D))

        # finding the next point for sampling
        ei_d = EI(len(D), max(f_s), mu, var)
        index = np.argmax(ei_d)
        s = np.append(s, [D[index]], axis=0)
        f_s = np.append(f_s, test_func.evaluate(cnv_prj.evaluate([D[index]])), axis=0)

        #Collecting data
        best_results[0,i]=np.max(f_s)

    # max_index = np.argmax(f_s)
    # max_point = s[max_index]
    # max_value = f_s[max_index]
    #
    # print('The best value is:  ', max_value,
    #       '\n \n at the point:  ', max_point,
    #       '\n \n with Ay value:  ', test_func.scale_domain(cnv_prj.evaluate([max_point])),
    #       '\n\nin the iteration:  ', max_index)
    return best_results, s, f_s



