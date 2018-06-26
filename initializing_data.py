import pickle
import numpy as np
from pyDOE import lhs
import projections
import functions

def initiate_branin_data(initial_n=20, effective_dim=2, high_dim=25, replications=100):

    fileObject=open('high_dim_branin_initial_data','wb')
    all_A=np.random.normal(0, 1, [replications,effective_dim,high_dim])
    all_s=np.empty((replications, initial_n, effective_dim))
    all_f_s=np.empty((replications,initial_n, 1))
    test_func = functions.Branin()
    for i in range(replications):
        cnv_prj = projections.ConvexProjection(all_A[i])
        all_s[i] = lhs(effective_dim, initial_n) * 2 * np.sqrt(effective_dim) - np.sqrt(effective_dim)
        all_f_s[i] = test_func.evaluate(cnv_prj.evaluate(all_s[i]))
    pickle.dump(all_A,fileObject)
    pickle.dump(all_s,fileObject)
    pickle.dump(all_f_s,fileObject)
    fileObject.close()

def initiate_rosenbrock_data(initial_n=20, effective_dim=2, high_dim=25, replications=100):

    fileObject=open('high_dim_rosenbrock_initial_data','wb')
    all_A=np.random.normal(0, 1, [replications,effective_dim,high_dim])
    all_s=np.empty((replications, initial_n, effective_dim))
    all_f_s=np.empty((replications,initial_n, 1))
    test_func = functions.Rosenbrock()
    for i in range(replications):
        cnv_prj = projections.ConvexProjection(all_A[i])
        all_s[i] = lhs(effective_dim, initial_n) * 2 * np.sqrt(effective_dim) - np.sqrt(effective_dim)
        all_f_s[i] = test_func.evaluate(cnv_prj.evaluate(all_s[i]))
    pickle.dump(all_A,fileObject)
    pickle.dump(all_s,fileObject)
    pickle.dump(all_f_s,fileObject)
    fileObject.close()

def initiate_hartmann6_data(initial_n=20, effective_dim=6, high_dim=25, replications=100):

    fileObject = open('high_dim_hartmann6_initial_data', 'wb')
    all_A = np.random.normal(0, 1, [replications, effective_dim, high_dim])
    all_s = np.empty((replications, initial_n, effective_dim))
    all_f_s = np.empty((replications, initial_n, 1))
    test_func = functions.Hartmann6()
    for i in range(replications):
        cnv_prj = projections.ConvexProjection(all_A[i])
        all_s[i] = lhs(effective_dim, initial_n) * 2 * np.sqrt(effective_dim) - np.sqrt(effective_dim)
        all_f_s[i] = test_func.evaluate(cnv_prj.evaluate(all_s[i]))
    pickle.dump(all_A, fileObject)
    pickle.dump(all_s, fileObject)
    pickle.dump(all_f_s, fileObject)
    fileObject.close()

if __name__=='__main__':
    initiate_branin_data(replications=500)
    initiate_rosenbrock_data(replications=500)
    # initiate_hartmann6_data(initial_n=60, replications=500)

