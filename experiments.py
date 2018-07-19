import REMBO
import numpy as np
import pickle
import timeit
import sys

def run_experiments(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100, effect_dim=6, high_dim=25, initial_n=20):
    if test_func=='Rosenbrock':
        file_name = 'high_dim_rosenbrock_initial_data'
    elif test_func=='Branin':
        file_name = 'high_dim_branin_initial_data'
    elif test_func == 'Hartmann6':
        file_name = 'high_dim_hartmann6_initial_data'

    fileObject = open(file_name, 'rb')
    all_A = pickle.load(fileObject)
    all_init_s = pickle.load(fileObject)
    all_init_f_s = pickle.load(fileObject)

    result_x_obj = np.empty((0, total_itr))
    result_y_obj = np.empty((0, total_itr))
    result_psi_obj = np.empty((0, total_itr))

    result_x_s = np.empty((0, initial_n + total_itr, effect_dim))
    result_x_f_s = np.empty((0, initial_n + total_itr, 1))
    result_y_s = np.empty((0, initial_n + total_itr, effect_dim))
    result_y_f_s = np.empty((0, initial_n + total_itr, 1))
    result_psi_s = np.empty((0, initial_n + total_itr, effect_dim))
    result_psi_f_s = np.empty((0, initial_n + total_itr, 1))

    for i in range(start_rep - 1, stop_rep):
        start = timeit.default_timer()
        # Running different algorithms to solve Hartmann6 function
        temp_result, temp_s, temp_f_s = REMBO.RunRembo(low_dim=effect_dim, high_dim=high_dim, initial_n=initial_n,
                                                       total_itr=total_itr, func_type='Hartmann6', A_input=all_A[i],
                                                       s=all_init_s[i], f_s=all_init_f_s[i],
                                                       kern_inp_type='Y', matrix_type='simple')
        result_y_obj = np.append(result_y_obj, temp_result, axis=0)
        result_y_s = np.append(result_y_s, [temp_s], axis=0)
        result_y_f_s = np.append(result_y_f_s, [temp_f_s], axis=0)

        temp_result, temp_s, temp_f_s = REMBO.RunRembo(low_dim=effect_dim, high_dim=high_dim, initial_n=initial_n,
                                                       total_itr=total_itr, func_type='Hartmann6', A_input=all_A[i],
                                                       s=all_init_s[i], f_s=all_init_f_s[i],
                                                       kern_inp_type='X', matrix_type='simple')
        result_x_obj = np.append(result_x_obj, temp_result, axis=0)
        result_x_s = np.append(result_x_s, [temp_s], axis=0)
        result_x_f_s = np.append(result_x_f_s, [temp_f_s], axis=0)

        temp_result, temp_s, temp_f_s = REMBO.RunRembo(low_dim=effect_dim, high_dim=high_dim, initial_n=initial_n,
                                                       total_itr=total_itr, func_type='Hartmann6', A_input=all_A[i],
                                                       s=all_init_s[i], f_s=all_init_f_s[i],
                                                       kern_inp_type='psi', matrix_type='normal')
        result_psi_obj = np.append(result_psi_obj, temp_result, axis=0)
        result_psi_s = np.append(result_psi_s, [temp_s], axis=0)
        result_psi_f_s = np.append(result_psi_f_s, [temp_f_s], axis=0)

        stop = timeit.default_timer()

        print(i)
        print(stop - start)

    # Saving the results for Hartmann6 in a pickle
    if test_func=='Rosenbrock':
        file_name = 'high_dim_rosenbrock_results_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func=='Branin':
        file_name = 'high_dim_branin_results_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'Hartmann6':
        file_name = 'high_dim_hartmann6_results_rep_' + str(start_rep) + '_' + str(stop_rep)

    fileObject = open(file_name, 'wb')
    pickle.dump(result_y_obj, fileObject)
    pickle.dump(result_x_obj, fileObject)
    pickle.dump(result_psi_obj, fileObject)

    pickle.dump(result_y_s, fileObject)
    pickle.dump(result_x_s, fileObject)
    pickle.dump(result_psi_s, fileObject)

    pickle.dump(result_y_f_s, fileObject)
    pickle.dump(result_x_f_s, fileObject)
    pickle.dump(result_psi_f_s, fileObject)
    fileObject.close()

if __name__=='__main__':

    run_experiments(start_rep=int(sys.argv[1]), stop_rep=int(sys.argv[2]), test_func='Hartmann6', total_itr=250, initial_n=60)
