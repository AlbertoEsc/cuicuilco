from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import os
import numpy as np
import scipy
import scipy.misc
from skopt import gp_minimize


def expansion_number_to_string(expansion):
    if expansion == 0:
        return "u08"
    elif expansion == 1:
        return "qt"
    else:
        ex = "invalid expansion number: " + str(expansion)
        raise Exception(ex)


def string_to_expansion_number(string):
    if string == "u08Exp":
        return 0
    elif string == "qtExp":
        return 1
    else:
        ex = "invalid expansion string: " + string
        raise Exception(ex)


def cuicuilco_f_CE_Gauss(arguments):
    return 1.0 - cuicuilco_evaluation(arguments, measure="CR_Gauss")


def cuicuilco_f_CE_Gauss_soft(arguments):
    return 1.0 - cuicuilco_evaluation(arguments, measure="CR_Gauss_soft")


def cuicuilco_f_CE_Gauss_mix(arguments):
    return 1.0 - cuicuilco_evaluation(arguments, measure="mix") 
 

def cuicuilco_evaluation(arguments, measure="CR_Gauss", verbose=False):
    (L0_pca_out_dim, L0_sfa_out_dim, L1H_sfa_out_dim, L1V_sfa_out_dim, L2H_sfa_out_dim, L2V_sfa_out_dim,
     L3H_sfa_out_dim, L3V_sfa_out_dim, L0_delta_threshold, L1H_delta_threshold, L1V_delta_threshold,
     L2H_delta_threshold, L2V_delta_threshold, L3H_delta_threshold, L3V_delta_threshold, L0_expansion,
     L1H_expansion, L1V_expansion, L2H_expansion, L2V_expansion, L3H_expansion, L3V_expansion,
     L4_degree_QT, L4_degree_CT) = arguments

    # Testing whether arguments are compatible
    incompatible = 0
    if L0_pca_out_dim + L0_delta_threshold < L0_sfa_out_dim:
        L0_delta_threshold = L0_sfa_out_dim - L0_pca_out_dim
        print("Attempting to solve incompatibility case 1", L0_pca_out_dim, L0_delta_threshold, L0_sfa_out_dim)
    if L0_delta_threshold < 1 or L0_delta_threshold > 20:
        incompatible = 21

    if 2 * L2H_sfa_out_dim + L2V_delta_threshold < L2V_sfa_out_dim:
        L2V_delta_threshold - 2 * L2H_sfa_out_dim
    if L2V_delta_threshold < 1 or L2V_delta_threshold > 20:
        incompatible = 22

    if L0_pca_out_dim + L0_delta_threshold < L0_sfa_out_dim:
        incompatible = 1
    elif 2 * L0_sfa_out_dim + L1H_delta_threshold < L1H_sfa_out_dim:  # This factor is 2 and not 3 due to overlap
        incompatible = 2
    elif 2 * L1H_sfa_out_dim + L1V_delta_threshold < L1V_sfa_out_dim:  # This factor is 2 and not 3 due to overlap
        incompatible = 3
    elif 2 * L1V_sfa_out_dim + L2H_delta_threshold < L2H_sfa_out_dim:
        incompatible = 4
    elif 2 * L2H_sfa_out_dim + L2V_delta_threshold < L2V_sfa_out_dim:
        incompatible = 5
    elif 2 * L2V_sfa_out_dim + L3H_delta_threshold < L3H_sfa_out_dim:
        incompatible = 6
    elif 2 * L3H_sfa_out_dim + L3V_delta_threshold < L3V_sfa_out_dim:
        incompatible = 7
    if L1H_delta_threshold >  (2 + 3) * L0_sfa_out_dim:
        incompatible = 8
    elif L1V_delta_threshold >  (2 + 3) * L1H_sfa_out_dim:
        incompatible = 9
    elif L2H_delta_threshold >  2 * L1V_sfa_out_dim: # the factor here should be actually 4, right?
        incompatible = 10
    elif L2V_delta_threshold >  2 * L2H_sfa_out_dim:
        incompatible = 11
    elif L3H_delta_threshold >  2 * L2V_sfa_out_dim:
        incompatible = 12
    elif L3V_delta_threshold >  2 * L3H_sfa_out_dim:
        incompatible = 13
    if L0_delta_threshold > L0_sfa_out_dim:
        incompatible = 14
    elif L1H_delta_threshold > L1H_sfa_out_dim:
        incompatible = 15
    elif L1V_delta_threshold > L1V_sfa_out_dim:
        incompatible = 16
    elif L2H_delta_threshold > L2H_sfa_out_dim:
        incompatible = 17
    elif L2V_delta_threshold > L2V_sfa_out_dim:
        incompatible = 18
    elif L3H_delta_threshold > L3H_sfa_out_dim:
        incompatible = 19
    elif L3V_delta_threshold > L3V_sfa_out_dim:
        incompatible = 20

    if incompatible:
        print("Configuration (before fixes):", arguments, " is incompatible (%d) and was skipped" % incompatible)
        return 0.0


    # Update arguments variable
    arguments = (L0_pca_out_dim, L0_sfa_out_dim, L1H_sfa_out_dim, L1V_sfa_out_dim, L2H_sfa_out_dim, L2V_sfa_out_dim,
     L3H_sfa_out_dim, L3V_sfa_out_dim, L0_delta_threshold, L1H_delta_threshold, L1V_delta_threshold,
     L2H_delta_threshold, L2V_delta_threshold, L3H_delta_threshold, L3V_delta_threshold, L0_expansion,
     L1H_expansion, L1V_expansion, L2H_expansion, L2V_expansion, L3H_expansion, L3V_expansion,
     L4_degree_QT, L4_degree_CT)
    print("Creating configuration file ")
    fd = open("MNISTNetwork_24x24_7L_Overlap_config.txt", "w")
    txt = ""
    for entry in arguments:
        txt += str(entry)+ " "
    fd.write(txt)
    fd.close()

    cuicuilco_experiment_seeds = [112233, 112244]
    metrics = []
    for cuicuilco_experiment_seed in cuicuilco_experiment_seeds:  #112233 #np.random.randint(2**25)  #     np.random.randn()
        os.putenv("CUICUILCO_EXPERIMENT_SEED", str(cuicuilco_experiment_seed))
        print("Setting CUICUILCO_EXPERIMENT_SEED: ", str(cuicuilco_experiment_seed))



        output_filename = "hyperparameter_tuning/MNIST_MNISTNetwork_24x24_7L_Overlap_config_L0cloneL_%dPC_%dSF_%sExp_%dF_" + \
                          "L1cloneL_%dSF_%sExp_%dF_L2clone_%dSF_%sExp_%dF_L3cloneL_%dSF_%sExp_%dF_" + \
                          "L4cloneL_%dSF_%sExp_%dF_L5_%dSF_%sExp_%dF_L6_%dSF_%sExp_%dF_NoHead_QT%dAP_CT%dAP_seed%d.txt"
        output_filename = output_filename % (L0_pca_out_dim, L0_delta_threshold, expansion_number_to_string(L0_expansion), L0_sfa_out_dim,
                                L1H_delta_threshold, expansion_number_to_string(L1H_expansion), L1H_sfa_out_dim,
                                L1V_delta_threshold, expansion_number_to_string(L1V_expansion), L1V_sfa_out_dim,
                                L2H_delta_threshold, expansion_number_to_string(L2H_expansion), L2H_sfa_out_dim,
                                L2V_delta_threshold, expansion_number_to_string(L2V_expansion), L2V_sfa_out_dim,
                                L3H_delta_threshold, expansion_number_to_string(L3H_expansion), L3H_sfa_out_dim,
                                L3V_delta_threshold, expansion_number_to_string(L3V_expansion), L3V_sfa_out_dim,
                                L4_degree_QT, L4_degree_CT, cuicuilco_experiment_seed)
        if os.path.isfile(output_filename):
            print("file %s already exists, skipping its computation" % output_filename)
        else:
            command = "time nice -n 19 python -u -m cuicuilco.cuicuilco_run --EnableDisplay=0 --CacheAvailable=0 " + \
                  "--NetworkCacheReadDir=/local/tmp/escalafl/Alberto/SavedNetworks " + \
                  "--NetworkCacheWriteDir=/local/tmp/escalafl/Alberto/SavedNetworks " + \
                  "--NodeCacheReadDir=/local/tmp/escalafl/Alberto/SavedNodes " + \
                  "--NodeCacheWriteDir=/local/tmp/escalafl/Alberto/SavedNodes " + \
                  "--ClassifierCacheWriteDir=/local/tmp/escalafl/Alberto/SavedClassifiers " + \
                  "--SaveSubimagesTraining=0 --SaveAverageSubimageTraining=0 --NumFeaturesSup=9 " + \
                  "--SaveSorted_AE_GaussNewid=0 --SaveSortedIncorrectClassGaussNewid=0 " + \
                  "--ComputeSlowFeaturesNewidAcrossNet=0 --UseFilter=0 --EnableGC=1 --SFAGCReducedDim=0 --EnableKNN=0 " + \
                  "--kNN_k=3 --EnableNCC=0 --EnableSVM=0 --SVM_C=0.125 --SVM_gamma=1.0 --EnableLR=0 " + \
                  "--AskNetworkLoading=0 --LoadNetworkNumber=-1 --NParallel=2 --EnableScheduler=0 " + \
                  "--EstimateExplainedVarWithInverse=0 --EstimateExplainedVarWithKNN_k=0 " + \
                  "--EstimateExplainedVarWithKNNLinApp=0 --EstimateExplainedVarLinGlobal_N=0 --AddNormalizationNode=0 " + \
                  "--MakeLastPCANodeWhithening=0 --FeatureCutOffLevel=-1.0 --ExportDataToLibsvm=0 " + \
                  "--IntegerLabelEstimation=0 --MapDaysToYears=0 --CumulativeScores=0 --DatasetForDisplayNewid=0 " + \
                  "--GraphExactLabelLearning=0 --OutputInsteadOfSVM2=0 --NumberTargetLabels=0 --EnableSVR=0 " + \
                  "--SVR_gamma=0.85 --SVR_C=48.0 --SVR_epsilon=0.075 --SVRInsteadOfSVM2=1 --ObjectiveLabel=0 " + \
                  "--ExperimentalDataset=ParamsMNISTFunc --HierarchicalNetwork=MNISTNetwork_24x24_7L_Overlap_config " + \
                  "--SleepM=0 2>&1 > " + output_filename

            print("excecuting command: ", command)
            os.system(command)

        if verbose:
            print("extracting performance metric from resulting file")
        metric = extract_performance_metric_from_file(output_filename)
        metrics.append(metric)
    return np.array(metric).mean()


def extract_performance_metric_from_file(output_filename, measure = "CR_Gauss", verbose=False):
    command_extract = "cat %s | grep New | grep CR_G > del_tmp.txt" % output_filename
    os.system(command_extract)
    fd = open("del_tmp.txt", "r")
    metrics = fd.readline().split(" ")
    fd.close()
    if verbose:
        print("metrics: ", metrics)
    if len(metrics) > 10 and metrics[6] == "CR_Gauss":
        metric_CR_Gauss = float(metrics[7].strip(","))
        metric_CR_Gauss_soft = float(metrics[9].strip(","))
    else:
        print("unable to find metrics in file (defaulting to 0.95)")
        metric_CR_Gauss = 0.95
        metric_CR_Gauss_soft = 0.95  
    if measure == "CR_Gauss":
        metric = metric_CR_Gauss
    elif measure == "CR_Gauss_soft":
        metric = metric_CR_Gauss_soft
    elif measure == "mix":
        metric = 0.5 * (metric_CR_Gauss + metric_CR_Gauss_soft)
  
    print("metric_CR_Gauss: ", metric_CR_Gauss, " metric_CR_Gauss_soft:", metric_CR_Gauss_soft)

    return metric


def load_saved_executions(measure="CR_Gauss", dimensions=None, verbose=False):
    path = "hyperparameter_tuning"
    only_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    only_files = [f for f in only_files if f.startswith("MNIST_MNISTNetwork_24x24_7L_Overlap_config")]
    arguments_list = []
    results_list = []
    for f in only_files:
        print("filename %s was found" % f)
        # MNIST_MNISTNetwork_24x24_7L_Overlap_config_L0cloneL_16PC_1SF_qtExp_25F_L1cloneL_1SF_u08Exp_20F_L2clone_30SF_u08Exp_80F_L3cloneL_1SF_u08Exp_100F_L4cloneL_20F_u08Exp_120F_L5_20F_u08Exp_90SF_L6_20F_u08Exp_250SF_NoHead_QT90AP_CT25AP_seed13153651.txt
        vals = f.split("_")
        vals = [val.strip("PCFSseedQTA.txt") for val in vals]
        if verbose:
            print("vals=", vals)
        if len(vals) >= 35:
            L0_pca_out_dim = int(vals[7])
            L0_sfa_out_dim = int(vals[10])
            L1H_sfa_out_dim = int(vals[14])
            L1V_sfa_out_dim = int(vals[18]) 
            L2H_sfa_out_dim = int(vals[22])
            L2V_sfa_out_dim = int(vals[26])
            L3H_sfa_out_dim = int(vals[30])
            L3V_sfa_out_dim = int(vals[34])
            L0_delta_threshold = int(vals[8])
            L1H_delta_threshold = int(vals[12])
            L1V_delta_threshold = int(vals[16]) 
            L2H_delta_threshold = int(vals[20])
            L2V_delta_threshold = int(vals[24])
            L3H_delta_threshold = int(vals[28])
            L3V_delta_threshold = int(vals[32])
            L0_expansion = string_to_expansion_number(vals[9])
            L1H_expansion = string_to_expansion_number(vals[13])
            L1V_expansion = string_to_expansion_number(vals[17])
            L2H_expansion = string_to_expansion_number(vals[21])
            L2V_expansion = string_to_expansion_number(vals[25])
            L3H_expansion = string_to_expansion_number(vals[29])
            L3V_expansion = string_to_expansion_number(vals[33])
            L4_degree_QT = int(vals[36])
            L4_degree_CT = int(vals[37])
            seed = int(vals[38])
            arguments = [L0_pca_out_dim, L0_sfa_out_dim, L1H_sfa_out_dim, L1V_sfa_out_dim, L2H_sfa_out_dim, L2V_sfa_out_dim, L3H_sfa_out_dim, L3V_sfa_out_dim, L0_delta_threshold, L1H_delta_threshold, L1V_delta_threshold, L2H_delta_threshold, L2V_delta_threshold, L3H_delta_threshold, L3V_delta_threshold, L0_expansion, L1H_expansion, L1V_expansion, L2H_expansion, L2V_expansion, L3H_expansion, L3V_expansion, L4_degree_QT, L4_degree_CT]
            if verbose:
                print("parsed arguments:", arguments)

            metric = extract_performance_metric_from_file(os.path.join(path, f), measure)
            arguments_list.append(arguments)
            results_list.append(metric)            
        else:
            print("Error parging values", vals)
    if len(arguments_list) == 0:
        arguments_list = None
        results_list = None
    else:
        results_list = np.array(results_list)
        # arguments_list = np.array(arguments_list, dtype=int)
        ordering = np.argsort(results_list)[::-1]
	results_list = results_list[ordering]
        sorted_arguments_list = []
        for i in range(len(ordering)):
            sorted_arguments_list.append(arguments_list[ordering[i]])
        arguments_list = sorted_arguments_list
        print("ordered results_list: ", results_list)
        print("ordered arguments_list: ")
        for arguments in arguments_list:
            print(arguments)
        if dimensions is not None:
            validity_values = []             
            for i, arguments in enumerate(arguments_list):
                valid = True
                for j, dim in enumerate(dimensions):
                    if len(dim) == 2 and isinstance(dim, tuple):
                        if dim[0] > arguments[j] or dim[1] < arguments[j]:
                            valid = False
                    #elif isinstance(dim, list):
                    #    if arguments[j] not in dim:
                    #        valid = False
       	        validity_values.append(valid)
            print("validity_values:", validity_values)
            filtered_arguments_list = []
            for i in range(len(validity_values)):
                if validity_values[i]:
                    filtered_arguments_list.append(arguments_list[i])
            arguments_list = filtered_arguments_list    
            results_list = results_list[validity_values]
    print("final ordered results_list: ", results_list)
    print("final ordered arguments_list: ")
    for arguments in arguments_list:
        print(arguments)

    return arguments_list, results_list


def progress_callback(res):
    print("C", end="")


#def gp_minimize(func, dimensions, base_estimator=None, n_calls=100, n_random_starts=10, acq_func='gp_hedge',
#                acq_optimizer='auto', x0=None, y0=None, random_state=None, verbose=False, callback=None,
#                n_points=10000, n_restarts_optimizer=5, xi=0.01, kappa=1.96, noise='gaussian', n_jobs=1)

# 13 20 28 50 70 90 120 200 9 19 10 26 6 6 9 0 0 0 0 0 0 0 90 25
range_L0_pca_out_dim = (12, 14)  # (10, 16) # 13
range_L0_sfa_out_dim = (15, 19)  # (15, 25) # [20] # (20, 21)
range_L1H_sfa_out_dim = (32, 36)  # (20, 36) # [28] # (28, 29)
range_L1V_sfa_out_dim = (45, 60) # [50] # (50, 51)
range_L2H_sfa_out_dim = (75, 110) # [70] # (70, 71)
range_L2V_sfa_out_dim = (60, 100) # [90] # (90, 91)
range_L3H_sfa_out_dim = (80, 120) # [120] # (120, 121)
range_L3V_sfa_out_dim = (90, 200) #[200] # (200, 201)
range_L0_delta_threshold = (5, 16)  # (1, 20) # [9] # #(9, 10) #
range_L1H_delta_threshold = (3, 20) # [19] # (19, 20)
range_L1V_delta_threshold = (1, 25) # [10] # (10, 11)
range_L2H_delta_threshold = (5, 35) # [26] # (26, 27)
range_L2V_delta_threshold = (0, 20) # [6] # (6, 7)
range_L3H_delta_threshold = (0, 10) # [6] # (6, 7)
range_L3V_delta_threshold = (0, 6) # [9] # (9, 10)
range_L0_expansion = (0, 1) # [0] # (0, 1)
range_L1H_expansion = [0] # (0, 1)
range_L1V_expansion = [0] # (0, 1)
range_L2H_expansion = [0] # (0, 1)
range_L2V_expansion = [0] # (0, 1)
range_L3H_expansion = [0] # (0, 0)
range_L3V_expansion = [0] # (0, 0)
range_L4_degree_QT = (70, 99) # [90] # (90, 90)
range_L4_degree_CT = (21, 25) # [25] # (25, 25)
cuicuilco_dimensions = [range_L0_pca_out_dim, range_L0_sfa_out_dim, range_L1H_sfa_out_dim, range_L1V_sfa_out_dim, range_L2H_sfa_out_dim, range_L2V_sfa_out_dim, range_L3H_sfa_out_dim, range_L3V_sfa_out_dim, range_L0_delta_threshold, range_L1H_delta_threshold, range_L1V_delta_threshold, range_L2H_delta_threshold, range_L2V_delta_threshold, range_L3H_delta_threshold, range_L3V_delta_threshold, range_L0_expansion, range_L1H_expansion, range_L1V_expansion, range_L2H_expansion, range_L2V_expansion, range_L3H_expansion, range_L3V_expansion, range_L4_degree_QT, range_L4_degree_CT]

#TODO: load previously computed results from saved files

np.random.seed(1234)

argument_list, results_list = load_saved_executions(measure="CR_Gauss", dimensions=cuicuilco_dimensions)
if results_list is not None:
    results_list = [1.0 - result for result in results_list]


t0 = time.time()
res = gp_minimize(func=cuicuilco_f_CE_Gauss_mix, dimensions=cuicuilco_dimensions, base_estimator=None, n_calls=100, n_random_starts=20,  # 20 10
                  acq_func='gp_hedge', acq_optimizer='auto', x0=argument_list, y0=results_list, random_state=None, verbose=False,
                  callback=progress_callback, n_points=100*10000, n_restarts_optimizer=5,   # n_points=10000
                  xi=0.01, kappa=1.96, noise='gaussian', n_jobs=1)
t1 = time.time()

print("res:", res)
print("Execution time: %0.3f s" % (t1 - t0))

