from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import os
import numpy as np
import scipy
import scipy.misc
from skopt import gp_minimize
from skopt.space import Categorical

def expansion_number_to_string(exp_n):
    if 0 <= exp_n < 100:
        if exp_n == 0:
            return "id"
        elif exp_n == 1:
            return "u08"
        elif exp_n == 2:
            return "s10QT"
        elif exp_n == 3:
            return "s12QT"
        elif exp_n == 4:
            return "s14QT"
        elif exp_n == 5:
            return "s16QT"
        elif exp_n == 6:
            return "s18QT"
        elif exp_n == 7:
            return "s20QT"
        elif exp_n == 8:
            return "s22QT"
        elif exp_n == 9:
            return "s24QT"
        elif exp_n == 10:
            return "s28QT"
        elif exp_n == 11:
            return "s30QT"
        elif exp_n == 12:
            return "s35QT"
        elif exp_n == 13:
            return "s10CT"
        elif exp_n == 14:
            return "s12CT"
        elif exp_n == 15:
            return "s14CT"
        elif exp_n == 16:
            return "s16CT"
        else:
            ex = "Invalid value for expansion:" + str(exp_n) + "-" + str(type(exp_n))
            raise Exception(ex)
    elif 200 <= exp_n < 300:
        if exp_n == 200:
            return "u08"
        elif exp_n == 201:
            return "ch2s10qt"
        elif exp_n == 202:
            return "ch2s15qt"
        elif exp_n == 203:
            return "ch2s20qt"
        elif exp_n == 204:
            return "ch2s25qt"
        elif exp_n == 205:
            return "ch2s30qt"
        elif exp_n == 206:
            return "ch2s35qt"
        elif exp_n == 207:
            return "ch2s40qt"
        elif exp_n == 208:
            return "ch2s45qt"
        elif exp_n == 209:
            return "ch2s50qt"
        elif exp_n == 210:
            return "ch2s55qt"
        elif exp_n == 211:
            return "ch2s60qt"
        elif exp_n == 212:
            return "ch2s65qt"
        elif exp_n == 213:
            return "ch2s70qt"
        elif exp_n == 214:
            return "ch2s75qt"
        elif exp_n == 215:
            return "ch2s80qt"
        else:
            ex = "invalid expansion number: " + str(exp_n)
            raise Exception(ex)
    elif 300 <= exp_n < 400:
        if exp_n == 300:
            return "u08"
        elif exp_n == 301:
            return "ch3s10qt"
        elif exp_n == 302:
            return "ch3s15qt"
        elif exp_n == 303:
            return "ch3s20qt"
        elif exp_n == 304:
            return "ch3s25qt"
        else:
            ex = "invalid expansion number: " + str(exp_n)
            raise Exception(ex)
    elif 400 <= exp_n < 500:
        if exp_n == 400:
            return "u08"
        elif exp_n == 401:
            return "ch4s10qt"
        elif exp_n == 402:
            return "ch4s15qt"
        elif exp_n == 403:
            return "ch4s20qt"
        elif exp_n == 404:
            return "ch4s25qt"
        else:
            ex = "invalid expansion number: " + str(exp_n)
            raise Exception(ex)
    else:
        ex = "invalid expansion number: " + str(exp_n)
        raise Exception(ex)

valid_expansion_numbers = range(0, 17) + range(200, 216) + range(300, 305) + range(400, 405)
map_exp_n_to_string = {}
map_string_to_exp_n = {}
for i in valid_expansion_numbers[::-1]:
    entry = map_exp_n_to_string[i] = expansion_number_to_string(i)
    map_string_to_exp_n[entry] = i
print("map_exp_n_to_string:", map_exp_n_to_string)
print("map_string_to_exp_n", map_string_to_exp_n)


def string_to_expansion_number(string):
    if string in map_string_to_exp_n.keys():
        return map_string_to_exp_n[string]
    else:
        ex = "Invalid expansion string: " + string + " of type " + str(type(string))
        raise Exception(ex)


def cuicuilco_f_CE_Gauss(arguments):
    return 1.0 - cuicuilco_evaluation(arguments, measure="CR_Gauss")


def cuicuilco_f_CE_Gauss_soft(arguments):
    return 1.0 - cuicuilco_evaluation(arguments, measure="CR_Gauss_soft")


def cuicuilco_f_CE_Gauss_mix(arguments):
    return 1.0 - cuicuilco_evaluation(arguments, measure="CR_Gauss_mix") 
 

def cuicuilco_evaluation(arguments, measure="CR_Gauss", verbose=False):
    (L0_pca_out_dim, L0_sfa_out_dim, L1H_sfa_out_dim, L1V_sfa_out_dim, L2H_sfa_out_dim,
     L2V_sfa_out_dim, L3H_sfa_out_dim, L3V_sfa_out_dim, L5_sfa_out_dim,
     L0_delta_threshold, L1H_delta_threshold, L1V_delta_threshold, L2H_delta_threshold,
     L2V_delta_threshold, L3H_delta_threshold, L3V_delta_threshold, L5_delta_threshold,
     L0_expansion, L1H_expansion, L1V_expansion, L2H_expansion, L2V_expansion, L3H_expansion,
     L3V_expansion, L5_expansion, L6_degree_QT, L6_degree_CT) = arguments
 
    print("invoking cuicuilco_evaluation with arguments:", arguments)

    # Testing whether arguments are compatible
    incompatible = 0
    if L0_pca_out_dim + L0_delta_threshold < L0_sfa_out_dim:
        L0_delta_threshold = L0_sfa_out_dim - L0_pca_out_dim
        print("Attempting to solve incompatibility case 1", L0_pca_out_dim, L0_delta_threshold, L0_sfa_out_dim)
    #if L0_delta_threshold < 1 or L0_delta_threshold > 36:
    #    incompatible = 21

    if 2 * L2H_sfa_out_dim + L2V_delta_threshold < L2V_sfa_out_dim:
        L2V_delta_threshold - 2 * L2H_sfa_out_dim
    #if L2V_delta_threshold < 1 or L2V_delta_threshold > 36:
    #    incompatible = 22

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
    if L1H_delta_threshold > (2 + 3) * L0_sfa_out_dim:
        incompatible = 8
    elif L1V_delta_threshold > (2 + 3) * L1H_sfa_out_dim:
        incompatible = 9
    elif L2H_delta_threshold > 2 * L1V_sfa_out_dim: # the factor here should be actually 4, right?
        incompatible = 10
    elif L2V_delta_threshold > 2 * L2H_sfa_out_dim:
        incompatible = 11
    elif L3H_delta_threshold > 2 * L2V_sfa_out_dim:
        incompatible = 12
    elif L3V_delta_threshold > 2 * L3H_sfa_out_dim:
        incompatible = 13
    if L0_delta_threshold > L0_sfa_out_dim:
        L0_delta_threshold = (L0_delta_threshold + L0_sfa_out_dim) // 2
        L0_sfa_out_dim = L0_delta_threshold
        print("Attempting to solve incompatibility case 14:", L0_delta_threshold, L0_sfa_out_dim)
    elif L1H_delta_threshold > L1H_sfa_out_dim:
        L1H_delta_threshold = (L1H_delta_threshold + L1H_sfa_out_dim) // 2
        L1H_sfa_out_dim = L1H_delta_threshold
        print("Attempting to solve incompatibility case 15:", L1H_delta_threshold, L1H_sfa_out_dim)
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

    if incompatible and False:  # WARNING, ignoring possibly incompatible neworks
        print("Configuration (before fixes):", arguments, " is incompatible (%d) and was skipped" % incompatible)
        return 0.0


    # Update arguments variable
    arguments = (L0_pca_out_dim, L0_sfa_out_dim, L1H_sfa_out_dim, L1V_sfa_out_dim, L2H_sfa_out_dim,
                 L2V_sfa_out_dim, L3H_sfa_out_dim, L3V_sfa_out_dim, L5_sfa_out_dim,
                 L0_delta_threshold, L1H_delta_threshold, L1V_delta_threshold, L2H_delta_threshold,
                 L2V_delta_threshold, L3H_delta_threshold, L3V_delta_threshold, L5_delta_threshold,
                 L0_expansion, L1H_expansion, L1V_expansion, L2H_expansion, L2V_expansion, L3H_expansion,
                 L3V_expansion, L5_expansion, L6_degree_QT, L6_degree_CT)
    print("arguments:", arguments)
    print("Creating configuration file ")
    fd = open("HiGSFA_CIFAR10_Network_9L_config.txt", "w")
    txt = ""
    for entry in arguments:
        txt += str(entry)+ " "
    fd.write(txt)
    fd.close()
    print("created configuration file with contents:", txt)

    cuicuilco_experiment_seeds = [112250, 112260] # 112210 112220, 112230] #, 112240] #[112244, 112255, 112266, 112277]  # , 112277]
    metrics = []
    for cuicuilco_experiment_seed in cuicuilco_experiment_seeds:  #112233 #np.random.randint(2**25)  #     np.random.randn()
        os.putenv("CUICUILCO_EXPERIMENT_SEED", str(cuicuilco_experiment_seed))
        print("Setting CUICUILCO_EXPERIMENT_SEED: ", str(cuicuilco_experiment_seed))

        output_filename = "hyper_t/CIFAR_9L_5cloneL_L0_%dPC_%dSF_%s_%dF_" + \
                          "L1H_%dSF_%s_%dF_L1V_%dSF_%s_%dF_L2H_%dSF_%s_%dF_" + \
                          "L2V_%dSF_%s_%dF_L3H_%dSF_%s_%dF_L3V_%dSF_%s_%dF_" + \
                          "L5_%dSF_%s_%dF_L6_QT%dAP_CT%dAP_seed%d.txt"
        output_filename = output_filename % (L0_pca_out_dim, L0_delta_threshold, expansion_number_to_string(L0_expansion), L0_sfa_out_dim,
                                L1H_delta_threshold, expansion_number_to_string(L1H_expansion), L1H_sfa_out_dim,
                                L1V_delta_threshold, expansion_number_to_string(L1V_expansion), L1V_sfa_out_dim,
                                L2H_delta_threshold, expansion_number_to_string(L2H_expansion), L2H_sfa_out_dim,
                                L2V_delta_threshold, expansion_number_to_string(L2V_expansion), L2V_sfa_out_dim,
                                L3H_delta_threshold, expansion_number_to_string(L3H_expansion), L3H_sfa_out_dim,
                                L3V_delta_threshold, expansion_number_to_string(L3V_expansion), L3V_sfa_out_dim,
                                L5_delta_threshold, expansion_number_to_string(L5_expansion), L5_sfa_out_dim,
                                L6_degree_QT, L6_degree_CT, cuicuilco_experiment_seed)
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
                  "--ExperimentalDataset=ParamsCIFAR10Func_32x32 " +\
                  "--HierarchicalNetwork=HiGSFA_CIFAR10_Network_9L_dd2_config " + \
                  "--SleepM=0 2>&1 > " + output_filename

            print("excecuting command: ", command)
            #quit()
            os.system(command)

        if verbose:
            print("extracting performance metric from resulting file")
        metric = extract_performance_metric_from_file(output_filename, measure=measure)
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
        if np.isnan(metric_CR_Gauss_soft):
            print("warning, nan metric was found and fixed as metric_CR_Gauss - 0.0001")
            metric_CR_Gauss_soft = metric_CR_Gauss - 0.0001
    else:
        print("unable to find metrics in file (defaulting to 0.4)")
        metric_CR_Gauss = 0.4
        metric_CR_Gauss_soft = 0.4
    if measure == "CR_Gauss":
        metric = metric_CR_Gauss
    elif measure == "CR_Gauss_soft":
        metric = metric_CR_Gauss_soft
    elif measure == "CR_Gauss_mix":
        metric = 0.5 * (metric_CR_Gauss + metric_CR_Gauss_soft)
    else:
        er = "invalid measure: " +  str(measure)
        raise Exception(er)
    # print("metric_CR_Gauss: ", metric_CR_Gauss, " metric_CR_Gauss_soft:", metric_CR_Gauss_soft)

    return metric


def load_saved_executions(measure="CR_Gauss", dimensions=None, verbose=False):
    path = "hyper_t"
    only_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    only_files = [f for f in only_files if f.startswith("CIFAR")]
    arguments_list = []
    results_list = []
    for f in only_files:
        # print("filename %s was found" % f)
        # MNIST_24x24_7L_L0cloneL_16PC_1SF_qtExp_25F_L1cloneL_1SF_u08Exp_20F_L2clone_30SF_u08Exp_80F_L3cloneL_1SF_u08Exp_100F_L4cloneL_20F_u08Exp_120F_L5_20F_u08Exp_90SF_L6_20F_u08Exp_250SF_NoHead_QT90AP_CT25AP_seed13153651.txt
        vals = f.split("_")
        #vals = [val.strip("PCFSseedQTA.txt") for val in vals]
        if verbose:
            print("vals=", vals)
        # quit()
        if len(vals) >= 36:
            L0_pca_out_dim = int(vals[4].strip("PC"))
            L0_sfa_out_dim = int(vals[7].strip("SF"))
            L1H_sfa_out_dim = int(vals[11].strip("SF"))
            L1V_sfa_out_dim = int(vals[15].strip("SF")) 
            L2H_sfa_out_dim = int(vals[19].strip("SF"))
            L2V_sfa_out_dim = int(vals[23].strip("SF"))
            L3H_sfa_out_dim = int(vals[27].strip("SF"))
            L3V_sfa_out_dim = int(vals[31].strip("SF"))
            L5_sfa_out_dim = int(vals[35].strip("SF")) # FIX
            L0_delta_threshold = int(vals[5].strip("SF"))
            L1H_delta_threshold = int(vals[9].strip("SF"))
            L1V_delta_threshold = int(vals[13].strip("SF")) 
            L2H_delta_threshold = int(vals[17].strip("SF"))
            L2V_delta_threshold = int(vals[21].strip("SF"))
            L3H_delta_threshold = int(vals[25].strip("SF"))
            L3V_delta_threshold = int(vals[29].strip("SF"))
            L5_delta_threshold = int(vals[33].strip("SF")) # FIX
            L0_expansion = string_to_expansion_number(vals[6])
            L1H_expansion = string_to_expansion_number(vals[10])
            L1V_expansion = string_to_expansion_number(vals[14])
            L2H_expansion = string_to_expansion_number(vals[18])
            L2V_expansion = string_to_expansion_number(vals[22])
            L3H_expansion = string_to_expansion_number(vals[26])
            L3V_expansion = string_to_expansion_number(vals[30])
            L5_expansion = string_to_expansion_number(vals[34]) # FIX
            L6_degree_QT = int(vals[37].strip("QTAP"))
            L6_degree_CT = int(vals[38].strip("CTAP"))
            seed = int(vals[39].strip("seed.txt"))
            if L1H_expansion == 1:
		        L1H_expansion = 300
            if L1V_expansion == 1:
                L1V_expansion = 300
            if L2H_expansion == 1:
                L2H_expansion = 300
            if L2V_expansion == 1:
                L2V_expansion = 300
            if L3H_expansion == 1:
                L3H_expansion = 200
            if L3V_expansion == 1:
                L3V_expansion = 200
            if L5_expansion == 1:
                L5_expansion = 400
            arguments = [L0_pca_out_dim, L0_sfa_out_dim, L1H_sfa_out_dim,
                         L1V_sfa_out_dim, L2H_sfa_out_dim, L2V_sfa_out_dim,
                         L3H_sfa_out_dim, L3V_sfa_out_dim, L5_sfa_out_dim,
                         L0_delta_threshold, L1H_delta_threshold,
                         L1V_delta_threshold, L2H_delta_threshold,
                         L2V_delta_threshold, L3H_delta_threshold,
                         L3V_delta_threshold, L5_delta_threshold,
                         L0_expansion, L1H_expansion, L1V_expansion,
                         L2H_expansion, L2V_expansion, L3H_expansion,
                         L3V_expansion, L5_expansion, L6_degree_QT, L6_degree_CT]
            if verbose:
                print("parsed arguments:", arguments)

            metric = extract_performance_metric_from_file(os.path.join(path, f), measure)
            arguments_list.append(arguments)
            results_list.append(metric)            
        else:
            print("Error parging values", vals)
    if len(arguments_list) > 0:
        results_list = np.array(results_list)
        # arguments_list = np.array(arguments_list, dtype=int)
        ordering = np.argsort(results_list)[::-1]
        results_list = results_list[ordering]
        sorted_arguments_list = []
        for i in range(len(ordering)):
            sorted_arguments_list.append(arguments_list[ordering[i]])
        arguments_list = sorted_arguments_list
        # print("ordered results_list: ", results_list)
        # print("ordered arguments_list: ")
        for arguments in arguments_list:
            print(arguments)
        if dimensions is not None:
            validity_values = []             
            for i, arguments in enumerate(arguments_list):
                valid = True
                for j, dim in enumerate(dimensions):
                    arg_value = arguments[j]
                    if isinstance(dim, Categorical):
                        if arguments[j] not in dim.categories:
                            valid = False
                            if verbose:
                                print("entry %d failed validity for argument %d with value %d" % (i, j, arg_value), dim.categories)
                    elif isinstance(dim, tuple) and len(dim) == 2:
                        if dim[0] > arguments[j] or dim[1] < arguments[j]:
                            valid = False
                            if verbose:
                                print("entry %d failed validity for argument %d with value %d" % (i, j, arg_value), dim)
                    elif isinstance(dim, list):
                        if arguments[j] not in dim:
                            valid = False
                            if verbose:
                                print("entry %d failed validity for argument %d with value %d" % (i, j, arg_value), dim)
       	        validity_values.append(valid)
            print("validity_values:", validity_values)
            filtered_arguments_list = []
            for i in range(len(validity_values)):
                if validity_values[i]:
                    filtered_arguments_list.append(arguments_list[i])
            arguments_list = filtered_arguments_list    
            results_list = results_list[validity_values]
    # if len(arguments_list) == 0:
    #     arguments_list = None
    #     results_list = None
    # print("final ordered results_list: ", results_list)
    # print("final ordered arguments_list: ")
    for arguments in arguments_list:
        print(arguments)
        # quit()
    if len(arguments_list) == 0:
        arguments_list = None
        results_list = None

    return arguments_list, results_list

def display_best_arguments(arguments_list, results_list, consider_std=True):
    if arguments_list is None:
        print("arguments_list is None")
        return None, None

    arguments_results_dict = {}
    for i, arguments in enumerate(arguments_list):
        arg_tuple = tuple(arguments)
        if arg_tuple in arguments_results_dict:
            arguments_results_dict[arg_tuple].append(results_list[i])
        else:
            arguments_results_dict[arg_tuple] = [results_list[i]]
    # Average all entries with the same key
    averaged_arguments_list = []
    averaged_results_list = []
    results_stds = []
    results_lens = []
    for arg in arguments_results_dict.keys():
        averaged_arguments_list.append(arg)
        averaged_results_list.append(np.array(arguments_results_dict[arg]).mean())
        results_stds.append(np.array(arguments_results_dict[arg]).std())
        results_lens.append(len(arguments_results_dict[arg]))
        # print("std: ", np.array(arguments_results_dict[arg]).std(), " len:", len(arguments_results_dict[arg]))
    # print("averaged_arguments_list=", averaged_arguments_list)
    # print("averaged_results_list=", averaged_results_list)

    # sort
    averaged_results_list = np.array(averaged_results_list)
    results_stds = np.array(results_stds)
    results_lens = np.array(results_lens)
    if consider_std:
        ordering = np.argsort(averaged_results_list - 0.5 * results_stds/(results_lens-1)**0.5)[::-1]
    else:
        ordering = np.argsort(averaged_results_list)[::-1]

    averaged_results_list = averaged_results_list[ordering]
    results_stds = results_stds[ordering]
    results_lens = results_lens[ordering]
    averaged_sorted_arguments_list = []
    for i in range(len(ordering)):
        averaged_sorted_arguments_list.append(averaged_arguments_list[ordering[i]])
    averaged_arguments_list = averaged_sorted_arguments_list
    print("averaged ordered results_list: ", averaged_results_list)
    print("results_stds: ", results_stds)
    corrected_results_list = averaged_results_list - 0.5 * results_stds/(results_lens-1)**0.5
    print("averaged ordered results_list - 0.5 * results_stds/factor: ", corrected_results_list)
    print("results_lens: ", results_lens)
    print("averaged ordered arguments_list: ")
    for arguments in averaged_arguments_list:
    	print("(", end="")
        for arg in arguments:
            print("%3d, "%arg, end="")
        print(")")

    if consider_std:
        final_results_list = corrected_results_list
    else:
        final_results_list = averaged_results_list
    return averaged_arguments_list, final_results_list

def progress_callback(res):
    print("C", end="")


#def gp_minimize(func, dimensions, base_estimator=None, n_calls=100, n_random_starts=10, acq_func='gp_hedge',
#                acq_optimizer='auto', x0=None, y0=None, random_state=None, verbose=False, callback=None,
#                n_points=10000, n_restarts_optimizer=5, xi=0.01, kappa=1.96, noise='gaussian', n_jobs=1)

# Output dimensionalities (PCA and iGSFA)
range_L0_pca_out_dim = (24, 28)
range_L0_sfa_out_dim = (28, 32)
range_L1H_sfa_out_dim = (68, 77)
range_L1V_sfa_out_dim = (75, 130)
range_L2H_sfa_out_dim = (130, 250)
range_L2V_sfa_out_dim = (190, 400)
range_L3H_sfa_out_dim = (200, 375)
range_L3V_sfa_out_dim = (130, 400)
range_L5_sfa_out_dim = (200, 450)

# Length of slow part
range_L0_delta_threshold = (26, 30)
range_L1H_delta_threshold = (21, 25)
range_L1V_delta_threshold = (24, 32)
range_L2H_delta_threshold = (28, 44)
range_L2V_delta_threshold = (15, 30) 
range_L3H_delta_threshold = (17, 37)
range_L3V_delta_threshold = (19, 26)
range_L5_delta_threshold = [0]

# WARNING two categories cannot be expressed as [n1, n2], instead use e.g., Categorical([0, 3])
#         otherwise interval (n1, n2) is assumed
# Expansions
range_L0_expansion = [1]
range_L1H_expansion = [300]
range_L1V_expansion = [300]
range_L2H_expansion = [300]
range_L2V_expansion = [300]
range_L3H_expansion = (201, 205)
range_L3V_expansion = (201, 202)
range_L5_expansion = [400] # (400, 402)
range_L4_degree_QT = (130, 169)
range_L4_degree_CT = (24, 37)
cuicuilco_dimensions = (range_L0_pca_out_dim, range_L0_sfa_out_dim, range_L1H_sfa_out_dim,
                        range_L1V_sfa_out_dim, range_L2H_sfa_out_dim, range_L2V_sfa_out_dim,
                        range_L3H_sfa_out_dim, range_L3V_sfa_out_dim, range_L5_sfa_out_dim,
                        range_L0_delta_threshold, range_L1H_delta_threshold,
                        range_L1V_delta_threshold, range_L2H_delta_threshold,
                        range_L2V_delta_threshold, range_L3H_delta_threshold,
                        range_L3V_delta_threshold, range_L5_delta_threshold,
                        range_L0_expansion, range_L1H_expansion, range_L1V_expansion,
                        range_L2H_expansion, range_L2V_expansion, range_L3H_expansion,
                        range_L3V_expansion, range_L5_expansion,
                        range_L4_degree_QT, range_L4_degree_CT)

print("cuicuilco_dimensions:", cuicuilco_dimensions)
# np.random.seed(1234) # use a new random seed each time to allow combination of executions on different systems
argument_list, results_list = load_saved_executions(measure="CR_Gauss_mix", dimensions=cuicuilco_dimensions, verbose=False)
display_best_arguments(argument_list, results_list, consider_std=True)
quit()

#argument_list = None
#results_list = None
#argument_list = [
#[24,  28,  70,  75, 230, 280, 237, 140, 430,  27,  24,  30,  40,  17,  35,  24,   0,   1, 300, 300, 300, 300, 201, 202, 401, 147,  31, ],
#[30,  28,  74, 109, 130, 251, 210, 399, 430,  28,  22,  26,  35,  28,  20,  20,   0,   1, 300, 300, 300, 300, 202, 201, 400, 161,  32, ],
#[ 27,  29,  73, 115, 210, 380, 230, 160, 200,  29,  23,  26,  36,  28,  25,  20,   0,   1, 300, 300, 300, 300, 203, 201, 401, 159,  36, ],
#[ 26,  28,  76,  89, 134, 195, 361, 220, 226,  28,  24,  30,  28,  27,  34,  19,   0,   1, 300, 300, 300, 300, 204, 202, 400, 167,  32, ],
#[ 24,  28,  77, 121, 228, 394, 370, 322, 222,  26,  22,  30,  32,  27,  35,  24,   0,   1, 300, 300, 300, 300, 205, 202, 400, 153,  28, ],
#[ 26,  31,  70, 124, 196, 369, 345, 167, 409,  29,  21,  24,  30,  21,  31,  23,   0,   1, 300, 300, 300, 300, 205, 201, 400, 167,  26, ],
#[ 24,  31,  77,  76, 225, 381, 362, 267, 362,  29,  22,  28,  37,  28,  18,  19,   0,   1, 300, 300, 300, 300, 205, 202, 400, 133,  33, ],
#[ 24,  31,  76, 114, 155, 226, 212, 141, 210,  27,  21,  30,  29,  27,  34,  19,   0,   1, 300, 300, 300, 300, 205, 201, 401, 149,  37, ],
#[ 24,  31,  76, 124, 180, 335, 251, 390, 201,  26,  23,  24,  28,  26,  18,  24,   0,   1, 300, 300, 300, 300, 201, 202, 401, 167,  37, ]
#]

if results_list is not None:
    results_list = [1.0 - result for result in results_list]

print("cuicuilco_dimensions:", cuicuilco_dimensions)
t0 = time.time()
res = gp_minimize(func=cuicuilco_f_CE_Gauss_mix, dimensions=cuicuilco_dimensions, base_estimator=None, n_calls=40, n_random_starts=10,  # 20 10
                  acq_func='gp_hedge', acq_optimizer='auto', x0=argument_list, y0=results_list, random_state=None, verbose=False,
                  callback=progress_callback, n_points=100*10000, n_restarts_optimizer=5,   # n_points=10000
                  xi=0.01, kappa=1.96, noise='gaussian', n_jobs=1)
t1 = time.time()

print("res:", res)
print("Execution time: %0.3f s" % (t1 - t0))

