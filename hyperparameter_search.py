from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import os
import numpy as np
import scipy
import scipy.misc
from skopt import gp_minimize


def exp_txt(expansion):
    if expansion == 0:
        return "u08"
    else:
        return "qt"


def cuicuilco_f_RMSE_Gauss(arguments):
    return cuicuilco_evaluation(arguments, measure="RMSE_Gauss")

 
def cuicuilco_evaluation(arguments, measure="RMSE_Gauss"):
    cuicuilco_experiment_seed = np.random.randint(2**25)  #     np.random.randn()
    os.putenv("CUICUILCO_EXPERIMENT_SEED", str(cuicuilco_experiment_seed))
    print("Setting CUICUILCO_EXPERIMENT_SEED: ", str(cuicuilco_experiment_seed))

    (L0_pca_out_dim, L0_sfa_out_dim, L1H_sfa_out_dim, L1V_sfa_out_dim, L2H_sfa_out_dim, L2V_sfa_out_dim,
     L3H_sfa_out_dim, L3V_sfa_out_dim, L0_delta_threshold, L1H_delta_threshold, L1V_delta_threshold,
     L2H_delta_threshold, L2V_delta_threshold, L3H_delta_threshold, L3V_delta_threshold, L0_expansion,
     L1H_expansion, L1V_expansion, L2H_expansion, L2V_expansion, L3H_expansion, L3V_expansion,
     L4_degree_QT, L4_degree_CT) = arguments


    output_filename = "hyperparameter_tuning/MNIST_MNISTNetwork_24x24_7L_Overlap_config_L0cloneL_%dPC_%dSF_%sExp_%dF_" + \
                      "L1cloneL_%dSF_%sExp_%dF_L2clone_%dSF_%sExp_%dF_L3cloneL_%dSF_%sExp_%dF_" + \
                      "L4cloneL_%dF_%sExp_%dF_L5_%dF_%sExp_%dSF_L6_%dF_%sExp_%dSF_NoHead_QT%dAP_CT%dAP_seed%d.txt" 
    output_filename = output_filename % (L0_pca_out_dim, L0_delta_threshold, exp_txt(L0_expansion), L0_sfa_out_dim,
                            L1H_delta_threshold, exp_txt(L1H_expansion), L1H_sfa_out_dim,
                            L1V_delta_threshold, exp_txt(L1V_expansion), L1V_sfa_out_dim,
                            L2H_delta_threshold, exp_txt(L2H_expansion), L2H_sfa_out_dim,
                            L2V_delta_threshold, exp_txt(L2V_expansion), L2V_sfa_out_dim,
                            L3H_delta_threshold, exp_txt(L3H_expansion), L3H_sfa_out_dim,
                            L3V_delta_threshold, exp_txt(L3V_expansion), L3V_sfa_out_dim,
                            L4_degree_QT, L4_degree_CT, cuicuilco_experiment_seed)
    if os.path.isfile(output_filename):
        print("file %s already exists, skipping its computation" % output_filename)
    else:
        command = "nice -n 19 time python -u -m cuicuilco.cuicuilco_run --EnableDisplay=0 --CacheAvailable=0 " + \
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

    print("extracting performance metrici from resulting file")
    command_extract = "cat %s | grep New | grep CR_G > del_tmp.txt" % output_filename
    os.system(command_extract)
    fd = open("del_tmp.txt", "r")
    metrics = fd.readline().split(" ")
    fd.close()
    print("metrics: ", metrics)
    if len(metrics) > 10 and metrics[6] == "CR_Gauss":
        metric_CR_Gauss = metrics[7].strip(",")
        metric_softCR_Gauss = metrics[9].strip(",")
    else:
        print("unable to find metrics in file")
        metric_CR_Gauss = 20.0
        metric_softCR_Gauss = 20.0    
    print("metric_CR_Gauss: ", metric_CR_Gauss, " metric_CR_Gauss:", metric_softCR_Gauss)

    return float(metric_CR_Gauss)



def progress_callback(res):
    print("C", end="")


#def gp_minimize(func, dimensions, base_estimator=None, n_calls=100, n_random_starts=10, acq_func='gp_hedge',
#                acq_optimizer='auto', x0=None, y0=None, random_state=None, verbose=False, callback=None,
#                n_points=10000, n_restarts_optimizer=5, xi=0.01, kappa=1.96, noise='gaussian', n_jobs=1)

# 13 20 28 50 70 90 120 200 9 19 10 26 6 6 9 0 0 0 0 0 0 0 90 25
range_L0_pca_out_dim = (10, 16) # 13
range_L0_sfa_out_dim = (15, 25) # [20] # (20, 21)
range_L1H_sfa_out_dim = (20, 36) # [28] # (28, 29)
range_L1V_sfa_out_dim = (30, 80) # [50] # (50, 51)
range_L2H_sfa_out_dim = (40, 100) # [70] # (70, 71)
range_L2V_sfa_out_dim = (60, 120) # [90] # (90, 91)
range_L3H_sfa_out_dim = (90, 150) # [120] # (120, 121)
range_L3V_sfa_out_dim = (100, 250) #[200] # (200, 201)
range_L0_delta_threshold = (1, 20) # [9] # #(9, 10)
range_L1H_delta_threshold = (1, 30) # [19] # (19, 20)
range_L1V_delta_threshold = (1, 30) # [10] # (10, 11)
range_L2H_delta_threshold = (1, 40) # [26] # (26, 27)
range_L2V_delta_threshold = (1, 20) # [6] # (6, 7)
range_L3H_delta_threshold = (1, 20) # [6] # (6, 7)
range_L3V_delta_threshold = (1, 20) # [9] # (9, 10)
range_L0_expansion = (0, 1) # [0] # (0, 1)
range_L1H_expansion = [0] # (0, 1)
range_L1V_expansion = [0] # (0, 1)
range_L2H_expansion = [0] # (0, 1)
range_L2V_expansion = [0] # (0, 1)
range_L3H_expansion = [0] # (0, 0)
range_L3V_expansion = [0] # (0, 0)
range_L4_degree_QT = (60, 90) # [90] # (90, 90)
range_L4_degree_CT = (20, 25) # [25] # (25, 25)
cuicuilco_dimensions = [range_L0_pca_out_dim, range_L0_sfa_out_dim, range_L1H_sfa_out_dim, range_L1V_sfa_out_dim, range_L2H_sfa_out_dim, range_L2V_sfa_out_dim, range_L3H_sfa_out_dim, range_L3V_sfa_out_dim, range_L0_delta_threshold, range_L1H_delta_threshold, range_L1V_delta_threshold, range_L2H_delta_threshold, range_L2V_delta_threshold, range_L3H_delta_threshold, range_L3V_delta_threshold, range_L0_expansion, range_L1H_expansion, range_L1V_expansion, range_L2H_expansion, range_L2V_expansion, range_L3H_expansion, range_L3V_expansion, range_L4_degree_QT, range_L4_degree_CT]

#TODO: load previously computed results from saved files

np.random.seed(1234)

t0 = time.time()
res = gp_minimize(func=cuicuilco_f_RMSE_Gauss, dimensions=cuicuilco_dimensions, base_estimator=None, n_calls=20, n_random_starts=10,
                  acq_func='gp_hedge', acq_optimizer='auto', x0=None, y0=None, random_state=None, verbose=False,
                  callback=progress_callback, n_points=10000, n_restarts_optimizer=5,
                  xi=0.01, kappa=1.96, noise='gaussian', n_jobs=1)
t1 = time.time()

print("res:", res)
print("Execution time: %0.3f s" % (t1 - t0))

