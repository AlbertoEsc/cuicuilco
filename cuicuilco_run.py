#! /usr/bin/env python

#####################################################################################################################
# CUICUILCO: A general purpose framework that allows the construction and evaluation of
#            hierarchical networks for supervised learning

# By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de 
# First Version 9 Dec 2009. Current version May 2017
# Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott


#####################################################################################################################
# USAGE EXAMPLES:
# python cuicuilco_run.py --Experiment=ParamsNatural --Network=u08expoNetwork1L --InputFilename=/scratch/escalafl/
# cooperations/igel/rbm_64/data_bin_4000.bin --OutputFilename=deleteme2.txt
#####################################################################################################################


import numpy
import scipy
import scipy.misc

# mpl.style.use('classic')

import PIL
import mdp
import more_nodes
import patch_mdp

import object_cache as cache
import os
import sys
import glob
import random
import sfa_libs
from sfa_libs import (scale_to, distance_squared_Euclidean, str3, wider_1Darray, ndarray_to_string, cutoff)
from exact_label_learning import (ConstructGammaFromLabels, RemoveNegativeEdgeWeights, MapGammaToEdgeWeights)
import system_parameters
from system_parameters import (scale_sSeq, take_first_02D, take_0_k_th_from_2D_list, sSeq_force_image_size,
                               sSeq_getinfo_format, convert_sSeq_to_funcs_params_sets)
from image_loader import *
import classifiers_regressions as classifiers
import network_builder
import time
from matplotlib.ticker import MultipleLocator
import copy
import string

import getopt
from lockfile import LockFile
from inspect import getmembers
import subprocess
import mkl

import matplotlib as mpl

mpl.use('Qt4Agg')
import matplotlib.pyplot as plt

__version__ = "0.8.0"

# benchmark is a list that contains benchmark information (running times) with entries: ("description", time as float
#  in seconds)
benchmark = None

mkl.set_num_threads(18)  # Number of threads used by mlk to parallelize matrix operations of numpy.
# Adjust according to the number of cores in your system.


#
random_seed = 123456  # Default seed used by hierarchical_networks
numpy.random.seed(random_seed)

enable_display = False
input_filename = None  # This field is only used by the ParamsNatural experimental dataset
output_filename = None  # The delta values of the extracted features are saved to this file
cache_available = True  # This should be enabled for any cache (network, node, signal, classifier caches) to work
load_and_append_output_features_dir = None
num_features_to_append_to_input = 0
save_output_features_dir = None
network_cache_read_dir = None  # "/local/tmp/escalafl/Alberto/SavedNetworks"
network_cache_write_dir = None  # "/local/tmp/escalafl/Alberto/SavedNetworks"
node_cache_read_dir = None  # "/local/tmp/escalafl/Alberto/SavedNodes"
node_cache_write_dir = None  # "/local/tmp/escalafl/Alberto/SavedNodes"
signal_cache_read_dir = None  # "/local/tmp/escalafl/Alberto/SavedSignals"
signal_cache_write_dir = None  # "/local/tmp/escalafl/Alberto/SavedSignals"
classifier_cache_read_dir = None  # "/local/tmp/escalafl/Alberto/SavedClassifiers"
classifier_cache_write_dir = None  # "/local/tmp/escalafl/Alberto/SavedClassifiers"

enable_command_line = True
reg_num_signals = 4
skip_num_signals = 0
use_full_sl_output = False
enable_kNN = False
enable_NCC = False
enable_GC = False
kNN_k = 1
enable_svm = False
svm_gamma = 0
svm_C = 1.0
svm_min = -1.0
svm_max = 1.0
enable_lr = False
load_network_number = None
ask_network_loading = True
n_parallel = None  # 5
enable_scheduler = False

save_subimages_training = False  # or True
save_images_training_supplementary_info = None
save_average_subimage_training = False  # or True
save_sorted_AE_Gauss_newid = False  # or True
save_sorted_incorrect_class_Gauss_newid = False  # or True
compute_slow_features_newid_across_net = 0  # or 1,2,3
estimate_explained_var_with_inverse = False
estimate_explained_var_with_kNN_k = 0
estimate_explained_var_with_kNN_lin_app_k = 0
estimate_explained_var_linear_global_N = 0
add_normalization_node = False
make_last_PCA_node_whithening = False
feature_cut_off_level = 0.0
use_filter = None
export_data_to_libsvm = False
integer_label_estimation = False
cumulative_scores = False
confusion_matrix = False
features_residual_information = 5000  # 0
compute_input_information = True
convert_labels_days_to_years = False
sfa_gc_reduced_dim = None

clip_seenid_newid_to_training = False
add_noise_to_seenid = False

dataset_for_display_train = 0
dataset_for_display_newid = 0
objective_label = 0

# ELL options
graph_exact_label_learning = False
output_instead_of_SVM2 = False
# The total number of labels is num_orig_labels * # number_of_target_labels_per_orig_label
number_of_target_labels_per_orig_label = 0
coherent_seeds = False or True

cuicuilco_queue = "queue_cuicuilco.txt"
cuicuilco_lock_file = "queue_cuicuilco"
minutes_sleep = 0

t0 = time.time()
print "LOADING INPUT/SETUP INFORMATION"

import hierarchical_networks
import experimental_datasets

print "Using mdp version:", mdp.__version__, "file:", mdp.__file__
print hierarchical_networks.__file__
print experimental_datasets.__file__

print "Attempting to retrieve hash of current git commit"
try:
    print "output of \"$git describe --tags\":", subprocess.check_output(["git", "describe", "--tags"]).strip()
    print "output of \"$git rev-parse HEAD\":", subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
except subprocess.CalledProcessError as e:
    print "\nFailed to determine current git commit:", str(e), "\n"

print "List of modules and their versions:"
obj_names = sys.modules.keys()
for obj_name in obj_names:
    obj_value = sys.modules[obj_name]
    obj_members = dir(obj_value)  # getmembers(obj_value)
    if "__version__" in obj_members:
        print "   using ", obj_name, " version: ", obj_value.__version__

available_experiments = {}
print "Creating list of available experiments:"
for (obj_name, obj_value) in getmembers(experimental_datasets):
    if isinstance(obj_value, system_parameters.ParamsSystem):
        print "   ", obj_name
        available_experiments[obj_name] = obj_value
# print "object", obj.__name__

available_networks = {}
print "Creating list of available networks:"
for (obj_name, obj_value) in getmembers(hierarchical_networks):
    if isinstance(obj_value, system_parameters.ParamsNetwork) and obj_name != "network":
        print "   ", obj_name
        available_networks[obj_name] = obj_value
# print "object", obj.__name__

name_default_experiment = "ParamsMNISTFunc"
name_default_network = "voidNetwork1L"

DefaultExperimentalDataset = available_experiments[name_default_experiment]
DefaultNetwork = available_networks[name_default_network]

# See also: ParamsGender, ParamsAngle, ParamsIdentity,  ParamsTransX, ParamsAge,
# ParamsRTransX, ParamsRTransY, ParamsRScale, ParamsRFace, ParamsRObject, ParamsNatural, ParamsRawNatural
# ParamsRFaceCentering, ParamsREyeTransX, ParamsREyeTransY
# Data Set based training data: ParamsRTransXFunc, ParamsRTransYFunc, ParamsRTransXY_YFunc
# ParamsRGTSRBFunc, ParamsRAgeFunc, ParamsMNISTFunc
# from experimental_datasets import ParamsMNISTFunc as DefaultExperimentalDataset #ParamsRAgeFunc, ParamsMNISTFunc
# ParamsRTransXYScaleFunc

# Networks available: voidNetwork1L, SFANetwork1L, PCANetwork1L, u08expoNetwork1L, quadraticNetwork1L
# Test_Network, linearNetwork4L, u08expoNetwork4L, NL_Network5L, linearNetwork5L, linearNetworkT6L, TestNetworkT6L, 
# linearNetworkU11L, TestNetworkU11L, nonlinearNetworkU11L, TestNetworkPCASFAU11L, linearPCANetworkU11L,
# u08expoNetworkU11L
# linearWhiteningNetwork11L, u08expo_m1p1_NetworkU11L, u08expoNetworkU11L, experimentalNetwork
# u08expo_pcasfaexpo_NetworkU11L, u08expoA2NetworkU11L/A3/A4, u08expoA3_pcasfaexpo_NetworkU11L, IEMNetworkU11L,
# SFANetwork1LOnlyTruncated
# from hierarchical_networks import NLIPCANetwork1L as DefaultNetwork ### using A4 lately, u08expoNetworkU11L,
# u08expo_pcasfaexpo_NetworkU11L, u08expoNetwork2T, GTSRBNetwork, u08expoNetworkU11L, u08expoS42NetworkU11L,
# u08expoNetwork1L, HeuristicEvaluationExpansionsNetworkU11L, HeuristicPaperNetwork
# HeuristicEvaluationExpansionsNetworkU11L, HardSFAPCA_u08expoNetworkU11L, HardSFAPCA_u08expoNetworkU11L
# GTSRBNetwork, u08expoNetworkU11L, IEVMLRecNetworkU11L, linearPCANetworkU11L, SFAAdaptiveNLNetwork32x32U11L,
# u08expoNetwork32x32U11L_NoTop
# TT4 PCANetwork
# u08expoS42NetworkU11L
# 5x5L0 Networks: u08expoNetworkU11L_5x5L0, linearPCANetworkU11L_5x5L0, IEVMLRecNetworkU11L_5x5L0, PCANetwork1L
# MNISTNetwork7L, SFANetwork1LOnlyTruncated, MNISTNetwork_24x24_7L, MNISTNetwork_24x24_7L_B, SFANetworkMNIST2L,
# (MNIST 8C:) SFANetwork1L, SFADirectNetwork1L (MNIST: semi-supervised, Gender: exact label)
# GENDER ELL: NetworkGender_8x8L0
# AGE MORPH-II: IEVMLRecNetworkU11L_Overlap6x6L0_1Label, HeadNetwork1L, IEVMLRecNetworkU11L_Overlap6x6L0_GUO_3Labels,
# IEVMLRecNetworkU11L_Overlap6x6L0_GUO_1Label, SFANetworkU11L_Overlap6x6L0_GUO_3Labels
# IEVMLRecNetworkU11L_Overlap6x6L0_3Labels <-Article HiGSFA, IEVMLRecNetworkU11L_Overlap6x6L0_2Labels

from experimental_datasets import experiment_seed
from experimental_datasets import DAYS_IN_A_YEAR


def my_sigmoid(x):
    return numpy.tanh(5 * x)


def svm_compute_range(data):
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    return mins, maxs


def svm_scale(data, mins, maxs, svm_min, svm_max):
    return (data - mins) * (svm_max - svm_min) / (maxs - mins) + svm_min


if coherent_seeds:
    print "experimental_datasets.experiment_seed=", experiment_seed
    numpy.random.seed(experiment_seed + 111111)

if __name__ == "__main__":  # ############## Parse command line arguments ####################
    if enable_command_line:
        argv = None
        if argv is None:
            argv = sys.argv
        print "Apparent command line arguments: \n", " ".join(argv)
        if len(argv) >= 2:
            try:
                opts, args = getopt.getopt(argv[1:], "", ["InputFilename=", "OutputFilename=", "EnableDisplay=",
                                                          "CacheAvailable=", "NumFeaturesSup=", "SkipFeaturesSup=",
                                                          'SVM_gamma=', 'SVM_C=', 'EnableSVM=', "LoadNetworkNumber=",
                                                          "AskNetworkLoading=",
                                                          'EnableLR=', "NParallel=", "EnableScheduler=",
                                                          "SaveOutputFeaturesDir=",
                                                          "LoadAndAppendOutputFeaturesDir=",
                                                          "NumFeaturesToAppendToInput=",
                                                          "NetworkCacheReadDir=", "NetworkCacheWriteDir=",
                                                          "NodeCacheReadDir=", "NodeCacheWriteDir=",
                                                          "SignalCacheReadDir=", "SignalCacheWriteDir=",
                                                          "ClassifierCacheReadDir=",
                                                          "ClassifierCacheWriteDir=", "SaveSubimagesTraining=",
                                                          "SaveAverageSubimageTraining=",
                                                          "SaveSorted_AE_GaussNewid=",
                                                          "SaveSortedIncorrectClassGaussNewid=",
                                                          "ComputeSlowFeaturesNewidAcrossNet=", "UseFilter=", "kNN_k=",
                                                          'EnableKNN=', "EnableNCC=",
                                                          "EnableGC=", "SaveSubimagesTrainingSupplementaryInfo=",
                                                          "EstimateExplainedVarWithInverse=",
                                                          "EstimateExplainedVarWithKNN_k=",
                                                          "EstimateExplainedVarWithKNNLinApp_k=",
                                                          "EstimateExplainedVarLinGlobal_N=", "AddNormalizationNode=",
                                                          "MakeLastPCANodeWhithening=", "FeatureCutOffLevel=",
                                                          "ExportDataToLibsvm=",
                                                          "IntegerLabelEstimation=", "CumulativeScores=",
                                                          "FeaturesResidualInformation=",
                                                          "ComputeInputInformation=", "SleepM=",
                                                          "DatasetForDisplayTrain=", "DatasetForDisplayNewid=",
                                                          "GraphExactLabelLearning=", "OutputInsteadOfSVM2=",
                                                          "NumberTargetLabels=", "ConfusionMatrix=",
                                                          "MapDaysToYears=", "AddNoiseToSeenid=", "ClipSeenidNewid=",
                                                          "HierarchicalNetwork=", "ExperimentalDataset=",
                                                          "SFAGCReducedDim=", "ObjectiveLabel=", "help"])
                print "opts=", opts
                print "args=", args

                if len(args) > 0:
                    print "Arguments not understood:", args
                    sys.exit(2)

                for opt, arg in opts:
                    if opt in ('--InputFilename',):
                        input_filename = arg
                        print "Using the following input file:", input_filename
                    elif opt in ('--OutputFilename',):
                        output_filename = arg
                        print "Using the following output file:", output_filename
                    elif opt in ('--EnableDisplay',):
                        if arg == '1':
                            enable_display = True
                        else:
                            enable_display = False
                        print "Setting enable_display to", enable_display
                    elif opt in ('--CacheAvailable',):
                        if arg == '1':
                            cache_available = True
                        else:
                            cache_available = False
                        print "Setting cache_available to", cache_available
                    elif opt in ('--NumFeaturesSup',):
                        reg_num_signals = int(arg)
                        print "Setting reg_num_signals to", reg_num_signals
                    elif opt in ('--SkipFeaturesSup',):
                        skip_num_signals = int(arg)
                        print "Setting skip_num_signals to", skip_num_signals
                    elif opt in ('--SVM_gamma',):
                        svm_gamma = float(arg)
                        print "Setting svm_gamma to", svm_gamma
                    elif opt in ('--SVM_C',):
                        svm_C = float(arg)
                        print "Setting svm_C to", svm_C
                    elif opt in ('--EnableSVM',):
                        enable_svm = int(arg)
                        print "Setting enable_svm to", enable_svm
                    elif opt in ('--LoadNetworkNumber',):
                        load_network_number = int(arg)
                        print "Setting load_network_number to", load_network_number
                    elif opt in ('--AskNetworkLoading',):
                        ask_network_loading = int(arg)
                        print "Setting ask_network_loading to", ask_network_loading
                    elif opt in ('--EnableLR',):
                        enable_lr = int(arg)
                        print "Setting enable_lr to", enable_lr
                    elif opt in ('--NParallel',):
                        n_parallel = int(arg)
                        print "Setting n_parallel to", n_parallel
                    elif opt in ('--EnableScheduler',):
                        enable_scheduler = int(arg)
                        print "Setting enable_scheduler to", enable_scheduler
                    elif opt in ('--SaveOutputFeaturesDir',):
                        if arg == "None":
                            save_output_features_dir = None
                        else:
                            save_output_features_dir = arg
                        print "Setting save_output_features_dir to", save_output_features_dir
                    elif opt in ('--LoadAndAppendOutputFeaturesDir',):
                        if arg == "None":
                            load_and_append_output_features_dir = None
                        else:
                            load_and_append_output_features_dir = arg
                        print "Setting load_and_append_output_features_dir to", load_and_append_output_features_dir
                    elif opt in ('--NumFeaturesToAppendToInput',):
                        if arg == "None":
                            num_features_to_append_to_input = None
                        else:
                            num_features_to_append_to_input = int(arg)
                        print "Setting num_features_to_append_to_input to", num_features_to_append_to_input
                    elif opt in ('--NetworkCacheReadDir',):
                        if arg == "None":
                            network_cache_read_dir = None
                        else:
                            network_cache_read_dir = arg
                        print "Setting network_cache_read_dir to", network_cache_read_dir
                    elif opt in ('--NetworkCacheWriteDir',):
                        if arg == "None":
                            network_cache_write_dir = None
                        else:
                            network_cache_write_dir = arg
                        print "Setting network_cache_write_dir to", network_cache_write_dir
                    elif opt in ('--NodeCacheReadDir',):
                        if arg == "None":
                            node_cache_read_dir = None
                        else:
                            node_cache_read_dir = arg
                        print "Setting node_cache_read_dir to", node_cache_read_dir
                    elif opt in ('--NodeCacheWriteDir',):
                        if arg == "None":
                            node_cache_write_dir = None
                        else:
                            node_cache_write_dir = arg
                        print "Setting node_cache_write_dir to", node_cache_write_dir
                    elif opt in ('--SignalCacheReadDir',):
                        if arg == "None":
                            signal_cache_read_dir = None
                        else:
                            signal_cache_read_dir = arg
                        print "Setting signal_cache_read_dir to", signal_cache_read_dir
                    elif opt in ('--SignalCacheWriteDir',):
                        if arg == "None":
                            signal_cache_write_dir = None
                        else:
                            signal_cache_write_dir = arg
                        print "Setting signal_cache_write_dir to", signal_cache_write_dir
                    elif opt in ('--ClassifierCacheReadDir',):
                        if arg == "None":
                            classifier_cache_read_dir = None
                        else:
                            classifier_cache_read_dir = arg
                        print "Setting classifier_cache_read_dir to", classifier_cache_read_dir
                        er = "ClassifierCacheReadDir: Option not supported yet"
                        raise Exception(er)
                    elif opt in ('--ClassifierCacheWriteDir',):
                        if arg == "None":
                            classifier_cache_write_dir = None
                        else:
                            classifier_cache_write_dir = arg
                        print "Setting classifier_cache_write_dir to", classifier_cache_write_dir
                        # er = "ClassifierCacheWriteDir: Option not supported yet"
                        # raise Exception(er)
                    elif opt in ('--SaveSubimagesTraining',):
                        save_subimages_training = bool(int(arg))
                        print "Setting save_subimages_training to", save_subimages_training
                    elif opt in ('--SaveAverageSubimageTraining',):
                        save_average_subimage_training = bool(int(arg))
                        print "Setting save_average_subimage_training to", save_average_subimage_training
                    elif opt in ('--SaveSorted_AE_GaussNewid',):
                        save_sorted_AE_Gauss_newid = bool(int(arg))
                        print "Setting save_sorted_AE_Gauss_newid to", save_sorted_AE_Gauss_newid
                    elif opt in ('--SaveSortedIncorrectClassGaussNewid',):
                        save_sorted_incorrect_class_Gauss_newid = bool(int(arg))
                        print "Setting save_sorted_incorrect_class_Gauss_newid to %d" % \
                              save_sorted_incorrect_class_Gauss_newid
                    elif opt in ('--ComputeSlowFeaturesNewidAcrossNet',):
                        compute_slow_features_newid_across_net = int(arg)
                        print "Setting compute_slow_features_newid_across_net to %d" % \
                              compute_slow_features_newid_across_net
                    elif opt in ('--UseFilter',):
                        use_filter = arg
                        print "Setting use_filter to", use_filter
                    elif opt in ('--kNN_k',):
                        kNN_k = int(arg)
                        print "Setting kNN_k to", kNN_k
                    elif opt in ('--EnableKNN',):
                        enable_kNN = bool(int(arg))
                        print "Setting enable_kNN to", enable_kNN
                    elif opt in ('--EnableNCC',):
                        enable_NCC = bool(int(arg))
                        print "Setting enable_NCC to", enable_NCC
                    elif opt in ('--EnableGC',):
                        enable_GC = bool(int(arg))
                        print "Setting enable_GC to", enable_GC
                    elif opt in ('--SaveSubimagesTrainingSupplementaryInfo',):
                        save_images_training_supplementary_info = arg
                        print "Setting save_images_training_supplementary_info to",
                        print save_images_training_supplementary_info
                    elif opt in ('--EstimateExplainedVarWithInverse',):
                        estimate_explained_var_with_inverse = bool(int(arg))
                        print "Setting estimate_explained_var_with_inverse to", estimate_explained_var_with_inverse
                    elif opt in ('--EstimateExplainedVarWithKNN_k',):
                        estimate_explained_var_with_kNN_k = int(arg)
                        print "Setting estimate_explained_var_with_kNN_k to", estimate_explained_var_with_kNN_k
                    elif opt in ('--EstimateExplainedVarWithKNNLinApp_k',):
                        estimate_explained_var_with_kNN_lin_app_k = int(arg)
                        print "Setting estimate_explained_var_with_kNN_lin_app_k to %d" % \
                              estimate_explained_var_with_kNN_lin_app_k
                    elif opt in ('--EstimateExplainedVarLinGlobal_N',):
                        estimate_explained_var_linear_global_N = int(arg)
                        print "Setting estimate_explained_var_linear_global_N to %d" % \
                              estimate_explained_var_linear_global_N
                    elif opt in ('--AddNormalizationNode',):
                        add_normalization_node = bool(int(arg))
                        print "Setting add_normalization_node to", add_normalization_node
                    elif opt in ('--MakeLastPCANodeWhithening',):
                        make_last_PCA_node_whithening = bool(int(arg))
                        print "Setting make_last_PCA_node_whithening to", make_last_PCA_node_whithening
                    elif opt in ('--FeatureCutOffLevel',):
                        feature_cut_off_level = float(arg)
                        print "Setting feature_cut_off_level to", feature_cut_off_level
                    elif opt in ('--ExportDataToLibsvm',):
                        export_data_to_libsvm = bool(int(arg))
                        print "Setting export_data_to_libsvm to", export_data_to_libsvm
                    elif opt in ('--IntegerLabelEstimation',):
                        integer_label_estimation = bool(int(arg))
                        print "Setting integer_label_estimation to", integer_label_estimation
                    elif opt in ('--CumulativeScores',):
                        cumulative_scores = bool(int(arg))
                        print "Setting cumulative_scores to", cumulative_scores
                    elif opt in ('--FeaturesResidualInformation',):
                        features_residual_information = int(arg)
                        print "Setting features_residual_information to", features_residual_information
                    elif opt in ('--ComputeInputInformation',):
                        compute_input_information = bool(int(arg))
                        print "Setting compute_input_information to", compute_input_information
                    elif opt in ('--SleepM',):
                        minutes_sleep = float(arg)
                        if minutes_sleep >= 0:
                            print "Sleeping for %f minutes..." % minutes_sleep
                            time.sleep(minutes_sleep * 60)
                            print "... and awoke"
                        else:
                            print "Sleeping until execution in cuicuilco queue"
                            t_wa = time.time()
                            lock = LockFile(cuicuilco_lock_file)
                            pid = os.getpid()
                            print "process pid is:", pid
                            # Add this process to the queue
                            print "adding process to queue..."
                            lock.acquire()
                            q = open(cuicuilco_queue, "a")
                            q.write("%d\n" % pid)
                            q.close()
                            lock.release()
                            served = False
                            while not served:
                                lock.acquire()
                                q = open(cuicuilco_queue, "r")
                                next_pid = int(q.readline())
                                print "top of queue:", next_pid,
                                q.close()
                                lock.release()

                                if next_pid == pid:
                                    print "our turn in queue"
                                    served = True
                                else:
                                    print "sleeping 60 seconds"
                                    time.sleep(60)  # sleep for 10 seconds
                            t_wb = time.time()
                            print "process is executing now. Total waiting time: %f min" % ((t_wb - t_wa) / 60.0)
                    elif opt in ('--DatasetForDisplayTrain',):
                        dataset_for_display_train = int(arg)
                        print "Setting dataset_for_display_train to", dataset_for_display_train
                    elif opt in ('--DatasetForDisplayNewid',):
                        dataset_for_display_newid = int(arg)
                        print "Setting dataset_for_display_newid to", dataset_for_display_newid
                    elif opt in ('--GraphExactLabelLearning',):
                        graph_exact_label_learning = bool(int(arg))
                        print "Setting graph_exact_label_learning to", graph_exact_label_learning
                    elif opt in ('--OutputInsteadOfSVM2',):
                        output_instead_of_SVM2 = bool(int(arg))
                        print "Setting output_instead_of_SVM2 to", output_instead_of_SVM2
                    elif opt in ('--NumberTargetLabels',):
                        number_of_target_labels_per_orig_label = int(arg)
                        print "Setting number_of_target_labels_per_orig_label to %d" % \
                              number_of_target_labels_per_orig_label
                    elif opt in ('--ConfusionMatrix',):
                        confusion_matrix = bool(int(arg))
                        print "Setting confusion_matrix to", confusion_matrix
                    elif opt in ('--MapDaysToYears',):
                        convert_labels_days_to_years = bool(int(arg))
                        print "Setting convert_labels_days_to_years to", convert_labels_days_to_years
                    elif opt in ('--AddNoiseToSeenid',):
                        add_noise_to_seenid = bool(int(arg))
                        print "Setting add_noise_to_seenid to", add_noise_to_seenid
                    elif opt in ('--ClipSeenidNewid',):
                        clip_seenid_newid_to_training = bool(int(arg))
                        print "Setting clip_seenid_newid_to_training to", clip_seenid_newid_to_training
                    elif opt in ('--HierarchicalNetwork',):
                        name_default_network = arg
                        print "Setting default_network to", name_default_network
                        DefaultNetwork = available_networks[name_default_network]
                    elif opt in ('--ExperimentalDataset',):
                        name_default_experiment = arg
                        print "Setting name_default_experiment to", name_default_experiment
                        DefaultExperimentalDataset = available_experiments[name_default_experiment]
                    elif opt in ('--SFAGCReducedDim',):
                        sfa_gc_reduced_dim = int(arg)
                        print "Setting sfa_gc_reduced_dim to", sfa_gc_reduced_dim
                    elif opt in ('--ObjectiveLabel',):
                        objective_label = int(arg)
                        print "Setting objective_label to", objective_label
                    elif opt in ('--help',):
                        txt = \
                            """Cuicuilco: displaying help information
    Usage: python cuicuilco_run.py [OPTION]...
    Executes a single run of the Cuicuilco framework. 
    The following global variables must be specified on beforehand (integer values):
        CUICUILCO_TUNING_PARAMETER        (value of the tuning parameter used by the datasets)
        CUICUILCO_EXPERIMENT_SEED         (seed used for the dataset radomizations)
        CUICUILCO_IMAGE_LOADING_NUM_PROC  (max number of processes used by MKL)
    The options below may be used:
        **General options
            --EnableDisplay={1/0}. Enables the graphical interface
            --ExperimentalDataset={ParamsRAgeFunc/ParamsMNISTFunc/ParamsRTransXYScaleFunc/...}. Selects a particular
                dataset
            --HierarchicalNetwork={voidNetwork1L/PCANetwork1L/u08expoNetworkU11L/...}. Selects a particular network
            --NumFeaturesSup=N. Specifies the number of output features N used in the supervised step
            --SkipFeaturesSup=S. Specifies number of output features S that are skipped (ignored)
            --SleepM=M. Specifies a delay before Cuicuilco starts loading the dataset. (useful to prevent memory or
                processor clogging).
                    if M>0 the current Cuicuilco process is paused for M minutes
                    if M=0 there is no delay
                    if M<0 the program joins a waiting list (specified by the lock file named queue_cuicuilco.txt),
                        sleeps until its turn is reached, and deletes itself from the list after the labels/classes
                        have been estimated
        **Network options
            --AddNormalizationNode={1/0} Adds a normalization node at the end of the network
            --MakeLastPCANodeWhithening={1/0} Changes the last PCANode into a WhitheningNode
            --FeatureCutOffLevel=f Trims the feature values between -f and f
        **Cache options
            --CacheAvailable={1/0} Specifies whether any type of cache might be available
            --NetworkCacheReadDir=directory Specifies a directory used to load previously trained networks
            --NetworkCacheWriteDir=directory Specifies a directory used to save trained networks
            --LoadNetworkNumber=M Loads the Mth network in cache instead of training a new network
            --AskNetworkLoading={1/0} If the option is enabled, Cuicuilco requests in the command 
                                      line the number of the network to be loaded
            --NodeCacheReadDir=directory Specifies a directory used to search for nodes trained previously on the same
                data and parameters (can significantly speed up network training)
            --NodeCacheWriteDir=directory Specifies a directory where trained nodes are saved
        **Feature options
            --AddNoiseToSeenid={1/0} Adds noise to the data used to train the supervised step
            --ClipSeenidNewid={1/0} Trims the range of the data used to train the supervised step and the test data
                according to the range of the training data of the network
        **Supervised step options  
            --EnableLR={1/0} Enables linear regression (OLS) as supervised step
            --EnableKNN={1/0} Enables k-nearest neighbors (kNN) as supervised step
            --kNN_k=k Sets the value of k if kNN is enabled
            --EnableNCC={1/0} Enables a nearest centroid classifier as supervised step
            --EnableGC={1/0} Enables a Gaussian classifier as supervised step
            --EnableSVM={1/0} Enables a support vector machine as supervised step (requires libsvm)
            --SVM_gamma=gamma Sets the value of gamma if SVM is enabled (RBF, multiclass, one against one)
            --SVM_C=C Sets the value of C if SVM is enabled
        **Result options
            --SaveSubimagesTraining={1/0} Saves (a fraction of) the training images to disk (after data distortion and
                other operations)
            --SaveSubimagesTrainingSupplementaryInfo={Class/Label} If the option above is enabled, this option adds the
                correct label or class information to the image filenames
            --SaveAverageSubimageTraining={1/0} Saves the average training image to disk (after data distortion and
                other operations)
            --SaveSorted_AE_GaussNewid={1/0} Saves (a fraction of) the training images to disk ordered by the absolute
                error for label estimation
            --SaveSortedIncorrectClassGaussNewid={1/0} Saves (a fraction of) the training images to disk that were
                classified incorrectly
            --ExportDataToLibsvm={1/0} Saves the output features and labels in the format of libsvm
        **Options to control computation of explained variance (1-reconstruction error). 
            --EstimateExplainedVarWithInverse={1/0} Reconstructions are computed using flow.inverse
            --EstimateExplainedVarWithKNN_k=k If k>0 reconstructions are computed as the average of the k nearest
                neighbors
            --EstimateExplainedVarWithKNNLinApp_k=k If k>0 reconstructions are a linear average of the k nearest
                neighbors
            --EstimateExplainedVarLinGlobal_N=N Reconstructions are given by a linear model trained with N samples
                chosen randomly from the training data. If N=-1, all training samples are used.
        **Label estimation options
            --MapDaysToYears={1/0} Divides the ground-truth labels and label estimations by 365.242
            --IntegerLabelEstimation={1/0} Truncates all label estimations to integer values
            --CumulativeScores={1/0} Computes cumulative scores for test data
            --ConfusionMatrix={1/0} Computes the confusion matrix for test data
        **Exact label learning graph options
            --GraphExactLabelLearning={1/0} Computes an ELL graph based on the available labels
            --NumberTargetLabels=N Defines the number of target labels (if N>1 there N-1 auxiliary labels are created)
            --OutputInsteadOfSVM2={1/0} If the option is enabled, the network output replaces the SVM2 label estimation
        **Undocumented or in development options (consult the source code)
            --InputFilename=filename, --OutputFilename=filename,
            --SignalCacheReadDir=directory, --SignalCacheWriteDir=directory,
            --ClassifierCacheReadDir=directory, --ClassifierCacheWriteDir=directory,
            --EnableScheduler={1/0}, --NParallel=N, --UseFilter={1/0},
            --FeaturesResidualInformation=N, --ComputeInputInformation={1/0},
            --ComputeSlowFeaturesNewidAcrossNet={1/0}, --DatasetForDisplayTrain=N, --DatasetForDisplayNewid=N   
        **Other options
            --help Displays this help information
    """
                        print txt
                        quit()
                    else:
                        print "Argument not handled: ", opt
                        quit()
            except getopt.GetoptError as err:
                print "Error parsing the arguments: ", argv[1:]
                print "Error: ", err
                # print "option:", getopt.GetoptError.opt, "message:", getopt.GetoptError.msg
                sys.exit(2)


def main():
    global benchmark, num_features_to_append_to_input, reg_num_signals, use_full_sl_output, svm_gamma, \
        compute_input_information

    if enable_svm:
        import svm as libsvm

    if load_and_append_output_features_dir is None:
        num_features_to_append_to_input = 0

    if coherent_seeds:
        print "experimental_datasets.experiment_seed=", experiment_seed
        numpy.random.seed(experiment_seed + 12121212)

    if enable_scheduler and n_parallel > 1:
        scheduler = mdp.parallel.ThreadScheduler(n_threads=n_parallel)
    else:
        scheduler = None

    if features_residual_information <= 0 and compute_input_information:
        print "ignoring flag compute_input_information=%d because  features_residual_information=%d <= 0" % \
              (compute_input_information, features_residual_information)
        compute_input_information = False

    Parameters = DefaultExperimentalDataset
    Parameters.create()
    Network = DefaultNetwork

    # Specific code for setting up the ParamsNatural experiment (requires run-time computations)
    if Parameters == experimental_datasets.ParamsNatural and input_filename is not None:
        (magic_num, iteration, numSamples, rbm_sfa_numHid, sampleSpan) = read_binary_header("", input_filename)
        print "Iteration Number=%d," % iteration, "numSamples=%d" % numSamples, "rbm_sfa_numHid=%d," % rbm_sfa_numHid

        Parameters.sTrain.subimage_width = rbm_sfa_numHid / 8
        Parameters.sTrain.subimage_height = rbm_sfa_numHid / Parameters.sTrain.subimage_width
        Parameters.sTrain.name = "RBM Natural. 8x8 (exp 64=%d), iter %d, num_images %d" % (rbm_sfa_numHid, iteration,
                                                                                           Parameters.sTrain.num_images)
        Parameters.sSeenid.subimage_width = rbm_sfa_numHid / 8
        Parameters.sSeenid.subimage_height = rbm_sfa_numHid / Parameters.sSeenid.subimage_width
        Parameters.sSeenid.name = "RBM Natural. 8x8 (exp 64=%d), iter %d, num_images %d" % \
                                  (rbm_sfa_numHid, iteration, Parameters.sSeenid.num_images)
        Parameters.sNewid.subimage_width = rbm_sfa_numHid / 8
        Parameters.sNewid.subimage_height = rbm_sfa_numHid / Parameters.sNewid.subimage_width
        Parameters.sNewid.name = "RBM Natural. 8x8 (exp 64=%d), iter %d, num_images %d" % \
                                 (rbm_sfa_numHid, iteration, Parameters.sNewid.num_images)

        Parameters.sTrain.data_base_dir = Parameters.sSeenid.data_base_dir = Parameters.sNewid.data_base_dir = ""
        Parameters.sTrain.base_filename = Parameters.sSeenid.base_filename = Parameters.sNewid.base_filename = \
            input_filename

        if numSamples != 5000:
            er = "wrong number of Samples %d, 5000 were assumed" % numSamples
            raise Exception(er)

    # Cutoff for final network output
    min_cutoff = -1.0e200  # -10 # -30.0
    max_cutoff = 1.0e200  # 10 # 30.0

    enable_reduced_image_sizes = Parameters.enable_reduced_image_sizes
    reduction_factor = Parameters.reduction_factor
    print "reduction_factor=", reduction_factor
    hack_image_size = Parameters.hack_image_size
    enable_hack_image_size = Parameters.enable_hack_image_size

    if enable_reduced_image_sizes:
        #    reduction_factor = 2.0 # (the inverse of a zoom factor)
        Parameters.name += "_Resized images"
        # for iSeq in (Parameters.iTrain, Parameters.iSeenid, Parameters.iNewid):
        #     # iSeq.trans = iSeq.trans / 2
        #     pass

        for sSeq in (Parameters.sTrain, Parameters.sSeenid, Parameters.sNewid):
            print "sSeq", sSeq
            if isinstance(sSeq, list):
                for i, sSeq_vect in enumerate(sSeq):
                    print "sSeq_vect", sSeq_vect
                    if sSeq_vect is not None:  # is not None:
                        for j, sSeq_entry in enumerate(sSeq_vect):
                            if isinstance(sSeq_entry, system_parameters.ParamsDataLoading):
                                # TODO: Avoid code repetition, even though readability compromised
                                scale_sSeq(sSeq_entry, reduction_factor)
                            else:
                                er = "Unexpected data structure"
                                raise Exception(er)
            else:
                scale_sSeq(sSeq, reduction_factor)

    if coherent_seeds:
        numpy.random.seed(experiment_seed + 34343434)

    iTrain_set = Parameters.iTrain
    sTrain_set = Parameters.sTrain
    iTrain = take_0_k_th_from_2D_list(iTrain_set, k=dataset_for_display_train)
    sTrain = take_0_k_th_from_2D_list(sTrain_set, k=dataset_for_display_train)

    # take k=1? or choose from command line? NOPE. Take always first label (k=0). sSeq must compute proper classes for
    # chosen label anyway.
    # TODO: let the user choose objective_label through a command line argument
      # = 0, = 1, = 2, = 3
    if graph_exact_label_learning:
        if isinstance(iTrain_set, list):
            iTrain0 = iTrain_set[len(iTrain_set) - 1][0]
        else:
            iTrain0 = take_0_k_th_from_2D_list(iTrain_set, k=0)
        Q = iTrain0.num_images

        if len(iTrain0.correct_labels.shape) == 2:
            num_orig_labels = iTrain0.correct_labels.shape[1]
        else:
            num_orig_labels = 1
            iTrain0.correct_labels.reshape((-1, num_orig_labels))

        # number_of_target_labels_per_orig_label = 2 #1 or more for auxiliary labels
        if number_of_target_labels_per_orig_label >= 1:
            min_label = iTrain0.correct_labels.min(axis=0)
            max_label = iTrain0.correct_labels.max(axis=0)
            plain_labels = iTrain0.correct_labels.reshape((-1, num_orig_labels))
            num_samples = len(plain_labels)
            auxiliary_labels = numpy.zeros((num_samples, num_orig_labels * number_of_target_labels_per_orig_label))
            auxiliary_labels[:, 0:num_orig_labels] = plain_labels
            for i in range(1, number_of_target_labels_per_orig_label):
                auxiliary_labels[:, i * num_orig_labels:(i + 1) * num_orig_labels] = numpy.cos(
                    (plain_labels - min_label) * (1.0 + i) * numpy.pi / (max_label - min_label))
            print auxiliary_labels
        else:
            auxiliary_labels = iTrain0.correct_labels.reshape((-1, num_orig_labels))

        print "iTrain0.correct_labels.shape", iTrain0.correct_labels.shape
        orig_train_label_min = auxiliary_labels[:, objective_label].min()
        orig_train_label_max = auxiliary_labels[:, objective_label].max()

        orig_train_labels_mean = numpy.array(auxiliary_labels).mean(axis=0)
        orig_train_labels_std = numpy.array(auxiliary_labels).std(axis=0)
        orig_train_label_mean = orig_train_labels_mean[objective_label]
        orig_train_label_std = orig_train_labels_std[objective_label]

        orig_train_labels = auxiliary_labels
        orig_train_labels_mean = numpy.array(orig_train_labels).mean(axis=0)
        orig_train_labels_std = numpy.array(orig_train_labels).std(axis=0)
        train_feasible_labels = (orig_train_labels - orig_train_labels_mean) / orig_train_labels_std
        print "original feasible (perhaps correlated) label.T: ", train_feasible_labels.T

        if len(iTrain0.correct_labels.shape) == 2:
            iTrain0.correct_labels = iTrain0.correct_labels[:, objective_label].flatten()
            Parameters.iSeenid.correct_labels = Parameters.iSeenid.correct_labels[:, objective_label].flatten()
            Parameters.iNewid[0][0].correct_labels = Parameters.iNewid[0][0].correct_labels[:,
                                                     objective_label].flatten()

            iTrain0.correct_classes = iTrain0.correct_classes[:, objective_label].flatten()
            Parameters.iSeenid.correct_classes = Parameters.iSeenid.correct_classes[:, objective_label].flatten()
            Parameters.iNewid[0][0].correct_classes = Parameters.iNewid[0][0].correct_classes[:,
                                                      objective_label].flatten()

        node_weights = numpy.ones(Q)
        Gamma = ConstructGammaFromLabels(train_feasible_labels, node_weights, constant_deltas=False)
        print "Resulting Gamma is", Gamma
        Gamma = RemoveNegativeEdgeWeights(node_weights, Gamma)
        print "Removed negative weighs. Gamma=", Gamma
        edge_weights = Gamma

        if isinstance(sTrain_set, list):
            sTrain0 = sTrain_set[len(sTrain_set) - 1][0]
        else:
            sTrain0 = take_0_k_th_from_2D_list(sTrain_set, k=0)

        sTrain0.train_mode = "graph"
        sTrain0.node_weights = node_weights
        sTrain0.edge_weights = edge_weights

    print "sTrain=", sTrain

    iSeenid = Parameters.iSeenid
    sSeenid = Parameters.sSeenid

    if coherent_seeds:
        print "Setting coherent seed"
        numpy.random.seed(experiment_seed + 56565656)

    iNewid_set = Parameters.iNewid
    sNewid_set = Parameters.sNewid
    print "dataset_for_display_newid=", dataset_for_display_newid
    iNewid = take_0_k_th_from_2D_list(iNewid_set, k=dataset_for_display_newid)
    sNewid = take_0_k_th_from_2D_list(sNewid_set, k=dataset_for_display_newid)

    image_files_training = iTrain.input_files
    # print image_files_training

    num_images_training = num_images = iTrain.num_images

    seq_sets = sTrain_set
    seq = sTrain

    hack_image_sizes = [135, 90, 64, 32, 16]

    # hack_image_size = 64
    # enable_hack_image_size = True
    if enable_hack_image_size:
        print "changing the native image size (width and height) to: ", hack_image_size
        sSeq_force_image_size(sTrain_set, hack_image_size, hack_image_size)
        sSeq_force_image_size(sSeenid, hack_image_size, hack_image_size)
        sSeq_force_image_size(sNewid_set, hack_image_size, hack_image_size)

    subimage_shape, max_clip, signals_per_image, in_channel_dim = sSeq_getinfo_format(sTrain)
    
    # Filter used for loading images with transparent background
    # filter = generate_color_filter2((seq.subimage_height, seq.subimage_width))
    if use_filter == "ColoredNoise" or use_filter == "1":
        alpha = 4.0  # mask 1 / f^(alpha/2) => power 1/f^alpha
        my_filter = filter_colored_noise2D_imp((seq.subimage_height, seq.subimage_width), alpha)
    # back_type = None
    # filter = None
    elif use_filter == "None" or (use_filter is None) or (use_filter == "0"):
        my_filter = None
    else:
        print "Unknown filter: ", use_filter
        quit()
    sTrain.filter = my_filter
    sSeenid.filter = my_filter
    sNewid.filter = my_filter

    network_read_enabled = True  # and False
    if network_read_enabled and cache_available:
        network_read = cache.Cache(network_cache_read_dir, "")
    else:
        network_read = None

    network_saving_enabled = True  # and False
    if network_saving_enabled and cache_available and (network_cache_write_dir is not None):
        network_write = cache.Cache(network_cache_write_dir, "")
    else:
        network_write = None

    node_cache_read_enabled = True  # and False
    if node_cache_read_enabled and cache_available and (node_cache_read_dir is not None):
        node_cache_read = cache.Cache(node_cache_read_dir, "")
    else:
        node_cache_read = None

    signal_cache_read_enabled = True  # and False
    if signal_cache_read_enabled and cache_available and (signal_cache_read_dir is not None):
        signal_cache_read = cache.Cache(signal_cache_read_dir, "")
    else:
        signal_cache_read = None

    node_cache_write_enabled = True
    if node_cache_write_enabled and cache_available and (node_cache_write_dir is not None):
        node_cache_write = cache.Cache(node_cache_write_dir, "")
    else:
        node_cache_write = None

    signal_cache_write_enabled = True  # and False #or network_saving_enabled
    if signal_cache_write_enabled and cache_available and (signal_cache_write_dir is not None):
        signal_cache_write = cache.Cache(signal_cache_write_dir, "")
    else:
        signal_cache_write = None

    classifier_read_enabled = False
    if classifier_read_enabled and cache_available and (classifier_cache_read_dir is not None):
        classifier_read = cache.Cache(classifier_cache_read_dir, "")
    else:
        classifier_read = None

    classifier_saving_enabled = True  # and False
    if classifier_saving_enabled and cache_available and (classifier_cache_write_dir is not None):
        classifier_write = cache.Cache(classifier_cache_write_dir, "")
    else:
        classifier_write = None

    network_hashes_base_filenames = []
    if network_cache_read_dir and network_read:
        network_filenames = cache.find_filenames_beginning_with(network_cache_read_dir, "Network", recursion=False,
                                                                extension=".pckl")
        for i, network_filename in enumerate(network_filenames):
            network_base_filename = string.split(network_filename, sep=".")[0]
            network_hash = string.split(network_base_filename, sep="_")[-1]
            network_hashes_base_filenames.append((network_base_filename, network_hash))
    else:
        network_hashes_base_filenames = []

    network_hashes_base_filenames.sort(lambda x, y: cmp(x[1], y[1]))

    print "%d networks found:" % len(network_hashes_base_filenames)
    for i, (network_filename, network_hash) in enumerate(network_hashes_base_filenames):
        print "[%d]" % i, network_filename

    network_filename = None
    if len(network_hashes_base_filenames) > 0 and (ask_network_loading or load_network_number is not None):
        #    flow, layers, benchmark, Network, subimages_train, sl_seq_training =
        # cache.unpickle_from_disk(network_filenames[-1])
        if ask_network_loading or load_network_number is None:
            selected_network = int(raw_input("Please select a network (-1=Train new network):"))
        else:
            print "Network selected from program parameters: ", load_network_number
            selected_network = load_network_number

        if selected_network == -1:
            print "Selected: Train new network"
        else:
            print "Selected: Load Network", selected_network
            network_filename = network_hashes_base_filenames[selected_network][0]

    if network_filename is not None:
        network_base_filename = string.split(network_filename, sep=".")[0]
        network_hash = string.split(network_base_filename, sep="_")[-1]

        print "******************************************"
        print "Loading Trained Network and Display Data from Disk         "
        print "******************************************"

        print "network_cach_read_dir", network_cache_read_dir
        print "network_cach_write_dir", network_cache_write_dir
        print "network_filename:", network_filename
        print "network_basefilename:", network_base_filename
        print "network_hash:", network_hash

        # network_write.update_cache([flow, layers, benchmark, Network], None, network_base_dir, "Network"+
        # Network.name+"_ParName"+Parameters.name+"_"+network_hash, overwrite=True, use_hash=network_hash, verbose=True)
        # network_write.update_cache([iSeq, sSeq], None, network_base_dir, "iSeqsSeqData", overwrite=True,
        # use_hash=network_hash, verbose=True)
        # network_write.update_cache(subimages_train, None, network_base_dir, "TrainData", overwrite=True,
        # use_hash=network_hash, verbose=True)
        # network_write.update_cache(sl_seq_training, None, network_base_dir, "SLSeqData", overwrite=True,
        # use_hash=network_hash, verbose=True)

        #    flow, layers, benchmark, Network = cache.unpickle_from_disk(network_filename)
        flow, layers, benchmark, Network = network_read.load_obj_from_cache(None, "", network_base_filename,
                                                                            verbose=True)
        print "Done loading network: " + Network.name
        print flow
        #    quit()

        iTrain, sTrain = network_read.load_obj_from_cache(network_hash, network_cache_read_dir, "iTrainsTrainData",
                                                          verbose=True)
        print "Done loading iTrain sTrain data: " + sTrain.name

        block_size = sTrain.block_size
        train_mode = iTrain.train_mode
        print "Train mode is:", train_mode

        subimages_train = network_read.load_array_from_cache(network_hash, network_cache_read_dir, "TrainData",
                                                             verbose=True)
        print "Done loading subimages_train: ", subimages_train.shape

        sl_seq_training = network_read.load_array_from_cache(network_hash, network_cache_read_dir, "SLSeqData",
                                                             verbose=True)
        print "Done loading sl_seq_training: ", sl_seq_training.shape

    else:
        print "Generating Network..."
        # Usually true for voidNetwork1L, but might be also activated for other networks
        use_full_sl_output = False

        # Network.patch_network_for_RGB = True and False
        # Expand network output dimensions in case the original data is RGB or HOG02 features
        if (sTrain.convert_format == "RGB" or sTrain.convert_format == "HOG02") and Parameters.patch_network_for_RGB:
            if sTrain.convert_format == "RGB":
                #            factors = [3, 2, 1.5]
                factors = [2, 1.7, 1.5]
                print "Big Fail!", Network.patch_network_for_RGB
                quit()
            elif sTrain.convert_format == "HOG02":
                factors = [8, 4, 2]
            else:
                er = "unknown conversion factor in network correction for in_channel_dim"
                raise Exception(er)

            for i, layer in enumerate(Network.layers[0:3]):  # L0, Network.L1, Network.L2)):
                factor = factors[i]
                if layer is not None:
                    if layer.pca_out_dim is not None and layer.pca_out_dim >= 1:
                        layer.pca_out_dim = int(factor * layer.pca_out_dim)
                    if layer.red_out_dim is not None and layer.red_out_dim >= 1:
                        layer.red_out_dim = int(factor * layer.red_out_dim)
                    # What about ord? usually it keeps dimension the same, thus it is not specified
                    if layer.sfa_out_dim is not None and layer.sfa_out_dim >= 1:
                        layer.sfa_out_dim = int(factor * layer.sfa_out_dim)
            print "testing..."
            # quit()

        # Possibly skip some of the last layers of the network if the data resolution has been artificially reduced
        skip_layers = 0
        trim_network_layers = False  # trim_network_layers is obsolete, one must manually chose the proper network and
        # data version
        if trim_network_layers:
            if (hack_image_size == 8) and enable_hack_image_size:
                Network.L3 = None
                Network.L4 = None
                Network.L5 = None
                Network.L6 = None
                Network.L7 = None
                Network.L8 = None
                Network.L9 = None
                Network.L10 = None
                skip_layers = 8
            if (hack_image_size == 16) and enable_hack_image_size:
                Network.L5 = None
                Network.L6 = None
                Network.L7 = None
                Network.L8 = None
                Network.L9 = None
                Network.L10 = None
                skip_layers = 6
            if (hack_image_size == 32) and enable_hack_image_size:
                Network.L7 = None
                Network.L8 = None
                Network.L9 = None
                Network.L10 = None
                skip_layers = 4
            if (hack_image_size == 64 or hack_image_size == 72 or hack_image_size == 80 or hack_image_size == 95 or
                        hack_image_size == 96) and enable_hack_image_size:
                Network.L9 = None
                Network.L10 = None
                skip_layers = 2

        for l in Network.layers:
            print l
        print "SL=", skip_layers

        if skip_layers > 0:
            for i, layer in enumerate(Network.layers):
                if i + skip_layers < len(Network.layers) and (layer is not None) and (Network.layers[i +
                        skip_layers] is not None):
                    if layer.pca_node_class == mdp.nodes.SFANode:
                        print "FIX PCA%d" % i
                        if Network.layers[i + skip_layers].pca_node_class == mdp.nodes.SFANode:
                            if "sfa_expo" in Network.layers[i + skip_layers].pca_args:
                                layer.pca_args["sfa_expo"] = Network.layers[i + skip_layers].pca_args["sfa_expo"]
                            if "pca_expo" in Network.layers[i + skip_layers].pca_args:
                                layer.pca_args["pca_expo"] = Network.layers[i + skip_layers].pca_args["pca_expo"]
                        else:
                            if "sfa_expo" in Network.layers[i + skip_layers].sfa_args:
                                layer.pca_args["sfa_expo"] = Network.layers[i + skip_layers].sfa_args["sfa_expo"]
                            if "pca_expo" in Network.layers[i + skip_layers].sfa_args:
                                layer.pca_args["pca_expo"] = Network.layers[i + skip_layers].sfa_args["pca_expo"]
                    if layer.ord_node_class == mdp.nodes.SFANode:
                        if "sfa_expo" in Network.layers[i + skip_layers].ord_args:
                            layer.ord_args["sfa_expo"] = Network.layers[i + skip_layers].ord_args["sfa_expo"]
                        if "pca_expo" in Network.layers[i + skip_layers].ord_args:
                            layer.ord_args["pca_expo"] = Network.layers[i + skip_layers].ord_args["pca_expo"]
                    if layer.red_node_class == mdp.nodes.SFANode:
                        layer.red_args["sfa_expo"] = Network.layers[i + skip_layers].red_args["sfa_expo"]
                        layer.red_args["pca_expo"] = Network.layers[i + skip_layers].red_args["pca_expo"]
                    if layer.sfa_node_class == mdp.nodes.SFANode:
                        print "FixSFA %d" % i
                        if "sfa_expo" in Network.layers[i + skip_layers].sfa_args:
                            layer.sfa_args["sfa_expo"] = Network.layers[i + skip_layers].sfa_args["sfa_expo"]
                        if "pca_expo" in Network.layers[i + skip_layers].sfa_args:
                            layer.sfa_args["pca_expo"] = Network.layers[i + skip_layers].sfa_args["pca_expo"]

        if skip_layers > 0:
            Network.layers = Network.layers[:-skip_layers]
        # Network.layers = []
        # for layer in [Network.L0, Network.L1, Network.L2, Network.L3, Network.L4, Network.L5, Network.L6, Network.L7,
        # Network.L8, Network.L9, Network.L10 ]:
        #    if layer is None:
        #        break
        #    else:
        #        Network.layers.append(layer)

        print "sfa_expo and pca_expo across the network:"
        for i, layer in enumerate(Network.layers):
            if "sfa_expo" in Network.layers[i].pca_args:
                print "pca_args[%d].sfa_expo=" % i, Network.layers[i].pca_args["sfa_expo"]
            if "pca_expo" in Network.layers[i].pca_args:
                print "pca_args[%d].pca_expo=" % i, Network.layers[i].pca_args["pca_expo"]
            if "sfa_expo" in Network.layers[i].sfa_args:
                print "sfa_args[%d].sfa_expo=" % i, Network.layers[i].sfa_args["sfa_expo"]
            if "pca_expo" in Network.layers[i].sfa_args:
                print "sfa_args[%d].pca_expo=" % i, Network.layers[i].sfa_args["pca_expo"]

        if make_last_PCA_node_whithening and (hack_image_size == 16) and enable_hack_image_size and (Network.L4 is not
                                                                                                             None):
            if Network.L4.sfa_node_class == mdp.nodes.PCANode:
                Network.L4.sfa_node_class = mdp.nodes.WhiteningNode
                Network.L4.sfa_out_dim = 50
        if make_last_PCA_node_whithening and (hack_image_size == 32) and enable_hack_image_size and (Network.L6 is not
                                                                                                             None):
            if Network.L6.sfa_node_class == mdp.nodes.PCANode:
                Network.L6.sfa_node_class = mdp.nodes.WhiteningNode
                Network.L6.sfa_out_dim = 100
        if make_last_PCA_node_whithening and (hack_image_size == 64 or hack_image_size == 80) and \
                enable_hack_image_size and Network.L8 is not None:
            if Network.L8.sfa_node_class == mdp.nodes.PCANode:
                Network.L8.sfa_node_class = mdp.nodes.WhiteningNode

        load_subimages_train_signal_from_cache = True
        enable_select_train_signal = True

        subimages_train_signal_in_cache = False

        if signal_cache_read and load_subimages_train_signal_from_cache and False:
            print "Looking for subimages_train in cache..."

            info_beginning_filename = "subimages_info"
            subimages_info_filenames = cache.find_filenames_beginning_with(network_cache_read_dir,
                                                                           info_beginning_filename, recursion=False,
                                                                           extension=".pckl")
            print "The following possible training sequences were found:"
            if len(subimages_info_filenames) > 0:
                for i, info_filename in enumerate(subimages_info_filenames):
                    info_base_filename = string.split(info_filename, sep=".")[0]  # Remove extension
                    (iTrainInfo, sTrainInfo) = subimages_info = signal_cache_read.load_obj_from_cache(base_dir="/",
                                                                        base_filename=info_base_filename, verbose=True)
                    print "%d: %s, with %d images of width=%d, height=%d" % (i, iTrainInfo.name, iTrainInfo.num_images,
                                                                             sTrainInfo.subimage_width,
                                                                             sTrainInfo.subimage_height)

                if enable_select_train_signal:
                    selected_train_sequence = int(raw_input("Please select a training sequence (-1=Reload new data):"))
                else:
                    selected_train_sequence = 0
                print "Training sequence %d was selected" % selected_train_sequence

                if selected_train_sequence >= 0:
                    info_filename = subimages_info_filenames[selected_train_sequence]
                    info_base_filename = string.split(info_filename, sep=".")[0]  # Remove extension
                    (iTrain_set, sTrain_set) = signal_cache_read.load_obj_from_cache(base_dir="/",
                                                                                     base_filename=info_base_filename,
                                                                                     verbose=True)

                    iTrain = take_0_k_th_from_2D_list(iTrain_set, dataset_for_display_train)
                    sTrain = take_0_k_th_from_2D_list(sTrain_set, dataset_for_display_train)

                    signal_base_filename = string.replace(info_base_filename, "subimages_info", "subimages_train")

                    if signal_cache_read.is_splitted_file_in_filesystem(base_dir="/",
                                                                        base_filename=signal_base_filename):
                        print "Subimages train signal found in cache..."
                        subimages_train = signal_cache_read.load_array_from_cache(base_dir="/",
                                                                                  base_filename=signal_base_filename,
                                                                                  verbose=True)
                        subimages_train_signal_in_cache = True
                        print "Subimages train signal loaded from cache with shape: ",
                        print subimages_train.shape
                        if signal_cache_write:
                            subimages_train_hash = cache.hash_object(subimages_train).hexdigest()
                    else:
                        print "Subimages training signal UNEXPECTEDLY NOT FOUND in cache:", signal_base_filename
                        quit()

                        # Conversion from sSeq to data_sets (array or function), param_sets
                        # Actually train_func_sets
        train_data_sets, train_params_sets = convert_sSeq_to_funcs_params_sets(seq_sets, verbose=False)
        if load_and_append_output_features_dir is not None:
            training_data_hash = cache.hash_object((iTrain, sTrain)).hexdigest()
            training_data_hash = "0"
            print "loading output features (training data) from dir: ", load_and_append_output_features_dir, \
                "and hash:", training_data_hash
            additional_features_training = cache.unpickle_array(base_dir=load_and_append_output_features_dir,
                                                                base_filename="output_features_training_TrainingD" +
                                                                              training_data_hash)
            additional_features_training = 100000 * \
                                           additional_features_training[:, 0:num_features_to_append_to_input] + \
                                           10000.0 * numpy.random.normal(size=(iTrain.num_images,
                                                                               num_features_to_append_to_input))
            train_data_sets = system_parameters.expand_dataset_with_additional_features(train_data_sets,
                                                                                        additional_features_training)

        print "now building network"
        train_data_sets, train_params_sets = network_builder.expand_iSeq_sSeq_Layer_to_Network(train_data_sets,
                                                                                               train_params_sets,
                                                                                               Network)

        print "train_params_sets=", train_params_sets
        print "dataset_for_display_train=", dataset_for_display_train
        print "calling take_first_02D"
        params_node = take_0_k_th_from_2D_list(train_params_sets, k=dataset_for_display_train)

        block_size = params_node["block_size"]
        train_mode = params_node["train_mode"]

        print "calling take_first_02D again"
        train_func = take_0_k_th_from_2D_list(train_data_sets, k=dataset_for_display_train)
        print "train_func=", train_func
        if coherent_seeds:
            numpy.random.seed(experiment_seed + 222222)
        subimages_train = train_func()
        # TODO: Here add pre computed features!!!??? Or do this during experiment definition???

        print "subimages_train[0,0]=%0.40f" % subimages_train[0, 0]

        # Avoid double extraction of data from files
        if isinstance(train_data_sets, list) and len(train_data_sets) >= 1:
            if isinstance(train_data_sets[0], list) and len(train_data_sets[0]) >= 1 and len(
                    train_data_sets[0]) > dataset_for_display_train:
                print "Correcting double loading"
                func = train_data_sets[0][dataset_for_display_train]
                print "substituting func=", func, "for loaded data"
                for i in range(len(train_data_sets)):
                    for j in range(len(train_data_sets[i])):
                        print "train_data_sets[%d][%d]=" % (i, j), train_data_sets[i][j]
                        if train_data_sets[i][j] is func:
                            print "Correction done"  # fdssf
                            train_data_sets[i][j] = subimages_train

        # TODO: Support train signal chache for generalized training
        if signal_cache_write and (subimages_train_signal_in_cache is False) and False:
            print "Caching Train Signal..."
            subimages_ndim = subimages_train.shape[1]
            subimages_time = str(int(time.time()))
            iTrain_hash = cache.hash_object(iTrain_sets).hexdigest()
            sTrain_hash = cache.hash_object(sTrain_sets).hexdigest()
            subimages_base_filename = "subimages_train_%s_%s_%s_%s" % (
            (subimages_ndim, subimages_time, iTrain_hash, sTrain_hash))
            subimages_train_hash = signal_cache_write.update_cache(subimages_train,
                                                                   base_filename=subimages_base_filename,
                                                                   overwrite=True, verbose=True)
            subimages_info = (iTrain, sTrain)
            subimages_info_filename = "subimages_info_%s_%s_%s_%s" % \
                                      (subimages_ndim, subimages_time, iTrain_hash, sTrain_hash)
            subimages_info_hash = signal_cache_write.update_cache(subimages_info, base_filename=subimages_info_filename,
                                                                  overwrite=True, verbose=True)

        t1 = time.time()
        print seq.num_images, "Training Images loaded in %0.3f s" % ((t1 - t0))
        # benchmark.append(("Load Info and Training Images", t1-t0))

        save_images_training_base_dir = "/local/tmp/escalafl/Alberto/saved_images_training"
        if save_subimages_training:
            print "saving images to directory:", save_images_training_base_dir
            decimate = 1  # 10
            for i, x in enumerate(subimages_train):
                if i % decimate == 0:
                    if seq.convert_format == "L":
                        im_raw = numpy.reshape(x, (seq.subimage_width, seq.subimage_height))
                        im = scipy.misc.toimage(im_raw, mode=seq.convert_format)
                    elif seq.convert_format == "RGB":
                        im_raw = numpy.reshape(x, (seq.subimage_width, seq.subimage_height, 3))
                        im = scipy.misc.toimage(im_raw, mode=seq.convert_format)
                    else:
                        im_raw = numpy.reshape(x, (seq.subimage_width, seq.subimage_height))
                        im = scipy.misc.toimage(im_raw, mode="L")

                    if save_images_training_supplementary_info is None:
                        filename = "image%05d.png" % (i / decimate)
                        # quit()
                    elif save_images_training_supplementary_info == "Class":
                        filename = "image%05d_gt%05d.png" % (i / decimate, iTrain.correct_classes[i])
                    elif save_images_training_supplementary_info == "Label":
                        filename = "image%05d_gt%05.5f.png" % (i / decimate, iTrain.correct_labels[i])
                        # quit()
                    else:
                        er = "Incorrect value of save_images_training_supplementary_info:" + str(
                            save_images_training_supplementary_info)
                        raise Exception(er)
                    fullname = os.path.join(save_images_training_base_dir, filename)
                    im.save(fullname)
                    # print "done, finishing"
                    # quit()

        if save_average_subimage_training:
            average_subimage_training = subimages_train.mean(axis=0)
            if seq.convert_format == "L":
                average_subimage_training = average_subimage_training.reshape(sTrain.subimage_height,
                                                                              sTrain.subimage_width)
            elif seq.convert_format == "RGB":
                average_subimage_training = average_subimage_training.reshape(sTrain.subimage_height,
                                                                              sTrain.subimage_width, 3)
            else:
                average_subimage_training = average_subimage_training.reshape(sTrain.subimage_height,
                                                                              sTrain.subimage_width)
            print "average_subimage_training.shape=", average_subimage_training.shape,
            print "seq.convert_format=", seq.convert_format
            average_subimage_training_I = scipy.misc.toimage(average_subimage_training, mode=seq.convert_format)
            average_subimage_training_I.save("average_image_trainingRGB.jpg", mode=seq.convert_format)
            # print "done, finishing"
            # quit()

        print "******************************************"
        print "Creating hierarchy through network_builder"
        print "******************************************"
        # TODO: more primitive but potentially powerful flow specification here should be possible
        flow, layers, benchmark = network_builder.create_network(Network, sTrain.subimage_width, sTrain.subimage_height,
                                                                benchmark=benchmark,
                                                                in_channel_dim=in_channel_dim,
                                                                num_features_appended_to_input = \
                                                                    num_features_to_append_to_input)

        print "Making sure the first switchboard does not add any noise (noise added during image loading)"
        if isinstance(flow[0], mdp.nodes.PInvSwitchboard):
            flow[0].additive_noise_std = 0.0

        # For display purposes we alter here the image shape artificially...
        # TODO: Improve this logic overall... shape should be really the shape, and in_channel_dim should be used
        print subimage_shape
        if in_channel_dim in [1, 3]:
            subimage_shape = subimage_shape
        else:
            print "Patching subimage_shape for display purposes"
            subimage_shape = (subimage_shape[0], subimage_shape[1] * in_channel_dim)


            #    add_normalization_node = True
        if add_normalization_node:
            normalization_node = mdp.nodes.NormalizeNode()
            flow += normalization_node

        print "flow=", flow
        print len(flow)
        for node in flow:
            print "Node: ", node, "out_dim=", node.output_dim, "input_dim", node.input_dim
        # quit()
        print "*****************************"
        print "Training hierarchy ..."
        print "*****************************"

        subimages_p = subimages = subimages_train
        # DEFINE TRAINING TYPE. SET ONE OF THE FOLLOWING VARIABLES TO TRUE
        # Either use special (most debugged and efficient) or storage_iterator (saves memory)
        special_training = True
        iterator_training = False
        storage_iterator_training = False

        if special_training is True:
            ttrain0 = time.time()
            # Think: maybe training_signal cache can become unnecessary if flow.train is intelligent enough to look
            # for the signal in the cache without loading it???
            # Use same seed as before for data loading... hope the results are the same. Nothing should be done before
            # data generation to ensure this!!!
            if coherent_seeds:
                numpy.random.seed(experiment_seed + 222222)

                # TODO: f train_data_sets is func() or [[func()]], use instead loaded images!!!!
            sl_seq = sl_seq_training = flow.special_train_cache_scheduler_sets(train_data_sets,
                                                                               params_sets=train_params_sets,
                                                                               verbose=True, benchmark=benchmark,
                                                                               node_cache_read=node_cache_read,
                                                                               signal_cache_read=signal_cache_read,
                                                                               node_cache_write=node_cache_write,
                                                                               signal_cache_write=signal_cache_write,
                                                                               scheduler=scheduler,
                                                                               n_parallel=n_parallel,
                                                                               immediate_stop_training=True)
            print "sl_seq is", sl_seq

            ttrain1 = time.time()
            print "Network trained (specialized way) in time %0.3f s" % (ttrain1 - ttrain0)
            if benchmark is not None:
                benchmark.append(("Network training  (specialized way)", ttrain1 - ttrain0))
        elif iterator_training is True:
            ttrain0 = time.time()
            # WARNING, introduce smart way of computing chunk_sizes
            input_iter = cache.chunk_iterator(subimages_p, 4, block_size, continuous=False)

            flow.iterator_train(input_iter, block_size, continuous=True)
            sl_seq = sl_seq_training = flow.execute(subimages_p)

            ttrain1 = time.time()
            print "Network trained (iterator way) in time %0.3f s" % ((ttrain1 - ttrain0))
            if benchmark is not None:
               benchmark.append(("Network training (iterator way)", ttrain1 - ttrain0))
        elif storage_iterator_training is True:
            ttrain0 = time.time()
            # Warning: introduce smart way of computing chunk_sizes
            #    input_iter = chunk_iterator(subimages_p, 15 * 15 / block_size, block_size, continuous=False)
            input_iter = cache.chunk_iterator(subimages_p, 4, block_size, continuous=False)

            #    sl_seq = sl_seq_training = flow.iterator_train(input_iter)
            # WARNING, continuous should not always be true
            flow.storage_iterator_train(input_iter, "/local/tmp/escalafl/simulations/gender", "trainseq", block_size,
                                        continuous=True)

            output_iterator = cache.UnpickleLoader2(path="/local/tmp/escalafl/simulations/gender",
                                                    basefilename="trainseq" + "_N%03d" % (len(flow) - 1))

            sl_seq = sl_seq_training = cache.from_iter_to_array(output_iterator, continuous=False,
                                                                block_size=block_size, verbose=0)
            del output_iterator

            ttrain1 = time.time()
            print "Network trained (storage iterator way) in time %0.3f s" % ((ttrain1 - ttrain0))
            if benchmark is not None:
                benchmark.append(("Network training (storage iterator way)", ttrain1 - ttrain0))
        else:
            ttrain0 = time.time()
            flow.train(subimages_p)
            y = flow.execute(subimages_p[0:1])  # stop training
            sl_seq = sl_seq_training = flow.execute(subimages_p)
            ttrain1 = time.time()
            print "Network trained (MDP way) in time %0.3f s" % (ttrain1 - ttrain0)
            if benchmark is not None:
                benchmark.append(("Network training (MDP way)", ttrain1 - ttrain0))

        nodes_in_flow = len(flow)
        last_sfa_node = flow[nodes_in_flow - 1]
        if isinstance(last_sfa_node, mdp.hinet.CloneLayer) or \
                isinstance(last_sfa_node, mdp.hinet.Layer):
            last_sfa_node = last_sfa_node.nodes[0]

        if isinstance(last_sfa_node, mdp.nodes.SFANode):
            if iTrain.correct_labels[0:10].mean() <= iTrain.correct_labels[-10:].mean():
                start_negative = True
            else:
                start_negative = False
            sl_seq = sl_seq_training = more_nodes.sfa_pretty_coefficients(last_sfa_node, sl_seq_training,
                                                                          start_negative=start_negative)
            # default start_negative=True #WARNING!
        else:
            print "SFA coefficients not made pretty, last node was not SFA!!!"

        print "Since training is finished, making sure the switchboards do not add any noise from now on"
        for node in flow:
            if isinstance(node, mdp.nodes.PInvSwitchboard):
                node.additive_noise_std = 0.0

        print "Executing for display purposes (subimages_train)..."
        #    For display purposes ignore output of training, and concentrate on display signal:
        sl_seq = sl_seq_training = flow.execute(subimages_train)
        if feature_cut_off_level > 0.0:
            sl_seq = sl_seq_training = cutoff(sl_seq_training, min_cutoff, max_cutoff)

        network_hash = str(int(time.time()))
        #    network_filename = "Network_" + network_hash + ".pckl"

        if network_write:
            print "Saving flow, layers, benchmark, Network ..."
            # update cache is not adding the hash to the filename,so we add it manually
            network_write.update_cache(flow, None, network_cache_write_dir,
                                       "JustFlow" + sTrain.name + "_" + network_hash, overwrite=True,
                                       use_hash=network_hash, verbose=True)
            network_write.update_cache(layers, None, network_cache_write_dir,
                                       "JustLayers" + sTrain.name + "_" + network_hash, overwrite=True,
                                       use_hash=network_hash, verbose=True)
            network_write.update_cache(benchmark, None, network_cache_write_dir,
                                       "JustBenchmark" + sTrain.name + "_" + network_hash, overwrite=True,
                                       use_hash=network_hash, verbose=True)
            network_write.update_cache(Network, None, network_cache_write_dir,
                                       "JustNetwork" + sTrain.name + "_" + network_hash, overwrite=True,
                                       use_hash=network_hash, verbose=True)

            network_write.update_cache([flow, layers, benchmark, Network], None, network_cache_write_dir,
                                       "Network" + Network.name + "_ParName" + Parameters.name + "_" + network_hash,
                                       overwrite=True, use_hash=network_hash, verbose=True)
            network_write.update_cache([iTrain, sTrain], None, network_cache_write_dir,
                                       "iTrainsTrainData" + "_" + network_hash, overwrite=True, use_hash=network_hash,
                                       verbose=True)
            network_write.update_cache(subimages_train, None, network_cache_write_dir, "TrainData" + "_" + network_hash,
                                       overwrite=True, use_hash=network_hash, verbose=True)
            network_write.update_cache(sl_seq_training, None, network_cache_write_dir, "SLSeqData" + "_" + network_hash,
                                       overwrite=True, use_hash=network_hash, verbose=True)
            # obj, obj_data=None, base_dir = None, base_filename=None, overwrite=True, use_hash=None, verbose=True

        if signal_cache_write:
            print "Caching sl_seq_training  Signal... (however, it's never read!)"
            signal_ndim = sl_seq_training.shape[1]
            signal_time = str(int(time.time()))
            flow_hash = cache.hash_object(flow).hexdigest()
            # TODO: Add subimages_train_hash to signal filename. Compute according to first data/parameter sets
            signal_base_filename = "sfa_signal_ndim%s_time%s_flow%s" % ((signal_ndim, signal_time, flow_hash))
            signal_cache_write.update_cache(sl_seq_training, base_filename=signal_base_filename, overwrite=True,
                                            verbose=True)

        if save_output_features_dir is not None:
            print "saving output features (training data)"
            training_data_hash = cache.hash_object((iTrain, sTrain)).hexdigest()
            cache.pickle_array(sl_seq_training, base_dir=save_output_features_dir,
                               base_filename="output_features_training_TrainingD" + training_data_hash, overwrite=True,
                               verbose=True)

    num_pixels_per_image = numpy.ones(subimage_shape, dtype=int).sum()
    print "taking into account objective_label=%d" % objective_label
    if len(iTrain.correct_labels.shape) == 2:
        print "correction..."
        iTrain.correct_labels = iTrain.correct_labels[:, objective_label].flatten()
        Parameters.iSeenid.correct_labels = Parameters.iSeenid.correct_labels[:, objective_label].flatten()
        Parameters.iNewid[0][0].correct_labels = Parameters.iNewid[0][0].correct_labels[:, objective_label].flatten()

        iTrain.correct_classes = iTrain.correct_classes[:, objective_label].flatten()
        Parameters.iSeenid.correct_classes = Parameters.iSeenid.correct_classes[:, objective_label].flatten()
        Parameters.iNewid[0][0].correct_classes = Parameters.iNewid[0][0].correct_classes[:, objective_label].flatten()
    print "iTrain.correct_classes=", iTrain.correct_classes
    print "iTrain.correct_labels=", iTrain.correct_labels
    print "Parameters.iNewid[0][0].correct_classes", Parameters.iNewid[0][0].correct_classes
    print "Parameters.iNewid[0][0].correct_labels", Parameters.iNewid[0][0].correct_labels
    print "Parameters.iSeenid.correct_classes=", Parameters.iSeenid.correct_classes
    print "Parameters.iSeenid.correct_labels=", Parameters.iSeenid.correct_labels
    print "done"

    if coherent_seeds:
        numpy.random.seed(experiment_seed + 333333)

    subimages_p = subimages = subimages_train
    sl_seq_training = sl_seq_training[:, skip_num_signals:]
    sl_seq = sl_seq_training

    print "subimages_train[0,0]=%0.40f" % subimages_train[0, 0]

    print "Done creating / training / loading network"
    y = flow.execute(subimages_p[0:1])
    print y.shape
    more_nodes.describe_flow(flow)
    more_nodes.display_eigenvalues(flow, mode="Average")  # mode="FirstNodeInLayer", "Average", "All"

    hierarchy_out_dim = y.shape[1] - skip_num_signals

    print "hierarchy_out_dim (real output data) =", hierarchy_out_dim
    print "last_node_out_dim=", flow[-1].output_dim
    if isinstance(flow[-1], (mdp.hinet.Layer, mdp.hinet.CloneLayer)):
        print "last_Layer_node_out_dim=", flow[-1][0].output_dim
        print "last node of network is a layer! is this a mistake?"
    if hierarchy_out_dim != flow[-1].output_dim:
        print "error!!! hierarchy_out_dim != flow[-1].output_dim"
        print "Perhaps caused by enable_reduced_image_sizes=True or enable_hack_image_size=True?"
        quit()

    results = system_parameters.ExperimentResult()
    results.name = Parameters.name
    results.network_name = Network.name
    results.layers_name = []
    for lay in layers:
        results.layers_name.append(lay.name)
    results.iTrain = iTrain
    results.sTrain = sTrain
    results.iSeenid = iSeenid
    results.sSeenid = sSeenid
    results.iNewid = iNewid
    results.sNewid = sNewid

    print "Computing typical delta, eta values for Train SFA Signal"
    t_delta_eta0 = time.time()
    results.typical_delta_train, results.typical_eta_train = sfa_libs.comp_typical_delta_eta(sl_seq_training,
                                                                                             block_size, num_reps=200,
                                                                                             training_mode=iTrain.train_mode)
    results.brute_delta_train = sfa_libs.comp_delta_normalized(sl_seq_training)
    results.brute_eta_train = sfa_libs.comp_eta(sl_seq_training)
    t_delta_eta1 = time.time()
    print "typical_delta_train=", results.typical_delta_train
    print "typical_delta_train[0:31].sum()=", results.typical_delta_train[0:31].sum()

    print "computed delta/eta in %0.3f ms" % ((t_delta_eta1 - t_delta_eta0) * 1000.0)
    if benchmark is not None:
        benchmark.append(("Computation of delta, eta values for Train SFA Signal", t_delta_eta1 - t_delta_eta0))

    print "Setting correct classes and labels for the Classifier/Regression, Train SFA Signal"
    correct_classes_training = iTrain.correct_classes
    print "correct_classes_training=", correct_classes_training
    correct_labels_training = iTrain.correct_labels

    if convert_labels_days_to_years:
        correct_labels_training = correct_labels_training / DAYS_IN_A_YEAR
        if integer_label_estimation:
            correct_labels_training = (correct_labels_training + 0.0006).astype(
                int) * 1.0  # Otherwise MSE computation is erroneous!

    print "Loading test images, seen ids..."
    t_load_images0 = time.time()

    print "LOADING KNOWNID TEST INFORMATION"
    image_files_seenid = iSeenid.input_files
    num_images_seenid = iSeenid.num_images
    block_size_seenid = iSeenid.block_size
    seq = sSeenid

    if coherent_seeds:
        numpy.random.seed(experiment_seed + 444444)

    if seq.input_files == "LoadBinaryData00":
        subimages_seenid = load_natural_data(seq.data_base_dir, seq.base_filename, seq.samples, verbose=False)
    elif seq.input_files == "LoadRawData":
        subimages_seenid = load_raw_data(seq.data_base_dir, seq.base_filename, input_dim=seq.input_dim, dtype=seq.dtype,
                                         select_samples=seq.samples, verbose=False)
    else:
        # subimages_seenid = experimental_datasets.load_data_from_sSeq(seq)
        subimages_seenid = seq.load_data(seq)

    if load_and_append_output_features_dir is not None:
        seenid_data_hash = cache.hash_object((iSeenid, sSeenid)).hexdigest()
        seenid_data_hash = "0"
        print "loading output features (seenid data) from dir: ", load_and_append_output_features_dir,
        print "and hash:", seenid_data_hash
        additional_features_seenid = cache.unpickle_array(base_dir=load_and_append_output_features_dir,
                                                          base_filename="output_features_training_SeenidD" +
                                                                        seenid_data_hash)
        additional_features_seenid = 100000 * additional_features_seenid[:, 0:num_features_to_append_to_input] + \
                                     0.0 * numpy.random.normal(size=(iSeenid.num_images,
                                                                     num_features_to_append_to_input))
        print additional_features_seenid.shape
        print subimages_seenid.shape
        subimages_seenid = numpy.concatenate((subimages_seenid, additional_features_seenid), axis=1)
        # train_data_sets = system_parameters.expand_dataset_with_additional_features(train_data_sets,
        # additional_features_training)

    t_load_images1 = time.time()
    print num_images_seenid, " Images loaded in %0.3f s" % (t_load_images1 - t_load_images0)

    t_exec0 = time.time()
    print "Execution over known id testing set..."
    print "Input Signal: Known Id test images"
    sl_seq_seenid = flow.execute(subimages_seenid)
    sl_seq_seenid = sl_seq_seenid[:, skip_num_signals:]

    if feature_cut_off_level > 0.0:
        print "before cutoff sl_seq_seenid= ", sl_seq_seenid
        sl_seq_seenid = cutoff(sl_seq_seenid, min_cutoff, max_cutoff)

    sl_seq_training_min = sl_seq_training.min(axis=0)
    sl_seq_training_max = sl_seq_training.max(axis=0)

    if clip_seenid_newid_to_training:
        print "clipping sl_seq_seenid"
        sl_seq_seenid_min = sl_seq_seenid.min(axis=0)
        sl_seq_seenid_max = sl_seq_seenid.max(axis=0)
        print "sl_seq_training_min=", sl_seq_training_min
        print "sl_seq_training_max=", sl_seq_training_max
        print "sl_seq_seenid_min=", sl_seq_seenid_min
        print "sl_seq_seenid_max=", sl_seq_seenid_max
        sl_seq_seenid = numpy.clip(sl_seq_seenid, sl_seq_training_min, sl_seq_training_max)

    if add_noise_to_seenid:  # Using uniform noise due to its speed over normal noise
        noise_amplitude = (3 ** 0.5) * 0.5  # standard deviation 0.00005
        print "adding noise to sl_seq_seenid, with noise_amplitude:", noise_amplitude
        sl_seq_seenid += noise_amplitude * numpy.random.uniform(-1.0, 1.0, size=sl_seq_seenid.shape)

    t_exec1 = time.time()
    print "Execution over Known Id in %0.3f s" % (t_exec1 - t_exec0)

    if save_output_features_dir is not None:
        print "saving output features (seenid data)"
        seenid_data_hash = cache.hash_object((iSeenid, sSeenid)).hexdigest()
        cache.pickle_array(sl_seq_seenid, base_dir=save_output_features_dir,
                           base_filename="output_features_training_SeenidD" + seenid_data_hash, overwrite=True,
                           verbose=True)

    print "Computing typical delta, eta values for Seen Id SFA Signal"
    t_delta_eta0 = time.time()
    results.typical_delta_seenid, results.typical_eta_seenid = sfa_libs.comp_typical_delta_eta(sl_seq_seenid,
                                                                                               iSeenid.block_size,
                                                                                               num_reps=200,
                                                                                               training_mode=iSeenid.train_mode)
    print "sl_seq_seenid=", sl_seq_seenid
    results.brute_delta_seenid = sfa_libs.comp_delta_normalized(sl_seq_seenid)
    results.brute_eta_seenid = sfa_libs.comp_eta(sl_seq_seenid)
    t_delta_eta1 = time.time()
    print "typical_delta_seenid=", results.typical_delta_seenid
    print "typical_delta_seenid[0:31].sum()=", results.typical_delta_seenid[0:31].sum()
    print "typical_eta_seenid=", results.typical_eta_seenid
    print "brute_delta_seenid=", results.brute_delta_seenid
    print "computed delta/eta in %0.3f ms" % ((t_delta_eta1 - t_delta_eta0) * 1000.0)

    print "Setting correct labels/classes data for seenid"
    correct_classes_seenid = iSeenid.correct_classes
    correct_labels_seenid = iSeenid.correct_labels
    correct_labels_seenid_real = correct_labels_seenid.copy()

    if convert_labels_days_to_years:
        correct_labels_seenid_real = correct_labels_seenid_real / DAYS_IN_A_YEAR
        correct_labels_seenid /= DAYS_IN_A_YEAR
        if integer_label_estimation:
            correct_labels_seenid = (correct_labels_seenid + 0.0006).astype(int)

    # t8 = time.time()
    t_classifier_train0 = time.time()
    print "*** Training Classifier/Regression"

    # W
    if use_full_sl_output or reg_num_signals == 0:
        results.reg_num_signals = reg_num_signals = sl_seq_training.shape[1]
    # else:
    #    results.reg_num_signals = reg_num_signals = 3  #42

    # cf_sl = sl_seq_training
    # cf_num_samples = cf_sl.shape[0]
    # cf_correct_labels = correct_labels_training
    # cf_correct_classes = iTrain.correct_classes
    # cf_spacing = cf_block_size = iTrain.block_size

    cf_sl = sl_seq_seenid
    cf_num_samples = cf_sl.shape[0]
    cf_correct_labels = correct_labels_seenid_real
    cf_correct_classes = iSeenid.correct_classes
    cf_spacing = cf_block_size = iSeenid.block_size

    all_classes = numpy.unique(cf_correct_classes)

    avg_labels = more_nodes.compute_average_labels_for_each_class(cf_correct_classes, cf_correct_labels)

    if reg_num_signals <= 128 and (Parameters.analysis is not False) and enable_NCC:
        enable_ncc_cfr = True
    else:
        enable_ncc_cfr = False

    if reg_num_signals <= 128 and (Parameters.analysis is not False) and enable_GC:
        enable_ccc_Gauss_cfr = True
        enable_gc_cfr = True
    else:
        enable_ccc_Gauss_cfr = False
        enable_gc_cfr = False

    if reg_num_signals <= 64 and (Parameters.analysis is not False) and enable_kNN:
        enable_kNN_cfr = True
    else:
        enable_kNN_cfr = False

    if reg_num_signals <= 120 and (Parameters.analysis is not False) and enable_svm:  # and False:
        enable_svm_cfr = True
    else:
        enable_svm_cfr = False

    if reg_num_signals <= 8192 and (Parameters.analysis is not False) and enable_lr:
        enable_lr_cfr = True
    else:
        enable_lr_cfr = False

    if enable_ncc_cfr:
        print "Training Classifier/Regression NCC"
        ncc_node = mdp.nodes.NearestMeanClassifier()
        ncc_node.train(x=cf_sl[:, 0:reg_num_signals], labels=cf_correct_classes)
        ncc_node.stop_training()

    if enable_ccc_Gauss_cfr:
        print "Training Classifier/Regression GC..."
        print "unique labels =", numpy.unique(cf_correct_classes)
        print "len(unique_labels)=", len(numpy.unique(cf_correct_classes))
        print "cf_sl[0,:]=", cf_sl[0, :]
        print "cf_sl[1,:]=", cf_sl[1, :]
        print "cf_sl[2,:]=", cf_sl[2, :]
        print "cf_sl[3,:]=", cf_sl[3, :]
        print "cf_sl[4,:]=", cf_sl[4, :]
        print "cf_sl[5,:]=", cf_sl[5, :]

        #    for c in numpy.unique(cf_correct_classes):
        #        print "class %f appears %d times"%(c, (cf_correct_classes==c).sum())
        #        print "mean(cf_sl[c=%d,:])="%c, cf_sl[cf_correct_classes==c, :].mean(axis=0)
        #        print "std(cf_sl[c=%d,:])="%c, cf_sl[cf_correct_classes==c, :].std(axis=0)

        GC_node = mdp.nodes.SFA_GaussianClassifier(reduced_dim=sfa_gc_reduced_dim, verbose=True)
        GC_node.train(x=cf_sl[:, 0:reg_num_signals],
                      labels=cf_correct_classes)  # Functions for regression use class values!!!
        GC_node.stop_training()
        GC_node.avg_labels = avg_labels

        t_classifier_train1 = time.time()
        if benchmark is not None:
            benchmark.append(("Training Classifier/Regression GC", t_classifier_train1 - t_classifier_train0))

    t_classifier_train1 = time.time()

    if enable_kNN_cfr:
        print "Training Classifier/Regression kNN, for k=%d..." % kNN_k
        kNN_node = mdp.nodes.KNNClassifier(k=kNN_k)
        kNN_node.train(x=cf_sl[:, 0:reg_num_signals], labels=cf_correct_classes)
        kNN_node.stop_training()

        t_classifier_train1b = time.time()
        if benchmark is not None:
            benchmark.append(("Training Classifier/Regression kNN", t_classifier_train1b - t_classifier_train1))

    t_classifier_train1 = time.time()

    if cf_block_size is not None:
        if isinstance(cf_block_size, (numpy.float, numpy.float64, numpy.int)):
            num_blocks = cf_sl.shape[0] / cf_block_size
        else:
            num_blocks = len(cf_block_size)
    else:
        num_blocks = cf_sl.shape[0]

    if enable_svm_cfr:
        print "Training SVM..."

        params = {"C": svm_C, "gamma": svm_gamma, "nu": 0.6, "eps": 0.0001}
        svm_node = mdp.nodes.LibSVMClassifier(kernel="RBF", classifier="C_SVC", params=params, probability=True)
        data_mins, data_maxs = svm_compute_range(cf_sl[:, 0:reg_num_signals])
        svm_node.train(svm_scale(cf_sl[:, 0:reg_num_signals], data_mins, data_maxs, svm_min, svm_max),
                       cf_correct_classes)
        if svm_gamma == 0:
            svm_gamma = 1.0 / (num_blocks)
        svm_node.stop_training()

    if enable_lr_cfr:
        print "Training LR..."
        lr_node = mdp.nodes.LinearRegressionNode(with_bias=True, use_pinv=False)
        lr_node.train(cf_sl[:, 0:reg_num_signals], cf_correct_labels.reshape((cf_sl.shape[0], 1)))
        lr_node.stop_training()

    if classifier_write and enable_ccc_Gauss_cfr:
        print "Saving Gaussian Classifier"
        cf_sl_hash = cache.hash_array(cf_sl).hexdigest()
        # update cache is not adding the hash to the filename,so we add it manually
        classifier_filename = "GaussianClassifier_NetName" + Network.name + "iTrainName" + iTrain.name + "_NetH" + \
                              network_hash + "_CFSlowH" + cf_sl_hash + "_NumSig%03d" % reg_num_signals + "_L" + \
                              str(objective_label)
        classifier_write.update_cache(GC_node, None, None, classifier_filename, overwrite=True, verbose=True)

    ############################################################
    # ###TODO: make classifier cash work!
    # ###TODO: review eban_svm model & implementation! beat normal svm!

    print "Executing/Executed over training set..."
    print "Input Signal: Training Data"
    subimages_training = subimages
    num_images_training = num_images

    print "Classification/Regression over training set..."
    t_class0 = time.time()

    if enable_ncc_cfr:
        print "ncc classify..."
        classes_ncc_training = numpy.array(ncc_node.label(sl_seq_training[:, 0:reg_num_signals]))
        labels_ncc_training = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_ncc_training)
        print classes_ncc_training
    else:
        classes_ncc_training = labels_ncc_training = numpy.zeros(num_images_training)

    if enable_ccc_Gauss_cfr:
        print "GC classify..."
        classes_Gauss_training = numpy.array(GC_node.label(sl_seq_training[:, 0:reg_num_signals]))
        labels_Gauss_training = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels,
                                                                          classes_Gauss_training)

        regression_Gauss_training = GC_node.regression(sl_seq_training[:, 0:reg_num_signals], avg_labels)
        regressionMAE_Gauss_training = GC_node.regressionMAE(sl_seq_training[:, 0:reg_num_signals], avg_labels)
        probs_training = GC_node.class_probabilities(sl_seq_training[:, 0:reg_num_signals])

        softCR_Gauss_training = GC_node.softCR(sl_seq_training[:, 0:reg_num_signals], correct_classes_training)
    else:
        classes_Gauss_training = labels_Gauss_training = regression_Gauss_training = regressionMAE_Gauss_training = numpy.zeros(
            num_images_training)
        probs_training = numpy.zeros((num_images_training, 2))
        softCR_Gauss_training = 0.0

    if enable_kNN_cfr:
        print "kNN classify... (k=%d)" % kNN_k
        classes_kNN_training = numpy.array(kNN_node.label(sl_seq_training[:, 0:reg_num_signals]))
        labels_kNN_training = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_kNN_training)
    else:
        classes_kNN_training = labels_kNN_training = numpy.zeros(num_images_training)

    skip_svm_training = False

    if enable_svm_cfr and skip_svm_training is False:
        print "SVM classify..."
        classes_svm_training = svm_node.label(
            svm_scale(sl_seq_training[:, 0:reg_num_signals], data_mins, data_maxs, svm_min, svm_max))
        regression_svm_training = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels,
                                                                            classes_svm_training)
        regression2_svm_training = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels,
                                                                             classes_svm_training)
        regression3_svm_training = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels,
                                                                             classes_svm_training)
    else:
        classes_svm_training = regression_svm_training = regression2_svm_training = regression3_svm_training = \
            numpy.zeros(num_images_training)

    if enable_lr_cfr:
        print "LR execute..."
        regression_lr_training = lr_node.execute(sl_seq_training[:, 0:reg_num_signals]).flatten()
    else:
        regression_lr_training = numpy.zeros(num_images_training)

    if output_instead_of_SVM2:
        regression2_svm_training = (sl_seq_training[:, 0] * orig_train_label_std) + orig_train_label_mean
        print "Applying cutoff to the label estimations for LR and Linear Scaling (SVM2)"
        regression2_svm_training = cutoff(regression2_svm_training, orig_train_label_min, orig_train_label_max)
        regression_lr_training = cutoff(regression_lr_training, orig_train_label_min, orig_train_label_max)

    print "Classification of training data: ", labels_kNN_training
    t_classifier_train2 = time.time()

    print "Classifier trained in time %0.3f s" % (t_classifier_train1 - t_classifier_train0)
    print "Training Images Classified in time %0.3f s" % (t_classifier_train2 - t_classifier_train1)
    if benchmark is not None:
        benchmark.append("Classification of Training Images", (t_classifier_train2 - t_classifier_train1) )

    t_class1 = time.time()
    print "Classification/Regression over Training Set in %0.3f s" % (t_class1 - t_class0)

    if integer_label_estimation:
        print "Making all label estimations for training data integer numbers"
        if convert_labels_days_to_years:
            labels_ncc_training = labels_ncc_training.astype(int)
            regression_Gauss_training = regression_Gauss_training.astype(int)
            regressionMAE_Gauss_training = regressionMAE_Gauss_training.astype(int)
            labels_kNN_training = labels_kNN_training.astype(int)
            regression_svm_training = regression_svm_training.astype(int)
            regression2_svm_training = regression2_svm_training.astype(int)
            regression3_svm_training = regression3_svm_training.astype(int)
            regression_lr_training = regression_lr_training.astype(int)
        else:
            labels_ncc_training = numpy.rint(labels_ncc_training)
            regression_Gauss_training = numpy.rint(regression_Gauss_training)
            regressionMAE_Gauss_training = numpy.rint(regressionMAE_Gauss_training)
            labels_kNN_training = numpy.rint(labels_kNN_training)
            regression_svm_training = numpy.rint(regression_svm_training)
            regression2_svm_training = numpy.rint(regression2_svm_training)
            regression3_svm_training = numpy.rint(regression3_svm_training)
            regression_lr_training = numpy.rint(regression_lr_training)
        print "regressionMAE_Gauss_training[0:5]=", regressionMAE_Gauss_training[0:5]

    t_class0 = time.time()
    if enable_ncc_cfr:
        print "NCC classify..."
        classes_ncc_seenid = numpy.array(ncc_node.label(sl_seq_seenid[:, 0:reg_num_signals]))
        labels_ncc_seenid = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_ncc_seenid)
        print classes_ncc_seenid
    else:
        classes_ncc_seenid = labels_ncc_seenid = numpy.zeros(num_images_seenid)

    if enable_ccc_Gauss_cfr:
        classes_Gauss_seenid = numpy.array(GC_node.label(sl_seq_seenid[:, 0:reg_num_signals]))
        labels_Gauss_seenid = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_Gauss_seenid)

        regression_Gauss_seenid = GC_node.regression(sl_seq_seenid[:, 0:reg_num_signals], avg_labels)
        regressionMAE_Gauss_seenid = GC_node.regressionMAE(sl_seq_seenid[:, 0:reg_num_signals], avg_labels)
        probs_seenid = GC_node.class_probabilities(sl_seq_seenid[:, 0:reg_num_signals])
        softCR_Gauss_seenid = GC_node.softCR(sl_seq_seenid[:, 0:reg_num_signals], correct_classes_seenid)
    else:
        classes_Gauss_seenid = labels_Gauss_seenid = regression_Gauss_seenid = regressionMAE_Gauss_seenid = numpy.zeros(
            num_images_seenid)
        probs_seenid = numpy.zeros((num_images_seenid, 2))
        softCR_Gauss_seenid = 0.0

    if enable_kNN_cfr:
        print "kNN classify... (k=%d)" % kNN_k
        classes_kNN_seenid = numpy.array(kNN_node.label(sl_seq_seenid[:, 0:reg_num_signals]))
        labels_kNN_seenid = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_kNN_seenid)
    else:
        classes_kNN_seenid = labels_kNN_seenid = numpy.zeros(num_images_seenid)

    if enable_svm_cfr:
        classes_svm_seenid = svm_node.label(
            svm_scale(sl_seq_seenid[:, 0:reg_num_signals], data_mins, data_maxs, svm_min, svm_max))
        regression_svm_seenid = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_svm_seenid)
        regression2_svm_seenid = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_svm_seenid)
        regression3_svm_seenid = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_svm_seenid)
    else:
        classes_svm_seenid = regression_svm_seenid = regression2_svm_seenid = regression3_svm_seenid = numpy.zeros(
            num_images_seenid)

    if enable_lr_cfr:
        regression_lr_seenid = lr_node.execute(sl_seq_seenid[:, 0:reg_num_signals]).flatten()
    else:
        regression_lr_seenid = numpy.zeros(num_images_seenid)

    if output_instead_of_SVM2:
        regression2_svm_seenid = (sl_seq_seenid[:, 0] * orig_train_label_std) + orig_train_label_mean
        print "Applying cutoff to the label estimations for LR and Linear Scaling (SVM2)"
        regression2_svm_seenid = cutoff(regression2_svm_seenid, orig_train_label_min, orig_train_label_max)
        regression_lr_seenid = cutoff(regression_lr_seenid, orig_train_label_min, orig_train_label_max)

    print "labels_kNN_seenid.shape=", labels_kNN_seenid.shape

    # correct_labels_seenid = wider_1Darray(numpy.arange(iSeenid.MIN_GENDER, iSeenid.MAX_GENDER, iSeenid.GENDER_STEP),
    # iSeenid.block_size)
    print "correct_labels_seenid.shape=", correct_labels_seenid.shape

    t_class1 = time.time()
    print "Classification/Regression over Seen Id in %0.3f s" % (t_class1 - t_class0)

    if integer_label_estimation:
        print "Making all label estimations for seenid data integer numbers"
        if convert_labels_days_to_years:
            labels_ncc_seenid = labels_ncc_seenid.astype(int)
            regression_Gauss_seenid = regression_Gauss_seenid.astype(int)
            regressionMAE_Gauss_seenid = regressionMAE_Gauss_seenid.astype(int)
            labels_kNN_seenid = labels_kNN_seenid.astype(int)
            regression_svm_seenid = regression_svm_seenid.astype(int)
            regression2_svm_seenid = regression2_svm_seenid.astype(int)
            regression3_svm_seenid = regression3_svm_seenid.astype(int)
            regression_lr_seenid = regression_lr_seenid.astype(int)
        else:
            labels_ncc_seenid = numpy.rint(labels_ncc_seenid)
            regression_Gauss_seenid = numpy.rint(regression_Gauss_seenid)
            regressionMAE_Gauss_seenid = numpy.rint(regressionMAE_Gauss_seenid)
            labels_kNN_seenid = numpy.rint(labels_kNN_seenid)
            regression_svm_seenid = numpy.rint(regression_svm_seenid)
            regression2_svm_seenid = numpy.rint(regression2_svm_seenid)
            regression3_svm_seenid = numpy.rint(regression3_svm_seenid)
            regression_lr_seenid = numpy.rint(regression_lr_seenid)
        print "regressionMAE_Gauss_seenid[0:5]=", regressionMAE_Gauss_seenid[0:5]

    # t10 = time.time()
    t_load_images0 = time.time()
    print "Loading test images, new ids..."

    if coherent_seeds:
        numpy.random.seed(experiment_seed + 555555)

    image_files_newid = iNewid.input_files
    num_images_newid = iNewid.num_images
    block_size_newid = iNewid.block_size
    seq = sNewid

    if seq.input_files == "LoadBinaryData00":
        subimages_newid = load_natural_data(seq.data_base_dir, seq.base_filename, seq.samples, verbose=False)
    elif seq.input_files == "LoadRawData":
        subimages_newid = load_raw_data(seq.data_base_dir, seq.base_filename, input_dim=seq.input_dim, dtype=seq.dtype,
                                        select_samples=seq.samples, verbose=False)
    else:
        subimages_newid = seq.load_data(seq)

    if load_and_append_output_features_dir is not None:
        newid_data_hash = cache.hash_object((iNewid, sNewid)).hexdigest()
        newid_data_hash = "0"
        print "loading output features (newid data) from dir: ", load_and_append_output_features_dir,
        print "and hash:", newid_data_hash
        additional_features_newid = cache.unpickle_array(base_dir=load_and_append_output_features_dir,
                                                         base_filename="output_features_training_TestD" +
                                                                       newid_data_hash)
        additional_features_newid = 100000 * additional_features_newid[:, 0:num_features_to_append_to_input]  #
        subimages_newid = numpy.concatenate((subimages_newid, additional_features_newid), axis=1)

    t_load_images1 = time.time()
    t11 = time.time()
    print num_images_newid, " Images loaded in %0.3f s" % (t_load_images1 - t_load_images0)

    t_exec0 = time.time()
    print "Execution over New Id testing set..."
    print "Input Signal: New Id test images"
    sl_seq_newid = flow.execute(subimages_newid)
    sl_seq_newid = sl_seq_newid[:, skip_num_signals:]
    if feature_cut_off_level > 0.0:
        sl_seq_newid = cutoff(sl_seq_newid, min_cutoff, max_cutoff)

    if clip_seenid_newid_to_training:
        print "clipping sl_seq_newid"
        sl_seq_newid_min = sl_seq_newid.min(axis=0)
        sl_seq_newid_max = sl_seq_newid.max(axis=0)
        print "sl_seq_training_min=", sl_seq_training_min
        print "sl_seq_training_max=", sl_seq_training_max
        print "sl_seq_newid_min=", sl_seq_newid_min
        print "sl_seq_newid_max=", sl_seq_newid_max
        sl_seq_newid = numpy.clip(sl_seq_newid, sl_seq_training_min, sl_seq_training_max)

    corr_factor = 1.0
    print corr_factor
    sl_seq_newid[:, 0:reg_num_signals] = sl_seq_newid[:, 0:reg_num_signals] * corr_factor

    t_exec1 = time.time()
    print "Execution over New Id in %0.3f s" % ((t_exec1 - t_exec0))

    if save_output_features_dir is not None:
        print "saving output features (test data)"
        testing_data_hash = cache.hash_object((iNewid, sNewid)).hexdigest()
        cache.pickle_array(sl_seq_newid, base_dir=save_output_features_dir,
                           base_filename="output_features_training_TestD" + testing_data_hash, overwrite=True,
                           verbose=True)

    t_class0 = time.time()

    correct_classes_newid = iNewid.correct_classes
    correct_labels_newid = iNewid.correct_labels

    if convert_labels_days_to_years:
        if correct_labels_newid.mean() < 200:
            print "correct_labels_newid appears to be already year values (mean %f)" % correct_labels_newid.mean()
        else:
            print "converting correct_labels_newid from days to years"
            correct_labels_newid = correct_labels_newid / DAYS_IN_A_YEAR

        if integer_label_estimation:
            if (correct_labels_newid - correct_labels_newid.astype(int)).mean() < 0.01:
                print "correct_labels_newid appears to be already integer, preserving its value"
            else:
                print "correct_labels_newid seem to be real values, converting them to years"
                correct_labels_newid = (correct_labels_newid + 0.0006).astype(int)
    print "correct_labels_newid=", correct_labels_newid

    if enable_ncc_cfr:
        print "NCC classify..."
        classes_ncc_newid = numpy.array(ncc_node.label(sl_seq_newid[:, 0:reg_num_signals]))
        labels_ncc_newid = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_ncc_newid)
        print classes_ncc_newid
    else:
        classes_ncc_newid = labels_ncc_newid = numpy.zeros(num_images_newid)

    if enable_ccc_Gauss_cfr:
        classes_Gauss_newid = numpy.array(GC_node.label(sl_seq_newid[:, 0:reg_num_signals]))
        labels_Gauss_newid = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_Gauss_newid)

        regression_Gauss_newid = GC_node.regression(sl_seq_newid[:, 0:reg_num_signals], avg_labels)
        regressionMAE_Gauss_newid = GC_node.regressionMAE(sl_seq_newid[:, 0:reg_num_signals], avg_labels)
        probs_newid = GC_node.class_probabilities(sl_seq_newid[:, 0:reg_num_signals])
        softCR_Gauss_newid = GC_node.softCR(sl_seq_newid[:, 0:reg_num_signals], correct_classes_newid)
    else:
        classes_Gauss_newid = labels_Gauss_newid = regression_Gauss_newid = regressionMAE_Gauss_newid = numpy.zeros(
            num_images_newid)
        probs_newid = numpy.zeros((num_images_newid, 2))
        softCR_Gauss_newid = 0.0

    if enable_kNN_cfr:
        print "kNN classify... (k=%d)" % kNN_k
        classes_kNN_newid = numpy.array(kNN_node.label(sl_seq_newid[:, 0:reg_num_signals]))
        labels_kNN_newid = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_kNN_newid)
    else:
        classes_kNN_newid = labels_kNN_newid = numpy.zeros(num_images_newid)

    if enable_svm_cfr:
        classes_svm_newid = svm_node.label(
            svm_scale(sl_seq_newid[:, 0:reg_num_signals], data_mins, data_maxs, svm_min, svm_max))
        regression_svm_newid = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_svm_newid)
        regression2_svm_newid = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_svm_newid)
        regression3_svm_newid = more_nodes.map_class_numbers_to_avg_label(all_classes, avg_labels, classes_svm_newid)

        probs_training[0, 10] = 1.0
        probs_newid[0, 10] = 1.0
        probs_seenid[0, 10] = 1.0
    else:
        classes_svm_newid = regression_svm_newid = regression2_svm_newid = regression3_svm_newid = numpy.zeros(
            num_images_newid)

    if enable_lr_cfr:
        regression_lr_newid = lr_node.execute(sl_seq_newid[:, 0:reg_num_signals]).flatten()
    else:
        regression_lr_newid = numpy.zeros(num_images_newid)

    if output_instead_of_SVM2:
        regression2_svm_newid = (sl_seq_newid[:, 0] * orig_train_label_std) + orig_train_label_mean
        print "Applying cutoff to the label estimations for LR and Linear Scaling (SVM2)"
        regression2_svm_newid = cutoff(regression2_svm_newid, orig_train_label_min, orig_train_label_max)
        regression_lr_newid = cutoff(regression_lr_newid, orig_train_label_min, orig_train_label_max)

    t_class1 = time.time()
    print "Classification/Regression over New Id in %0.3f s" % ((t_class1 - t_class0))

    if integer_label_estimation:
        #     print "WARNING, ADDING A BIAS OF -0.5 TO ESTIMATION OF NEWID ONLY!!!"
        #     regression_Gauss_newid += -0.5
        #     regressionMAE_Gauss_newid += -0.5
        print "Making all label estimations for newid data integer numbers"
        if convert_labels_days_to_years:  # 5.7 should be mapped to 5 years because age estimation is exact (days)
            labels_ncc_newid = labels_ncc_newid.astype(int)
            regression_Gauss_newid = regression_Gauss_newid.astype(int)
            regressionMAE_Gauss_newid = regressionMAE_Gauss_newid.astype(int)
            labels_kNN_newid = labels_kNN_newid.astype(int)
            regression_svm_newid = regression_svm_newid.astype(int)
            regression2_svm_newid = regression2_svm_newid.astype(int)
            regression3_svm_newid = regression3_svm_newid.astype(int)
            regression_lr_newid = regression_lr_newid.astype(int)
        else:  # 5.7 should be mapped to 6 years because age estimation is already based on years
            labels_ncc_newid = numpy.rint(labels_ncc_newid)
            regression_Gauss_newid = numpy.rint(regression_Gauss_newid)
            regressionMAE_Gauss_newid = numpy.rint(regressionMAE_Gauss_newid)
            labels_kNN_newid = numpy.rint(labels_kNN_newid)
            regression_svm_newid = numpy.rint(regression_svm_newid)
            regression2_svm_newid = numpy.rint(regression2_svm_newid)
            regression3_svm_newid = numpy.rint(regression3_svm_newid)
            regression_lr_newid = numpy.rint(regression_lr_newid)
        print "regressionMAE_Gauss_newid[0:5]=", regressionMAE_Gauss_newid[0:5]

    # print "Saving train/test_data for external analysis"
    # ndarray_to_string(sl_seq_training, "/local/tmp/escalafl/training_samples.txt")
    # ndarray_to_string(correct_labels_training, "/local/tmp/escalafl/training_labels.txt")
    # ndarray_to_string(sl_seq_seenid, "/local/tmp/escalafl/seenid_samples.txt")
    # ndarray_to_string(correct_labels_seenid, "/local/tmp/escalafl/seenid_labels.txt")
    # ndarray_to_string(sl_seq_newid, "/local/tmp/escalafl/newid_samples.txt")
    # ndarray_to_string(correct_labels_newid, "/local/tmp/escalafl/newid_labels.txt")

    print "Computing typical delta, eta values for Training SFA Signal"
    # t_delta_eta0 = time.time()
    results.typical_delta_train, results.typical_eta_newid = sfa_libs.comp_typical_delta_eta(sl_seq_training,
                                                                                             iTrain.block_size,
                                                                                             num_reps=200,
                                                                                             training_mode=iTrain.train_mode)
    results.brute_delta_train = sfa_libs.comp_delta_normalized(sl_seq_training)
    results.brute_eta_train = sfa_libs.comp_eta(sl_seq_training)
    t_delta_eta1 = time.time()
    print "delta_train=", results.typical_delta_train
    print "eta_train=", results.typical_eta_train
    print "brute_delta_train=", results.brute_delta_train

    print "Computing typical delta, eta values for New Id SFA Signal"
    t_delta_eta0 = time.time()
    results.typical_delta_newid, results.typical_eta_newid = sfa_libs.comp_typical_delta_eta(sl_seq_newid,
                                                                                             iNewid.block_size,
                                                                                             num_reps=200,
                                                                                             training_mode=iNewid.train_mode)
    results.brute_delta_newid = sfa_libs.comp_delta_normalized(sl_seq_newid)
    results.brute_eta_newid = sfa_libs.comp_eta(sl_seq_newid)
    t_delta_eta1 = time.time()
    print "typical_delta_newid=", results.typical_delta_newid
    print "typical_delta_newid[0:31].sum()=", results.typical_delta_newid[0:31].sum()
    print "typical_eta_newid=", results.typical_eta_newid
    print "brute_delta_newid=", results.brute_delta_newid
    # print "brute_eta_newid=", results.brute_eta_newid
    print "computed delta/eta in %0.3f ms" % ((t_delta_eta1 - t_delta_eta0) * 1000.0)

    if isinstance(block_size, int):
        print "virtual sequence length complete = ", num_images_training * (block_size - 1) / 2
        print "virtual sequence length sequence = ", (num_images_training - block_size) * block_size
        print "virtual sequence length mixed = ", num_images_training * (block_size - 1) / 2 + \
                                                  (num_images_training - block_size) * block_size
    else:
        print "length of virtual sequence not computed = "

    save_train_data = True and False  # This fails for large datasets :( TODO: Make this an option
    if save_train_data:
        uniqueness = numpy.random.randint(32000)
        save_dir_subimages_features_training = "/local/tmp/escalafl/Alberto/saved_images_features_training"
        print "Using uniqueness %d for saving subimage and feature data" % uniqueness
        cache.pickle_array(subimages_train, base_dir=save_dir_subimages_features_training,
                           base_filename="subimages_train%5d" % uniqueness, chunk_size=5000, block_size=1,
                           continuous=False, overwrite=True, verbose=True)
        cache.pickle_array(sl_seq_training, base_dir=save_dir_subimages_features_training,
                           base_filename="sl_seq_training%5d" % uniqueness, chunk_size=5000, block_size=1,
                           continuous=False, overwrite=True, verbose=True)
    # Then unpicke with unpickle_array(base_dir="", base_filename="subimages_train%5d"%uniqueness):


    print "Estimating explained variance for Train SFA Signal"
    number_samples_explained_variance = 9000  # 1000 #4000 #2000
    # fast_inverse_available = True and False

    if estimate_explained_var_with_inverse:
        print "Estimated explained variance with inverse (train) is: ", more_nodes.estimate_explained_variance(
            subimages_train, flow, sl_seq_training, number_samples_explained_variance)
        print "Estimated explained variance with inverse (newid) is: ", more_nodes.estimate_explained_variance(
            subimages_newid, flow, sl_seq_newid, number_samples_explained_variance)
    else:
        print "Fast inverse not available, not estimating explained variance"

    if estimate_explained_var_with_kNN_k:
        k = estimate_explained_var_with_kNN_k  # k=64
        print "Estimated explained variance with kNN (train, %d features) is: " % reg_num_signals,
        print more_nodes.estimate_explained_var_with_kNN(subimages_train, sl_seq_training[:, 0:reg_num_signals],
                                                         max_num_samples_for_ev=10000, max_test_samples_for_ev=10000,
                                                         k=k, ignore_closest_match=True, operation="average")
    else:
        print "Not estimating explained variance with kNN"

    if estimate_explained_var_with_kNN_lin_app_k:
        k = estimate_explained_var_with_kNN_lin_app_k  # k=64
        print "Estimated explained variance with kNN_lin_app (train, %d features) is: " % reg_num_signals, \
            more_nodes.estimate_explained_var_with_kNN(subimages_train, sl_seq_training[:, 0:reg_num_signals],
                                                       max_num_samples_for_ev=10000, max_test_samples_for_ev=10000,
                                                       k=k, ignore_closest_match=True, operation="lin_app")
    else:
        print "Not estimating explained variance with kNN_lin_app"

    if estimate_explained_var_linear_global_N:
        if estimate_explained_var_linear_global_N > 0:
            number_samples_EV_linear_global = estimate_explained_var_linear_global_N
        else:
            number_samples_EV_linear_global = sl_seq_training.shape[0]
        number_samples_EV_linear_global = sl_seq_seenid.shape[0]
        num_features_linear_model = 75
        EVLinGlobal_train1, EVLinGlobal_train2, EVLinGlobal_newid = more_nodes.estimate_explained_var_linear_global(
            subimages_seenid, sl_seq_seenid[:, 0:num_features_linear_model], subimages_newid,
            sl_seq_newid[:, 0:num_features_linear_model], num_features_linear_model, number_samples_EV_linear_global)

        print "Explained Variance Linear Global for training data (%d features, subset of size %d) is: " % (
        num_features_linear_model, number_samples_EV_linear_global), EVLinGlobal_train1,
        print "for training data (new random subset) is: ", EVLinGlobal_train2
        print "for newid (all_samples FORCED %d) is: " % num_features_linear_model, EVLinGlobal_newid
    else:
        print "Not estimating explained variance with global linear reconstruction"
        num_features_linear_model = 75

    print "Computing chance levels for newid data"
    chance_level_RMSE_newid = correct_labels_newid.std()
    correct_labels_newid_sorted = correct_labels_newid + 0.0
    correct_labels_newid_sorted.sort()
    median_estimation = numpy.ones(len(correct_labels_newid)) * correct_labels_newid_sorted[
        len(correct_labels_newid) / 2]
    chance_level_MAE_newid = classifiers.mean_average_error(correct_labels_newid, median_estimation)
    print "chance_level_RMSE_newid=", chance_level_RMSE_newid, "chance_level_MAE_newid=", chance_level_MAE_newid
    print "correct_labels_newid.mean()=", correct_labels_newid.mean(),
    print "correct_labels_newid median() ", median_estimation
    print "correct_labels_newid.min()=", correct_labels_newid.min(),
    print "correct_labels_newid.max() ", correct_labels_newid.max()
    print "Computations Finished!"

    print "** Displaying Benchmark data: **"
    if benchmark is not None:
        for task_name, task_time in benchmark:
            print "     ", task_name, " done in %0.3f s" % task_time

    print "Classification/Regression Performance: "
    if Parameters.analysis or True:
        print correct_classes_training
        print classes_kNN_training
        # MSE
        results.class_ncc_rate_train = classifiers.correct_classif_rate(correct_classes_training, classes_ncc_training)
        results.class_kNN_rate_train = classifiers.correct_classif_rate(correct_classes_training, classes_kNN_training)
        results.class_Gauss_rate_train = classifiers.correct_classif_rate(correct_classes_training,
                                                                          classes_Gauss_training)
        results.class_svm_rate_train = classifiers.correct_classif_rate(correct_classes_training, classes_svm_training)
        results.mse_ncc_train = distance_squared_Euclidean(correct_labels_training, labels_ncc_training) / len(
            labels_kNN_training)
        results.mse_kNN_train = distance_squared_Euclidean(correct_labels_training, labels_kNN_training) / len(
            labels_kNN_training)
        results.mse_gauss_train = distance_squared_Euclidean(correct_labels_training, regression_Gauss_training) / len(
            labels_kNN_training)
        results.mse_svm_train = distance_squared_Euclidean(correct_labels_training, regression_svm_training) / len(
            labels_kNN_training)
        results.mse2_svm_train = distance_squared_Euclidean(correct_labels_training, regression2_svm_training) / len(
            labels_kNN_training)
        results.mse3_svm_train = distance_squared_Euclidean(correct_labels_training, regression3_svm_training) / len(
            labels_kNN_training)
        results.mse_lr_train = distance_squared_Euclidean(correct_labels_training, regression_lr_training) / len(
            labels_kNN_training)
        # MAE
        results.maeOpt_gauss_train = classifiers.mean_average_error(correct_labels_training,
                                                                    regressionMAE_Gauss_training)
        results.mae_gauss_train = classifiers.mean_average_error(regression_Gauss_training, correct_labels_training)
        # RMSE
        results.rmse_ncc_train = results.mse_ncc_train ** 0.5
        results.rmse_kNN_train = results.mse_kNN_train ** 0.5
        results.rmse_gauss_train = results.mse_gauss_train ** 0.5
        results.rmse_svm_train = results.mse_svm_train ** 0.5
        results.rmse2_svm_train = results.mse2_svm_train ** 0.5
        results.rmse3_svm_train = results.mse3_svm_train ** 0.5
        results.rmse_lr_train = results.mse_lr_train ** 0.5

        results.class_ncc_rate_seenid = classifiers.correct_classif_rate(correct_classes_seenid, classes_ncc_seenid)
        results.class_kNN_rate_seenid = classifiers.correct_classif_rate(correct_classes_seenid, classes_kNN_seenid)
        results.class_Gauss_rate_seenid = classifiers.correct_classif_rate(correct_classes_seenid, classes_Gauss_seenid)
        results.class_svm_rate_seenid = classifiers.correct_classif_rate(correct_classes_seenid, classes_svm_seenid)
        results.mse_ncc_seenid = distance_squared_Euclidean(correct_labels_seenid, labels_ncc_seenid) / len(
            labels_kNN_seenid)
        results.mse_kNN_seenid = distance_squared_Euclidean(correct_labels_seenid, labels_kNN_seenid) / len(
            labels_kNN_seenid)
        results.mse_gauss_seenid = distance_squared_Euclidean(correct_labels_seenid, regression_Gauss_seenid) / len(
            labels_kNN_seenid)
        results.maeOpt_gauss_seenid = classifiers.mean_average_error(correct_labels_seenid, regressionMAE_Gauss_seenid)
        results.mse_svm_seenid = distance_squared_Euclidean(correct_labels_seenid, regression_svm_seenid) / len(
            labels_kNN_seenid)
        results.mse2_svm_seenid = distance_squared_Euclidean(correct_labels_seenid, regression2_svm_seenid) / len(
            labels_kNN_seenid)
        results.mse3_svm_seenid = distance_squared_Euclidean(correct_labels_seenid, regression3_svm_seenid) / len(
            labels_kNN_seenid)
        results.mse_lr_seenid = distance_squared_Euclidean(correct_labels_seenid, regression_lr_seenid) / len(
            labels_kNN_seenid)
        results.mae_gauss_seenid = classifiers.mean_average_error(regression_Gauss_seenid, correct_labels_seenid)

        results.rmse_ncc_seenid = results.mse_ncc_seenid ** 0.5
        results.rmse_kNN_seenid = results.mse_kNN_seenid ** 0.5
        results.rmse_gauss_seenid = results.mse_gauss_seenid ** 0.5
        results.rmse_svm_seenid = results.mse_svm_seenid ** 0.5
        results.rmse2_svm_seenid = results.mse2_svm_seenid ** 0.5
        results.rmse3_svm_seenid = results.mse3_svm_seenid ** 0.5
        results.rmse_lr_seenid = results.mse_lr_seenid ** 0.5

        print correct_classes_newid.shape, classes_kNN_newid.shape
        results.class_ncc_rate_newid = classifiers.correct_classif_rate(correct_classes_newid, classes_ncc_newid)
        results.class_kNN_rate_newid = classifiers.correct_classif_rate(correct_classes_newid, classes_kNN_newid)
        results.class_Gauss_rate_newid = classifiers.correct_classif_rate(correct_classes_newid, classes_Gauss_newid)
        results.class_svm_rate_newid = classifiers.correct_classif_rate(correct_classes_newid, classes_svm_newid)
        results.mse_ncc_newid = distance_squared_Euclidean(correct_labels_newid, labels_ncc_newid) / len(
            labels_kNN_newid)
        results.mse_kNN_newid = distance_squared_Euclidean(correct_labels_newid, labels_kNN_newid) / len(
            labels_kNN_newid)
        results.mse_gauss_newid = distance_squared_Euclidean(correct_labels_newid, regression_Gauss_newid) / len(
            labels_kNN_newid)
        results.maeOpt_gauss_newid = classifiers.mean_average_error(correct_labels_newid, regressionMAE_Gauss_newid)
        results.mse_svm_newid = distance_squared_Euclidean(correct_labels_newid, regression_svm_newid) / len(
            labels_kNN_newid)
        results.mse2_svm_newid = distance_squared_Euclidean(correct_labels_newid, regression2_svm_newid) / len(
            labels_kNN_newid)
        results.mse3_svm_newid = distance_squared_Euclidean(correct_labels_newid, regression3_svm_newid) / len(
            labels_kNN_newid)
        results.mse_lr_newid = distance_squared_Euclidean(correct_labels_newid, regression_lr_newid) / len(
            labels_kNN_newid)
        results.mae_gauss_newid = classifiers.mean_average_error(correct_labels_newid, regression_Gauss_newid)

        results.rmse_ncc_newid = results.mse_ncc_newid ** 0.5
        results.rmse_kNN_newid = results.mse_kNN_newid ** 0.5
        results.rmse_gauss_newid = results.mse_gauss_newid ** 0.5
        results.rmse_svm_newid = results.mse_svm_newid ** 0.5
        results.rmse2_svm_newid = results.mse2_svm_newid ** 0.5
        results.rmse3_svm_newid = results.mse3_svm_newid ** 0.5
        results.rmse_lr_newid = results.mse_lr_newid ** 0.5

    print "Comparisson of MAE for RMSE estimation and MAE estimation"
    print "regression_Gauss_newid[100:150] =", regression_Gauss_newid[100:150]
    print "regressionMAE_Gauss_newid[100:150] =", regressionMAE_Gauss_newid[100:150]
    print "diff MAE-RMSE= ", regressionMAE_Gauss_newid[100:150] - regression_Gauss_newid[100:150]
    worst = numpy.argsort(numpy.abs(regressionMAE_Gauss_newid - regression_Gauss_newid))
    print "worst[-50:] diff MAE-RMSE= ", worst[-50:]
    print "regression_Gauss_newid[worst[-50:]]=", regression_Gauss_newid[worst[-50:]]
    print "regressionMAE_Gauss_newid[worst[-50:]]=", regressionMAE_Gauss_newid[worst[-50:]]
    print "correct_labels_newid[worst[-50:]]=", correct_labels_newid[worst[-50:]]

    results.maeOpt_gauss_newid = classifiers.mean_average_error(correct_labels_newid, regressionMAE_Gauss_newid)
    results.mae_gauss_newid = classifiers.mean_average_error(correct_labels_newid, regression_Gauss_newid)

    print "N1=", classifiers.mean_average_error(correct_labels_newid, regressionMAE_Gauss_newid)
    print "N2=", classifiers.mean_average_error(correct_labels_newid, regression_Gauss_newid)
    numpy.savetxt("regressionMAE_Gauss_newid", regressionMAE_Gauss_newid)
    numpy.savetxt("regression_Gauss_newid", regression_Gauss_newid)
    numpy.savetxt("correct_labels_newid", correct_labels_newid)

    cs_list = {}
    if cumulative_scores:
        largest_errors = numpy.arange(0, 31)
        print "Computing cumulative errors cs()= {",
        for largest_error in largest_errors:
            cs = more_nodes.cumulative_score(ground_truth=correct_labels_newid, estimation=regression_Gauss_newid,
                                             largest_error=largest_error, integer_rounding=True)
            cs_list[largest_error] = cs
            print "%d:%0.7f, " % (largest_error, cs),
        print "}"
    results.cs_list = cs_list

    error_table = {}
    maes = {}
    estimation_means = {}
    print "Error table:"
    if cumulative_scores:
        different_labels = numpy.unique(correct_labels_newid)
        for label in different_labels:
            error_table[label] = {}
            indices = (correct_labels_newid == label)
            errors = correct_labels_newid[indices] - regression_Gauss_newid[indices]
            abs_errors = numpy.abs(errors)

            estimation_means[label] = regression_Gauss_newid[indices].mean()

            for error in numpy.unique(errors):
                error_table[label][error] = 0

            maes[label] = abs_errors.mean()

            print ""
            print "label=", label,
            for error in errors:
                error_table[label][error] += 1

            for error in numpy.unique(errors):
                print "e[%d]=%d " % (error, error_table[label][error]),

        # Now compute error frequencies for all labels simultaneuous
        signed_errors = {}
        for error in numpy.arange(-200, 200):
            signed_errors[error] = 0
        for label in different_labels:
            for error in error_table[label].keys():
                signed_errors[error] += error_table[label][error]

        print "\n Global signed errors:"
        signed_error_keys = numpy.array(signed_errors.keys())
        signed_error_keys.sort()
        for error in signed_error_keys:
            print "e[%d]=%d " % (error, signed_errors[error]),
        print "."

        print "\n MAES for each GT year. maes(%d) to maes(%d)" % (different_labels[0], different_labels[-1])
        for label in different_labels:
            print "%f, " % maes[label],

        print "\n estimation means for each GT year. estim_means(%d) to estim_means(%d)" % (
        different_labels[0], different_labels[-1])
        for label in different_labels:
            print "%f, " % estimation_means[label],

    if confusion_matrix and integer_label_estimation:
        print "Computing confusion matrix"
        min_label = numpy.min(
            (correct_labels_training.min(), correct_labels_seenid.min(), correct_labels_newid.min())).astype(int)
        max_label = numpy.max(
            (correct_labels_training.max(), correct_labels_seenid.max(), correct_labels_newid.max())).astype(int)
        print "overriding min/max label from data (%d, %d) to (16,77)" % (min_label, max_label)
        min_label = 16
        max_label = 77

        different_labels = numpy.arange(min_label, max_label + 1, dtype=int)
        num_diff_labels = max_label - min_label + 1

        confusion = numpy.zeros((num_diff_labels, num_diff_labels))
        mask_gt = {}
        for ii, l_gt in enumerate(different_labels):
            mask_gt[ii] = (correct_labels_newid == l_gt)
        mask_est = {}
        for jj, l_est in enumerate(different_labels):
            mask_est[jj] = (regression_Gauss_newid == l_est)
        for ii, l_gt in enumerate(different_labels):
            for jj, l_est in enumerate(different_labels):
                confusion[ii, jj] = (mask_gt[ii] * mask_est[jj]).sum()

        print "confusion:=", confusion

        # Output confusion matrix to standard output
        for label in different_labels:
            print "%f, " % label,
        print ""
        for ii in range(num_diff_labels):
            print "[",
            for jj in range(num_diff_labels):
                print "%d, " % confusion[ii, jj],
            print "]"

        print "sums:",
        for ii in range(num_diff_labels):
            print "%d, " % mask_gt[ii].sum(),
        print "]"
    save_results = False
    if save_results:
        cache.pickle_to_disk(results, "results_" + str(int(time.time())) + ".pckl")

    if False:
        print "sl_seq_training.mean(axis=0)=", sl_seq_training.mean(axis=0)
        print "sl_seq_seenid.mean(axis=0)=", sl_seq_seenid.mean(axis=0)
        print "sl_seq_newid.mean(axis=0)=", sl_seq_newid.mean(axis=0)
        print "sl_seq_training.var(axis=0)=", sl_seq_training.var(axis=0)
        print "sl_seq_seenid.var(axis=0)=", sl_seq_seenid.var(axis=0)
        print "sl_seq_newid.var(axis=0)=", sl_seq_newid.var(axis=0)

    print "Train: %0.3f CR_NCC, %0.3f CR_kNN, CR_Gauss %0.5f, softCR_Gauss=%0.5f, CR_SVM %0.3f, MSE_NCC %0.3f, MSE_kNN %0.3f, MSE_Gauss= %0.3f MSE3_SVM %0.3f, MSE2_SVM %0.3f, MSE_SVM %0.3f, MSE_LR %0.3f, MAE= %0.5f MAE(Opt)= %0.3f" % (
        results.class_ncc_rate_train, results.class_kNN_rate_train, results.class_Gauss_rate_train,
        softCR_Gauss_training, results.class_svm_rate_train, results.mse_ncc_train, results.mse_kNN_train,
        results.mse_gauss_train,
        results.mse3_svm_train, results.mse2_svm_train, results.mse_svm_train, results.mse_lr_train,
        results.mae_gauss_train, results.maeOpt_gauss_train)
    print "Seen Id: %0.3f CR_NCC, %0.3f CR_kNN, CR_Gauss %0.5f, softCR_Gauss=%0.5f, CR_SVM %0.3f, MSE_NCC %0.3f, MSE_kNN %0.3f, MSE_Gauss= %0.3f MSE3_SVM %0.3f, MSE2_SVM %0.3f, MSE_SVM %0.3f, MSE_LR %0.3f, MAE= %0.5f MAE(Opt)= %0.3f" % (
        results.class_ncc_rate_seenid, results.class_kNN_rate_seenid, results.class_Gauss_rate_seenid,
        softCR_Gauss_seenid, results.class_svm_rate_seenid, results.mse_ncc_seenid, results.mse_kNN_seenid,
        results.mse_gauss_seenid,
        results.mse3_svm_seenid, results.mse2_svm_seenid, results.mse_svm_seenid, results.mse_lr_seenid,
        results.mae_gauss_seenid, results.maeOpt_gauss_seenid)
    print "New Id: %0.3f CR_NCC, %0.3f CR_kNN, CR_Gauss %0.5f, softCR_Gauss=%0.5f, CR_SVM %0.3f, MSE_NCC %0.3f, MSE_kNN %0.3f, MSE_Gauss= %0.3f MSE3_SVM %0.3f, MSE2_SVM %0.3f, MSE_SVM %0.3f, MSE_LR %0.3f , MAE= %0.5f MAE(Opt)= %0.3f" % (
        results.class_ncc_rate_newid, results.class_kNN_rate_newid, results.class_Gauss_rate_newid, softCR_Gauss_newid,
        results.class_svm_rate_newid, results.mse_ncc_newid, results.mse_kNN_newid, results.mse_gauss_newid,
        results.mse3_svm_newid, results.mse2_svm_newid, results.mse_svm_newid, results.mse_lr_newid,
        results.mae_gauss_newid, results.maeOpt_gauss_newid)

    print "Train:   RMSE_NCC %0.3f, RMSE_kNN %0.3f, RMSE_Gauss= %0.5f RMSE3_SVM %0.3f, RMSE2_SVM %0.3f, RMSE_SVM %0.3f, RMSE_LR %0.3f" % (
        results.rmse_ncc_train, results.rmse_kNN_train, results.rmse_gauss_train,
        results.rmse3_svm_train, results.rmse2_svm_train, results.rmse_svm_train, results.rmse_lr_train)
    print "Seen Id: RMSE_NCC %0.3f, RMSE_kNN %0.3f, RMSE_Gauss= %0.5f RMSE3_SVM %0.3f, RMSE2_SVM %0.3f, RMSE_SVM %0.3f, RMSE_LR %0.3f" % (
        results.rmse_ncc_seenid, results.rmse_kNN_seenid, results.rmse_gauss_seenid,
        results.rmse3_svm_seenid, results.rmse2_svm_seenid, results.rmse_svm_seenid, results.rmse_lr_seenid)
    print "New Id:  RMSE_NCC %0.3f, RMSE_kNN %0.3f, RMSE_Gauss= %0.5f RMSE3_SVM %0.3f, RMSE2_SVM %0.3f, RMSE_SVM %0.3f, RMSE_LR %0.3f" % (
        results.rmse_ncc_newid, results.rmse_kNN_newid, results.rmse_gauss_newid,
        results.rmse3_svm_newid, results.rmse2_svm_newid, results.rmse_svm_newid, results.rmse_lr_newid)

    if False:
        starting_point = "Sigmoids"  # None, "Sigmoids", "Identity"
        print "Computations useful to determine information loss and feature obfuscation IL/FO... starting_point=",
        print starting_point
        if iTrain.train_mode == "clustered":
            if features_residual_information > 0:
                results_e1_e2_from_feats = []
                for num_feats_residual_information in numpy.arange(40, features_residual_information, 5):
                    e1, e2 = more_nodes.compute_classification_performance(sl_seq_seenid, correct_classes_seenid,
                                                                           sl_seq_newid, correct_classes_newid,
                                                                           num_feats_residual_information,
                                                                           starting_point=starting_point)
                    # print "Classification rates on expanded slow features with dimensionality %d are %f for training
                    # and %f for test"%(num_feats_residual_information, e1, e2)
                    results_e1_e2_from_feats.append((num_feats_residual_information, e1, e2))
                print "\n num_feats_residual_information =",
                for (num_feats_residual_information, e1, e2) in results_e1_e2_from_feats:
                    print num_feats_residual_information, ", ",
                print "\n classification rate on seenid(feats) =",
                for (num_feats_residual_information, e1, e2) in results_e1_e2_from_feats:
                    print "%f, " % e1,
                print "\n classification rate on newid(feats) =",
                for (num_feats_residual_information, e1, e2) in results_e1_e2_from_feats:
                    print "%f, " % e2,

            if compute_input_information:
                results_e1_e2_from_pca = []
                pca_node = mdp.nodes.PCANode(output_dim=0.99)
                pca_node.train(subimages_seenid)
                pca_node.stop_training()
                subimages_seenid_pca = pca_node.execute(subimages_seenid)
                subimages_newid_pca = pca_node.execute(subimages_newid)
                print "\n *****subimages_newid_pca.shape", subimages_newid_pca.shape
                for num_feats_residual_information in numpy.arange(40, 150, 5):  # (40,150,5):
                    e1, e2 = more_nodes.compute_classification_performance(subimages_seenid_pca, correct_classes_seenid,
                                                                           subimages_newid_pca, correct_classes_newid,
                                                                           num_feats_residual_information,
                                                                           starting_point=starting_point)
                    # print "Classification rates on expanded input data with dimensionality %d are %f for training
                    # and %f for test"%(num_feats_residual_information, e1, e2)
                    results_e1_e2_from_pca.append((num_feats_residual_information, e1, e2))
                print "\n num_feats_residual_information =",
                for (num_feats_residual_information, e1, e2) in results_e1_e2_from_pca:
                    print num_feats_residual_information, ", ",
                print "\n classification rate on seenid(input) =",
                for (num_feats_residual_information, e1, e2) in results_e1_e2_from_pca:
                    print "%f, " % e1,
                print "\n classification rate on newid(input) =",
                for (num_feats_residual_information, e1, e2) in results_e1_e2_from_pca:
                    print "%f, " % e2,
                print "\n max CR on newid(input) = ", numpy.max([e2 for (nf, e1, e2) in results_e1_e2_from_pca])
        else:
            if features_residual_information > 0:
                for num_feats_residual_information in numpy.linspace(60, features_residual_information, 11):
                    e1, e2 = more_nodes.compute_regression_performance(sl_seq_seenid, correct_labels_seenid,
                                                                       sl_seq_newid, correct_labels_newid,
                                                                       num_feats_residual_information,
                                                                       starting_point=starting_point)
                    print "RMSE on expanded slow features with dimens. %d are %f for training and %f for test" % \
                          (num_feats_residual_information, e1, e2)

            if compute_input_information:
                for num_feats_residual_information in numpy.linspace(144, features_residual_information, 11):
                    e1, e2 = more_nodes.compute_regression_performance(subimages_seenid, correct_labels_seenid,
                                                                       subimages_newid, correct_labels_newid,
                                                                       num_feats_residual_information,
                                                                       starting_point=starting_point)
                    print "RMSE on expanded input data with dimensionality %d are %f for training and %f for test" % \
                          (num_feats_residual_information, e1, e2)

    scale_disp = 1

    print "Computing average SFA..."
    if isinstance(iTrain.block_size, int):
        num_blocks_train = iTrain.num_images / iTrain.block_size
    elif iTrain.block_size is not None:
        num_blocks_train = len(iTrain.block_size)
    else:
        num_blocks_train = iTrain.num_images

    print num_blocks_train, hierarchy_out_dim, sl_seq_training.shape, block_size, iTrain.block_size

    sl_seq_training_mean = numpy.zeros((num_blocks_train, hierarchy_out_dim))
    if isinstance(iTrain.block_size, int):
        for block in range(num_blocks_train):
            sl_seq_training_mean[block] = sl_seq_training[block * block_size:(block + 1) * block_size, :].mean(axis=0)
    else:
        counter_sl = 0
        for block in range(num_blocks_train):
            sl_seq_training_mean[block] = sl_seq_training[counter_sl:counter_sl + iTrain.block_size[block], :].mean(
                axis=0)
            counter_sl += iTrain.block_size[block]

    print "%d Blocks used for averages" % num_blocks_train

    # Function for gender label estimation. (-3 to -0.1) = Masculine, (0.0 to 2.9) = femenine. Midpoint between classes
    # is -0.05 and not zero
    def binarize_array(arr):
        arr2 = arr.copy()
        arr2[arr2 < -0.05] = -1
        arr2[arr2 >= -0.05] = 1
        return arr2

    if Parameters == experimental_datasets.ParamsGenderFunc or experimental_datasets.ParamsRAgeFunc:
        print "Computing effective gender recognition:"
        binary_gender_estimation_training = binarize_array(regression_Gauss_training)
        binary_correct_labels_training = binarize_array(correct_labels_training)
        binary_gender_estimation_rate_training = classifiers.correct_classif_rate(binary_correct_labels_training,
                                                                                  binary_gender_estimation_training)
        print "Binary gender classification rate (training) from continuous labels is %f" % \
              binary_gender_estimation_rate_training
        binary_gender_estimation_seenid = binarize_array(regression_Gauss_seenid)
        binary_correct_labels_seenid = binarize_array(correct_labels_seenid)
        binary_gender_estimation_rate_seenid = classifiers.correct_classif_rate(binary_correct_labels_seenid,
                                                                                binary_gender_estimation_seenid)
        print "Binary gender classification rate (seenid) from continuous labels is %f" % \
              binary_gender_estimation_rate_seenid
        binary_gender_estimation_newid = binarize_array(regression_Gauss_newid)
        binary_correct_labels_newid = binarize_array(correct_labels_newid)
        binary_gender_estimation_rate_newid = classifiers.correct_classif_rate(binary_correct_labels_newid,
                                                                               binary_gender_estimation_newid)
        print "Binary gender classification rate (newid) from continuous labels is %f" % \
              binary_gender_estimation_rate_newid

    if Parameters == experimental_datasets.ParamsRGTSRBFunc and output_filename is not None:
        fd = open("G" + output_filename, "w")
        txt = ""
        for i, filename in enumerate(iNewid.input_files):
            txt += filename[-9:] + "; " + str(int(classes_Gauss_newid[i])) + "\n"
        fd.write(txt)
        fd.close()

        fd = open("C" + output_filename, "w")
        txt = ""
        for i, filename in enumerate(iNewid.input_files):
            txt += filename[-9:] + "; " + str(int(classes_kNN_newid[i])) + "\n"
        fd.write(txt)
        fd.close()

        fd = open("NN" + output_filename, "w")
        txt = ""
        for i, filename in enumerate(iNewid.input_files):
            txt += filename[-9:] + "; " + str(int(classes_svm_newid[i])) + "\n"
        fd.write(txt)
        fd.close()

        save_SFA_Features = True and (
        Parameters.activate_HOG is False)  # Will only overwrite SFA features when running in SFA mode

        sfa_output_dir = "/local/escalafl/Alberto/GTSRB/Online-Test/SFA/SFA_02/"
        GTSRB_SFA_dir_training = "/local/escalafl/Alberto/GTSRB/GTSRB_Features_SFA/training"
        GTSRB_SFA_dir_OnlineTest = "/local/escalafl/Alberto/GTSRB/Online-Test/SFA"

        # /local/escalafl/Alberto/GTSRB/Online-Test/SFA/SFA_02
        number_SFA_features = 98
        if save_SFA_Features:
            iSeq_sl_tuples = [(iTrain, sl_seq_training), (iSeenid, sl_seq_seenid), (iNewid, sl_seq_newid)]
            for iSeq, sl in iSeq_sl_tuples:
                online_base_dir = "/local/escalafl/Alberto/GTSRB/Online-Test/Images"
                if iSeq.input_files[0][0:len(online_base_dir)] == online_base_dir:
                    base_SFA_dir = GTSRB_SFA_dir_OnlineTest
                    online_test = True
                else:
                    base_SFA_dir = GTSRB_SFA_dir_training
                    online_test = False

                base_GTSRB_dir = "/local/escalafl/Alberto/GTSRB"

                sample_img_filename = "00000/00001_00000.ppm"
                fset = "02"
                for ii, image_filename in enumerate(iSeq.input_files):
                    if online_test:
                        sfa_filename = GTSRB_SFA_dir_OnlineTest + "/SFA_" + fset + "/" + image_filename[-9:-3] + "txt"
                    else:
                        sfa_filename = GTSRB_SFA_dir_training + "/SFA_" + fset + "/" + image_filename[-len(
                            sample_img_filename):-3] + "txt"
                    if ii == 0:
                        print sfa_filename
                    filed = open(sfa_filename, "wb")
                    for i, xx in enumerate(sl[ii, 0:number_SFA_features]):
                        if i == 0:
                            filed.write('%f' % xx)
                        else:
                            filed.write('\n%f' % xx)
                    filed.close()

    elif output_filename is not None:
        fd = open(output_filename, "w")
        txt = ""
        for i, val in enumerate(results.typical_delta_newid):
            if i == 0:
                txt = "%f" % results.typical_delta_newid[i]
            else:
                txt += " %f" % results.typical_delta_newid[i]
        print "writing \"%s\" to; %s" % (txt, output_filename)
        fd.write(txt)
        fd.close()

    if save_sorted_AE_Gauss_newid:
        error_Gauss_newid = numpy.abs(correct_labels_newid - regression_Gauss_newid)
        sorting_error_Gauss_newid = error_Gauss_newid.argsort()

        save_images_sorted_error_Gauss_newid_base_dir = "/local/tmp/escalafl/Alberto/saved_images_sorted_AE_Gauss_newid"
        print "saving images to directory:", save_images_sorted_error_Gauss_newid_base_dir
        decimate = 1  # 100
        for i, i_x in enumerate(sorting_error_Gauss_newid):
            x = subimages_newid[i_x]
            if i % decimate == 0:
                if seq.convert_format == "RGB":
                    im_raw = numpy.reshape(x, (sNewid.subimage_width, sNewid.subimage_height, 3))  # Remove ,3 for L
                    im = scipy.misc.toimage(im_raw, mode=seq.convert_format)
                else:
                    im_raw = numpy.reshape(x, (sNewid.subimage_width, sNewid.subimage_height))  # Remove ,3 for L
                    im = scipy.misc.toimage(im_raw, mode="L")

                fullname = os.path.join(save_images_sorted_error_Gauss_newid_base_dir,
                                        "image%05d_gt%+05f_e%+05f.png" % (
                                        i / decimate, correct_labels_newid[i_x], regression_Gauss_newid[i_x]))
                im.save(fullname)
        total_samples = len(sorting_error_Gauss_newid)
        worst_samples_nr = min(150, total_samples)
        for i in range(worst_samples_nr):
            i_x = sorting_error_Gauss_newid[total_samples - i - 1]
            x = subimages_newid[i_x]
            if seq.convert_format == "L":
                im_raw = numpy.reshape(x, (sNewid.subimage_width, sNewid.subimage_height))  # Remove ,3 for L
                im = scipy.misc.toimage(im_raw, mode=seq.convert_format)
            elif seq.convert_format == "RGB":
                im_raw = numpy.reshape(x, (sNewid.subimage_width, sNewid.subimage_height, 3))  # Remove ,3 for L
                im = scipy.misc.toimage(im_raw, mode=seq.convert_format)
            else:
                im = scipy.misc.toimage(im_raw, mode="L")

            fullname = os.path.join(save_images_sorted_error_Gauss_newid_base_dir,
                                    "image_worst%05d_gt%+05f_e%+05f.png" % (
                                    i, correct_labels_newid[i_x], regression_Gauss_newid[i_x]))
            im.save(fullname)

    if save_sorted_incorrect_class_Gauss_newid:
        d1 = numpy.array(correct_classes_newid, dtype="int")
        d2 = numpy.array(classes_Gauss_newid, dtype="int")

        incorrect_Gauss_newid = (d1 == d2)
        sorting_incorrect_Gauss_newid = incorrect_Gauss_newid.argsort()

        save_images_sorted_incorrect_class_Gauss_newid_base_dir = \
            "/local/tmp/escalafl/Alberto/saved_images_sorted_incorrect_class_Gauss_newid"
        print "saving images to directory:", save_images_sorted_incorrect_class_Gauss_newid_base_dir
        decimate = 1
        num_signals_per_image = subimage_shape[0] * subimage_shape[1] * in_channel_dim
        for i, i_x in enumerate(sorting_incorrect_Gauss_newid):
            x = subimages_newid[i_x][0:num_signals_per_image]
            if i % decimate == 0:
                if seq.convert_format == "L":
                    im_raw = numpy.reshape(x, (sNewid.subimage_width, sNewid.subimage_height))
                    im = scipy.misc.toimage(im_raw, mode=seq.convert_format)
                elif seq.convert_format == "RGB":
                    im_raw = numpy.reshape(x, (sNewid.subimage_width, sNewid.subimage_height, 3))
                    im = scipy.misc.toimage(im_raw, mode=seq.convert_format)
                else:
                    im = scipy.misc.toimage(im_raw, mode="L")

                fullname = os.path.join(save_images_sorted_incorrect_class_Gauss_newid_base_dir,
                                        "image%05d_gt%03d_c%03d.png" % (
                                        i / decimate, correct_classes_newid[i_x], classes_Gauss_newid[i_x]))
                im.save(fullname)

    # Save training, known_id, and new_id data in the format of libsvm
    if export_data_to_libsvm:
        this_time = int(time.time())
        print "Saving output features in the format of libsvm (all components included). Timestamp:", this_time
        more_nodes.export_to_libsvm(correct_classes_training, sl_seq_training, "libsvm_data/training2%011d" % this_time)
        more_nodes.export_to_libsvm(correct_classes_seenid, sl_seq_seenid, "libsvm_data/seenid2%011d" % this_time)
        more_nodes.export_to_libsvm(correct_classes_newid, sl_seq_newid, "libsvm_data/newid2%011d" % this_time)

    compute_background_detection = True  # and False
    cutoff_backgrounds = [0.975, 0.98, 0.985, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.99995]
    if compute_background_detection and Parameters == experimental_datasets.ParamsRFaceCentering2Func:
        print "false_background should be as small as possible <= 0.01, correct_background should be large >= 0.8"
        for cutoff_background in cutoff_backgrounds:
            print "for cutoff_background = %f" % cutoff_background
            for seq, regression in [(iTrain, regression_Gauss_training), (iSeenid, regression_Gauss_seenid),
                                    (iNewid, regression_Gauss_newid)]:
                bs = seq.block_size
                correct_background = (regression[-bs:] > cutoff_background).sum() * 1.0 / bs
                false_background = (regression[0:-bs] > cutoff_background).sum() * 1.0 / len(regression[0:-bs])
                print "correct_background = ", correct_background, "false_background =", false_background

    if minutes_sleep < 0:
        lock.acquire()
        q = open(cuicuilco_queue, "r")
        next_pid = q.readline()
        print "queue_top was: ", next_pid, "we are:", pid,
        queue_rest = q.readlines()
        print "queue_rest=", queue_rest
        # served = True
        q.close()

        print "removing us from the queue"
        q2 = open(cuicuilco_queue, "w")
        for line in queue_rest:
            q2.write(line)
        q2.close()
        lock.release()

    if enable_display:
        print "Creating GUI..."

        ###############################################################################################################
        ############################## Plot for (examples of) the training, supervised, and test images ###############
        ###############################################################################################################
        print "****** Displaying Typical Images used for Training and Testing **********"
        tmp_fig = plt.figure()
        plt.suptitle(Parameters.name + ". Image Datasets")

        num_images_per_set = 4

        subimages_training = subimages
        # ## num_images_training

        sequences = [subimages_training, subimages_seenid, subimages_newid]
        messages = ["Training Images", "Seen Id Images", "New Id Images"]
        nums_images = [num_images_training, num_images_seenid, num_images_newid]
        sizes = [subimage_shape, subimage_shape, subimage_shape]
        for seqn in range(3):
            for im in range(num_images_per_set):
                tmp_sb = plt.subplot(3, num_images_per_set, num_images_per_set * seqn + im + 1)
                y = im * (nums_images[seqn] - 1) / (num_images_per_set - 1)
                subimage_im = sequences[seqn][y][0:num_pixels_per_image].reshape(sizes[seqn])

                tmp_sb.imshow(subimage_im.clip(0, max_clip), norm=None, vmin=0, vmax=max_clip, aspect='auto',
                              interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
                if im == 0:
                    plt.ylabel(messages[seqn])
                else:
                    tmp_sb.axis('off')

        #############################################################################################################
        # ############### Plot of all extracted features from the training, supervised, and test images #############
        #############################################################################################################
        print "************ Displaying SFA Output Signals **************"
        # Create Figure
        f0 = plt.figure()
        plt.suptitle(Parameters.name + ". Slow Signals")

        # display SFA of Training Set
        p11 = plt.subplot(1, 3, 1)
        plt.title("Output Signals (Training Set)")
        sl_seqdisp = sl_seq_training[:, range(0, hierarchy_out_dim)]
        sl_seqdisp = scale_to(sl_seqdisp, sl_seq_training.mean(axis=0),
                              sl_seq_training.max(axis=0) - sl_seq_training.min(axis=0), 127.5, 255.0, scale_disp,
                              'tanh')
        p11.imshow(sl_seqdisp.clip(0, 255), aspect='auto', interpolation='nearest', origin='upper',
                   cmap=mpl.pyplot.cm.gray)
        plt.xlabel("min[0]=%.3f, max[0]=%.3f, scale=%.3f\n e[]=" % \
                   (sl_seq_training.min(axis=0)[0], sl_seq_training.max(axis=0)[0], scale_disp) + str3(
            sfa_libs.comp_eta(sl_seq_training)[0:5]))
        plt.ylabel("Training Images")

        # display SFA of Known Id testing Set
        p12 = plt.subplot(1, 3, 2)
        plt.title("Output Signals (Seen Id Test Set)")
        sl_seqdisp = sl_seq_seenid[:, range(0, hierarchy_out_dim)]
        sl_seqdisp = scale_to(sl_seqdisp, sl_seq_training.mean(axis=0),
                              sl_seq_training.max(axis=0) - sl_seq_training.min(axis=0), 127.5, 255.0, scale_disp,
                              'tanh')
        p12.imshow(sl_seqdisp.clip(0, 255), aspect='auto', interpolation='nearest', origin='upper',
                   cmap=mpl.pyplot.cm.gray)
        plt.xlabel("min[0]=%.3f, max[0]=%.3f, scale=%.3f\n e[]=" % (sl_seq_seenid.min(axis=0)[0],
                                                                    sl_seq_seenid.max(axis=0)[0], scale_disp) +
                   str3(sfa_libs.comp_eta(sl_seq_seenid)[0:5]))
        plt.ylabel("Seen Id Images")

        # display SFA of Known Id testing Set
        p13 = plt.subplot(1, 3, 3)
        plt.title("Output Signals (New Id Test Set)")
        sl_seqdisp = sl_seq_newid[:, range(0, hierarchy_out_dim)]
        sl_seqdisp = scale_to(sl_seqdisp, sl_seq_training.mean(axis=0),
                              sl_seq_training.max(axis=0) - sl_seq_training.min(axis=0), 127.5, 255.0, scale_disp,
                              'tanh')
        p13.imshow(sl_seqdisp.clip(0, 255), aspect='auto', interpolation='nearest', origin='upper',
                   cmap=mpl.pyplot.cm.gray)
        plt.xlabel("min[0]=%.3f, max[0]=%.3f, scale=%.3f\n e[]=" % (
        sl_seq_newid.min(axis=0)[0], sl_seq_newid.max(axis=0)[0], scale_disp) + str3(
            sfa_libs.comp_eta(sl_seq_newid)[0:5]))
        plt.ylabel("New Id Images")

        ###############################################################################################################
        ########### Plot of the first 3 extracted features from the training, supervised, and test images #############
        ###############################################################################################################
        print "************ Plotting Relevant SFA Output Signals **************"
        relevant_out_dim = 3
        if hierarchy_out_dim < relevant_out_dim:
            relevant_out_dim = hierarchy_out_dim

        ax_5 = plt.figure()
        ax_5.subplots_adjust(hspace=0.5)
        plt.suptitle(Parameters.name + ". Most Relevant Slow Signals")

        sp11 = plt.subplot(2, 2, 1)
        plt.title("SFA Outputs. (Training Set)")

        relevant_sfa_indices = numpy.arange(relevant_out_dim)
        reversed_sfa_indices = range(relevant_out_dim)
        reversed_sfa_indices.reverse()

        # # r_color = (1 - relevant_sfa_indices * 1.0 / relevant_out_dim) * 0.8 + 0.2
        # # g_color = (relevant_sfa_indices * 1.0 / relevant_out_dim) * 0.8 + 0.2
        # r_color = (0.5*numpy.cos(relevant_sfa_indices * numpy.pi / relevant_out_dim) + 0.5).clip(0.0,1.0)
        # g_color = (0.5*numpy.cos(relevant_sfa_indices * numpy.pi / relevant_out_dim + numpy.pi)+0.5).clip(0.0,1.0)
        # b_color = relevant_sfa_indices * 0.0
        r_color = [1.0, 0.0, 0.0]
        g_color = [28 / 255.0, 251 / 255.0, 55 / 255.0]
        b_color = [33 / 255.0, 79 / 255.0, 196 / 255.0]
        max_amplitude_sfa = 3.0  # 2.0

        sfa_text_labels = ["Slowest Signal", "2nd Slowest Signal", "3rd Slowest Signal"]

        print num_images_training, sl_seq_training.shape
        for sig in reversed_sfa_indices:
            plt.plot(numpy.arange(num_images_training), sl_seq_training[:, sig], ".",
                     color=(r_color[sig], g_color[sig], b_color[sig]), label=sfa_text_labels[sig], markersize=6,
                     markerfacecolor=(r_color[sig], g_color[sig], b_color[sig]))
        # plt.plot(numpy.arange(num_images_training), sl_seq_training[:, sig], ".")
        plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
        plt.xlim(0, num_images_training)
        plt.xlabel("Input Image, Training Set ")
        plt.ylabel("Slow Signals")

        sp12 = plt.subplot(2, 2, 2)
        if Parameters.train_mode == 'serial' or Parameters.train_mode == 'mixed':
            plt.title("Example of Ideal SFA Outputs")
            num_blocks = num_images_training / block_size
            sl_optimal = numpy.zeros((num_images, relevant_out_dim))
            factor = -1.0 * numpy.sqrt(2.0)
            t_opt = numpy.linspace(0, numpy.pi, num_blocks)
            for sig in range(relevant_out_dim):
                sl_optimal[:, sig] = wider_1Darray(factor * numpy.cos((sig + 1) * t_opt), block_size)

            for sig in reversed_sfa_indices:
                colour = sig * 0.6 / relevant_out_dim
                plt.plot(numpy.arange(num_images_training), sl_optimal[:, sig], ".",
                         color=(r_color[sig], g_color[sig], b_color[sig]), label=sfa_text_labels[sig], markersize=6,
                         markerfacecolor=(r_color[sig], g_color[sig], b_color[sig]))
            plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
            plt.xlim(0, num_images_training)
            plt.xlabel("Input Image, Training Set ")
            plt.ylabel("Slow Signals")
        else:
            plt.title("Example Ideal SFA Outputs Not Available")

        sp13 = plt.subplot(2, 2, 3)
        plt.title("SFA Outputs. (Seen Id Test Set)")
        for sig in reversed_sfa_indices:
            colour = sig * 1.0 / relevant_out_dim
            plt.plot(numpy.arange(num_images_seenid), sl_seq_seenid[:, sig], ".",
                     color=(r_color[sig], g_color[sig], b_color[sig]), label=sfa_text_labels[sig], markersize=6,
                     markerfacecolor=(r_color[sig], g_color[sig], b_color[sig]))
        plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
        plt.xlim(0, num_images_seenid)
        # plt.ylim(0, 1)
        plt.xlabel("Input Image, Seen Id Test")
        plt.ylabel("Slow Signals")
        # plt.legend( (sfa_text_labels[2], sfa_text_labels[1], sfa_text_labels[0]), loc=4)

        sp14 = plt.subplot(2, 2, 4)
        plt.title("SFA Outputs. (New Id Test Set)")
        for sig in reversed_sfa_indices:
            # colour = sig * 1.0 / relevant_out_dim
            plt.plot(numpy.arange(num_images_newid), sl_seq_newid[:, sig], ".",
                     color=(r_color[sig], g_color[sig], b_color[sig]), label=sfa_text_labels[sig], markersize=6,
                     markerfacecolor=(r_color[sig], g_color[sig], b_color[sig]))
        plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
        plt.xlim(0, num_images_newid)
        plt.xlabel("Input Image, New Id Test")
        plt.ylabel("Slow Signals")

        # ######################### Some options to control the following plots ################################
        show_linear_inv = True
        show_linear_masks = True
        show_linear_masks_ext = True
        show_linear_morphs = True
        show_localized_masks = True
        show_localized_morphs = True
        show_progressive_morph = False

        show_translation_x = False

        #############################################################################################################
        # ##### Interactive plot that allows the user to perform network inverse for a given output vector ##########
        #############################################################################################################
        print "************ Displaying Training Set SFA and Inverses **************"
        # Create Figure
        f1 = plt.figure()
        ax_5.subplots_adjust(hspace=0.5)
        plt.suptitle(Network.name)

        # display SFA
        f1a11 = plt.subplot(2, 3, 1)
        plt.title("Output Unit (Top Node)")
        sl_seqdisp = sl_seq[:, range(0, hierarchy_out_dim)]
        sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max() - sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
        f1a11.imshow(sl_seqdisp.clip(0, 255), aspect='auto', interpolation='nearest', origin='upper',
                     cmap=mpl.pyplot.cm.gray)
        plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp) + str3(
            sfa_libs.comp_eta(sl_seq)[0:5]))

        # display first image
        # Alternative: im1.show(command="xv")
        f1a12 = plt.subplot(2, 3, 2)
        plt.title("A particular image in the sequence")
        # im_smalldisp = im_small.copy()
        # f1a12.imshow(im_smalldisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
        f1a13 = plt.subplot(2, 3, 3)
        plt.title("Reconstructed Image")

        f1a21 = plt.subplot(2, 3, 4)
        plt.title("Reconstruction Error")

        f1a22 = plt.subplot(2, 3, 5)
        plt.title("DIfferential Reconstruction y_pinv_(t+1) - y_pinv_(t)")

        f1a23 = plt.subplot(2, 3, 6)
        plt.title("Pseudoinverse of 0 / PINV(y) - PINV(0)")
        if show_linear_inv:
            sfa_zero = numpy.zeros((1, hierarchy_out_dim))
            pinv_zero = flow.inverse(sfa_zero)
            # WARNING L and RGB
            #    pinv_zero = pinv_zero.reshape((sTrain.subimage_height, sTrain.subimage_width))
            pinv_zero = pinv_zero[0, 0:num_pixels_per_image].reshape(subimage_shape)

            error_scale_disp = 1.5
            # WARNING L and RGB
            pinv_zero_disp = scale_to(pinv_zero, pinv_zero.mean(), pinv_zero.max() - pinv_zero.min(), max_clip / 2.0,
                                      max_clip, error_scale_disp, 'tanh')
            #    pinv_zero_disp = scale_to(pinv_zero/256.0, pinv_zero.mean()/256.0, pinv_zero.max()/256.0 -
            # pinv_zero.min()/256.0, 0.5, 1.0, error_scale_disp, 'tanh')
            f1a23.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, zero" % (
            pinv_zero_disp.min(), pinv_zero_disp.max(), pinv_zero_disp.std(), error_scale_disp))
            # WARNING L and RGB
            #    f1a23.imshow(pinv_zero_disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper',
            # cmap=mpl.pyplot.cm.gray)
            f1a23.imshow(pinv_zero_disp.clip(0, max_clip), aspect='auto', interpolation='nearest', origin='upper',
                         cmap=mpl.pyplot.cm.gray)
        else:
            pinv_zero = None

        # Retrieve Image in Sequence
        def on_press_inv(event):
            #### global plt, f1, f1a12, f1a13, f1a21, f1a22, fla23, L2, sTrain, sl_seq, pinv_zero, flow,
            # error_scale_disp #subimages
            print 'you pressed', event.button, event.xdata, event.ydata
            y = int(event.ydata)
            if y < 0:
                y = 0
            if y >= num_images:
                y = num_images - 1
            print "y=" + str(y)
            print "num_pixels_per_image=", num_pixels_per_image
            # Display Original Image
            # WARNING L and RGB
            #    subimage_im = subimages[y].reshape((sTrain.subimage_height, sTrain.subimage_width)) + 0.0
            subimage_im = subimages[y][0:num_pixels_per_image].reshape(subimage_shape) + 0.0

            if show_translation_x:
                if sTrain.trans_sampled:
                    subimage_im[:, sTrain.subimage_width / 2.0 - sTrain.translations_x[y]] = max_clip
                    subimage_im[:, sTrain.subimage_width / 2.0 - regression_Gauss_training[y] / reduction_factor] = 0
                else:
                    subimage_im[:,
                    sTrain.subimage_width / 2.0 - sTrain.translations_x[y] * sTrain.pixelsampling_x] = max_clip
                    subimage_im[:, sTrain.subimage_width / 2.0 - regression_Gauss_training[
                                                                     y] / reduction_factor * sTrain.pixelsampling_x] = 0

            f1a12.imshow(subimage_im.clip(0, max_clip), aspect='auto', interpolation='nearest', origin='upper',
                         cmap=mpl.pyplot.cm.gray)

            if show_linear_inv is False:
                f1.canvas.draw()
                return

                # Display Reconstructed Image
            data_out = sl_seq[y].reshape((1, hierarchy_out_dim))
            inverted_im = flow.inverse(data_out)
            inverted_im = inverted_im[0][0:num_pixels_per_image].reshape(subimage_shape)
            f1a13.imshow(inverted_im.clip(0, max_clip), aspect='auto', interpolation='nearest', origin='upper',
                         cmap=mpl.pyplot.cm.gray)

            # Display Reconstruction Error
            error_scale_disp = 1.5
            error_im = subimages[y][0:num_pixels_per_image].reshape(subimage_shape) - inverted_im
            error_im_disp = scale_to(error_im, error_im.mean(), error_im.max() - error_im.min(), max_clip / 2.0,
                                     max_clip, error_scale_disp, 'tanh')
            f1a21.imshow(error_im_disp.clip(0, max_clip), aspect='auto', interpolation='nearest', origin='upper',
                         cmap=mpl.pyplot.cm.gray)
            plt.axis = f1a21
            f1a21.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y=%d" % (
            error_im.min(), error_im.max(), error_im.std(), error_scale_disp, y))
            # Display Differencial change in reconstruction
            error_scale_disp = 1.5
            if y >= sTrain.num_images - 1:
                y_next = 0
            else:
                y_next = y + 1
            print "y_next=" + str(y_next)
            data_out2 = sl_seq[y_next].reshape((1, hierarchy_out_dim))
            inverted_im2 = flow.inverse(data_out2)[0, 0:num_pixels_per_image].reshape(subimage_shape)
            diff_im = inverted_im2 - inverted_im
            diff_im_disp = scale_to(diff_im, diff_im.mean(), diff_im.max() - diff_im.min(), max_clip / 2.0, max_clip,
                                    error_scale_disp, 'tanh')
            f1a22.imshow(diff_im_disp.clip(0, max_clip), aspect='auto', interpolation='nearest', origin='upper',
                         cmap=mpl.pyplot.cm.gray)
            plt.axis = f1a22
            f1a22.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y=%d" % (
            diff_im.min(), diff_im.max(), diff_im.std(), error_scale_disp, y))
            # Display Difference from PINV(y) and PINV(0)
            error_scale_disp = 1.0
            dif_pinv = inverted_im - pinv_zero
            dif_pinv_disp = scale_to(dif_pinv, dif_pinv.mean(), dif_pinv.max() - dif_pinv.min(), max_clip / 2.0,
                                     max_clip, error_scale_disp, 'tanh')
            f1a23.imshow(dif_pinv_disp.clip(0, max_clip), aspect='auto', interpolation='nearest', origin='upper',
                         cmap=mpl.pyplot.cm.gray)
            plt.axis = f1a23
            f1a23.set_xlabel("PINV(y) - PINV(0): min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y=%d" % (
            dif_pinv.min(), dif_pinv.max(), dif_pinv.std(), error_scale_disp, y))

            f1.canvas.draw()

        f1.canvas.mpl_connect('button_press_event', on_press_inv)

        ###############################################################################################################
        # ##### Interactive plot that allows the user to approximate an inverse using kNN and a global linear model ###
        ###############################################################################################################
        print "************ Displaying Test Set SFA and Inverses through kNN from Seen Id data **************"
        # TODO:Make this a parameter!
        kNN_k = 30
        # Create Figure
        fkNNinv = plt.figure()
        ax_5.subplots_adjust(hspace=0.5)
        plt.suptitle(Network.name + "Reconstruction using kNN (+ avg) from output features. k=" + str(kNN_k))

        # display SFA
        f_kNNinv_a11 = plt.subplot(2, 4, 1)
        plt.title("Output Unit (Top Node) New Id")
        sl_seqdisp = sl_seq_newid[:, range(0, hierarchy_out_dim)]
        sl_seqdisp = scale_to(sl_seqdisp, sl_seq_newid.mean(), sl_seq_newid.max() - sl_seq_newid.min(), 127.5, 255.0,
                              scale_disp, 'tanh')
        f_kNNinv_a11.imshow(sl_seqdisp.clip(0, 255), aspect='auto', interpolation='nearest', origin='upper',
                            cmap=mpl.pyplot.cm.gray)
        plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp) + str3(
            sfa_libs.comp_eta(sl_seq)[0:5]))

        # display first image
        # Alternative: im1.show(command="xv")
        f_kNNinv_a12 = plt.subplot(2, 4, 2)
        plt.title("A particular image in the sequence")
        # im_smalldisp = im_small.copy()
        # f_kNNinv_a12.imshow(im_smalldisp, aspect='auto', interpolation='nearest', origin='upper',
        # cmap=mpl.pyplot.cm.gray)
        f_kNNinv_a13 = plt.subplot(2, 4, 3)
        plt.title("Reconstructed Image using kNN")

        if estimate_explained_var_linear_global_N == -1:
            training_data_lmodel = subimages_seenid
            features_lmodel = sl_seq_seenid[:, 0:num_features_linear_model]
            indices_all_train_lmodel = more_nodes.random_subindices(training_data_lmodel.shape[0],
                                                                    number_samples_EV_linear_global)
            indices_all_newid = numpy.arange(subimages_newid.shape[0])  # Select all images of newid

            lr_node = mdp.nodes.LinearRegressionNode()
            sl_seq_training_sel = features_lmodel[indices_all_train_lmodel, :]
            subimages_train_sel = training_data_lmodel[indices_all_train_lmodel]
            lr_node.train(sl_seq_training_sel,
                          subimages_train_sel)  # Notice the input is "x"=n_sfa_x and the output to learn is "y" = x_pca
            lr_node.stop_training()

            sl_seq_newid_sel = sl_seq_newid[indices_all_newid, 0:num_features_linear_model]
            subimages_newid_app = lr_node.execute(sl_seq_newid_sel)
        else:
            subimages_newid_app = subimages_newid

        f_kNNinv_a14 = plt.subplot(2, 4, 4)
        if estimate_explained_var_linear_global_N == -1:
            plt.title("Linearly Reconstrion. #F=%d" % num_features_linear_model)
        else:
            plt.title("Linearly Reconstrion NOT Enabled. #F=%d" % num_features_linear_model)

        f_kNNinv_a21 = plt.subplot(2, 4, 5)
        plt.title("Reconstruction Error, k=1")

        f_kNNinv_a22 = plt.subplot(2, 4, 6)
        plt.title("Reconstructed Image using kNN, k=1")

        f_kNNinv_a23 = plt.subplot(2, 4, 7)
        plt.title("kNN Reconstruction Error")

        f_kNNinv_a24 = plt.subplot(2, 4, 8)
        plt.title("Linear Reconstruction Error")

        # Retrieve Image in Sequence
        def on_press_kNN(event):
            # Why is this declared global???
            # global plt, f_kNNinv_a12, f_kNNinv_a13, f_kNNinv_a14, f_kNNinv_a21, f_kNNinv_a22, f_kNNinv_a23,
            # f_kNNinv_a24, subimages_seenid, subimages_newid, sl_seq_seenid, sl_seq_newid, subimages_newid_app,
            # error_scale_disp
            print 'you pressed', event.button, event.xdata, event.ydata
            y = int(event.ydata)
            if y < 0:
                y = 0
            if y >= num_images_newid:
                y = num_images_newid - 1
            print "y=" + str(y)

            # Display Original Image
            # WARNING L and RGB
            #    subimage_im = subimages[y].reshape((sTrain.subimage_height, sTrain.subimage_width)) + 0.0
            subimage_im = subimages_newid[y][0:num_pixels_per_image].reshape(subimage_shape) + 0.0
            f_kNNinv_a12.imshow(subimage_im.clip(0, max_clip), aspect='auto', interpolation='nearest', origin='upper',
                                cmap=mpl.pyplot.cm.gray)
            f_kNNinv_a12.set_xlabel("Selected image y=%d" % y)

            # Display Reconstructed Image kNN
            data_out = sl_seq_newid[y].reshape((1, hierarchy_out_dim))
            x_app_test = more_nodes.approximate_kNN_op(subimages_seenid, sl_seq_seenid, data_out, k=kNN_k,
                                                       ignore_closest_match=True, operation="average")
            inverted_im_kNNavg = x_app_test[0, 0:num_pixels_per_image].reshape(subimage_shape)
            f_kNNinv_a13.imshow(inverted_im_kNNavg.clip(0, max_clip), aspect='auto', interpolation='nearest',
                                origin='upper', cmap=mpl.pyplot.cm.gray)

            # Display Linearly Reconstructed Image
            data_out = sl_seq_newid[y].reshape((1, hierarchy_out_dim))
            x_app_test = subimages_newid_app[y]
            inverted_im_LRec = x_app_test[0:num_pixels_per_image].reshape(subimage_shape)
            f_kNNinv_a14.imshow(inverted_im_LRec.clip(0, max_clip), aspect='auto', interpolation='nearest',
                                origin='upper', cmap=mpl.pyplot.cm.gray)

            # Display Reconstructed Image for kNN_k=1
            data_out = sl_seq_newid[y].reshape((1, hierarchy_out_dim))
            x_app_test = more_nodes.approximate_kNN_op(subimages_seenid, sl_seq_seenid, data_out, k=1,
                                                       ignore_closest_match=False, operation="average")
            inverted_im_kNN1 = x_app_test[0, 0:num_pixels_per_image].reshape(subimage_shape)
            f_kNNinv_a22.imshow(inverted_im_kNN1.clip(0, max_clip), aspect='auto', interpolation='nearest',
                                origin='upper', cmap=mpl.pyplot.cm.gray)

            # Display Reconstruction Error, kNN1
            error_scale_disp = 1.5
            error_im = subimage_im - inverted_im_kNN1
            error_im_disp = scale_to(error_im, error_im.mean(), error_im.max() - error_im.min(), max_clip / 2.0,
                                     max_clip, error_scale_disp, 'tanh')
            f_kNNinv_a21.imshow(error_im_disp.clip(0, max_clip), aspect='auto', interpolation='nearest', origin='upper',
                                cmap=mpl.pyplot.cm.gray)
            plt.axis = f_kNNinv_a21
            f_kNNinv_a21.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f" % \
                                    (error_im.min(), error_im.max(), error_im.std(), error_scale_disp))

            # Display Reconstruction Error, kNN average
            error_scale_disp = 1.5
            error_im = subimage_im - inverted_im_kNNavg
            error_im_disp = scale_to(error_im, error_im.mean(), error_im.max() - error_im.min(), max_clip / 2.0,
                                     max_clip, error_scale_disp, 'tanh')
            f_kNNinv_a23.imshow(error_im_disp.clip(0, max_clip), aspect='auto', interpolation='nearest', origin='upper',
                                cmap=mpl.pyplot.cm.gray)
            plt.axis = f_kNNinv_a23
            f_kNNinv_a23.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f" % (error_im.min(), error_im.max(),
                                                                                  error_im.std(), error_scale_disp))

            # Display Linear Reconstruction Error
            error_scale_disp = 1.5
            error_im = subimage_im - inverted_im_LRec
            error_im_disp = scale_to(error_im, error_im.mean(), error_im.max() - error_im.min(), max_clip / 2.0,
                                     max_clip, error_scale_disp, 'tanh')
            f_kNNinv_a24.imshow(error_im_disp.clip(0, max_clip), aspect='auto', interpolation='nearest', origin='upper',
                                cmap=mpl.pyplot.cm.gray)
            plt.axis = f_kNNinv_a24
            f_kNNinv_a24.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f" %
                                    (error_im.min(), error_im.max(), error_im.std(), error_scale_disp))

            fkNNinv.canvas.draw()

        fkNNinv.canvas.mpl_connect('button_press_event', on_press_kNN)

        ##############################################################################################################
        # ##### Plot that shows the classification of the training, supervised, and test datasets ####################
        ##############################################################################################################

        print "************ Displaying Classification Results **************"
        f2 = plt.figure()
        plt.suptitle("Classification Results (Class Numbers)  using %d Slow Signals" % reg_num_signals)
        # plt.title("Training Set")
        p11 = f2.add_subplot(311, frame_on=False)
        xlabel = "Image Number, Training Set."
        p11.plot(numpy.arange(len(correct_classes_training)), correct_classes_training, 'r.', markersize=2,
                 markerfacecolor='red')
        if enable_ncc_cfr:
            p11.plot(numpy.arange(len(classes_ncc_training)), classes_ncc_training, 'k.', markersize=2,
                     markerfacecolor='black')
            xlabel += " CR_ncc=%.3f," % results.class_ncc_rate_train
        if enable_kNN_cfr:
            p11.plot(numpy.arange(len(classes_kNN_training)), classes_kNN_training, 'b.', markersize=2,
                     markerfacecolor='blue')
            xlabel += " CR_kNN=%.3f," % results.class_kNN_rate_train
        p11.plot(numpy.arange(len(classes_Gauss_training)), classes_Gauss_training, 'm.', markersize=2,
                 markerfacecolor='magenta')
        xlabel += " CR_Gauss=%.3f," % results.class_Gauss_rate_train
        if enable_svm_cfr:
            p11.plot(numpy.arange(len(classes_svm_training)), classes_svm_training, 'g.', markersize=2,
                     markerfacecolor='green')
            xlabel += " CR_SVM=%.3f," % results.class_svm_rate_train
        plt.xlabel(xlabel)
        plt.ylabel("Class Number")
        p11.grid(True)
        # draw horizontal and vertical lines
        # majorLocator_x   = MultipleLocator(block_size)
        # majorLocator_y   = MultipleLocator(1)
        # p11.xaxis.set_major_locator(majorLocator_x)
        ##p11.yaxis.set_major_locator(majorLocator_y)
        # plt.xticks(numpy.arange(0, len(labels_kNN_training), block_size))
        # plt.yticks(numpy.arange(0, len(labels_kNN_training), block_size))
        # print "Block_size is: ", block_size

        p12 = f2.add_subplot(312, frame_on=False)
        xlabel = "Image Number, Seen Id Set."
        p12.plot(numpy.arange(len(correct_classes_seenid)), correct_classes_seenid, 'r.', markersize=2,
                 markerfacecolor='red')
        if enable_ncc_cfr:
            p12.plot(numpy.arange(len(classes_ncc_seenid)), classes_ncc_seenid, 'k.', markersize=2,
                     markerfacecolor='black')
            xlabel += " CR_ncc=%.3f," % results.class_ncc_rate_seenid
        if enable_kNN_cfr:
            p12.plot(numpy.arange(len(classes_kNN_seenid)), classes_kNN_seenid, 'b.', markersize=2,
                     markerfacecolor='blue')
            xlabel += " CR_kNN=%.3f," % results.class_kNN_rate_seenid
        p12.plot(numpy.arange(len(classes_Gauss_seenid)), classes_Gauss_seenid, 'm.', markersize=2,
                 markerfacecolor='magenta')
        xlabel += " CR_Gauss=%.3f," % results.class_Gauss_rate_seenid
        if enable_svm_cfr:
            p12.plot(numpy.arange(len(classes_kNN_seenid)), classes_svm_seenid, 'g.', markersize=2,
                     markerfacecolor='green')
            xlabel += " CR_SVM=%.3f," % results.class_svm_rate_seenid
        plt.xlabel(xlabel)
        plt.ylabel("Class Number")
        p12.grid(True)
        # p12.plot(numpy.arange(len(labels_ccc_seenid)), correct_classes_seenid, 'mo', markersize=3,
        # markerfacecolor='magenta')
        # majorLocator_y   = MultipleLocator(block_size)
        ##majorLocator_x   = MultipleLocator(block_size_seenid)
        # majorLocator_x   = MultipleLocator(block_size_seenid)
        # p12.xaxis.set_major_locator(majorLocator_x)
        # p12.yaxis.set_major_locator(majorLocator_y)
        # majorLocator_y   = MultipleLocator(block_size)

        p13 = f2.add_subplot(313, frame_on=False)
        xlabel = "Image Number, New Id Set."
        p13.plot(numpy.arange(len(correct_classes_newid)), correct_classes_newid, 'r.', markersize=2,
                 markerfacecolor='red')
        if enable_ncc_cfr:
            p13.plot(numpy.arange(len(classes_ncc_newid)), classes_ncc_newid, 'k.', markersize=2,
                     markerfacecolor='black')
            xlabel += " CR_ncc=%.3f," % results.class_ncc_rate_newid
        if enable_kNN_cfr:
            p13.plot(numpy.arange(len(classes_kNN_newid)), classes_kNN_newid, 'b.', markersize=2,
                     markerfacecolor='blue')
            xlabel += " CR_kNN=%.3f," % results.class_kNN_rate_newid
        p13.plot(numpy.arange(len(classes_Gauss_newid)), classes_Gauss_newid, 'm.', markersize=2,
                 markerfacecolor='magenta')
        xlabel += " CR_Gauss=%.3f," % results.class_Gauss_rate_newid
        if enable_svm_cfr:
            p13.plot(numpy.arange(len(classes_svm_newid)), classes_svm_newid, 'g.', markersize=2,
                     markerfacecolor='green')
            xlabel += " CR_svm=%.3f," % results.class_svm_rate_newid
        plt.xlabel(xlabel)
        plt.ylabel("Class Number")
        p13.grid(True)
        # majorLocator_y = MultipleLocator(block_size)
        ##majorLocator_x   = MultipleLocator(block_size_seenid)
        # majorLocator_x   = MultipleLocator(block_size_newid)
        # p13.xaxis.set_major_locator(majorLocator_x)
        ##p13.yaxis.set_major_locator(majorLocator_y)


        ################################################################################################################
        # ##### Plot that shows the label estimations for the training, supervised, and test datasets ##################
        ################################################################################################################
        print "************ Displaying Regression Results **************"
        f3 = plt.figure()

        plt.suptitle("Regression Results (Labels) using %d Slow Signals" % reg_num_signals)
        # plt.title("Training Set")
        p11 = f3.add_subplot(311, frame_on=False)
        # correct_classes_training = numpy.arange(len(labels_ccc_training)) / block_size
        xlabel = "Image Number, Training Set."
        regression_text_labels = []
        if enable_ncc_cfr:
            p11.plot(numpy.arange(len(labels_ncc_training)), labels_ncc_training, 'k.', markersize=3,
                     markerfacecolor='black')
            xlabel += " MSE_ncc=%f," % results.mse_ncc_train
            regression_text_labels.append("Nearest Centroid Class.")
        if enable_kNN_cfr:
            p11.plot(numpy.arange(len(labels_kNN_training)), labels_kNN_training, 'b.', markersize=3,
                     markerfacecolor='blue')
            xlabel += " MSE_kNN=%f," % results.mse_kNN_train
            regression_text_labels.append("kNN")
        if enable_svm_cfr:
            p11.plot(numpy.arange(len(regression_svm_training)), regression_svm_training, 'g.', markersize=3,
                     markerfacecolor='green')
            xlabel += " MSE_svm=%f," % results.mse_svm_train
            regression_text_labels.append("SVM")
        if enable_gc_cfr:
            p11.plot(numpy.arange(len(regression_Gauss_training)), regression_Gauss_training, 'm.', markersize=3,
                     markerfacecolor='magenta')
            xlabel += " MSE_Gauss=%f," % results.mse_gauss_train
            regression_text_labels.append("Gaussian Class/Regr.")
        if enable_lr_cfr:
            #        p11.plot(numpy.arange(len(correct_labels_training)), regression_lr_training, 'b.', markersize=3,
            # markerfacecolor='blue')
            p11.plot(numpy.arange(len(correct_labels_training)), regression_lr_training, 'c.', markersize=3,
                     markerfacecolor='cyan')
            xlabel += " MSE_lr=%f," % (results.mse_lr_train)
            regression_text_labels.append("LR")

        p11.plot(numpy.arange(len(correct_labels_training)), correct_labels_training, 'k.', markersize=3,
                 markerfacecolor='black')
        #    p11.plot(numpy.arange(len(correct_labels_training)), correct_labels_training, 'r.', markersize=3,
        # markerfacecolor='red')
        regression_text_labels.append("Ground Truth")
        # #draw horizontal and vertical lines
        # majorLocator   = MultipleLocator(block_size)
        # p11.xaxis.set_major_locator(majorLocator)
        # #p11.yaxis.set_major_locator(majorLocator)
        # plt.xticks(numpy.arange(0, len(labels_ccc_training), block_size))
        # plt.yticks(numpy.arange(0, len(labels_ccc_training), block_size))
        plt.xlabel(xlabel)
        plt.ylabel("Label")
        plt.legend(regression_text_labels, loc=2)
        p11.grid(True)

        p12 = f3.add_subplot(312, frame_on=False)
        xlabel = "Image Number, Seen Id Set."
        # correct_classes_seenid = numpy.arange(len(labels_ccc_seenid)) * len(labels_ccc_training) /
        # len(labels_ccc_seenid) / block_size
        if enable_ncc_cfr:
            p12.plot(numpy.arange(len(labels_ncc_seenid)), labels_ncc_seenid, 'k.', markersize=4,
                     markerfacecolor='black')
            xlabel += " MSE_ncc=%f," % results.mse_ncc_seenid
        if enable_kNN_cfr:
            p12.plot(numpy.arange(len(labels_kNN_seenid)), labels_kNN_seenid, 'b.', markersize=4,
                     markerfacecolor='blue')
            xlabel += " MSE_kNN=%f," % results.mse_kNN_seenid
        if enable_svm_cfr:
            p12.plot(numpy.arange(len(regression_svm_seenid)), regression_svm_seenid, 'g.', markersize=4,
                     markerfacecolor='green')
            xlabel += " MSE_svm=%f," % results.mse_svm_seenid
        if enable_gc_cfr:
            p12.plot(numpy.arange(len(regression_Gauss_seenid)), regression_Gauss_seenid, 'm.', markersize=4,
                     markerfacecolor='magenta')
            xlabel += " MSE_Gauss=%f," % results.mse_gauss_seenid
        if enable_lr_cfr:
            #        p12.plot(numpy.arange(len(regression_lr_seenid)), regression_lr_seenid, 'b.', markersize=4,
            # markerfacecolor='blue')
            p12.plot(numpy.arange(len(regression_lr_seenid)), regression_lr_seenid, 'c.', markersize=4,
                     markerfacecolor='cyan')
            xlabel += " MSE_lr=%f," % results.mse_lr_seenid

        p12.plot(numpy.arange(len(correct_labels_seenid)), correct_labels_seenid, 'k.', markersize=4,
                 markerfacecolor='black')
        #    p12.plot(numpy.arange(len(correct_labels_seenid)), correct_labels_seenid, 'r.', markersize=4,
        # markerfacecolor='red')

        # #majorLocator_y   = MultipleLocator(block_size)
        # #majorLocator_x   = MultipleLocator(block_size_seenid)
        # majorLocator_x   = MultipleLocator( len(labels_ccc_seenid) * block_size / len(labels_ccc_training))
        # p12.xaxis.set_major_locator(majorLocator_x)
        # #p12.yaxis.set_major_locator(majorLocator_y)
        plt.xlabel(xlabel)
        plt.ylabel("Label")
        plt.legend(regression_text_labels, loc=2)
        p12.grid(True)

        p13 = f3.add_subplot(313, frame_on=False)
        xlabel = "Image Number, New Id Set"
        if enable_ncc_cfr:
            p13.plot(numpy.arange(len(labels_kNN_newid)), labels_kNN_newid, 'k.', markersize=8, markerfacecolor='black')
            xlabel += " MSE_ncc=%f," % results.mse_ncc_newid
        if enable_kNN_cfr:
            p13.plot(numpy.arange(len(labels_kNN_newid)), labels_kNN_newid, 'b.', markersize=8, markerfacecolor='blue')
            xlabel += " MSE_kNN=%f," % results.mse_kNN_newid
        if enable_svm_cfr:
            p13.plot(numpy.arange(len(regression_svm_newid)), regression_svm_newid, 'g.', markersize=8,
                     markerfacecolor='green')
            xlabel += " MSE_svm=%f," % results.mse_svm_newid
        if enable_gc_cfr:
            p13.plot(numpy.arange(len(regression_Gauss_newid)), regression_Gauss_newid, 'm.', markersize=8,
                     markerfacecolor='magenta')
            xlabel += " MSE_Gauss=%f," % results.mse_gauss_newid

        if enable_lr_cfr:
            #        p13.plot(numpy.arange(len(regression_lr_newid)), regression_lr_newid, 'b.', markersize=8,
            # markerfacecolor='blue')
            p13.plot(numpy.arange(len(regression_lr_newid)), regression_lr_newid, 'c.', markersize=8,
                     markerfacecolor='cyan')
            xlabel += " MSE_lr=%f," % results.mse_lr_newid

        p13.plot(numpy.arange(len(correct_labels_newid)), correct_labels_newid, 'k.', markersize=8,
                 markerfacecolor='black')
        #    p13.plot(numpy.arange(len(correct_labels_newid)), correct_labels_newid, 'r.', markersize=8,
        # markerfacecolor='red')

        # #majorLocator_y   = MultipleLocator(block_size)
        # #majorLocator_x   = MultipleLocator(block_size_seenid)
        # majorLocator_x   = MultipleLocator( len(labels_ccc_newid) * block_size / len(labels_ccc_training))
        # p13.xaxis.set_major_locator(majorLocator_x)
        # #p12.yaxis.set_major_locator(majorLocator_y)
        plt.xlabel(xlabel)
        plt.ylabel("Label")
        plt.legend(regression_text_labels, loc=2)
        p13.grid(True)

        ###############################################################################################################
        # ##### Plot that displays the class probabilities estimated by a gaussian classifier #########################
        ###############################################################################################################
        print "************** Displaying Probability Profiles ***********"
        f4 = plt.figure()
        plt.suptitle(
            "Probability Profiles of Gaussian Classifier Using %d Signals for Classification" % reg_num_signals)

        # display Probability Profile of Training Set
        ax11 = plt.subplot(1, 3, 1)
        plt.title("(Network) Training Set")
        # cax = p11.add_axes([1, 10, 1, 10])
        pic = ax11.imshow(probs_training, aspect='auto', interpolation='nearest', origin='upper',
                          cmap=mpl.pyplot.cm.hot)
        plt.xlabel("Class Number")
        plt.ylabel("Image Number, Training Set")
        f4.colorbar(pic)

        # display Probability Profile of Seen Id
        ax11 = plt.subplot(1, 3, 2)
        plt.title("Seen Id Test Set")
        pic = ax11.imshow(probs_seenid, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.hot)
        plt.xlabel("Class Number")
        plt.ylabel("Image Number, Seen Id Set")
        f4.colorbar(pic)

        # display Probability Profile of New Id
        ax11 = plt.subplot(1, 3, 3)
        plt.title("New Id Test Set")
        pic = ax11.imshow(probs_newid, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.hot)
        plt.xlabel("Class Number")
        plt.ylabel("Image Number, New Id Set")
        f4.colorbar(pic)

        ###############################################################################################################
        # ##### Plot that displays  #################################
        ###############################################################################################################
        print "************ Displaying Linear (or Non-Linear) Morphs and Masks Learned by SFA **********"
        # Create Figure
        ax6 = plt.figure()
        ax6.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)

        # ax_6.subplots_adjust(hspace=0.5)
        plt.suptitle("Linear (or Non-Linear) Morphs using SFA")

        # display SFA
        ax6_11 = plt.subplot(4, 5, 1)
        plt.title("Train-Signals in Slow Domain")
        sl_seqdisp = sl_seq[:, range(0, hierarchy_out_dim)]
        sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max() - sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
        ax6_11.imshow(sl_seqdisp.clip(0, 255), aspect='auto', interpolation='nearest', origin='upper',
                      cmap=mpl.pyplot.cm.gray)
        plt.ylabel("Image number")
        # plt.xlabel("Slow Signal S[im][sl]")

        ax6_12 = plt.subplot(4, 5, 2)
        plt.title("Selected Original Image")
        ax6_12.axis('off')

        ax6_13 = plt.subplot(4, 5, 3)
        plt.title("Approx. Image x'")
        ax6_13.axis('off')

        ax6_14 = plt.subplot(4, 5, 4)
        plt.title("Re-Approx Image x''")
        ax6_14.axis('off')

        if show_linear_inv:
            ax6_15 = plt.subplot(4, 5, 5)
            plt.title("Avg. Image H-1(0)=z'")
            error_scale_disp = 1.0
            # z_p = pinv_zero
            pinv_zero_disp = scale_to(pinv_zero, pinv_zero.mean(), pinv_zero.max() - pinv_zero.min(), max_clip / 2.0,
                                      max_clip, error_scale_disp, 'tanh')
            ax6_15.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, zero" %
                              (pinv_zero_disp.min(), pinv_zero_disp.max(), pinv_zero_disp.std(), error_scale_disp))
            ax6_15.imshow(pinv_zero_disp.clip(0, max_clip), aspect='equal', interpolation='nearest', origin='upper',
                          cmap=mpl.pyplot.cm.gray)
            ax6_15.axis('off')

        ax6_21 = plt.subplot(4, 5, 6)
        plt.title("H-1 (y*), y*[sl]=-2")
        plt.yticks([])
        plt.xticks([])
        plt.ylabel("Modified Projection")

        ax6_22 = plt.subplot(4, 5, 7)
        plt.title("H-1 (y*), y*[sl]=-1")
        ax6_22.axis('off')

        ax6_23 = plt.subplot(4, 5, 8)
        plt.title("H-1 (y*), y*[sl]=0")
        ax6_23.axis('off')

        ax6_24 = plt.subplot(4, 5, 9)
        plt.title("H-1 (y*), y*[sl]=1")
        ax6_24.axis('off')

        ax6_25 = plt.subplot(4, 5, 10)
        # plt.title("x' - rec S[im][sl]=2")
        plt.title("H-1 (y*), y*[sl]=2")
        ax6_25.axis('off')

        ax6_31 = plt.subplot(4, 5, 11)
        plt.title("x-x'+H-1(S*-S), S*[sl]=-2")
        plt.yticks([])
        plt.xticks([])
        plt.ylabel("Morph")

        ax6_32 = plt.subplot(4, 5, 12)
        plt.title("x-x'+H-1(S*-S), S*[sl]=-1")
        ax6_32.axis('off')

        ax6_33 = plt.subplot(4, 5, 13)
        plt.title("x-x'+H-1(S*-S), S*[sl]=0")
        ax6_33.axis('off')

        ax6_34 = plt.subplot(4, 5, 14)
        plt.title("x-x'+H-1(S*-S), S*[sl]=1")
        ax6_34.axis('off')

        ax6_35 = plt.subplot(4, 5, 15)
        plt.title("x-x'+H-1(S*-S), S*[sl]=2")
        ax6_35.axis('off')

        ax6_41 = plt.subplot(4, 5, 16)
        plt.title("x-x'+H-1(SFA_train[0]-S)")
        plt.yticks([])
        plt.xticks([])
        plt.ylabel("Morph from SFA_Train")

        ax6_42 = plt.subplot(4, 5, 17)
        plt.title("x-x'+H-1(SFA_train[1/4]-S)")
        ax6_42.axis('off')

        ax6_43 = plt.subplot(4, 5, 18)
        plt.title("x-x'+H-1(SFA_train[2/4]-S)")
        ax6_43.axis('off')

        ax6_44 = plt.subplot(4, 5, 19)
        plt.title("x-x'+H-1(SFA_train[3/4]-S)")
        ax6_44.axis('off')

        ax6_45 = plt.subplot(4, 5, 20)
        plt.title("x-x'+H-1(SFA_train[4/4]-S)")
        ax6_45.axis('off')

        print "************ Displaying Linear (or Non-Linear) Masks Learned by SFA **********"
        # Create Figure
        ax7 = plt.figure()
        ax7.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)

        # ax_6.subplots_adjust(hspace=0.5)
        plt.suptitle("Linear (or Non-Linear) Masks Learned by SFA [0 - 4]")

        global mask_normalize

        mask_normalize = False
        lim_delta_sfa = 0.01
        num_masks = 4
        slow_values = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
        axes = range(num_masks)
        for ma in range(num_masks):
            axes[ma] = range(len(slow_values))

        for ma in range(num_masks):
            for sl, slow_value in enumerate(slow_values):
                tmp_ax = plt.subplot(4, 6, ma * len(slow_values) + sl + 1)
                plt.axes(tmp_ax)
                plt.title("H-1( S[%d]=%d ) - z'" % (ma, slow_value))

                if sl == 0:
                    plt.yticks([])
                    plt.xticks([])
                    plt.ylabel("Mask[%d]" % ma)
                else:
                    tmp_ax.axis('off')
                axes[ma][sl] = tmp_ax

        # Create Figure
        ax8 = plt.figure()
        ax8.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)

        # ax_6.subplots_adjust(hspace=0.5)
        plt.suptitle("Linear (or Non-Linear) Masks Learned by SFA [4 to 15]")

        masks2 = range(4, 16)
        num_masks2 = len(masks2)
        slow_values2 = [-1.0, 1.0]
        axes2 = range(num_masks2)
        for ma in range(num_masks2):
            axes2[ma] = range(len(slow_values2))

        for ma, mask in enumerate(masks2):
            for sl, slow_value in enumerate(slow_values2):
                tmp_ax = plt.subplot(4, 6, ma * len(slow_values2) + sl + 1)
                plt.axes(tmp_ax)
                plt.title("H-1( S[%d]=%d ) - z'" % (mask, slow_value))
                if sl == 0:
                    plt.yticks([])
                    plt.xticks([])
                    plt.ylabel("Mask[%d]" % mask)
                else:
                    tmp_ax.axis('off')
                axes2[ma][sl] = tmp_ax

        def mask_on_press(event):
            # #global plt, ax6, ax6_11, ax6_12, ax6_13, ax6_14, ax6_21, ax6_22, ax6_23, ax6_24, ax6_25, ax6_31,
            # ax6_32, ax6_33, ax6_34, ax6_35, ax6_41, ax6_42, ax6_43, ax6_44, ax6_45
            # #global ax7, axes, num_masks, slow_values, mask_normalize, lim_delta_sfa
            global mask_normalize
            # #global ax8, axes2, masks2, slow_values2
            # ###global ax9, ax9_11, ax9_12, ax9_13, ax9_14, ax9_21, ax9_22, ax9_23, ax9_24, ax9_25
            # ##global subimages, sTrain, sl_seq, pinv_zero, flow, error_scale_disp

            print 'you pressed', event.button, event.xdata, event.ydata

            if event.xdata is None or event.ydata is None:
                mask_normalize = not mask_normalize
                print "mask_normalize is: ", mask_normalize
                return

            y = int(event.ydata)
            if y < 0:
                y = 0
            if y >= num_images:
                y = num_images - 1
            x = int(event.xdata)
            if x < 0:
                x = 0
            if x >= hierarchy_out_dim:
                x = hierarchy_out_dim - 1
            print "Image Selected=" + str(y) + " , Slow Component Selected=" + str(x)

            print "Displaying Original and Reconstructions"
            # Display Original Image
            subimage_im = subimages[y][0:num_pixels_per_image].reshape(subimage_shape)
            ax6_12.imshow(subimage_im.clip(0, max_clip), vmin=0, vmax=max_clip, aspect='equal', interpolation='nearest',
                          origin='upper', cmap=mpl.pyplot.cm.gray)

            if show_linear_inv:
                # Display Reconstructed Image
                data_out = sl_seq[y].reshape((1, hierarchy_out_dim))
                inverted_im = flow.inverse(data_out)
                inverted_im = inverted_im[0, 0:num_pixels_per_image].reshape(subimage_shape)
                x_p = inverted_im
                inverted_im_ori = inverted_im.copy()
                ax6_13.imshow(inverted_im.clip(0, max_clip), vmin=0, vmax=max_clip, aspect='equal',
                              interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)

                # Display Re-Reconstructed Image
                re_data_out = flow.execute(inverted_im_ori.reshape((1, signals_per_image)))
                re_inverted_im = flow.inverse(re_data_out)
                re_inverted_im = re_inverted_im[0, 0:num_pixels_per_image].reshape(subimage_shape)
                ax6_14.imshow(re_inverted_im.clip(0, max_clip), vmin=0, vmax=max_clip, aspect='equal',
                              interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)

            if show_linear_morphs:
                print "Displaying Morphs, Original Version, no localized inverses"
                # Display: Altered Reconstructions
                # each tuple has the form: (val of slow_signal, remove, axes for display)
                # where remove is None, "avg" or "ori"
                error_scale_disp = 1.0
                disp_data = [(-2, "inv", ax6_21), (-1, "inv", ax6_22), (0, "inv", ax6_23), (1, "inv", ax6_24),
                             (2, "inv", ax6_25), (-2, "mor", ax6_31), (-1, "mor", ax6_32), (0, "mor", ax6_33),
                             (1, "mor", ax6_34), (2, "mor", ax6_35), (-2, "mo2", ax6_41), (-1, "mo2", ax6_42),
                             (0, "mo2", ax6_43), (1, "mo2", ax6_44), (2, "mo2", ax6_45)]

                work_sfa = data_out.copy()

                for slow_value, display_type, fig_axes in disp_data:
                    work_sfa[0][x] = slow_value
                    inverted_im = flow.inverse(work_sfa)
                    inverted_im = inverted_im[0, 0:num_pixels_per_image].reshape(subimage_shape)
                    if display_type == "inv":
                        fig_axes.imshow(inverted_im.clip(0, max_clip), vmin=0, vmax=max_clip, aspect='equal',
                                        interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
                    elif display_type == "mor":
                        # delta_sfa = sfa*-sfa
                        delta_sfa = numpy.zeros((1, hierarchy_out_dim))
                        delta_sfa[0][x] = slow_value
                        delta_im = flow.inverse(delta_sfa)
                        delta_im = delta_im[0, 0:num_pixels_per_image].reshape(subimage_shape)
                        morphed_im = subimage_im - x_p + delta_im
                        #            morphed_im = subimage_im - x_p + inverted_im - z_p
                        #            morphed_im = morphed.reshape((sTrain.subimage_height, sTrain.subimage_width))
                        #            inverted_im = inverted_im - pinv_zero
                        #            inverted_im_disp = scale_to(inverted_im, inverted_im.mean(), inverted_im.max() -
                        # inverted_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
                        #            morphed_im_disp = scale_to(morphed_im, morphed_im.mean(), morphed_im.max() -
                        # morphed_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
                        morphed_im_disp = morphed_im
                        fig_axes.imshow(morphed_im_disp.clip(0, max_clip), vmin=0, vmax=max_clip, aspect='equal',
                                        interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
                    elif display_type == "mo2":
                        # delta_sfa = sfa*-sfa
                        sfa_asterix = sl_seq[(slow_value + 2) * (num_images - 1) / 4].reshape((1, hierarchy_out_dim))
                        delta_sfa = sfa_asterix - data_out
                        delta_im = flow.inverse(delta_sfa)
                        delta_im = delta_im[0, 0:num_pixels_per_image].reshape(subimage_shape)
                        morphed_im = subimage_im - x_p + delta_im
                        morphed_im_disp = morphed_im
                        fig_axes.imshow(morphed_im_disp.clip(0, max_clip), vmin=0, vmax=max_clip, aspect='equal',
                                        interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
            ax6.canvas.draw()

            if show_linear_masks:
                print "Displaying Masks [0-3]"
                for ma in range(num_masks):
                    for sl, slow_value in enumerate(slow_values):
                        tmp_ax = axes[ma][sl]

                        print "Computing mask %d, slow_value %d" % (ma, slow_value)
                        work_sfa = data_out.copy()
                        work_sfa[0][ma] = work_sfa[0][ma] + slow_value * lim_delta_sfa
                        mask_im = flow.inverse(work_sfa)
                        mask_im = (mask_im[0, 0:num_pixels_per_image].reshape(subimage_shape) - x_p) / lim_delta_sfa
                        if mask_normalize:
                            mask_im_disp = scale_to(mask_im, 0.0, mask_im.max() - mask_im.min(), max_clip / 2.0,
                                                    max_clip / 2.0, error_scale_disp, 'tanh')
                        else:
                            mask_im_disp = mask_im + max_clip / 2.0
                        axes[ma][sl].imshow(mask_im_disp.clip(0, max_clip), vmin=0, vmax=max_clip, aspect='equal',
                                            interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
                ax7.canvas.draw()

            if show_linear_masks_ext:
                print "Displaying Masks [4-15]"
                for ma, mask in enumerate(masks2):
                    for sl, slow_value in enumerate(slow_values2):
                        tmp_ax = axes2[ma][sl]
                        work_sfa = data_out.copy()
                        work_sfa[0][mask] += slow_value * lim_delta_sfa
                        mask_im = flow.inverse(work_sfa)
                        mask_im = (mask_im[0, 0:num_pixels_per_image].reshape(subimage_shape) - x_p) / lim_delta_sfa
                        if mask_normalize:
                            mask_im_disp = scale_to(mask_im, 0.0, mask_im.max() - mask_im.min(), max_clip / 2.0,
                                                    max_clip, error_scale_disp, 'tanh')
                        else:
                            mask_im_disp = mask_im + max_clip / 2.0
                        axes2[ma][sl].imshow(mask_im_disp.clip(0, max_clip), vmin=0, vmax=max_clip, aspect='equal',
                                             interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
                ax8.canvas.draw()

        ax6.canvas.mpl_connect('button_press_event', mask_on_press)

        # # #         print "************ Displaying Localized Morphs and Masks **********"
        # # #         #Create Figure
        # # #         ax9 = plt.figure()
        # # #         ax9.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
        # # #
        # # #         #ax_6.subplots_adjust(hspace=0.5)
        # # #         plt.suptitle("Localized Linear (or Non-Linear) Morphs")
        # # #
        # # #         ax9_11 = plt.subplot(4,5,1)
        # # #         plt.title("Train-Signals in Slow Domain")
        # # #         sl_seqdisp = sl_seq[:, range(0,hierarchy_out_dim)]
        # # #         sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0,
        # # # scale_disp, 'tanh')
        # # #         ax9_11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper',
        # # # cmap=mpl.pyplot.cm.gray)
        # # #         plt.ylabel("Image number")
        # # #
        # # #         ax9_12 = plt.subplot(4,5,2)
        # # #         plt.title("Selected Original Image")
        # # #         ax9_11.axis('off')
        # # #
        # # #         ax9_13 = plt.subplot(4,5,3)
        # # #         plt.title("Loc. Approx. Image x'")
        # # #         ax9_12.axis('off')
        # # #
        # # #         ax9_14 = plt.subplot(4,5,4)
        # # #         plt.title("Loc. Re-Approx Image x''")
        # # #         ax9_13.axis('off')
        # # #
        # # #         ax9_21 = plt.subplot(4,5,6)
        # # #         plt.title("-8*Mask(cl -> cl_prev)")
        # # #         plt.yticks([])
        # # #         plt.xticks([])
        # # #         plt.ylabel("Modified Loc. Inv")
        # # #
        # # #         ax9_22 = plt.subplot(4,5,7)
        # # #         plt.title("-4*Mask(cl -> cl_prev)")
        # # #         ax9_22.axis('off')
        # # #
        # # #         ax9_23 = plt.subplot(4,5,8)
        # # #         plt.title("2*Mask(cl -> cl_prev)")
        # # #         ax9_23.axis('off')
        # # #
        # # #         ax9_24 = plt.subplot(4,5,9)
        # # #         plt.title("4*Mask(cl -> cl_prev)")
        # # #         ax9_24.axis('off')
        # # #
        # # #         ax9_25 = plt.subplot(4,5,10)
        # # #         plt.title("8*Mask(cl -> cl_prev)")
        # # #         ax9_25.axis('off')
        # # #
        # # #         print "************ Displaying Localized Morphs Learned by SFA **********"
        # # #         #Create Figure
        # # #         ax10 = plt.figure()
        # # #         ax10.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
        # # #
        # # #         num_morphs = 20
        # # #         morph_step = 0.5
        # # #         all_axes_morph_inc = []
        # # #         all_classes_morph_inc = numpy.arange(num_morphs)*morph_step
        # # #         for i in range(len(all_classes_morph_inc)):
        # # #             tmp_ax = plt.subplot(4,5,i+1)
        # # #             plt.title("Morph(cl* -> cl*)")
        # # #             tmp_ax.axis('off')
        # # #             all_axes_morph_inc.append(tmp_ax)
        # # #
        # # #
        # # #         ax11 = plt.figure()
        # # #         ax11.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
        # # #
        # # #         all_axes_morph_dec = []
        # # #         all_classes_morph_dec = numpy.arange(0, -1 * num_morphs, -1) * morph_step
        # # #         for i in range(len(all_classes_morph_dec)):
        # # #             tmp_ax = plt.subplot(4, 5,20-i)
        # # #             plt.title("Morph (cl*-> cl*)")
        # # #             tmp_ax.axis('off')
        # # #             all_axes_morph_dec.append(tmp_ax)

        # morphed sequence in SFA domain
        ax12 = plt.figure()
        # ##        ax12_1 = plt.subplot(2, 2, 1)
        # ##        plt.title("SFA of Morphed Images")
        ax12_2 = plt.subplot(1, 1, 1)
        plt.title("Average SFA for each Class")
        sl_seq_meandisp = scale_to(sl_seq_training_mean, sl_seq_training_mean.mean(),
                                   sl_seq_training_mean.max() - sl_seq_training_mean.min(), 127.5, 255.0, scale_disp,
                                   'tanh')
        ax12_2.imshow(sl_seq_meandisp.clip(0, 255), aspect='auto', interpolation='nearest', origin='upper',
                      cmap=mpl.pyplot.cm.gray)
        # ##        ax12_3 = plt.subplot(2, 2, 3)
        # ##        plt.title("SFA of Selected Image")

        # # #         print "************ Displaying Localized) Morphs Learned by SFA **********"
        # # #         #Create Figure
        # # #         ax13 = plt.figure()
        # # #         ax13.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
        # # #
        # # #         all_axes_mask_inc = []
        # # #         for i in range(len(all_classes_morph_inc)):
        # # #             tmp_ax = plt.subplot(4,5,i+1)
        # # #             plt.title("Mask(cl* -> cl*)")
        # # #             tmp_ax.axis('off')
        # # #             all_axes_mask_inc.append(tmp_ax)
        # # #
        # # #         ax14 = plt.figure()
        # # #         ax14.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
        # # #
        # # #         all_axes_mask_dec = []
        # # #         for i in range(len(all_classes_morph_dec)):
        # # #             tmp_ax = plt.subplot(4, 5,20-i)
        # # #             plt.title("Mask (cl*-> cl*)")
        # # #             tmp_ax.axis('off')
        # # #             all_axes_mask_dec.append(tmp_ax)
        # # #
        # Retrieve Image in Sequence

        # # #
        # # #         def localized_on_press(event):
        # # #             global plt, ax9, ax9_11, ax9_12, ax9_13, ax9_14, ax9_21, ax9_22, ax9_23, ax9_24, ax9_25
        # # #             global ax9_31, ax9_32, ax9_33, ax9_34, ax9_35
        # # #             global ax9_41, ax9_42, ax9_43, ax9_44, ax9_45
        # # #             global subimages, sTrain, sl_seq, flow, error_scale_disp, hierarchy_out_dim
        # # #             global mask_normalize, lim_delta_sfa, correct_classes_training, S2SC, block_size
        # # #             global ax10, ax11, all_axes_morph
        # # #             global ax13, ax14
        # # #
        # # #             print 'you pressed', event.button, event.xdata, event.ydata
        # # #
        # # #             if event.xdata is None or event.ydata is None:
        # # #                 mask_normalize = not mask_normalize
        # # #                 print "mask_normalize was successfully changed to: ", mask_normalize
        # # #                 return
        # # #
        # # #             y = int(event.ydata)
        # # #             if y < 0:
        # # #                 y = 0
        # # #             if y >= num_images:
        # # #                 y = num_images - 1
        # # #             x = int(event.xdata)
        # # #             if x < 0:
        # # #                 x = 0
        # # #             if x >= hierarchy_out_dim:
        # # #                 x = hierarchy_out_dim -1
        # # #             print "Image Selected=" + str(y) + " , Slow Component Selected=" + str(x)
        # # #
        # # #             print "Displaying Original and Reconstructions"
        # # #         #Display Original Image
        # # #             subimage_im = subimages[y][0:num_pixels_per_image].reshape(subimage_shape)
        # # #             ax9_12.imshow(subimage_im.clip(0,max_clip), vmin=0, vmax=max_clip, aspect='equal',
        # interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
        # # #
        # # #             if show_localized_morphs:
        # # #                 #Display Localized Reconstructed Image
        # # #                 data_in = subimages[y][0:num_pixels_per_image].reshape((1, signals_per_image))
        # # #                 data_out = sl_seq[y].reshape((1, hierarchy_out_dim))
        # # #                 loc_inverted_im = flow.localized_inverse(data_in, data_out)
        # # #                 loc_inverted_im = loc_inverted_im[0, 0:num_pixels_per_image].reshape(subimage_shape)
        # # #                 loc_inverted_im_ori = loc_inverted_im.copy()
        # # #                 ax9_13.imshow(loc_inverted_im.clip(0,max_clip), vmin=0, vmax=max_clip, aspect='equal',
        # interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
        # # #
        # # #                 #Display Re-Reconstructed Image
        # # #                 loc_re_data_in = loc_inverted_im_ori.reshape((1, signals_per_image))
        # # #                 loc_re_data_out = flow.execute(loc_re_data_in)
        # # #                 loc_re_inverted_im = flow.localized_inverse(loc_re_data_in, loc_re_data_out)
        # # #                 loc_re_inverted_im = loc_re_inverted_im[0, 0:num_pixels_per_image].reshape(subimage_shape)
        # # #                 ax9_14.imshow(loc_re_inverted_im.clip(0,max_clip), vmin=0, vmax=max_clip, aspect='equal',
        # interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
        # # #
        # # #                 print "Displaying Masks Using Localized Inverses"
        # # #                 error_scale_disp=1.0
        # # #                 disp_data = [(-8, "lmsk", ax9_21), (-4.0, "lmsk", ax9_22), (2.0, "lmsk", ax9_23),
        # (4.0, "lmsk", ax9_24), (8.0, "lmsk", ax9_25)]
        # # #
        # # #                 data_in = subimages[y][0:num_pixels_per_image].reshape((1, signals_per_image))
        # # #                 data_out = sl_seq[y].reshape((1, hierarchy_out_dim))
        # # #                 work_sfa = data_out.copy()
        # # #
        # # #                 for scale_factor, display_type, fig_axes in disp_data:
        # # #                     #WARNING, this should not be computed this way!!!!
        # # #                     current_class = y/block_size
        # # #                     print "Current classs is:", current_class
        # # #                     if scale_factor < 0:
        # # #                         next_class = current_class-1
        # # #                         if next_class < 0:
        # # #                             next_class = 0
        # # #                     else:
        # # #                         next_class = current_class+1
        # # #                         if next_class >= hierarchy_out_dim:
        # # #                             next_class = hierarchy_out_dim-1
        # # #
        # # #                     current_avg_sfa = sl_seq[current_class *
        # block_size:(current_class+1)*block_size,:].mean(axis=0)
        # # #                     next_avg_sfa = sl_seq[next_class*block_size:(next_class+1)*block_size,:].mean(axis=0)
        # # #
        # # #                     print "Current class is ", current_class
        # # #                     #print "Current class_avg is ", current_avg_sfa
        # # #                     #print "Next class_avg is ", next_avg_sfa
        # # #
        # # #                     data_out_next = next_avg_sfa
        # # #                     print "Computing from class %d to class %d, slow_value %d"%(current_class,
        # next_class, scale_factor)
        # # #                     work_sfa = data_out * (1-lim_delta_sfa) + data_out_next * lim_delta_sfa
        # # #                     t_loc_inv0 = time.time()
        # # #                     mask_im = flow.localized_inverse(data_in, work_sfa, verbose=False)[:,
        # 0:num_pixels_per_image]
        # # #                     t_loc_inv1 = time.time()
        # # #                     print "Localized inverse computed in %0.3f s"% ((t_loc_inv1-t_loc_inv0))
        # # #                     mask_im = (mask_im - data_in)[0, 0:num_pixels_per_image].reshape(subimage_shape) /
        # lim_delta_sfa
        # # #                     if mask_normalize:
        # # #                         mask_im_disp = abs(scale_factor) * scale_to(mask_im, 0.0, mask_im.max() -
        # mask_im.min(), max_clip/2.0, max_clip/2.0, error_scale_disp, 'tanh')
        # # #                     else:
        # # #                         mask_im_disp = abs(scale_factor) * mask_im + max_clip/2.0
        # # #                     fig_axes.imshow(mask_im_disp.clip(0,max_clip), vmin=0, vmax=max_clip, aspect='equal',
        # interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
        # # #                     fig_axes.set_title('%0.2f x Mask: cl %d => %d'%(abs(scale_factor), current_class,
        # next_class))
        # # #                 ax9.canvas.draw()
        # # #
        # # #                 error_scale_disp=1.0
        # # #                 print "Displaying Morphs Using Localized Inverses Incrementing Class"
        # # #
        # # #                 num_morphs_inc = len(all_classes_morph_inc)
        # # #                 num_morphs_dec = len(all_classes_morph_dec)
        # # #                 #make a function specialized in morphs, use this variable,
        # # #                 morph_outputs =  range(num_morphs_inc + num_morphs_dec - 1)
        # # #                 morph_sfa_outputs = range(num_morphs_inc + num_morphs_dec - 1)
        # # #                 original_class = y/block_size
        # # #
        # # #
        # # #           #WARNING!!! THERE IS A BUG, IN WHICH THE LAST INC MORPHS ARE INCORRECTLY COMBINED USING ZERO???
        # # #                 for ii, action in enumerate(["inc", "dec"]):
        # # #                     current_class = original_class
        # # #                     if ii==0:
        # # #                         all_axes_morph = all_axes_morph_inc
        # # #                         all_axes_mask = all_axes_mask_inc
        # # #                         num_morphs = len(all_classes_morph_inc)
        # # #                         desired_next_classes = all_classes_morph_inc + current_class
        # # #                         max_class = num_images/block_size
        # # #                         for i in range(len(desired_next_classes)):
        # # #                             if desired_next_classes[i] >= max_class:
        # # #                                 desired_next_classes[i] = -1
        # # #                     else:
        # # #                         all_axes_morph = all_axes_morph_dec
        # # #                         all_axes_mask = all_axes_mask_dec
        # # #                         num_morphs = len(all_classes_morph_dec)
        # # #                         desired_next_classes = all_classes_morph_dec + current_class
        # # #                         for i in range(len(desired_next_classes)):
        # # #                             if desired_next_classes[i] < 0:
        # # #                                 desired_next_classes[i] = -1
        # # #
        # # #                     desired_next_sfa=[]
        # # #                     for next_class in desired_next_classes:
        # # #                         if next_class >= 0 and next_class < max_class:
        # # #                             c1 = numpy.floor(next_class)
        # # #                             c2 = c1  + 1
        # # #                             if c2 >= max_class:
        # # #                                 c2 = max_class-1
        # # #                             desired_next_sfa.append(sl_seq_training_mean[c1] * (1+c1-next_class) +
        # sl_seq_training_mean[c2]*(next_class-c1))
        # # #                         else: #just in case something goes wrong
        # # #                             desired_next_sfa.append(sl_seq_training_mean[0])
        # # #                         #sl_seq[next_class*block_size:(next_class+1)*block_size,:].mean(axis=0))
        # # #
        # # #                     data_in = subimages[y][0:num_pixels_per_image].reshape((1, signals_per_image))
        # # #                     data_out = sl_seq[y].reshape((1, hierarchy_out_dim))
        # # #                     for i, next_class in enumerate(desired_next_classes):
        # # #                         if next_class == -1:
        # # #                             if ii==0:
        # # #                                 morph_sfa_outputs[i+num_morphs_dec-1] = numpy.zeros(len(data_out[0]))
        # # #                             else:
        # # #                                 morph_sfa_outputs[num_morphs_dec-i-1] = numpy.zeros(len(data_out[0]))
        # # #                             break
        # # #                         data_out_next = desired_next_sfa[i]
        # # #                         print "Morphing to desired class %.2f..."%next_class
        # # #
        # # #                         work_sfa = data_out * (1-lim_delta_sfa) + data_out_next * lim_delta_sfa
        # # #
        # # #                         t_loc_inv0 = time.time()
        # # #                         morphed_data = flow.localized_inverse(data_in, work_sfa, verbose=False)[0,
        # 0:num_pixels_per_image] #TODO: last part might be unnecessary
        # # #                         t_loc_inv1 = time.time()
        # # #                         print "Localized inverse computed in %0.3f s"% ((t_loc_inv1-t_loc_inv0))
        # # #
        # # #                         morphed_data = data_in + (morphed_data - data_in)/lim_delta_sfa
        # # #                         morphed_im_disp = morphed_data[0:num_pixels_per_image].reshape(subimage_shape)
        # # #
        # # #                         if all_axes_morph[i] is not None:
        # # #                             all_axes_morph[i].imshow(morphed_im_disp.clip(0,max_clip), vmin=0,
        # vmax=max_clip, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
        # # #                             all_axes_morph[i].set_title("Morph(cl %.1f -> %.1f)"%(current_class,
        # next_class))
        # # #                         else:
        # # #                             print "No plotting Morph. (Reason: axes = None)"
        # # #
        # # #                         if all_axes_mask[i] is not None:
        # # #                             loc_mask_data = morphed_data[0] - data_in[0]
        # # #                             loc_mask_disp = loc_mask_data[0:num_pixels_per_image].reshape(\
        # subimage_shape) + max_clip/2.0
        # # #                             loc_mask_disp = scale_to(loc_mask_disp, loc_mask_disp.mean(),
        # loc_mask_disp.max() - loc_mask_disp.min(), 127.5, 255.0, scale_disp, 'tanh')
        # # #                             all_axes_mask[i].imshow(loc_mask_disp.clip(0,max_clip), vmin=0, vmax=max_clip,
        #  aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
        # # #                             all_axes_mask[i].set_title("Mask(cl %.1f -> %.1f)"%(current_class,next_class))
        # # #                         else:
        # # #                             print "No plotting Mask. (Reason: axes = None)"
        # # #
        # # #                         current_class = next_class
        # # #                         data_in = morphed_data
        # # #                         data_out = flow.execute(data_in)
        # # #                         if ii==0: #20-29
        # # #                             morph_sfa_outputs[i+num_morphs_dec-1] = data_out[0]
        # # #                         else: #0-19
        # # #                             morph_sfa_outputs[num_morphs_dec-i-1] = data_out[0]
        # # #
        # # #                 ax10.canvas.draw()
        # # #                 ax11.canvas.draw()
        # # #                 ax13.canvas.draw()
        # # #                 ax14.canvas.draw()
        # # #
        # # #         #        for i, sfa_out in enumerate(morph_sfa_outputs):
        # # #         #            print "elem %d: "%i, "has shape", sfa_out.shape, "and is= ", sfa_out
        # # #
        # # #         #        morph_sfa_outputs[num_morphs_dec] = sl_seq[y]
        # # #                 sl_morph = numpy.array(morph_sfa_outputs)
        # # #                 sl_morphdisp = scale_to(sl_morph, sl_morph.mean(), sl_morph.max()-sl_morph.min(), 127.5,
        #  255.0, scale_disp, 'tanh')
        # # #         #        extent = (L, R, B, U)
        # # #                 extent = (0, hierarchy_out_dim-1, all_classes_morph_inc[-1]+original_class,
        # all_classes_morph_dec[-1]+original_class-0.25)
        # # #                 ax12_1.imshow(sl_morphdisp.clip(0,255), aspect='auto', interpolation='nearest',
        # origin='upper', cmap=mpl.pyplot.cm.gray, extent=extent)
        # # #         #        majorLocator_y   = MultipleLocator(0.5)
        # # #         #        ax12_1.yaxis.set_major_locator(majorLocator_y)
        # # #                 plt.ylabel("Morphs")
        # # #
        # # #
        # # #                 sl_selected = sl_seq[y][:].reshape((1, hierarchy_out_dim))
        # # #                 sl_selected = scale_to(sl_selected, sl_selected.mean(), sl_selected.max() -
        # sl_selected.min(), 127.5, 255.0, scale_disp, 'tanh')
        # # #         #        extent = (0, hierarchy_out_dim-1, all_classes_morph_inc[-1]+original_class+0.5,
        # all_classes_morph_dec[-1]+original_class-0.5)
        # # #                 ax12_3.imshow(sl_selected.clip(0,255), aspect=8.0, interpolation='nearest',
        # origin='upper', cmap=mpl.pyplot.cm.gray)
        # # #         #        majorLocator_y   = MultipleLocator(0.5)
        # # #         #        ax12_1.yaxis.set_major_locator(majorLocator_y)
        # # #         #        plt.ylabel("Morphs")
        # # #
        # # #
        # # #         #        morphed_classes = numpy.concatenate((all_classes_morph_dec[::-1], [0],
        # all_classes_morph_inc))
        # # #         #        print "morphed_classes=", morphed_classes
        # # #         #        morphed_classes = morphed_classes + original_class
        # # #         #        majorLocator_y   = MultipleLocator(1)
        # # #         #        #ax12_1.yticks(morphed_classes)
        # # #         #        ax12_1.yaxis.set_major_locator(majorLocator_y)
        # # #                 ax12.canvas.draw()
        # # #
        # # #         ax9.canvas.mpl_connect('button_press_event', localized_on_press)

        ###############################################################################################################
        # ##### Plot that displays the slow features of some central node of each layer of the network ################
        ###############################################################################################################
        if compute_slow_features_newid_across_net:
            if compute_slow_features_newid_across_net == 2:
                print "Displaying slowest component across network"
                relevant_out_dim = 1
                first_feature = 0
                transpose_plot = False
                two_feats = False
            elif compute_slow_features_newid_across_net == 3:
                print "Displaying second slowest component across network, vertically"
                relevant_out_dim = 1
                first_feature = 1
                transpose_plot = True
                two_feats = False
            elif compute_slow_features_newid_across_net == 4:
                print "Displaying second against first slowest component across network"
                relevant_out_dim = 2
                first_feature = 0
                transpose_plot = False
                two_feats = True
            else:
                print "Displaying first %d slowest components across network" % relevant_out_dim
                transpose_plot = False
                first_feature = 0
                two_feats = False

            print "************ Displaying Slow Features Across Network (Newid) **********"
            ax = plt.figure()
            # ax13.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)

            sfa_nodes_or_layers_indices = []
            for i, node in enumerate(flow.flow):
                if isinstance(node, (mdp.hinet.Layer, mdp.hinet.CloneLayer)):
                    if isinstance(node.nodes[0], mdp.nodes.SFANode):
                        sfa_nodes_or_layers_indices.append(i)
                elif isinstance(node, mdp.nodes.SFANode):
                    sfa_nodes_or_layers_indices.append(i)

            num_sfa_nodes_or_layers = len(sfa_nodes_or_layers_indices)
            print "num_sfa_nodes_or_layers=", num_sfa_nodes_or_layers
            for plot_nr, node_i in enumerate(sfa_nodes_or_layers_indices):
                flow_partial = flow[0:node_i + 1]
                sl_partial = flow_partial.execute(subimages_newid)

                if transpose_plot:
                    plt.subplot((num_sfa_nodes_or_layers - 1) / 3 + 1, 3, plot_nr + 1)
                else:
                    plt.subplot(2, (num_sfa_nodes_or_layers - 1) / 2 + 1, plot_nr + 1)

                node = flow.flow[node_i]
                if isinstance(node, (mdp.hinet.Layer, mdp.hinet.CloneLayer)):
                    num_nodes = len(node.nodes)
                    central_node_nr = num_nodes / 2 + int((num_nodes ** 0.5)) / 2
                    z = sl_partial.shape[1] / num_nodes
                    sl_partial = sl_partial[:, z * central_node_nr:z * (central_node_nr + 1)]
                    print "num_nodes=", num_nodes, "central_node_nr=", central_node_nr,
                    print "sl_partial.shape=", sl_partial.shape
                    plt.title("SFA Outputs. Node %d, subnode %d " % (node_i, central_node_nr))
                else:
                    plt.title("SFA Outputs. Node %d" % (node_i))

                if two_feats is False:
                    # TODO: Notice conflict between (relevant_sfa_indices deprecated) and reversed_sfa_indices
                    relevant_sfa_indices = numpy.arange(relevant_out_dim)
                    reversed_sfa_indices = range(first_feature, relevant_out_dim + first_feature)
                    reversed_sfa_indices.reverse()

                    # r_color = (1 - relevant_sfa_indices * 1.0 / relevant_out_dim) * 0.8 + 0.2
                    # g_color = (relevant_sfa_indices * 1.0 / relevant_out_dim) * 0.8 + 0.2
                    # b_color = relevant_sfa_indices * 0.0
                    r_color = [1.0, 0.0, 0.0]
                    g_color = [28 / 255.0, 251 / 255.0, 55 / 255.0]
                    b_color = [33 / 255.0, 79 / 255.0, 196 / 255.0]
                    if relevant_out_dim == 1:
                        r_color = [0.0] * 3
                        g_color = [55 / 255.0] * 3
                        b_color = [196 / 255.0] * 3

                    max_amplitude_sfa = 3.0  # 2.0

                    sfa_text_labels = ["Slowest Signal", "2nd Slowest Signal", "3rd Slowest Signal"]

                    for sig in reversed_sfa_indices:
                        if transpose_plot is False:
                            plt.plot(numpy.arange(num_images_newid), sl_partial[:, sig], ".",
                                     color=(r_color[sig], g_color[sig], b_color[sig]), label=sfa_text_labels[sig],
                                     markersize=6, markerfacecolor=(r_color[sig], g_color[sig], b_color[sig]))
                            plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
                        else:
                            plt.plot(sl_partial[:, sig], numpy.arange(num_images_newid), ".",
                                     color=(r_color[sig], g_color[sig], b_color[sig]), label=sfa_text_labels[sig],
                                     markersize=6, markerfacecolor=(r_color[sig], g_color[sig], b_color[sig]))
                            plt.xlim(-max_amplitude_sfa, max_amplitude_sfa)

                            # plt.xlabel("Input Image, Training Set (red=slowest signal, light green=fastest signal)")
                            # plt.ylabel("Slow Signals")
                else:  # Second component vs First component
                    r_color = 0.0
                    g_color = 55 / 255.0
                    b_color = 196 / 255.0
                    max_amplitude_sfa = 3.0  # 2.0

                    if transpose_plot is False:
                        # Typically: plot 2nd slowest feature against first one
                        plt.plot(sl_partial[:, 1], sl_partial[:, 0], ".", color=(r_color, g_color, b_color),
                                 markersize=6, markerfacecolor=(r_color, g_color, b_color))
                    else:
                        plt.plot(sl_partial[:, 0], sl_partial[:, 1], ".", color=(r_color, g_color, b_color),
                                 markersize=6, markerfacecolor=(r_color, g_color, b_color))

                    plt.xlim(-max_amplitude_sfa, max_amplitude_sfa)
                    plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
                    # plt.xlabel("Input Image, Training Set (red=slowest signal, light green=fastest signal)")
                    # plt.ylabel("Slow Signals")

        print "GUI Created, showing!!!!"
        plt.show()
        print "GUI Finished!"

    print "Program successfully finished"


if __name__ == "__main__":
    main()
