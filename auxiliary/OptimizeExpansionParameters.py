#Script to optimize the parameters 'k' and 'd' of a predefined expansion
#By Alberto Escalante. Alberto.Escalante@ini.ruhr-uni-bochum.de First Version 6 August 2012
#Ruhr-University-Bochum, Institute of Neurocomputation, Theory of Neural Systems, Group of Prof. Dr. Wiskott

import numpy
#import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import Image
import mdp
import os
import glob
import random
import sys
import subprocess
sys.path.append("/home/escalafl/workspace4/cuicuilco_MDP3.2/src")
sys.path.append("/home/escalafl/workspace4/cuicuilco_MDP3.2/src/auxiliary")
from nonlinear_expansion import *
#import sfa_libs
import getopt
import string


#Execution Parameters
#expansion parameter (usually offset)
k=1.0
#expansion parameter (usually exponent)
d=2.0
#num_samples
num_steps = 100 #10000   1000 for L, 148 for Q at 6 signals
#num features before expansion
num_slow_signals = 60 #60   or 1667 for L and 349 for Q at 10 signals
#max dim considered after expansion
#noise_dim = 15

#noise level for each feature
std_noises = [0.25, 0.5, 0.75]

std_noise = 0.75
different_train_test = True #do testing on different data
showFunctions = True and False
showFunctions2D = True and False
showOutliers = True and False
enableDisplay = True and False
seed = None
#seed = 123456
selected_function = None
only_selected_func = False
fraction_outliers = 0.05 # 0.05
num_outliers = int(num_steps * fraction_outliers)
problem_weights = [0.02, 0.2, 0.02, 0.14, 0.02, 0.02, 0.02, 0.02, 0.5]
#beta P1P2P6POv,POA: -1.41906081e+06,  -3.31440465e+01, 0.0, 0.0, 0.0,   1.86231842e+06, 0.0, -4.43240732e+05, 9.68825545e+00, 6.30713102e-02
problem_weights = [-3.31440465e+01, 1.86231842e+06, 0.0, 0.0, 0.0, -4.43240732e+05, 0.0, 9.68825545e+00, 6.30713102e-02]
#beta P1P2POv,POA: -1.84818322e+06,  -1.38619330e+01,   1.84819909e+06,   1.93113426e+01,  -2.15521186e-01
problem_weights = [-1.38619330e+01, 1.84819909e+06, 0.0, 0.0, 0.0, 0.0, 0.0, 1.93113426e+01,  -2.15521186e-01]
problem_weights = [0.02, 0.2, 0.02, 0.14, 0.02, 0.02, 0.02, 0.02, 0.5]

                      
enable_overfitting = True #and False
enable_outlier_amplification = True #and False
outlier_amplification_method = "Magnitude"

verbose=True

#print "showFunctions", showFunctions
#quit()

argv = None
if argv is None:
    argv = sys.argv
if len(argv) >= 2:
    try:
        opts, args = getopt.getopt(argv[1:], "", ["k=", "d=", "std_noise=", "num_steps=", "num_slow_signals=", "DifferentTrainTest=", 
                                                  "EnableDisplay=", "ShowFunctions=", "ShowFunctions2D=", "Seed=", 
                                                  "SelectFunction=", "OnlySelectedFunction=", "EnableOverfitting=", 
                                                  "EnableOutlierAmp=","verbose=", "Weights="])
        print "opts=", opts
        print "args=", args

        if len(args)>0:
            print "Arguments not understood:", args
            sys.exit(2)
                               
        for opt, arg in opts:
            if opt in ('--k'):
                k = float(arg)
            elif opt in ('--d'):
                d = float(arg)
            elif opt in ('--std_noise'):
                std_noise = float(arg)
            elif opt in ('--num_steps'):
                num_steps = int(arg)
            elif opt in ('--num_slow_signals'):
                num_slow_signals = int(arg)
            elif opt in ('--SelectFunction'):
                selected_function = arg
            elif opt in ('--OnlySelectedFunction'):
                only_selected_func = int(arg)
            elif opt in ('--DifferentTrainTest'):
                if arg == '1':
                    different_train_test = True
                else:
                    different_train_test = False
            elif opt in ('--EnableDisplay'):
                if arg == '1':
                    enableDisplay = True
                else:
                    enableDisplay = False
            elif opt in ('--ShowFunctions'):
                if arg == '1':
                    showFunctions = True
                else:
                    showFunctions = False
            elif opt in ('--ShowFunctions2D'):
                if arg == '1':
                    showFunctions2D = True
                else:
                    showFunctions2D = False
            elif opt in ('--Seed'):
                seed = int(arg)
            elif opt in ('--EnableOutlierAmp'):
                enable_outlier_amplification =  bool(int(arg))
            elif opt in ('--EnableOverfitting'):
                enable_overfitting = bool(int(arg))
            elif opt == '--verbose':
                verbose = bool(int(arg))                
            elif opt == '--Weights':   #Weights for evaluation of expansions!
                problem_weights  = map(float, string.split(arg, ","))
                print "Changing problem_weights to", problem_weights
            else:
                print "Argument not handled: ", opt
    except getopt.GetoptError:
        print "Error parsing the arguments: ", argv[1:]
        sys.exit(2)


selected_function = '0.8Exp'

if verbose:
    print "PARAMETERS. k=", k, ",d=", d, ",std_noise=", std_noise, ",num_steps=", num_steps, 
    print ",num_slow_signals=", num_slow_signals, ",DifferentTrainTest=", different_train_test,
    print ",EnableDisplay=",enableDisplay, ",ShowFunctions=",showFunctions, ",ShowFunctions2D=",showFunctions2D, ",Seed=", seed
    print ",SelectFunction=",selected_function, ",OnlySelectedFunction=",only_selected_func, ",EnableOverfitting=",enable_overfitting, 
    print ",EnableOutlierAmp=",enable_outlier_amplification, ",verbose=",verbose, ",problem_weights=",problem_weights

numpy.random.seed(seed)

#expansion parameter (usually offset)
k_set = [1.0]
#expansion parameter (usually exponent)
d_set = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
std_noise_set = [0.25, 0.5, 0.75]
w1, w2, w3, w4, w5, w6, w7, w_OV, w_OA = problem_weights
num_iterations = 5

performances = {}
for d in d_set:
    for k in k_set:
        for std_noise in std_noise_set:
            for iteration in range(num_iterations):
                """
                "k=", "d=", "std_noise=", "num_steps=", "num_slow_signals=", "DifferentTrainTest=", 
                                                      "EnableDisplay=", "ShowFunctions=", "ShowFunctions2D=", "Seed=", 
                                                      "SelectFunction=", "OnlySelectedFunction=", "EnableOverfitting=", 
                                                      "EnableOutlierAmp=","verbose=", "Weights="
                """           
                weight_string = "%f,%f,%f,%f,%f,%f,%f,%f,%f"%(w1, w2, w3, w4, w5, w6, w7, w_OV, w_OA)
                p = subprocess.Popen(["python", "FunctionApproximationForSFA_experiments.py", "--k=%f"%k, "--d=%f"%d, "--std_noise=%f"%std_noise, "--num_steps=%d"%num_steps, 
                                      "--num_slow_signals=%d"%num_slow_signals, "--OnlySelectedFunction=1", "--SelectFunction=%s"%selected_function,
                                      "--EnableOverfitting=1", "--EnableOutlierAmp=%d", "--Weights=%s"%weight_string], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for i, line in enumerate(p.stdout.readlines()):
                    print "[%d]="%i, line, 
                    last_line = line
                perf = float(line)
                performances[(d, k, std_noise, iteration)] = perf
                print "performances[(d=%f, k=%f, std_noise=%f, iteration=%d)] = perf = %f"%(d,k,std_noise,iteration, perf)

#Display all performances
for d in d_set:
    for k in k_set:
        dk_avg_perf = 0.0
        for std_noise in std_noise_set:
            dks_avg_perf = 0.0
            for iteration in range(num_iterations):
                dks_avg_perf += performances[(d, k, std_noise, iteration)]
                print "Perf[(d=%f, k=%f, std_noise=%f, iteration=%d)] = perf = %f"%(d,k,std_noise,iteration, performances[(d, k, std_noise, iteration)])
            dks_avg_perf /= num_iterations
            dk_avg_perf += dks_avg_perf
            print "Avg Perf[(d=%f, k=%f, std_noise=%f)] = %f"%(d,k,std_noise,dks_avg_perf)
        dk_avg_perf /= len(std_noise_set)
        print "Avg_Perf[(d=%f, k=%f)] = %f"%(d,k, dk_avg_perf)

#Find optimal one        
d_opt = None
k_opt = None
perf_opt = None

for d in d_set:
    for k in k_set:
        dk_avg_perf = 0.0
        for std_noise in std_noise_set:
            dks_avg_perf = 0.0
            for iteration in range(num_iterations):
                dks_avg_perf += performances[(d, k, std_noise, iteration)]
            dks_avg_perf /= num_iterations
            dk_avg_perf += dks_avg_perf
        dk_avg_perf /= len(std_noise_set)
        if perf_opt==None or dk_avg_perf < perf_opt:
            print "optimal improved to %f"%dk_avg_perf, "with d=%f, k=%f"%(d, k)
            perf_opt = dk_avg_perf
            d_opt = d
            k_opt = k
            