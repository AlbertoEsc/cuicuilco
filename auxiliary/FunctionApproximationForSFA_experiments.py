#Experiments regarding comparison of basis functions used for NL expansion
#By Alberto Escalante. Alberto.Escalante@ini.ruhr-uni-bochum.de First Version 4 May 2011
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
sys.path.append("/home/escalafl/workspace4/cuicuilco_MDP3.2/src")
from nonlinear_expansion import *
#import sfa_libs
import getopt
import string

#Note, warning due to a divide by zero is handled correctly and can be ignored
#numpy.seterr(all='raise')

#Metrics applied to each sample separately, require 2D data
def metric_max_var(x):
    return numpy.abs(x).max(axis=1)

def metric_magnitude(x):
    return (x**2).sum(axis=1)**0.5

def metric_RMSA(x):
    return (x**2).mean(axis=1)**0.5

def metric_corrected_RMSA(x):
    return (x**2).mean(axis=1)**0.5 * numpy.sqrt(x.shape[1])

#Metric for whole data
#TODO: Unused, consider deletion
def metric_individual_RMSE(x):
    comp_RMSA =  ((x**2).mean(axis=0))**0.5
    print "comp_RMSA=", comp_RMSA
    x = x / comp_RMSA
    return (x**2).mean(axis=1)**0.5




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
max_expanded_dim = 2000 #2000   or 
#noise_dim = 15

#noise level for each feature
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

if verbose:
    print "PARAMETERS. k=", k, ",d=", d, ",std_noise=", std_noise, ",num_steps=", num_steps, 
    print ",num_slow_signals=", num_slow_signals, ",DifferentTrainTest=", different_train_test,
    print ",EnableDisplay=",enableDisplay, ",ShowFunctions=",showFunctions, ",ShowFunctions2D=",showFunctions2D, ",Seed=", seed
    print ",SelectFunction=",selected_function, ",OnlySelectedFunction=",only_selected_func, ",EnableOverfitting=",enable_overfitting, 
    print ",EnableOutlierAmp=",enable_outlier_amplification, ",verbose=",verbose, ",problem_weights=",problem_weights


numpy.random.seed(seed)

###########################################
#Create artificial data. Train and test.
t = numpy.linspace(0, numpy.pi, num_steps)
factor = numpy.sqrt(2)

sl = numpy.zeros((num_steps, num_slow_signals))
for i in range(num_slow_signals):
    # *(numpy.random.randint(2)*2 -1)
    sl[:,i] = factor * numpy.cos((i+1)*t)  + std_noise * numpy.random.normal(size=(num_steps)) 
    print "sl[:,i].mean()=", sl[:,i].mean(), "  sl[:,i].var()=", sl[:,i].var()
sl = fix_mean_var(sl)

sl_test = numpy.zeros((num_steps, num_slow_signals))
for i in range(num_slow_signals):
    # *(numpy.random.randint(2)*2 -1)
    sl_test[:,i] = factor * numpy.cos((i+1)*t)  + std_noise * numpy.random.normal(size=(num_steps)) 
    print "sl_test[:,i].mean()=", sl_test[:,i].mean(), "  sl[:,i].var()=", sl_test[:,i].var()
sl_test = fix_mean_var(sl_test)

#Auxiliar test data
sl1sl2 = numpy.cos(t) * numpy.cos(2*t)
sl1sl2 = fix_mean_var(sl1sl2)
sl1sl2 += std_noise * numpy.random.normal(size=(num_steps))
sl1sl2 = fix_mean_var(sl1sl2)
sl1sl2 = sl1sl2.reshape((num_steps,1))

sl1sl2_test = numpy.cos(t) * numpy.cos(2*t)
sl1sl2_test = fix_mean_var(sl1sl2_test)
sl1sl2_test += std_noise * numpy.random.normal(size=(num_steps))
sl1sl2_test = fix_mean_var(sl1sl2_test)
sl1sl2_test= sl1sl2_test.reshape((num_steps,1))

#Just Noise signals
#noise = numpy.random.normal(size=(num_steps,noise_dim))
#noise = fix_mean_var(noise)
#noise_test = numpy.random.normal(size=(num_steps,noise_dim))
#noise_test = fix_mean_var(noise_test)

####################################
#Noise-free signals

sl1 = numpy.cos(t)
sl1 = fix_mean_var(sl1)
#What function do we want to approximate?

sh_sl1 = sl1 + 0.0
numpy.random.shuffle(sh_sl1)

sin = numpy.sin(t)
sin = fix_mean_var(sin)

sl2 = numpy.cos(2*t)
sl2 = fix_mean_var(sl2)

sl3 = numpy.cos(3*t)
sl3 = fix_mean_var(sl3)

const = numpy.ones((num_steps,1))

##############################################
#Computational problems
all_problems = ["s1->s1", "s1->s2", "s1,s2->s3", "cos(t)->sin(t)", "s2,s3->s1","s2,s3,s4->s1", "s3,s4,s5->s1", "s1,s2,s1s2->s1","s*->sh(s1)"]
problems = ["s1->s1", "s1->s2", "s1,s2->s3", "cos(t)->sin(t)", "s2,s3->s1","s2,s3,s4->s1", "s3,s4,s5->s1"]
#problems = []
if enable_overfitting:
    problems += ["s*->sh(s1)"]

if enable_outlier_amplification:
    problems += ["OutlierAmp"]

problem_outlier_amp = "s*->sh(s1)"

#problem_weights is a list
#prob_weights = {"s1->s1":0.02, "s1->s2":0.2, "s1,s2->s3":0.02, "cos(t)->sin(t)":0.14, 
#                "s2,s3->s1":0.02, "s2,s3,s4->s1":0.02, "s3,s4,s5->s1":0.02, "s1,s2,s1s2->s1":0.02,"s*->sh(s1)":0.5 }

prob_weights = {}
for i, problem in enumerate(problems):
    prob_weights[problem] = problem_weights[i]


#problems = ["s1->s2", "s1->s1"]
num_problems = len(problems)

#exponents =  [2.0, 1.0, 0.8, 0.5, -1]
displayable_basis = ["identity","SExp","0.8Exp", "QExp", "TExp", "Q_AN_exp", "T_AN_exp", "Q_N_exp", "T_N_exp", 
             "Q_E_exp", "T_E_exp","Q_AE_exp", "T_AE_exp", "Q_AP_exp", "T_AP_exp",]
optimized_basis = ["identity","SExp", "QExp", "0.8Exp", "TExp", "Q_AN_exp", "T_AN_exp", "Q_N_exp", "T_N_exp", 
         "Q_E_exp", "T_E_exp", "Q_AE_exp", "T_AE_exp", "Q_AP_exp", "T_AP_exp",]

#paper_basis = ["identity", "QExp", "TExp", "Q_N_exp", "T_N_exp", "QExp2", "TExp2", "0.8Exp", "SExp"]
#basis = ["identity", "QExp", "Q_N_exp", "QExp2", "0.8Exp", "SExp"]
#basis = ["identity", "QExp", "Q_N_exp", "QExp2"]
#poster_basis = ["identity", "0.8Exp", "T_N_exp", "SExp"]
basis = ["identity", "QExp", "Q_N_exp", "QExp2", "0.8Exp", "SExp"]
#basis = paper_basis

displayable_basis=basis

#####################################
#List of expansions
ExpansionFromName = {"identity":I_exp, "SExp":S_exp, "QExp":Q_exp, "0.8Exp":S_exp, "TExp": T_exp, "Q_AN_exp":Q_AN_exp, "T_AN_exp":T_AN_exp, 
             "Q_N_exp":Q_N_exp, "T_N_exp":T_N_exp, "Q_E_exp":Q_E_exp, "T_E_exp":T_E_exp, "Q_AE_exp":Q_AE_exp, "T_AE_exp":T_AE_exp, 
             "Q_AP_exp":Q_AP_exp, "T_AP_exp":T_AP_exp, "QExp2":Q_exp, "TExp2": T_exp,}

if only_selected_func:
    if selected_function in basis:
        basis = [selected_function]
    else:
        er = "Selected Function %r should be in basis"%selected_function, basis
        raise Exception(er)
    
nan = numpy.nan
optimized_basis_k_d = {"identity":(nan, nan), "SExp":(nan,2.0), "QExp":(nan,0.85), "0.8Exp":(nan,0.77), "TExp":(nan, 3.0), "Q_AN_exp":(1.15, 1.08), 
             "T_AN_exp":(1.1, 1.05), "Q_N_exp":(1.0, 0.6), "T_N_exp":(1.0, 0.7333), "Q_E_exp":(1.0, 1.0), "T_E_exp":(1.0, 1.0),
             "Q_AE_exp":(1.1, 0.6), "T_AE_exp":(1.1, 0.6), "Q_AP_exp":(nan, 0.4), "T_AP_exp":(nan, 0.3)}
basis_k_d = {"identity":(nan, nan), "QExp":(nan,2.0), "TExp":(nan, 3.0),  
             "Q_N_exp":(1.0, 2.0), "T_N_exp":(1.0, 3.0), "QExp2":(nan, 0.8), "TExp2":(nan, 0.9),
             "0.8Exp":(nan,0.8), "SExp":(nan,2.0)}
num_all_basis = len(displayable_basis)
num_basis = len(basis)

if selected_function != None:
    print "Setting parameters for basis function: ", selected_function, "to: (k, d)= ", (k, d)
    if selected_function in basis_k_d.keys():
        basis_k_d[selected_function] = (k, d)
    else:
        er = "Selected function ", selected_function, "not in known basis"
        raise Exception(er)
    
#Display expansion functions over 1 dimensional data
if showFunctions:
    data = numpy.random.normal(size=(1000,1))
    ww = data.flatten()
    ww.sort()
    data = ww.reshape((1000,1))
    
    f0 = plt.figure()
    plt.suptitle("Expansion functions for different base functions")
    for i, base in enumerate(displayable_basis):
        print "Using base:", base
        kk, dd=basis_k_d[base]
        expansion = ExpansionFromName[base]
        exp = expansion(data, k=kk, d=dd)
    
        for j in range(exp.shape[1]):
            v = exp[:,j].std()
            if v > 0:
                exp[:,j] =  (exp[:,j] - exp[:,j].mean()) / exp[:,j].std()
           
        ax = plt.subplot(3,(num_all_basis+2)/3,i+1)
        for j in range(exp.shape[1]):
            plt.plot(data[:,0], exp[:,j], "-")
        plt.xlabel("Basis:"+base+" (k=%f, d=%f)"%(kk,dd))


#Display expansion functions over 2-dimensional data
if showFunctions2D:
    data1 = numpy.linspace(-3,3,100)
    data2 = numpy.linspace(-3,3,100)
    
    mesh = numpy.array(numpy.meshgrid(data1, data2))
    mesh = numpy.swapaxes(mesh, 0, 2) #?????? needed????
    data = mesh.reshape(100*100, 2)
    print "data.shape =", mesh.shape
     
    f0 = plt.figure()
    plt.suptitle("Expansion functions for different base functions: base (k,d)")
    for i, base in enumerate(displayable_basis):
        print "Using base:", base
        kk, dd=basis_k_d[base]
        expansion = ExpansionFromName[base]
        exp = expansion(data, k=kk, d=dd)
    
        for j in range(exp.shape[1]):
            v = exp[:,j].std()
            if v > 0:
                exp[:,j] =  (exp[:,j] - exp[:,j].mean()) / exp[:,j].std()
           
        for j in range(exp.shape[1]):
            ax = plt.subplot(9, num_all_basis, j*num_all_basis+i+1)
            #plt.plot(data[:,0], exp[:,j], "-")
            im = exp[:,j].reshape(100,100).T
            ax.imshow(im, cmap=mpl.cm.jet, interpolation='nearest')
            if j == exp.shape[1]-1:
                plt.xlabel(base+"(%2.2f, %2.2f)"%(kk,dd))
                ax.axis('off')
            else:
                ax.axis('off')

enable_normalize = False
perf = {}
outlier_info = {}
sphered_data = None
sphered_exp = None

long_description={"s1->s1":"Approximation: 1, sl_1 -> s1_1", 
                  "s1->s2":"Approximation: 1, sl_1 -> sl_2", 
                  "s1,s2->s3":"Approximation:  1, sl_1, sl2 -> sl_3",
                  "cos(t)->sin(t)":"Approximation: 1, sl_1 (cos(t)) -> sin(t)", 
                  "s2,s3->s1":"Approximations 1, sl_2, sl_3 ->  sl_1", 
                  "s2,s3,s4->s1":"Approximations 1, sl_2, sl_3, sl_4 ->  sl_1", 
                  "s3,s4,s5->s1":"Approximations 1, sl_3, sl_4, sl_5 ->  sl_1",
                  "s1,s2,s1s2->s1":"Approximations 1, sl_1, sl_2, sl_1 * sl2 ->  sl_1", 
                  "s*->sh(s1)":"Approximations 1, noise ->  sh_sl1"}

for problem in problems:
    if problem == "OutlierAmp":
        print "Skipping problem 'OutlierAmp', will be computer during 's*->sh(s1)'"
        continue
    print ""
    print "Solving problem:", problem, "(", long_description[problem], ")"
    if enableDisplay:
        f1 = plt.figure()
        plt.suptitle(long_description[problem])
    if problem == "s1->s1":
        data = sl[:,0:1]
        data_test = sl[:,0:1]
        print "data=", data
        goal = sl[:,0:1]
    elif problem == "s1->s2":
        data = sl[:,0:1]
        data_test = sl_test[:,0:1]
        print "data=", data
        goal = sl2
    elif problem == "s1,s2->s3":
        data = sl[:,0:2]
        data_test = sl_test[:,0:2]
        goal = sl3
    elif problem == "cos(t)->sin(t)":
        data = sl[:,0:1]
        data_test = sl_test[:,0:1]
        goal = sin
    elif problem == "s2,s3->s1":
        data = sl[:,1:3]
        data_test = sl_test[:,1:3]
        goal = sl1
    elif problem == "s2,s3,s4->s1":
        data = sl[:,1:4]
        data_test = sl_test[:,1:4]
        goal = sl1
    elif problem == "s3,s4,s5->s1":
        data = sl[:,2:5]
        data_test = sl_test[:,2:5]
        goal = sl1        
    elif problem == "s1,s2,s1s2->s1":
        data = numpy.concatenate((sl[:,0:2], sl1sl2),axis=1)
        data_test = numpy.concatenate((sl_test[:,0:2], sl1sl2_test), axis=1)
        goal = sl1        
    elif problem == "s*->sh(s1)":
        data = sl
        data_test = sl_test
        goal = sl1+0.0
        numpy.random.shuffle(goal)        
    else:
        er = "Problem unknown:", problem
        raise Exception(er)
        
    for i, base in enumerate(basis):
        print "Using base:", base
        kk, dd=basis_k_d[base]

        expansion = ExpansionFromName[base]
        exp = numpy.concatenate((const, expansion(data, k=kk, d=dd)),axis=1)
        exp_test = numpy.concatenate((const, expansion(data_test, k=kk, d=dd)),axis=1)

        if exp.shape[1] > max_expanded_dim:
            print "Skipping base because expanded dim to high: ", exp.shape
            length = exp.shape[1]
            perf[(problem, base)] = (0, 0, 0, length)
            if enable_outlier_amplification:
                outlier_info[(problem, base)] = (0, 0, 0, 0, 0, 0)
            continue

        exp = fix_mean_var(exp)
        exp[:,0]=1.0
        exp_test = fix_mean_var(exp_test)
        exp_test[:,0]=1.0

        print "exp.shape=", exp.shape
        print "exp_test.shape=", exp_test.shape
        print "data[0,:]=", data[0,:]
        print "data_test[0,:]=", data_test[0,:]
        #print "exp[0,:]=", exp[0,:]
        #print "exp_test[0,:]=", exp_test[0,:]
#        print "exp:", exp

        num_base_funcs = exp.shape[1]  
        pinv = numpy.linalg.pinv(exp)
        #print "pinv.shape is", pinv.shape
        #print "goal.shape is", goal.shape
        coefs = numpy.dot(pinv, goal)
        #print "coefs: ", coefs
        #print "exp.shape", exp.shape

      
        if different_train_test==False:
            data_test = data
            exp_test = exp 

        app = numpy.dot(exp_test, coefs)

        #Measure "rareness" amplification for computation of function
        #TODO: Lots of this code is useless, only one method worked in practice, consider cleaning the code and keeping the only working solution
        if enable_outlier_amplification and problem == problem_outlier_amp: # "s*->sh(s1)"           
            if outlier_amplification_method == "Old":
                if sphered_data is None:
                    print "whitening data..."
                    whitening_node = mdp.nodes.WhiteningNode(reduce=True)
                    whitening_node.train(data)
                    sphered_data =  whitening_node.execute(data)
                
                    print "computing data NN..."
                    d_in = dist_to_closest_neighbour(sphered_data, sphered_data)
                    d_in_avg = d_in.mean()
                    outliers_in_index = numpy.argsort(d_in)[-int(num_steps*fraction_outliers):]
                    outliers_in = d_in[outliers_in_index]
                    d_in_outlier_avg = outliers_in.mean()
                    outlier_ratio_in = d_in_outlier_avg / d_in_avg
              
                print "whitening exp..."
                whitening_node = mdp.nodes.WhiteningNode(reduce=True)
                whitening_node.train(exp)
                sphered_exp =  whitening_node.execute(exp)
    
                print "computing exp NN..."
                d_out = dist_to_closest_neighbour(sphered_exp,sphered_data )
                d_out_avg = d_out.mean()
                outliers_out_index = numpy.argsort(d_out)[-int(num_steps*fraction_outliers):]
                outliers_out = d_out[outliers_out_index]
                d_out_outlier_avg = outliers_out.mean()
                outlier_ratio_out = d_out_outlier_avg / d_out_avg
                
                outlier_amplification = outlier_ratio_out / outlier_ratio_in
 
                print "In: ", d_in_avg, d_in_outlier_avg, outlier_ratio_in
                print "Out: ", d_out_avg, d_out_outlier_avg, outlier_ratio_out
                outlier_info[(problem, base)] = (outlier_amplification, d_in_avg, d_in_outlier_avg, outlier_ratio_in, d_out_avg, d_out_outlier_avg, outlier_ratio_out)
           
            elif outlier_amplification_method == "New":
                print "whitening exp..."
                whitening_node = mdp.nodes.WhiteningNode(reduce=True)
                whitening_node.train(exp)
                sphered_exp =  whitening_node.execute(exp)
                magnitudes_sphered_exp = ((sphered_exp**2).sum(axis=1))**0.5
                magnitudes_sphered_exp.sort()
                
                print "computing exp NN..."
                d_exp = dist_to_closest_neighbour(sphered_exp, sphered_exp)
                d_exp_avg = d_exp.mean()
                outliers_exp_index = numpy.argsort(d_exp)[-int(num_steps*fraction_outliers):]
                outliers_exp = d_exp[outliers_exp_index]
                d_exp_outlier_avg = outliers_exp.mean()
                outlier_ratio_exp = d_exp_outlier_avg / d_exp_avg

                print "whitening exp_test..."
                sphered_exp_test =  whitening_node.execute(exp_test)
                magnitudes_sphered_exp_test = ((sphered_exp_test**2).sum(axis=1))**0.5
                magnitudes_sphered_exp_test.sort()
    
                print "computing exp_test NN..."
                d_exp_test = dist_to_closest_neighbour(sphered_exp_test, sphered_exp)
                d_exp_test_avg = d_exp_test.mean()
                outliers_exp_test_index = numpy.argsort(d_exp_test)[-int(num_steps*fraction_outliers):]
                outliers_exp_test = d_exp_test[outliers_exp_test_index]
                d_exp_test_outlier_avg = outliers_exp_test.mean()
                outlier_ratio_exp_test = d_exp_test_outlier_avg / d_exp_test_avg
                
                outlier_amplification = d_exp_test_outlier_avg / d_exp_outlier_avg
                print "*********outlier_amplification=", outlier_amplification
                print "Exp (d_avg, d_o_avg, o_r): ", d_exp_avg, d_exp_outlier_avg, outlier_ratio_exp
                print "Exp_test (d_avg, d_o_avg, o_r): ", d_exp_test_avg, d_exp_test_outlier_avg, outlier_ratio_exp_test
                outlier_info[(problem, base)] = (outlier_amplification, d_exp_avg, d_exp_outlier_avg, outlier_ratio_exp, d_exp_test_avg, d_exp_test_outlier_avg, outlier_ratio_exp_test)
            elif outlier_amplification_method == "Magnitude":
                #metric_max_var, metric_magnitude, metric_RMSA, metric_corrected_RMSA, metric_individual_RMSE
                metric = metric_RMSA
                print "Using metric:", metric
                print "whitening data..."
                whitening_node_data = mdp.nodes.WhiteningNode(reduce=True)
                whitening_node_data.train(data)
                sphered_data =  whitening_node_data.execute(data)
                metric_sphered_data = metric(sphered_data)
                metric_sphered_data.sort()
                
                print "whitening data_test..."
                sphered_data_test =  whitening_node_data.execute(data_test)
                metric_sphered_data_test = metric(sphered_data_test)
                metric_sphered_data_test.sort()

                print "whitening exp..."
                whitening_node = mdp.nodes.WhiteningNode(reduce=True)
                whitening_node.train(exp)
                sphered_exp =  whitening_node.execute(exp)
                metric_sphered_exp = metric(sphered_exp)
                metric_sphered_exp.sort()
                
                print "whitening exp_test..."
                sphered_exp_test =  whitening_node.execute(exp_test)
                metric_sphered_exp_test = metric(sphered_exp_test)
                metric_sphered_exp_test.sort() 
                      
                m_data_avg = metric_sphered_data.mean()
                m_data_test_avg = metric_sphered_data_test.mean()
                m_data_outlier_avg = metric_sphered_data[-num_outliers:].mean()
                m_data_test_outlier_avg = metric_sphered_data_test[-num_outliers:].mean()
                m_exp_avg = metric_sphered_exp.mean()
                m_exp_test_avg = metric_sphered_exp_test.mean()
                m_exp_outlier_avg = metric_sphered_exp[-num_outliers:].mean()
                m_exp_test_outlier_avg = metric_sphered_exp_test[-num_outliers:].mean()
                
                plt.figure()
                plt.suptitle("Squared norms of vectors before/after expansion: "+base+"for noise:"+str(std_noise))
                plt.plot(numpy.arange(num_steps), metric_sphered_exp, "m.")
                plt.plot(numpy.arange(num_steps), metric_sphered_exp_test, "r.")
                plt.plot(numpy.arange(num_steps), metric_sphered_data, "k.")
                plt.plot(numpy.arange(num_steps), metric_sphered_data_test, "b.")
#                plt.plot(numpy.arange(num_steps), extremes_sphered_data, "o")
#                plt.plot(numpy.arange(num_steps), extremes_sphered_exp_test, "+")

####if enable_fit:
####    plt.plot(sl[:,0], ap2, "m.")
####    plt.plot(sl[:,0], ap1,"r.")
#
#                outlier_amplification1 = m_exp_test_outlier_avg / m_exp_outlier_avg
                outlier_amplification1 = m_exp_test_outlier_avg / m_data_outlier_avg
                print "*********outlier_amplification=", outlier_amplification1
                outlier_amplification2 = m_exp_test_outlier_avg / m_data_test_outlier_avg
                print "*********outlier_amplification2=", outlier_amplification2
                outlier_amplification3 = m_exp_outlier_avg / m_data_outlier_avg
                print "*********outlier_amplification3=", outlier_amplification3
                print "Exp (m_exp_avg, m_exp_o_avg, o_r): ", m_exp_avg, m_exp_outlier_avg, m_exp_test_avg/m_exp_avg
                print "Exp_test (m_exp_avg, m_exp_o_avg, o_r): ", m_exp_test_avg, m_exp_test_outlier_avg, m_exp_test_outlier_avg/m_exp_test_avg
                outlier_info[(problem, base)] = (outlier_amplification1, outlier_amplification2, outlier_amplification3, None, None, None, None)
                perf[("OutlierAmp", base)] = (outlier_amplification3, kk, dd, 0)
            else:
                er = "Unknown outlier_amplification_method=", outlier_amplification_method
                raise Expansion(er)
 
        if enable_normalize:
            app = app-app.mean()
            app = app/app.std()

        #TODO: Make computation of overfitting metric more clean
        if problem == "s*->sh(s1)":
            error = app+0.0 #For overfitting, instead of difference goal-app we have just app, so that we compute <y^2>^0.5 instead of <(y-sh(s1))^2>^0.5
        else:
            error = goal-app
        
        if enableDisplay:
            ax = plt.subplot(3,(num_basis+2)/3,i+1)
#            plt.title("c=%f"%exponent)
            plt.plot(t, data_test, ".")
#            plt.plot(t, sl[:,1], "g.")
            plt.plot(t, app, "m.")
            plt.plot(t, goal, "k.")
            plt.xlabel("Approx. error std for %s is %f"%(base, error.std()))
    
        length = exp.shape[1]
        perf[(problem, base)] = (error.std(), kk, dd, length)

verbose=False
if verbose:
    for base in basis:
        for problem in problems:
            ee = perf[(problem,base)][0]
            kk = perf[(problem,base)][1]
            dd = perf[(problem,base)][2]
            ll = perf[(problem,base)][3]      
            print "Error for base: ", base, "(k=%2.4f, d=%2.4f, l=%3d) problem: "%(kk, dd, ll), problem ," is: ", ee

#Printing all results for outlier amplification metric
if enable_outlier_amplification:
    print ""
    print "Outlier Amplific",
    for base in basis:
        print base, " "*(10-len(base)),
    print ""
    for problem in [problem_outlier_amp]: #Basically only one problem, which is "s*->sh(s1)"
        print problem, " "*(14-len(problem)), "[",
        for base in basis:
            ov1 = outlier_info[(problem,base)][0]
            ov2 = outlier_info[(problem,base)][1]
            ov3 = outlier_info[(problem,base)][2]
            print "% 2.4f/ % 2.4f/% 2.4f,    "%(ov1,ov2,ov3),
            print "]"
    
#    joint_outlier_amp={}
#    for base in basis:
#        joint_outlier_amp[base]=0.0
#        for problem in problems_outlier_amp:
#    #        print perf[(problem,base)]
#    #        print perf[(problem,base)][0]
#            joint_outlier_amp[base] += prob_weights[problem] * (outlier_info[(problem,base)][0])
#    #    print "Joint error for base: ", base, " is: ", joint_error[base]
#    print "Joint Outlier  ",
#    for base in basis:
#        ov = joint_outlier_amp[base]
#        print "% 2.4f    "%ov,

print "Weights:", 
sum_prob_weights=0.0
for problem in problems:
    print "Problem:", problem, ": %2.5f, "%prob_weights[problem], 
    sum_prob_weights += prob_weights[problem]
print ""
print "Sum of prob_weights=%2.5f"%sum_prob_weights

print ""
print ""
print "Error            ",
for base in basis:
    print base, " "*(9-len(base)),
print ""
for problem in problems:
    print problem, " "*(14-len(problem)), "[",
    for base in basis:
        ee = perf[(problem,base)][0]
        kk = perf[(problem,base)][1]
        dd = perf[(problem,base)][2]
        ll = perf[(problem,base)][3]      
        print "%2.5f,  "%ee,
    print "],"


        
joint_error={}
for base in basis:
    joint_error[base]=0.0
    for problem in problems:
#        print perf[(problem,base)]
#        print perf[(problem,base)][0]
        joint_error[base] += prob_weights[problem] * perf[(problem,base)][0]
#    print "Joint error for base: ", base, " is: ", joint_error[base]
print "Joint Error     [",
for base in basis:
    ee = joint_error[base]
    print "%2.5f,  "%ee,
print "]"

for base in basis:
    ee = joint_error[base]
    print "%2.5f    "%ee,

if showFunctions or enableDisplay or showOutliers:
    print "Displaying..."
    plt.show()





##################### Older display stuff, not needed for the moment
####f2 = plt.figure()
####plt.suptitle("Actual mapping we are interested in to generate cos(2t) from x(t)")
####plt.plot(sl[:,0], sl2, "g.")
####if enable_fit:
####    plt.plot(sl[:,0], ap2, "m.")
####    plt.plot(sl[:,0], ap1,"r.")
####    plt.legend(["Desired mapping", "fit solution (normalized)", "approx solution (normalized)"], loc=4)


###Code for approximating slower frequency armonics
##f3 = plt.figure()
##for i, exponent in enumerate(exponents):
##    nl1 = numpy.abs(sl[:,1]) ** exponent
##    nl1 = (nl1 - nl1.mean())/nl1.std()
##    nl2 = numpy.abs(sl[:,2]) ** exponent
##    nl2 = (nl2 - nl2.mean())/nl2.std()
##
##    nl3 = numpy.abs(sl[:,1]+sl[:,2]) ** exponent
##    nl3 = (nl3 - nl3.mean())/nl3.std()
##    nl4 = numpy.abs(sl[:,2]+sl[:,3]) ** exponent
##    nl4 = (nl4 - nl4.mean())/nl4.std()
##    nl5 = numpy.abs(sl[:,3]+sl[:,4]) ** exponent
##    nl5 = (nl5 - nl5.mean())/nl5.std()
##    nl6 = numpy.abs(sl[:,4]+sl[:,5]) ** exponent
##    nl6 = (nl6 - nl6.mean())/nl6.std()
##    nl7 = numpy.abs(sl[:,5]+sl[:,6]) ** exponent
##    nl7 = (nl7 - nl7.mean())/nl7.std()
##    nl8 = numpy.abs(sl[:,6]+sl[:,7]) ** exponent
##    nl8 = (nl8 - nl8.mean())/nl8.std()
##    nl9 = numpy.abs(sl[:,7]+sl[:,8]) ** exponent
##    nl9 = (nl9 - nl9.mean())/nl9.std()
##    nl10 = numpy.abs(sl[:,8]+sl[:,9]) ** exponent
##    nl10 = (nl10 - nl10.mean())/nl10.std()
##
##
###    nl3 = numpy.abs(sl[:,3]) ** exponent
###    nl3 = (nl3 - nl3.mean())/nl3.std()
###    nl4 = numpy.abs(sl[:,4]) ** exponent
###    nl4 = (nl4 - nl4.mean())/nl4.std()
##
##
##    mat = numpy.zeros((num_steps, 9))
##    mat[:,0]=1
##    mat[:,1]=nl3
##    mat[:,2]=nl4
##    mat[:,3]=nl5
##    mat[:,4]=nl6
##    mat[:,5]=nl7
##    mat[:,6]=nl8
##    mat[:,7]=nl9
##    mat[:,8]=nl10
##
##
##    pinv = numpy.linalg.pinv(mat)
##    coefs = numpy.dot(pinv, sl[:,0])
##    
##    ap2 = numpy.dot(mat, coefs)
##
##    ax = plt.subplot(num_exponents,1,i+1)
###    plt.title("c=%f"%exponent)
##    plt.plot(t, sl[:,0], "b.")
##    plt.plot(t, ap2, "r.")
###    plt.plot(t, sl[:,1], "g.")
##
##    error = sl[:,0]-ap2
##    plt.xlabel("approximation error from sl[:,1]... to sl[:,0] for c=%f is %f, coefs=%s"%(exponent, error.std(), str(coefs)))




####
####exponents1 = [0.4]
####exponents2 = [0.03, 0.04, 0.2]
####
####num_exponents = len(exponents)
####
#####f4 = plt.figure()
####for i, exponent1 in enumerate(exponents1):
####    for j, exponent2 in enumerate(exponents2):
####        nl_func = lambda x: new_signed_nl_func(x, exponent1, exponent2)
####    #    nl_func = lambda x: numpy.sign(x)
####    #    nl_func = lambda x: x
####        
####        nl3 = nl_func(sl[:,1]*sl[:,2])
####    #    nl3 = numpy.sqrt((1.0/2 + sl[:,1]/(1.4*2)).clip(0,5)) * numpy.sign(sl[:,1]*sl[:,2])+8for i in range(len(divisions_sl2)-1):
####    
####    #    other_signed_nl_func
####    #    nl3 = other_signed_nl_func(sl[:,1], sl[:,2], 1.0, 1.0)
####        nl3 = (nl3 - nl3.mean())/nl3.std()
####        
####        nl4 = nl_func(sl[:,2]*sl[:,3])
####        nl4 = (nl4 - nl4.mean())/nl4.std()
####        nl5 = nl_func(sl[:,3]*sl[:,4])
####        nl5 = (nl5 - nl5.mean())/nl5.std()
####        nl6 = nl_func(sl[:,4]*sl[:,5])
####        nl6 = (nl6 - nl6.mean())/nl6.std()
####        nl7 = nl_func(sl[:,5]*sl[:,6])
####        nl7 = (nl7 - nl7.mean())/nl7.std()
####        nl8 = nl_func(sl[:,6]*sl[:,7])
####        nl8 = (nl8 - nl8.mean())/nl8.std()
####        nl9 = nl_func(sl[:,7]*sl[:,8])    
####        nl9 = (nl9 - nl9.mean())/nl9.std()
####        nl10 = nl_func(sl[:,8]*sl[:,9])    
####        nl10 = (nl10 - nl10.mean())/nl10.std()
####    
####    
####    #    nl3 = numpy.abs(sl[:,3]) ** exponent
####    #    nl3 = (nl3 - nl3.mean())/nl3.std()
####    #    nl4 = numpy.abs(sl[:,4]) ** exponent
####    #    nl4 = (nl4 - nl4.mean())/nl4.std()
####    
####    
####        mat = numpy.zeros((num_steps, 9))
####        mat[:,0]=1
####        mat[:,1]=nl3
####        mat[:,2]=nl4
####        mat[:,3]=nl5
####        mat[:,4]=nl6
####        mat[:,5]=nl7
####        mat[:,6]=nl8
####        mat[:,7]=nl9
####        mat[:,8]=nl10
####    
####    
####        pinv = numpy.linalg.pinv(mat)
####        coefs = numpy.dot(pinv, sl[:,0])
####        
####        ap2 = numpy.dot(mat, coefs)
####        ap2 = (ap2-ap2.mean())/ap2.std()
####        
####        delta = sfa_libs.comp_delta(ap2.reshape((num_steps,1)))[0]
####          
#####        ax = plt.subplot(num_exponents,1,i+1)
#####    #    plt.title("c=%f"%exponent)
#####        plt.plot(t, sl[:,0], "b.")
#####        plt.plot(t, ap2, "r.")
#####    #    plt.plot(t, sl[:,1], "g.")
####    
####        error = sl[:,0]-ap2
#####        plt.xlabel("approximation error from sl[:,1]... to sl[:,0] for c=%f is %f, coefs=%s"%(exponent, error.std(), str(coefs)))
####        print "approximation error from sl[:,1], sl[:,2] to sl[:,0] for exp1=%f, exp2=%f is %f, sl=%f, coefs=%s"%(exponent1, exponent2, error.std(), delta, str(coefs))
####
####
####
#####Conclusions:
#####signed expo 0.4 (product) seems to be the best nl function
#####signed new_nl(0.4, 1.0, 0.4) is less good ...
#####but signed new_nl(0.4, 0.1-0.3, 0.4) improves (thought slowness might be compromised)
#####incredible!! also decreasing the second exponent helps!!!
#####ex: signed new_nl(0.4, 0.1-0.3, 0.05)  => 0.3974 error
#####however numpy.sign() has less good performance (still similar! => 0.404052 error)
#####Note: taking the square root of the first one and fixing the sign is poor and has sharp jumps
####
####
####
#####f5 = plt.figure()
#####plt.suptitle("Actual mapping we are interested in to generate cos(t) from cos(2t) and cos(3t)")
#####divisions = [-1.7,-1.25, -0.7, -0.5, -0.25, 0.0]
#####enable_fit = True
####
####divisions_sl2 = numpy.arange(-1.45, 1.45, 0.1)
#####divisions_sl2 = [-1.0, -0.75]
#####[-1.5,-1.0, -0.5, 0.0, 0.5]
#####divisions_sl3 = numpy.arange(-1.5, 0.2, 0.2)
#####divisions_sl3 = [-1.5,-1.0, -0.5, 0.0, 0.5]
####divisions_sl3 = [-1.0, -0.9, -0.8]
####divisions_sl3 = numpy.arange(-1.45, 1.45, 0.1)
####
####
####
####goal_out = {}
####for i in range(len(divisions_sl2)-1):
####    for j in range(len(divisions_sl3)-1):
####        mask = (sl[:,1] > divisions_sl2[i]) & (sl[:,1]<divisions_sl2[i+1]) & (sl[:,2] > divisions_sl3[j]) & (sl[:,2]<divisions_sl3[j+1])
####        mean = sl1[mask].mean()
####        if numpy.isnan(mean):
####            mean = 0
####        goal_out[(i,j)] = mean
####
####fig=f6 = plt.figure()
####ax = Axes3D(f6)
####
####centroids_sl2 = (divisions_sl2[:-1]+divisions_sl2[1:])/2
####centroids_sl3 = (divisions_sl3[:-1]+divisions_sl3[1:])/2
####xx, yy = numpy.meshgrid(centroids_sl2, centroids_sl3)
####zz = numpy.zeros((len(centroids_sl2),len(centroids_sl3)))
####for i in range(len(centroids_sl2)):
####    for j in range(len(centroids_sl3)):
####        zz[i,j] = goal_out[(i,j)]
####                
####surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=mpl.cm.jet,
####        linewidth=0, antialiased=False)
####ax.set_zlim3d(-1.01, 1.01)
####ax.w_zaxis.set_major_locator(mpl.ticker.LinearLocator(10))
####ax.w_zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.03f'))
####
####fig.colorbar(surf, shrink=0.5, aspect=5)
####        
####ax.set_xlabel('c(2)')
####ax.set_ylabel('c(3)')
####ax.set_zlabel('c(1)')
####
####
####fig=f7 = plt.figure()
####ax = Axes3D(f7)
####app_out = numpy.zeros((len(centroids_sl2), len(centroids_sl3)))
####for i in range(len(centroids_sl2)):
####    for j in range(len(centroids_sl3)):
####        xx_sl2 = centroids_sl2[i]
####        xx_sl3 = centroids_sl3[j]
####        www = numpy.array([xx_sl2*xx_sl3]).reshape((1,1))
####        www = nl_func(www)
####        app_out[i,j] = www[0,0]
####      
####        
####surf = ax.plot_surface(xx, yy, app_out, rstride=1, cstride=1, cmap=mpl.cm.jet,
####        linewidth=0, antialiased=False)
####ax.set_zlim3d(-1.01, 1.01)
####ax.w_zaxis.set_major_locator(mpl.ticker.LinearLocator(10))
####ax.w_zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.03f'))
####
####fig.colorbar(surf, shrink=0.5, aspect=5)
####        
####ax.set_xlabel('c(2)')
####ax.set_ylabel('c(3)')
####ax.set_zlabel('app c(1)')
####plt.show()
####
####
####
####
####plt.show()
####
####plt.suptitle("Actual/approx mapping we need to generate cos(t) from cos(2t) and cos(3t)")
####plt.subplot(2,2,1)
####for i in range(len(divisions_sl2)-1):
####    y_out = []
####    for j in range(len(divisions_sl3)-1):
####        y_out.append(goal_out[(i,j)])
####    plt.plot(divisions_sl3[:-1], y_out)
#####plt.legend(["Desired mapping for fixed sl2 >= %f"%divisions_sl2[i] for i in range(len(divisions_sl2)-1)], loc=3)
####
####
####f7 = plt.figure()
####plt.subplot(2,2,2)
#####plt.suptitle("Actual mapping we are interested in to generate cos(t) from cos(2t) and cos(3t)")
####for j in range(len(divisions_sl3)-1):
####    y_out = []
####    for i in range(len(divisions_sl2)-1):
####        y_out.append(goal_out[(i,j)])
####    plt.plot(divisions_sl2[:-1], y_out)
#####plt.legend(["Desired mapping for fixed sl3 >= %f"%divisions_sl3[i] for i in range(len(divisions_sl3)-1)], loc=3)
####
####
####
####
####plt.subplot(2,2,3)
####for i in range(len(divisions_sl2)-1):
####    y_out = []
####    for j in range(len(divisions_sl3)-1):
####        y_out.append(app_out[i,j])
####    plt.plot(divisions_sl3[:-1], y_out)
#####plt.legend(["Approximate mapping for fixed sl2 >= %f"%divisions_sl2[i] for i in range(len(divisions_sl2)-1)], loc=3)
####
####plt.subplot(2,2,4)
####for j in range(len(divisions_sl3)-1):
####    y_out = []
####    for i in range(len(divisions_sl2)-1):
####        y_out.append(app_out[i,j])
####    plt.plot(divisions_sl2[:-1], y_out)
#####plt.legend(["Approximate mapping for fixed sl3 >= %f"%divisions_sl3[i] for i in range(len(divisions_sl3)-1)], loc=3)
####
####
######fixed_index = 2
######variable_index = 1
######for i in range(len(divisions)-1):
######    mask = (sl[:,fixed_index] > divisions[i]) & (sl[:,fixed_index]<divisions[i+1]) 
######    goal_bin = sl1[mask]
######    sl2_bin = sl[:,1][mask]
######    sl3_bin = sl[:,2][mask]
######
######    if fixed_index == 2:
######        plt.plot(sl2_bin, goal_bin, ".")
######    else:
######        plt.plot(sl3_bin, goal_bin, ".")
######        
######    if enable_fit: #and False:
######        var2_divisions = [-2.0, -1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0]
######        posx = []
######        posy = []
######        for j in range(len(var2_divisions)-1):
######            if variable_index == 0:
######                mask2 = (sl2_bin > var2_divisions[j]) & (sl2_bin <var2_divisions[j+1]) 
######                sl1_bin_bin = goal_bin[mask2]
######            else:
######                mask2 = (sl3_bin > var2_divisions[j]) & (sl3_bin <var2_divisions[j+1]) 
######                sl1_bin_bin = goal_bin[mask2]
######                
#######            plt.plot([(var2_divisions[j]+var2_divisions[j+1])/2], [sl1_bin_bin.mean()], "o")
######            posx.append((var2_divisions[j]+var2_divisions[j+1])/2)
######            posy.append(sl1_bin_bin.mean())
######        plt.plot(posx, posy)
######            
#######        plt.plot(sl[:,0], ap1,"r.")
#######        plt.legend(["Desired mapping", "fit solution (normalized)", "approx solution (normalized)"], loc=4)
######plt.legend(["Desired mapping for bin %d"%i for i in range(len(divisions)-1)], loc=5)
####
#######Fourth Experiment: Shows taylor approximation of sqrt
######t=numpy.arange(0, 3, 0.01)
######x=t-1
######y=1+0.5*x+(1.0/8)*x**2-(1.0/16)*x**3-(5.0/128)*x**4+(7.0/256)*x**5-(21.0/1024)*x**6
######plt.figure()
######plt.plot(t,y)
######plt.plot(t,numpy.sqrt(t))
####
####


#DUMPSTER: Code might be useful in the future as reference, but now it is not used
#def products_2_old(x, y, func):
#    x_height, x_width = x.shape
#    y_height, y_width = y.shape
#
#    if x_height != y_height:
#        er = "Incompatible shape of x and y: ",  x.shape,   y.shape
#        raise Exception(er)
#
#    k=0
#    mask = numpy.triu(numpy.ones((x_width,x_width)), k) > 0.5
#    print mask
##    mask = mask.reshape((1,x_width,x_width))
##    mask = numpy.ones((x_width, y_width)) > 0.5
#    z1 = x.reshape(x_height, x_width, 1)
#    z2 = y.reshape(y_height, 1, y_width)
#    yexp = func(z1, z2)
#    
#    print "yexp.shape=", yexp.shape
#    print "mask.shape=", mask.shape
##    out = yexp[:, mask]
#    out = yexp[:, mask]
##    print "out.shape=", out.shape
#    #yexp.reshape((x_height, N*N))
#    return out 
#
##Full expansion functions
#
#
#
##x = numpy.array(([[2.0, 3.0],[1.0, 1.0]]))
##expo =  Q_expo(x, expo=2.0)
##print expo
##quit()
# 
##Warning! for now only working for max_expo=3
##typically max_expo=2 or 3, d = 0.6 or 0.73 (1-0.8/max_expo)
#   
#def new_nl_func(data, expo1=2, expo2=0.5):
#    mask = numpy.abs(data) < 1
#    res = numpy.zeros(data.shape)
#
#    res[mask] = (numpy.abs(data) ** expo1)[mask]
#    res[mask^True] = (numpy.abs(data) ** expo2)[mask^True]   
#    return res
#
#def signed_expo(data, expo):
#    signs = numpy.sign(data)
#    return signs * numpy.abs(data) ** expo
#
#
#def new_signed_nl_func(data, expo1=2, expo2=0.5):
#    mask = numpy.abs(data) < 1
#    res = numpy.zeros(data.shape)
#
#    res[mask] = (signed_expo(data, expo1))[mask]
#    res[mask^True] = (signed_expo(data, expo2))[mask^True]   
#    return res
#
#def other_signed_nl_func(data1, data2, expo1=2, expo2=0.5):
#    sqrts = numpy.sqrt(numpy.abs(data1))
#
#    res1 = numpy.sign(data1 * data2)*sqrts
#    
#    return new_signed_nl_func(res1, 1.0, 1.0)


