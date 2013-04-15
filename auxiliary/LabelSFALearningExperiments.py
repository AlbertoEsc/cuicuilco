#Basic Experiments for direct label learning with SFA 
#By Alberto Escalante. Alberto.Escalante@ini.ruhr-uni-bochum.de First Version 10 June 2010
#Ruhr-University-Bochum, Institute of Neurocomputation, Teory of Neural Systems (Prof. Dr. Wiskott)

import numpy
from numpy import linalg as LA
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import Image
import sys
sys.path.append("/home/escalafl/usr/lib/python2.6/site-packages")
import mdp
import os
import glob 
import random
sys.path.append("/home/escalafl/workspace4/cuicuilco_MDP3.2/src")
import patch_mdp
import sfa_libs 

repetitions = 1 #1000
num_samples = 20 #20
dim = 18 #3--17
sort = True #and False
last_slow_signal = 3 #1/3
handling_negative_weights =  "Offset" # "Preserve"# "Preserve", "Truncate", "Offset"
consistency_constraints = True #and False
lagrange_multipliers = True and False #seems like solution worsens when lambdas introduced! 
default_lambda1 = 0.0 # -2.09545818 #lambda1 and lambda2, usually lambda_2 = 0
default_lambda2 = 0.0 # -1.90758595e-15 
Q = R = num_samples * 1.0 #(assuming w_i=1)
plot_eigenvectors = True
add_weight_noise = True
weight_noise_amplitude = 1e-16
#direct_graph_construction = True and False
dg_enable_2_labels = True and False
dg_a1 = -0.4 
dg_a2 = -0.2
dg_b = 0.5
#EV1 =  - dg_a1 (num_samples-1)+dg_b

numpy.random.seed(123456789)

def delta_sfa_graph(y, node_weights, edge_weights, count_loops=False):
    delta = 0.0

    if count_loops:    
        for i, y_i in enumerate(y):
            for j, y_j in enumerate(y):
                w = edge_weights[(j,i)]
                delta += w * (y_i-y_j)**2
    else:
        for i, y_i in enumerate(y):
            for j, y_j in enumerate(y):
                if i != j:
                    w = edge_weights[(j,i)]
                    delta += w * (y_i-y_j)**2       
    return delta

error = {}
for repetition in range(repetitions):
    samples = numpy.random.normal(size=(num_samples,dim))
    samples = samples-samples.mean(axis=0)
    samples = samples / samples.std(axis=0)
    
    labels = numpy.random.random(num_samples)
    labels = labels-labels.mean()
    labels_std2 = ((labels**2).sum()/(num_samples-1))**0.5
    labels = labels/labels_std2
    if sort:
        labels.sort()

    labels1 = labels
    labels2 = numpy.random.random(num_samples)
    labels2 = labels2-labels2.mean()
    labels2 = labels2 - numpy.dot(labels2,labels1)*labels1 / numpy.dot(labels1, labels1)
    labels2_std2 = ((labels2**2).sum()/(num_samples-1))**0.5
    labels2 = labels2/labels2_std2
    if labels2[0] > 0:
        labels2 *= -1

    if consistency_constraints:
        num_equations = 3* num_samples
    else:
        num_equations = num_samples + 1

    if lagrange_multipliers:
        M = numpy.zeros((num_equations, num_samples*num_samples+2))
    else:
        M = numpy.zeros((num_equations, num_samples*num_samples))        
    C = numpy.zeros((num_equations, 1))
    
    #coef(wj,i) = M(i*num_samples+j)
    if consistency_constraints:
        #Alternative formulation, removed to promote symmetry, why is it better????
        #First restriction: for each node: sum(weights_in) - sum(weights_out) = 0 (could be changed to sum(weights_out)=1)
        eq = 0
        for i in range(num_samples):
            for j in range(num_samples):
                if j != i:
                    M[eq+i, i*num_samples+j] = 1
                    M[eq+i, j*num_samples+i] = -1

#        #First restriction: for each node: sum(weights_out) = 1 (assuming w_i=1)
#        eq = 0
#        for i in range(num_samples):
#            for j in range(num_samples):
#                if j != i:
#                    M[eq+i, j*num_samples+i] = 1
#            C[eq+i] = Q
                
        #Second restriction: for each node: sum(weights_in) = Q (assuming w_i=1).   W[i*num_samples+j] = w_{j,i}
        eq = num_samples
        for i in range(num_samples):
            for j in range(num_samples):
                if j != i:
                    M[eq+i, i*num_samples+j] = 1
            C[eq+i] = Q
    
        eq = 2*num_samples
    else:
        #Only condition: sum of all weights (without loops) = 2*num_samples
        for i in range(num_samples):
            for j in range(num_samples):
                if j != i:
                    M[0, i*num_samples+j] = 1
        C[0] = 2.0*num_samples 
        eq = 1
    
    #Third restriction: for each partial derivative: flow(i,j0)*d(j0,i)+flow(i,j1)*d(j1,i)+...=0
    for i in range(num_samples):
        for j in range(num_samples):
            diff = labels[i]-labels[j]
            if j != i:
                M[eq+i, i*num_samples+j] = 2 * diff / R
                M[eq+i, j*num_samples+i] = 2 * diff / R
        if lagrange_multipliers: #BUG HERE??? no eq???
            M[eq+i, num_samples**2] = 2*1*labels[i] # 2* default_lambda1*labels[i] 
            M[eq+i, num_samples**2+1] =  1 # default_lambda2 * 1
        else:
            C[eq+i] = -(default_lambda1*2*1*labels[i]+default_lambda2*1) 
    Mpinv = numpy.linalg.pinv(M)
    Wfull = numpy.dot(Mpinv, C)
    
    W = Wfull[0:num_samples**2].flatten()
    L = Wfull[num_samples**2:num_samples**2+2].flatten()

    if handling_negative_weights == "Truncate":
        print "Truncating negative edge weights"
        W[W<0]=0.0
    elif handling_negative_weights == "Offset": #here offset should be improved to keep magnitude!
        print "Offseting negative edge weights"
        w_min = W.min()
        if w_min < 0:
            W = (W - w_min)/(1-num_samples*w_min)
    elif handling_negative_weights == "Preserve":
        print "Preserving negative weights"
    else:
        ex = "handling_negative_weights method unknown:", handling_negative_weights
        raise Exception(ex)

    W2 = W.reshape((num_samples,num_samples))
    if add_weight_noise:
        weight_noise = weight_noise_amplitude * numpy.random.normal(size=(num_samples,num_samples))
        weight_noise = weight_noise + weight_noise.T
        W2 = W2+weight_noise
    print "W2=", W2
    T = W2 + W2.T
    N = -2.0 / R * T
    evals, evects = LA.eig(W2)
    evects = evects # *(num_samples-1)/6
    signs = numpy.sign(evects[0,:])
    evects[:,:] = evects[:,:]*signs*-1
    
    print "N.eigenvalues:", evals 
    lll1 = -1 * (evals + 4/R) / 2
    print "Corresponds to lambdas:",  -1 * (evals + 4/R) / 2
    print "Corresponds to deltas:",  -1 * lll1 * (Q-1)
    print "N.eigenvectors.T:"
    print evects.T
    evects_means = evects.mean(axis=0)

    if plot_eigenvectors:
        f1 = plt.figure()
        plt.suptitle("Eigenvectors of  N = -2 * Diag(wi**-0.5) T Diag(wi**-0.5)")
        plt.plot(range(num_samples), labels/((labels**2).sum())**0.5, "ko-")
        plt.plot(range(num_samples), evects[:, 0:3], "-") 
        plt.plot(range(num_samples), evects[:, dim-1], "-") 

        plt.legend(["label/||label||", "Eigenvector 1, EV=%f, Lam1=%f, mean=%f"%(evals[0], lll1[0], evects_means[0]), "Eigenvector 2, EV=%f, Lam1=%f, mean=%f"%(evals[1], lll1[1], evects_means[1]), 
                    "Eigenvector 3, EV=%f, Lam1=%f, mean=%f"%(evals[2], lll1[2], evects_means[2]), "Eigenvector %d, EV=%f, Lam1=%f, mean=%f"%(dim, evals[dim-1], lll1[dim-1], evects_means[dim-1])], loc=4)
    
    
    if lagrange_multipliers:
        print "lambda1=", L[0], "lambda2=", L[1]

    node_weights = numpy.ones(num_samples)
    edge_weights={}
    for i in range(num_samples):
        for j in range(num_samples):
            edge_weights[(j,i)] = W[i*num_samples+j]
        
    sfa_node = mdp.nodes.SFANode()
    sfa_node.train(samples, train_mode = "graph", block_size = 1, node_weights=node_weights, edge_weights=edge_weights)
    sl = sfa_node.execute(samples)
    signs = numpy.sign(sl[0,:])
    sl[:,:] = sl[:,:] * signs * -1

    delta = delta_sfa_graph(sl[:,0], node_weights=node_weights, edge_weights=edge_weights)
    delta_labels = delta_sfa_graph(labels, node_weights=node_weights, edge_weights=edge_weights)
    print "Computed delta of slowest feature=", delta, "corresponds to lambda1=", -1 * delta / (Q-1)
    print "Computed delta of labels=", delta_labels, "corresponds to lambda1=", -1 * delta_labels / (Q-1)
    print "sl[:,0]=", sl[:,0]
    print "labels=",labels
    error_sfa = labels- sl[:,0]
    rmse_sfa = ((error_sfa**2).sum()/num_samples)**0.5
    print "rmse_sfa=", rmse_sfa
    
    print sl.shape
    labels_lr = labels.reshape((num_samples,1))

    error[(repetition, "SFA")] = rmse_sfa
    
    y_lr = {}
    for dim_lr in range(1,dim+1):
        lr_node = mdp.nodes.LinearRegressionNode()
        print "dim_lr=", dim_lr, 
        if dim_lr < dim or True:
            lr_node.train(sl[:, 0:dim_lr], labels_lr)
            y_lr[dim_lr] = lr_node.execute(sl[:, 0:dim_lr])
        else:
            lr_node.train(samples, labels_lr)
            y_lr[dim_lr] = lr_node.execute(samples)
            
        
        error_lr = labels_lr - y_lr[dim_lr]
        #print ", error_lr[:,0]=", error_lr[:,0],
        rmse_lr = ((error_lr**2).sum()/num_samples)**0.5
        print ", rmse_lr[%d]="%dim_lr, rmse_lr
        error[(repetition, "LR", dim_lr)] = rmse_lr


    f0 = plt.figure()
    plt.suptitle("Linear SFA learning arbitrary functions. %d random samples of dimension %d"%(num_samples, dim))
    plt.plot(range(num_samples), sl[:,0:last_slow_signal])
    plt.plot(range(num_samples), labels, "ko-")
    if dg_enable_2_labels:
        plt.plot(range(num_samples), labels2, "k+-") 
    plt.plot(range(num_samples), y_lr[1], "*-")
    plt.plot(range(num_samples), y_lr[2], "*-")
    plt.plot(range(num_samples), y_lr[dim], "*-")

    if dg_enable_2_labels:
        if last_slow_signal >= 3:
            plt.legend(["slow feature 1","slow feature 2","slow feature 3","label1", "label2", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
        elif last_slow_signal == 2:
                plt.legend(["slow feature 1","slow feature 2","label1", "label2", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
        elif last_slow_signal == 1:
                plt.legend(["slow feature 1", "label1", "label2", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
    else:
        if last_slow_signal >= 3:
            plt.legend(["slow feature 1","slow feature 2","slow feature 3","label", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
        elif last_slow_signal == 2:
                plt.legend(["slow feature 1","slow feature 2","label", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
        elif last_slow_signal == 1:
                plt.legend(["slow feature 1", "label", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
    
    ########################################################   
    #Direct Graph Construction
    labels_2D = labels.reshape((num_samples,1))
    labels2_2D = labels2.reshape((num_samples,1))

    if dg_enable_2_labels:
        dg_N = dg_a1 * numpy.dot(labels_2D, labels_2D.T) + dg_a2 * numpy.dot(labels2_2D, labels2_2D.T) + dg_b * numpy.identity(num_samples)
    else:
        dg_N = dg_a1 * numpy.dot(labels_2D, labels_2D.T) + dg_b * numpy.identity(num_samples)
    dg_T = - (2.0 / R) * dg_N # Assuming wi=1
    print "dg_N[0:3,0:3]=", dg_N[0:3,0:3], "R=", R

    
    evals, evects = LA.eig(dg_N)
    evects = evects # *(num_samples-1)/6
    signs = numpy.sign(evects[0,:])
    evects[:,:] = evects[:,:]*signs*-1
        
    print "dg_N.eigenvalues:", evals 
    #quit()
    lll1 = -1 * (evals + 4/R) / 2
    print "Corresponds to lambdas:",  -1 * (evals + 4/R) / 2
    print "Corresponds to deltas:",  -1 * lll1 * (Q-1)
    print "dg_N.eigenvectors.T:"
    print evects.T
    evects_means = evects.mean(axis=0)

    if plot_eigenvectors:
        f1 = plt.figure()
        plt.suptitle("Eigenvectors of dg_N = -(2/R) * Diag(wi**-0.5) T Diag(wi**-0.5)")
        plt.plot(range(num_samples), labels/((labels**2).sum())**0.5, "ko-")
        if dg_enable_2_labels:
            plt.plot(range(num_samples), labels2/((labels2**2).sum())**0.5, "k+-")
        plt.plot(range(num_samples), evects[:, 0:3], "-") 
        plt.plot(range(num_samples), evects[:, dim-1], "-") 

        if dg_enable_2_labels:
            plt.legend(["label1/||label1||", "label2/||label2||", "Eigenvector 1, EV=%f, Lam1=%f, mean=%f"%(evals[0], lll1[0], evects_means[0]), "Eigenvector 2, EV=%f, Lam1=%f, mean=%f"%(evals[1], lll1[1], evects_means[1]), 
                    "Eigenvector 3, EV=%f, Lam1=%f, mean=%f"%(evals[2], lll1[2], evects_means[2]), "Eigenvector %d, EV=%f, Lam1=%f, mean=%f"%(dim, evals[dim-1], lll1[dim-1], evects_means[dim-1])], loc=4)
        else:
            plt.legend(["label/||label||", "Eigenvector 1, EV=%f, Lam1=%f, mean=%f"%(evals[0], lll1[0], evects_means[0]), "Eigenvector 2, EV=%f, Lam1=%f, mean=%f"%(evals[1], lll1[1], evects_means[1]), 
                    "Eigenvector 3, EV=%f, Lam1=%f, mean=%f"%(evals[2], lll1[2], evects_means[2]), "Eigenvector %d, EV=%f, Lam1=%f, mean=%f"%(dim, evals[dim-1], lll1[dim-1], evects_means[dim-1])], loc=4)
  
  
    if add_weight_noise:
        weight_noise = 100*weight_noise_amplitude * numpy.random.normal(size=(num_samples,num_samples))
        weight_noise = weight_noise + weight_noise.T
        dg_W2 = dg_T/2.0+weight_noise
    
    
    if handling_negative_weights == "Truncate":
        print "Truncating negative edge weights"
        dg_W2[dg_W2<0]=0.0
    elif handling_negative_weights == "Offset": #here offset should be improved to keep magnitude!
        print "Offseting negative edge weights"
        w_min = dg_W2.min()
        if w_min < 0:
            dg_W2 = (dg_W2 - w_min)/(1-num_samples*w_min)
    elif handling_negative_weights == "Preserve":
        print "Preserving negative weights"
    else:
        ex = "handling_negative_weights method unknown:", handling_negative_weights
        raise Exception(ex)
    
    dg_W2 = (dg_W2+dg_W2.T)/2.0
    node_weights = numpy.ones(num_samples)
    edge_weights={}
    for i in range(num_samples):
        for j in range(num_samples):
            edge_weights[(j,i)] = dg_W2[j,i]
            
    sfa_node = mdp.nodes.SFANode()
    sfa_node.train(samples, train_mode = "graph", block_size = 1, node_weights=node_weights, edge_weights=edge_weights)
    sl = sfa_node.execute(samples)
    signs = numpy.sign(sl[0,:])
    sl[:,:] = sl[:,:] * signs * -1
    delta = delta_sfa_graph(sl[:,0], node_weights=node_weights, edge_weights=edge_weights)
    delta_labels = delta_sfa_graph(labels, node_weights=node_weights, edge_weights=edge_weights)
    print "Computed delta of slowest feature=", delta, "corresponds to lambda1=", -1 * delta / (Q-1)
    print "Computed delta of labels=", delta_labels, "corresponds to lambda1=", -1 * delta_labels / (Q-1)
    print "sl[:,0]=", sl[:,0]
    print "labels=",labels
    error_sfa = labels- sl[:,0]
    rmse_sfa = ((error_sfa**2).sum()/num_samples)**0.5
    print "rmse_sfa=", rmse_sfa
    
    f0 = plt.figure()
    plt.suptitle("DG: Linear SFA learning arbitrary functions. %d random samples of dimension %d"%(num_samples, dim))
    plt.plot(range(num_samples), sl[:,0:last_slow_signal])
    plt.plot(range(num_samples), labels, "ko-")
    if dg_enable_2_labels:   
        plt.plot(range(num_samples), labels2, "k+-")
    plt.plot(range(num_samples), y_lr[1], "*-")
    plt.plot(range(num_samples), y_lr[2], "*-")
    plt.plot(range(num_samples), y_lr[dim], "*-")


    if dg_enable_2_labels:
        if last_slow_signal >= 3:
            plt.legend(["slow feature 1","slow feature 2","slow feature 3","label1", "label2", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
        elif last_slow_signal == 2:
                plt.legend(["slow feature 1","slow feature 2","label1", "label2", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
        elif last_slow_signal == 1:
                plt.legend(["slow feature 1", "label1", "label2", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
    else:
        if last_slow_signal >= 3:
            plt.legend(["slow feature 1","slow feature 2","slow feature 3","label", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
        elif last_slow_signal == 2:
                plt.legend(["slow feature 1","slow feature 2","label", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
        elif last_slow_signal == 1:
                plt.legend(["slow feature 1", "label", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4) 
    
for repetition in range(repetitions):
    for dim_lr in range(1,dim+1):   
        error[("LR", dim_lr)] = 0

error_sfa_avg_norm = 0
for repetition in range(repetitions):
    error_sfa_avg_norm += error[(repetition, "SFA")]/ error[(repetition, "LR", dim)]
    for dim_lr in range(1,dim+1):   
        error[("LR", dim_lr)] += error[(repetition, "LR", dim_lr)] / error[(repetition, "LR", dim)]
error_sfa_avg_norm  /= repetitions
for dim_lr in range(1,dim+1):   
    error[("LR", dim_lr)] /= repetitions

print "rms_sfa_avg_norm =", error_sfa_avg_norm 
for dim_lr in range(1,dim+1):   
    print "rms_lr_avg_norm [%d]="%dim_lr, error[("LR", dim_lr)]

##f0 = plt.figure()
##plt.suptitle("Linear SFA learning arbitrary functions. %d random samples of dimension %d"%(num_samples, dim))
##plt.plot(range(num_samples), sl[:,0:last_slow_signal])
##plt.plot(range(num_samples), labels, "ko-")
##plt.plot(range(num_samples), y_lr[1], "*-")
##plt.plot(range(num_samples), y_lr[2], "*-")
##plt.plot(range(num_samples), y_lr[dim], "*-")
##
##
##if last_slow_signal >= 3:
##    plt.legend(["slow feature 1","slow feature 2","slow feature 3","label", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
##elif last_slow_signal == 2:
##    plt.legend(["slow feature 1","slow feature 2","label", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
##elif last_slow_signal == 1:
##    plt.legend(["slow feature 1", "label", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)



plt.show()

#1000 repetitions, 20 samples, dim=3
#"Preserve"
#rms_sfa_avg_norm = 1.33769693847
#rms_lr_avg_norm [1]= 1.00153008565
#rms_lr_avg_norm [2]= 1.00082425569
#rms_lr_avg_norm [3]= 1.0
#"Truncate"
#rms_sfa_avg_norm = 1.36967580204
#rms_lr_avg_norm [1]= 1.0146706488
#rms_lr_avg_norm [2]= 1.00734331683
#rms_lr_avg_norm [3]= 1.0
#"Offset"
#rms_sfa_avg_norm = 1.32357226611
#rms_lr_avg_norm [1]= 1.00177744968
#rms_lr_avg_norm [2]= 1.00097686945
#rms_lr_avg_norm [3]= 1.0

#1000 repetitions, 20 samples, dim=17
#"Preserve"
#rms_sfa_avg_norm = 1.01862825171
#rms_lr_avg_norm [1]= 1.00140899153
#rms_lr_avg_norm [2]= 1.00140546389
#rms_lr_avg_norm [3]= 1.00139819279
#rms_lr_avg_norm [4]= 1.00138531213
#rms_lr_avg_norm [5]= 1.00136553436
#rms_lr_avg_norm [6]= 1.00134200451
#rms_lr_avg_norm [7]= 1.00130723727
#rms_lr_avg_norm [8]= 1.00126413493
#rms_lr_avg_norm [9]= 1.00121550388
#rms_lr_avg_norm [10]= 1.00115307043
#rms_lr_avg_norm [11]= 1.00107615231
#rms_lr_avg_norm [12]= 1.00098722719
#rms_lr_avg_norm [13]= 1.00087835969
#rms_lr_avg_norm [14]= 1.00075024523
#rms_lr_avg_norm [15]= 1.00056942089
#rms_lr_avg_norm [16]= 1.00035439729
#rms_lr_avg_norm [17]= 1.0
#"Truncate"
#rms_sfa_avg_norm = 1.59885113342
#rms_lr_avg_norm [1]= 1.56437101474
#rms_lr_avg_norm [2]= 1.56426068032
#rms_lr_avg_norm [3]= 1.56392541691
#rms_lr_avg_norm [4]= 1.56324065096
#rms_lr_avg_norm [5]= 1.56196134136
#rms_lr_avg_norm [6]= 1.55879424672
#rms_lr_avg_norm [7]= 1.55199254185
#rms_lr_avg_norm [8]= 1.53961461607
#rms_lr_avg_norm [9]= 1.52130133725
#rms_lr_avg_norm [10]= 1.49625409906
#rms_lr_avg_norm [11]= 1.46276327176
#rms_lr_avg_norm [12]= 1.41384804853
#rms_lr_avg_norm [13]= 1.35363554242
#rms_lr_avg_norm [14]= 1.29994730044
#rms_lr_avg_norm [15]= 1.21242417825
#rms_lr_avg_norm [16]= 1.12623681116
#rms_lr_avg_norm [17]= 1.0
#"Offset"
#rms_sfa_avg_norm = 1.01564307681
#rms_lr_avg_norm [1]= 1.00142750257
#rms_lr_avg_norm [2]= 1.00142441784
#rms_lr_avg_norm [3]= 1.00141726558
#rms_lr_avg_norm [4]= 1.00140493762
#rms_lr_avg_norm [5]= 1.00138721078
#rms_lr_avg_norm [6]= 1.00136455322
#rms_lr_avg_norm [7]= 1.00133150606
#rms_lr_avg_norm [8]= 1.00129187696
#rms_lr_avg_norm [9]= 1.00124503053
#rms_lr_avg_norm [10]= 1.0011805057
#rms_lr_avg_norm [11]= 1.00110670796
#rms_lr_avg_norm [12]= 1.00102088188
#rms_lr_avg_norm [13]= 1.00091001917
#rms_lr_avg_norm [14]= 1.00077996731
#rms_lr_avg_norm [15]= 1.00063559136
#rms_lr_avg_norm [16]= 1.00041919781
#rms_lr_avg_norm [17]= 1.0

##First Experiment: Extract slow signals from noise using regular training graph, but with arbitrary edge weights
##Here we approximate different functions by modifying the edge weights following an heuristic
##Signals are extracted from noise, which is possible only because of extreme overfitting
#activate_first_experiment = False
#
#if activate_first_experiment:
#    num_steps = 500
#    num_slow_signals = 10
#    
#    std_noise = .15
#    
#    t = numpy.linspace(0, numpy.pi, num_steps)
#    
#    dim = 499
#    noise = numpy.random.normal(size = (num_steps, dim))
#    #noise[:,0] += numpy.arange(num_steps)
#    noise = noise-noise.mean(axis=0)
#    noise = noise / noise.std(axis=0)
#    
#    #node_weights = numpy.sin(t)
#    node_weights = numpy.ones(num_steps)
#    
#    weight_shapes = ["normal", "sine", "sine**2", "normal_sine", "cos+1", "sin/(t+0.01)"] # , "sin/t**2", "sine+0.5", "normal_sine2", "normal_sine2_rep", sin/(t+0.01)
#    
#    max_amplitude_sfa = 3.0
#    for plt_num, weight_shape in enumerate(weight_shapes):
#    #weight_shape = "normal_sine2"
#    
#        edge_weights={}
#        
#        for i in range(0, num_steps-1):
#            if weight_shape == "normal":
#                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = 1.0 
#            elif weight_shape == "sine":
#                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])+0.0005
#            elif weight_shape == "sine+0.5":
#                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])+0.5
#            elif weight_shape == "normal_sine":
#                if i >= num_steps/2:
#                    edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])+0.0005
#                else:
#                    edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = 1.0
#            elif weight_shape == "cos+1":
#                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.cos(t[i])+1.0005            
#            elif weight_shape == "sine**2":
#                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])**2+0.0005
#            elif weight_shape == "normal_sine**2":
#                if i >= num_steps/2:
#                    edge_weights[(i+1,i)] =  edge_weights[(i,i+1)] = numpy.sin(t[i])**2+0.0005
#                else:
#                    edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = 1.0
#            elif weight_shape == "normal_sine2_rep":
#                if i >= num_steps/2:
#                    edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])**2+0.0005
#                    edge_weights[(i+1,i-1)] = edge_weights[(i-1,i+1)] = numpy.sin(t[i])**2+0.0005
#                else:
#                    edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = 1.0
#                    edge_weights[(i+2,i)] = edge_weights[(i,i+2)] = 1.0                
#            elif weight_shape == "sin/t**2":
#                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])/ (2* (i * numpy.pi/num_steps)+0.01)**2 + 0.00025
#            elif weight_shape == "sin/(t+0.01)":
#                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])/ (t[i] + 0.01) + 0.000025
#            elif weight_shape == "sin**0.5":
#                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])**0.5 + 0.00025
#    
#        sfa_node = mdp.nodes.SFANode(output_dim=10)
#        sfa_node.train(noise, train_mode = "graph", block_size = 1, node_weights=node_weights, edge_weights=edge_weights)
#        
#        sl = sfa_node.execute(noise)
#        print sl.shape
#        sl_signs = numpy.sign(sl[0,:])
#        sl = sl * sl_signs * -1
#        
#        ax = plt.subplot(2,3,plt_num+1)
#        plt.plot(t, sl[:,3], "y.")
#        plt.plot(t, sl[:,2], "g.")
#        plt.plot(t, sl[:,1], "r.")
#        plt.plot(t, sl[:,0], "b.")
#        plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
#        plt.title(weight_shape+", D[0:3]=[%f,%f,%f]"%(sfa_node.d[0], sfa_node.d[1], sfa_node.d[2]))
#    #    plt.plot(t, ap1, "r.")
#    #    if enable_fit:
#    #        plt.plot(t, ap2, "m.")
#    #    plt.plot(t, sl2, "g.")
#    #    if enable_fit:    
#    #        plt.legend(["x(t)=cos(t)+%f*n(t)"%std_noise, "white(|x|^%f)"%exponent, "white(fit(|x|))", "cos(2t)"], loc=4)
#    #    else:
#    #        plt.legend(["x(t)=cos(t)+%f*n(t)"%std_noise, "white(|x|^%f)"%exponent, "cos(2t)"], loc=4)
#    #
#    #    error = sl2-ap1
#    #    error2 = sl2-ap2
#    #    if enable_fit:
#    #        plt.xlabel("approximation error white(|x|^%f) and white(fit(|x|)) vs cos(2t): %f and %f"%(exponent, error.std(), error2.std()))
#    #    else:
#    #        plt.xlabel("approximation error white(|x|^%f) vs cos(2t): %f"%(exponent, error.std()))        
#
#activate_second_experiment=True
#
#if activate_second_experiment:
#    #Second Experiment: Extract slow signals from noise using regular training graph, but with arbitrary NODE weights
#    #Here we approximate different functions by modifying the NODE weights following an heuristic
#    #Signals are extracted from noise, which is possible only because of extreme overfitting
#    num_steps = 500
#    num_slow_signals = 10
#    
#    std_noise = .15  
#    t = numpy.linspace(0, numpy.pi, num_steps)    
#    dim = 499
#    noise = numpy.random.normal(size = (num_steps, dim))
#    #noise[:,0] += numpy.arange(num_steps)
#    noise = noise-noise.mean(axis=0)
#    noise = noise / noise.std(axis=0)
#    
#    #node_weights = numpy.sin(t)
#    node_weights = numpy.ones(num_steps)
#
#    edge_weights = {}
#    for i in range(0, num_steps-1):
#        edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = 1.0
#                
#    goal = 1.0*t**2+t
#    goal = goal-goal.mean(axis=0)
#    goal = goal / goal.std(axis=0)
#    dgoal = goal[1:]-goal[0:-1]
#    dgoal = numpy.concatenate((dgoal, [dgoal[-1]]))
#
#    weight_shapes = ["normal2", "test7", "test8", "test9", "test10","test6"] 
#    # , "sin/t**2", "sine+0.5", "normal_sine2", "normal_sine2_rep", sin/(t+0.01)
#    
#    max_amplitude_sfa = 3.0
#    for plt_num, weight_shape in enumerate(weight_shapes):
#    #weight_shape = "normal_sine2"
#        if weight_shape == "normal":
#            node_weights = numpy.ones(num_steps)
#        if weight_shape == "normal2":
#            node_weights = numpy.ones(num_steps)*2
#        elif weight_shape == "(|c(i)|+k)/(|l(i)|+k)":
#            node_weights = (numpy.abs(numpy.cos(t))+0.001) / (numpy.abs(goal)+0.001)
#        elif weight_shape == "test1":
#            node_weights = 1.0 / (numpy.abs(goal)+0.001)
#        elif weight_shape == "test2":
#            node_weights = 1.0 / ((numpy.abs(goal)**2+0.001))
#        elif weight_shape == "test3":
#            node_weights = 1.0 / ((numpy.abs(goal)**0.5+0.001))
#        elif weight_shape == "test4":
#            node_weights = (numpy.abs(numpy.cos(t))+0.0001) / ((numpy.abs(goal)**0.5+0.0001))
#        elif weight_shape == "test5":
#            node_weights = (numpy.abs(numpy.cos(t))**2+0.0001) / ((numpy.abs(goal)**0.5+0.0001))
#        elif weight_shape == "test6":
#            node_weights = 1.0 / ((numpy.abs(goal)**0.5+0.0001)*(numpy.abs(numpy.sin(t))+0.001))
#        elif weight_shape == "test7":
#            node_weights = 1.0 / (numpy.abs(numpy.sin(t))+0.001)
#        elif weight_shape == "test8":
#            node_weights = 1.0 / (numpy.abs(numpy.sin(t))+0.001)
#        elif weight_shape == "test9":
#            node_weights = 1.0 / (numpy.abs(numpy.sin(t))**2+0.001)
#        elif weight_shape == "test10":
#            node_weights = (numpy.abs(numpy.sin(t))**0.01+0.001)
#
#        sfa_node = mdp.nodes.SFANode(output_dim=10)
#        sfa_node.train(noise, train_mode = "graph", block_size = 1, node_weights=node_weights, edge_weights=edge_weights)
#        
#        sl = sfa_node.execute(noise)
#        print sl.shape
#        sl_signs = numpy.sign(sl[0,:])
#        sl = sl * sl_signs * -1
#        
#        ax = plt.subplot(2,3,plt_num+1)
#        plt.plot(t, sl[:,3], "y.")
#        plt.plot(t, sl[:,2], "g.")
#        plt.plot(t, sl[:,1], "r.")
#        plt.plot(t, sl[:,0], "b.")
#        plt.plot(t, goal, "k.")
#        plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
#        plt.title(weight_shape+", D[0:3]=[%f,%f,%f]"%(sfa_node.d[0], sfa_node.d[1], sfa_node.d[2]))
#
#plt.show()


