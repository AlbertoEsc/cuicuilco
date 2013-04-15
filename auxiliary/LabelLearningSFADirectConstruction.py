#Basic Experiments for direct label learning with SFA 
#By Alberto Escalante. Alberto.Escalante@ini.ruhr-uni-bochum.de First Version 10 June 2010
#Important update for direct graph construction added on 25 April 2012
#Ruhr-University-Bochum, Institute of Neurocomputation, Teory of Neural Systems (Prof. Dr. Wiskott)

#TODO: Decide on magnitude of u and make sure the free responses (theory) fulfill their supposed magnitude

import numpy
from numpy import linalg as LA
import scipy
#from mpl_toolkits.mplot3d import Axes3D
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

repetitions = 0 #1000
num_samples = 32 #20
dim = 30 #3--17,18
num_labels = 3
generate_armonic_labels = True and False
random_node_weights=False #or True
sort = True #and False
last_slow_signal = 5 #1/3
handling_negative_weights =  "Preserve" #"Truncate"#"Smart_Fix"#"Offset"#"Preserve" # "Preserve"# "Preserve", "Truncate", "Offset" "Smart_Fix"
consistency_constraints = True #and False
lagrange_multipliers = True and False #seems like solution worsens when lambdas introduced! 
Q = R = num_samples * 1.0 #(assuming w_i=1)
plot_eigenvectors = True #and False
plot_eigenvectors_regular_sfa = True and False
add_weight_noise = True
weight_noise_amplitude = 1e-15
first_label_lr = 0 # 4 # 0 to num_labels-1
num_labels_lr = 1
num_eigenvectors_show = 20 # min(num_samples, num_labels + 2)
numpy.random.seed(123456789)

#TODO: loops do not make any difference!, remove the parameter!
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

def dot(x,y): #1D arrays
    return (x*y).sum()

def norm(x): #1D array
    return ((x**2).sum())**0.5

def normalize(x): #1D array
    return x / norm(x)

def decorrelate_from_second(y, x): #1D arrays
    nx = normalize(x)
    return y - nx * numpy.dot(nx,y)

def construct_M_from_eigenvectors(eigenvectors, eigenvalues):
    dim = eigenvectors.shape[1]
    M = numpy.zeros((dim,dim))
    if len(eigenvectors) != len(eigenvalues):
        er = "Different number of eigenvectors and eigenvalues"
        raise Exception(er)
    for i, evector in enumerate(eigenvectors):
        ev_2D = normalize(evector)
        ev_2D = ev_2D.reshape((dim,1))
        M += eigenvalues[i] * numpy.dot(ev_2D, ev_2D.T)
    
    return M

def construct_Gamma_from_M(M, node_weights):
    DV_12 = numpy.diag(node_weights**0.5)
    return numpy.dot(DV_12, numpy.dot(M, DV_12))

def construct_M_from_Gamma(Gamma, node_weights):
    DV_m12 = numpy.diag(node_weights**(-0.5))
    return numpy.dot(DV_m12, numpy.dot(Gamma, DV_m12))

#TODO: verify normalization constraints are enforced on labels, 
#Why does matrix M/Gamma has complex/negative eigenvalues???
#Are extracted features slower than the desired features?
#Is the number of samples/dimension doing something nasty here?

error = {}
for repetition in range(repetitions):
    samples = numpy.random.normal(size=(num_samples,dim))
    samples = samples-samples.mean(axis=0)
    samples = samples / samples.std(axis=0)

    if random_node_weights:
        node_weights = numpy.random.normal(size=num_samples)
        node_weights = node_weights + -1 * node_weights.min()+1.0/num_samples
        node_weights = Q * node_weights/node_weights.sum()
    else:
        node_weights = Q*numpy.ones(num_samples)*1.0/num_samples
 
    constr_eigenvectors = numpy.random.normal(size=(num_labels+1, num_samples))

    s = (R ** 0.5 / Q) * node_weights ** 0.5
    constr_eigenvectors[0] = s
    offset_eigenvectors=1

    if generate_armonic_labels:
        for j in range(1, num_labels+1):
            constr_eigenvectors[j] = -1 * numpy.cos(numpy.pi * j *numpy.arange(num_samples)/num_samples)

    for i in range(1,num_labels+1):
        for j in range(0,i):
            print "i=", i, "j=",j
            constr_eigenvectors[i] = decorrelate_from_second(constr_eigenvectors[i], constr_eigenvectors[j]) 
            print "constr_eigenvectors[i]=", constr_eigenvectors[i]
            print "decorrelated from constr_eigenvectors[j]=", constr_eigenvectors[j]
            print numpy.dot(constr_eigenvectors[j], constr_eigenvectors[i])
    for i in range(1,num_labels+1):
        constr_eigenvectors[i] = normalize(constr_eigenvectors[i])

#    signs = numpy.sign(constr_eigenvectors[:,0])
#    print "fixing sign of constr_eigenvectors"
#    constr_eigenvectors[:,:] = constr_eigenvectors[:,:] * signs #* -1
    
    constr_eigenvalues = numpy.random.uniform(0.0, R*1.0/Q, size=num_labels+1)         
    constr_eigenvalues.sort()
    constr_eigenvalues = constr_eigenvalues[-1::-1]+0.0 #Eigenvalues are decreasing
#    constr_eigenvalues[0] = R*1.0/Q


    M = construct_M_from_eigenvectors(constr_eigenvectors, constr_eigenvalues)   
    print "M="
    print M
    Gamma = construct_Gamma_from_M(M, node_weights)
    print "Gamma="
    print Gamma

    labels = numpy.dot(numpy.diag(node_weights**-0.5), constr_eigenvectors[1:,:].T)
    #Normalizing amplitude of labels
    label_variances = numpy.diag(numpy.dot(labels.T,  numpy.dot(numpy.diag(node_weights), labels)))
    print "label_variances=", label_variances
    print "labels.T=", labels.T
    labels = labels * (Q/label_variances)**0.5
    label_variances = numpy.dot(labels.T,  numpy.dot(numpy.diag(node_weights), labels)) 
    print "label_variances2=", label_variances
    print "labels2.T=", labels.T

    print "(Before handling negative weights) Gamma.min()=", Gamma.min(), "Gamma.max()=", Gamma.max()
    if handling_negative_weights == "Truncate":
        print "Truncating negative edge weights"
        Gamma[Gamma<0]=0.0
    elif handling_negative_weights == "Offset": #here offset should be improved to keep magnitude!
        print "Offseting negative edge weights"
        w_min = Gamma.min()
        if w_min < 0:
            Gamma = (Gamma - w_min)/(1-num_samples*w_min)
    elif handling_negative_weights == "Preserve":
        print "Preserving negative weights"
    elif handling_negative_weights == "Smart_Fix":
        print "Fixing negative weights in a smart way"
        MM = -1*numpy.dot(numpy.dot(numpy.diag(1.0/node_weights), Gamma), numpy.diag(1.0/node_weights))
        c = MM.max()
        node_weights_2D = numpy.reshape(node_weights, (num_samples,1))
        Gamma_prime = 1/(1+c*(Q**2)/R) * (Gamma + c * numpy.dot(node_weights_2D, node_weights_2D.T))
        Gamma = Gamma_prime + 0.0
    else:
        ex = "handling_negative_weights method unknown:", handling_negative_weights
        raise Exception(ex)
    print "(After handling negative weights) Gamma.min()=", Gamma.min(), "Gamma.max()=", Gamma.max()

    Gamma2 = Gamma + 0.0
    if add_weight_noise:
        weight_noise = weight_noise_amplitude * numpy.random.normal(size=(num_samples,num_samples))
        weight_noise = weight_noise + weight_noise.T
        Gamma2 + weight_noise
##    W.reshape((num_samples,num_samples))
##    if add_weight_noise:
##        weight_noise = weight_noise_amplitude * numpy.random.normal(size=(num_samples,num_samples))
##        weight_noise = weight_noise + weight_noise.T
##        W2 = W2+weight_noise
##    print "W2=", W2
##    T = W2 + W2.T
##    N = -2.0 / R * T
    evals, evects = LA.eig(M)
    evects = evects # *(num_samples-1)/6
    signs = numpy.sign(evects[0,:])
    evects[:,:] = evects[:,:]*signs*-1
    
    print "M.eigenvalues:", evals 
    print "M.eigenvectors.T:"
    print evects.T
    evects_means = evects.mean(axis=0)

##    if plot_eigenvectors:
##        f1 = plt.figure()
##        plt.suptitle("Eigenvectors of  M")
##        plt.plot(range(num_samples), labels/((labels**2).sum())**0.5, "ko-")
##        plt.plot(range(num_samples), evects[:, 0:3], "-") 
##        plt.plot(range(num_samples), evects[:, dim-1], "-") 
##
##        plt.legend(["label/||label||", "Eigenvector 1, EV=%f, Lam1=%f, mean=%f"%(evals[0], lll1[0], evects_means[0]), "Eigenvector 2, EV=%f, Lam1=%f, mean=%f"%(evals[1], lll1[1], evects_means[1]), 
##                    "Eigenvector 3, EV=%f, Lam1=%f, mean=%f"%(evals[2], lll1[2], evects_means[2]), "Eigenvector %d, EV=%f, Lam1=%f, mean=%f"%(dim, evals[dim-1], lll1[dim-1], evects_means[dim-1])], loc=4)
##    
##    
##    if lagrange_multipliers:
##        print "lambda1=", L[0], "lambda2=", L[1]

    edge_weights={}
    for i in range(num_samples):
        for j in range(num_samples):
            edge_weights[(i,j)] = Gamma2[i,j]
        
    sfa_node = mdp.nodes.SFANode()
    sfa_node.train(samples, train_mode = "graph", block_size = 1, node_weights=node_weights, edge_weights=edge_weights)
    sl = sfa_node.execute(samples)
    signs = numpy.sign(sl[0,:])
    print "fixing sign of extracted features"
    sl[:,:] = sl[:,:] * signs * -1

    delta = delta_sfa_graph(sl[:,0], node_weights=node_weights, edge_weights=edge_weights)
    delta_labels = delta_sfa_graph(labels, node_weights=node_weights, edge_weights=edge_weights)
    #print "Computed delta of slowest feature=", delta, "corresponds to lambda1=", -1 * delta / (Q-1)
    print "Computed delta of labels=", delta_labels
    #print "corresponds to lambda1=", -1 * delta_labels / (Q-1)
    print "sl[:,0]=", sl[:,0]
    print "constr_eigenvectors[0:2]=", constr_eigenvectors[0:2]
    print "labels[:,0:num_labels].T=",labels[:,0:num_labels].T
    error_sfa = labels[:,0]- sl[:,0]
    rmse_sfa = ((error_sfa**2).sum()/num_samples)**0.5
    print "rmse_sfa=", rmse_sfa
    
    print sl.shape
    labels_lr = labels[:,first_label_lr:first_label_lr+num_labels_lr]
#    labels.reshape((num_samples,num_labels))

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
    plt.plot(range(num_samples), sl[:,0:last_slow_signal], "v-")
    plt.plot(range(num_samples), labels, "o-")
##    if dg_enable_2_labels:
##        plt.plot(range(num_samples), labels2, "k+-") 
    
    plt.plot(range(num_samples), y_lr[1], "*-")
    plt.plot(range(num_samples), y_lr[2], "*-")
    plt.plot(range(num_samples), y_lr[dim], "*-")
    plt.xlabel("Sample")

##    if dg_enable_2_labels:
##        if last_slow_signal >= 3:
##            plt.legend(["slow feature 1","slow feature 2","slow feature 3","label1", "label2", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
##        elif last_slow_signal == 2:
##                plt.legend(["slow feature 1","slow feature 2","label1", "label2", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)
##        elif last_slow_signal == 1:
##                plt.legend(["slow feature 1", "label1", "label2", "LR 1 ft", "LR 2 ft", "LR %d ft"%dim], loc=4)

    legend = []
    for i in range(last_slow_signal):
        legend.append("slow feature %d"%(i+1))
    for i in range(num_labels):
        legend.append("label %d"%(i+1))
    for i in range(num_labels_lr):
        legend.extend(["LR[%d] 1 ft"%(i+1), "LR[%d] 2 ft"%(i+1), "LR[%d] %d ft"%(i+1,dim)])

    plt.legend(legend, loc=4)
    
#Code that displays free responses as predicted from theory
#training_graphs = ["Serial", "Mixed"]
training_graphs = ["Serial", "Mixed"]
for training_graph in training_graphs:
    print "Processing training graph for free responses:", training_graph

    Ng = 10
    L = 50
    num_samples = Ng * L
    num_eigenvectors_show = 4#None #25 
    first_eigenvector_show = 0 #Eigenvector 0 is the weighted zero mean constraint
    legends = False or True
    show_eigenvalues = False 
    show_scaled_eigenvectors = False
    show_center_line = True
    
    print "Ng=%d, L=%d, num_samples=%d,"%(Ng,L,num_samples) + " num_eigenvectors_show=", num_eigenvectors_show
    if training_graph == "Serial":
        #Serious warning: either 1.0 & 2.0 (as in article now), or 0.5 & 1.0 (nicer figures)
        node_weights = numpy.ones(num_samples)*0.5  
        #MEGAWARNING!!!!
        #node_weights = numpy.ones(num_samples)*1.0 #(Wrong code)
        node_weights[Ng:-Ng]=1.0
        #print node_weights
        #quit()
        
        Gamma = numpy.zeros((num_samples,num_samples)) 
        for l in range(L-1):
            for i1 in range(Ng):
                for i2 in range(Ng):
                    Gamma[l*Ng+i1, (l+1)*Ng+i2]=1.0
                    Gamma[(l+1)*Ng+i2, l*Ng+i1]=1.0    
    elif training_graph == "Mixed":
        node_weights = numpy.ones(num_samples)*1.0  
        Gamma = numpy.zeros((num_samples,num_samples)) 
        #Serial graph edges:
        for l in range(L-1):
            for i1 in range(Ng):
                for i2 in range(Ng):
                    Gamma[l*Ng+i1, (l+1)*Ng+i2]=1.0    
                    Gamma[(l+1)*Ng+i2, l*Ng+i1]=1.0    
        #Complete graph edges:
        for l in range(L):
            if l in [0,L-1]:
                edge_weight=2.0
            else:
                edge_weight=1.0
            for i1 in range(Ng):
                for i2 in range(Ng):
                    Gamma[l*Ng+i1, l*Ng+i2]= edge_weight  
                    Gamma[l*Ng+i2, l*Ng+i1]= edge_weight          
    else:
        er= "Graph unsupported:"+training_graph
        raise Exception(er)
    
    print "node_weights=", node_weights
    print "Gamma=", Gamma

#MEGAWARNING, should be:
    M = construct_M_from_Gamma(Gamma, node_weights)
#    M = construct_Gamma_from_M(Gamma, node_weights) #(WRONG code)

    evals, evects = LA.eig(M)
    signs = numpy.sign(evects[0,:])
    evects[:,:] = evects[:,:]*signs*-1
    order = numpy.argsort(evals)[-1::-1]
    evects = evects[:,order]
    evals = evals[order]
    norm_evects = numpy.sqrt((evects**2).sum(axis=0))
    Q = node_weights.sum()
    evects[:,:] = evects[:,:]*numpy.sqrt(Q) / norm_evects


    positive_eigenvalues = (evals > 0.0001).sum()
    print "TGraph:"+training_graph+" has %d clearly positive eigenvalues"%positive_eigenvalues
    if num_eigenvectors_show == None:
        num_eigenvectors_show = positive_eigenvalues
        
    if show_eigenvalues:
        f1 = plt.figure()
        plt.suptitle("Eigenvalues of M = Diag(V^-1/2) Gamma Diag(V^-1/2). TGraph:"+training_graph)
        plt.plot(range(num_samples), evals, "*-") 
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        
    show_eigenvectors_instead_of_free_responses = False
    f2 = plt.figure()
    if legends:
        if show_eigenvectors_instead_of_free_responses:
            plt.suptitle("Eigenvectors of M = Diag(V^-1/2) Gamma Diag(V^-1/2) TGraph:"+training_graph)
            signals = evects
        else:
            plt.suptitle("Free responses corresponding to M = Diag(V^-1/2) Gamma Diag(V^-1/2) TGraph:"+training_graph)
            DV_m12 = numpy.diag(node_weights**(-0.5))
            signals = numpy.dot(DV_m12, evects)

    if show_center_line:
        for i in range(num_eigenvectors_show):
            eigenvector_index = first_eigenvector_show+num_eigenvectors_show-i-1      
            sample_indices = numpy.arange(L)*Ng+ Ng/2
            sample_x_positions = numpy.arange(L)*Ng+ (Ng-1.0)/2
            plt.plot(sample_x_positions, signals[sample_indices,eigenvector_index], "k-", linewidth=1.5) 

#    styles = ["v-", "*--", "o:", "s-."]
    styles = ["v", "*", "o", "^"]
    for i in range(num_eigenvectors_show):
        eigenvector_index = first_eigenvector_show+num_eigenvectors_show-i-1      
        if i<4:
            plt.plot(range(num_samples), signals[:,eigenvector_index], styles[i], linewidth=1.5) 
        else:
            plt.plot(range(num_samples), signals[:,eigenvector_index], "x", linewidth=1.5) 
    plt.xlim(0-0.5, L*Ng-1+0.5)
    plt.ylim(-1.6, 1.6)


    legend = []
    for i in range(num_eigenvectors_show):
        if show_eigenvectors_instead_of_free_responses:
            legend.extend(["Eigenvector[%d]"%(i+1)])
        else:
            legend.extend(["Free_response[%d]"%(i)])
    if legends:
        plt.legend(legend)

    if show_scaled_eigenvectors:
        f3 = plt.figure()
        plt.suptitle("Scaled Eigenvectors of M = Diag(V^-1/2) Gamma Diag(V^-1/2) TGraph:"+training_graph)
        scaled_evects = evects * evals
        plt.plot(range(num_samples), scaled_evects[:,first_eigenvector_show:first_eigenvector_show+num_eigenvectors_show], "x-") 
    
        legend = []
        for i in range(num_eigenvectors_show):
            legend.extend(["ScaledEigenvector[%d]"%(i+1)])
        plt.legend(legend)


plot_free_responses_standard_SFA = True and False
if plot_free_responses_standard_SFA:
    training_graph = "Standard"
    print "Processing training graph:", training_graph

    f1 = plt.figure()
    num_samples = 16
    num_eigenvectors_show = 16 

    Q = num_samples
    R = num_samples -1

    lambda_1 = 0.00 # -1.9 / (num_samples * (num_samples-1)) # suggested value: (-2, -2N, -4N+2)/(N * (N-1))
    I = numpy.diag(numpy.ones(num_samples))
    
    X = numpy.zeros((num_samples,num_samples))
    X[:,0] = 1
    X[:,num_samples-1] = 1
    #X = X.T #* 0.0 + 0.0 #WARNING!!!!
    
    Gamma = 0.5 * (numpy.diag(numpy.ones(num_samples-1), 1) +numpy.diag(numpy.ones(num_samples-1), -1))
    
    M = (4.0/R) * Gamma + (2.0 / (Q*R)) * X
    
    Z = M  - (4.0/Q + 2*lambda_1)*I

    print "M=", M
    print "Z=", Z
    
    evals, evects = LA.eig(M)
    signs = numpy.sign(evects[0,:])
    evects[:,:] = evects[:,:]*signs*-1
    order = numpy.argsort(evals)[-1::-1]
    evects = evects[:,order]
    evals = evals[order]
    norm_evects = numpy.sqrt((evects**2).sum(axis=0))
    evects[:,:] = evects[:,:]*numpy.sqrt(Q) / norm_evects
    norm_evects = numpy.sqrt((evects**2).sum(axis=0))
    print "norm_evects after normalization", norm_evects
    
    print "evects[:,0:2]", evects[:,0:2]
    print "evals[0:2]", evals[0:2]
    
    y2 = numpy.dot(M, evects[:,0:2])
    norm_y2 = numpy.sqrt((y2**2).sum(axis=0))
    y2 = y2*numpy.sqrt(Q) / norm_y2 
    print "norm_evects after normalization", norm_evects
    print "y2=norm(numpy.dot(M, evects[:,0:2]))", y2
        
    evects_mean = evects.mean(axis=0)
    print "evects_mean:", evects_mean
    print "abs(evects_mean).argmin = ", numpy.abs(evects_mean).argmin()
    
    print "(original) desired_l1=", lambda_1
    real_l1 = 1.0/(2*Q) * (4.0 / R * numpy.dot(numpy.dot(evects.T, Gamma), evects)-4)
    real_l1 = numpy.diag(real_l1)
    print "real_l1=", real_l1
    
    plt.suptitle("Eigenvalues of Z = 4/R * Gamma + 2 / (Q*R) * X -(4.0/Q + 2*lambda_1)*I. lambda1=%f. TGraph:"%lambda_1+training_graph)
    plt.plot(range(num_samples), evals, "*-") 
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
        
    f2 = plt.figure()
    plt.suptitle("Eigenvectors of Z = 4/R * Gamma + 2 / (Q*R) * X-(4.0/Q + 2*lambda_1)*I. lambda1=%f.TGraph:"%lambda_1+training_graph)
    plt.plot(range(num_samples), evects[:,0:num_eigenvectors_show], "x-") 

    legend = []
    for i in range(num_eigenvectors_show):
        legend.extend(["Eigenvector[%d]"%(i+1)])
    plt.legend(legend)

    f3 = plt.figure()
    plt.suptitle("Scaled Eigenvectors of Z = 4/R * Gamma + 2 / (Q*R) * X-(4.0/Q + 2*lambda_1)*I. lambda1=%f.TGraph:"%lambda_1+training_graph)
    scaled_evects = evects * evals
    plt.plot(range(num_samples), scaled_evects[:,0:num_eigenvectors_show], "x-") 

    legend = []
    for i in range(num_eigenvectors_show):
        legend.extend(["ScaledEigenvector[%d]"%(i+1)])
    plt.legend(legend)


plot_free_responses_GSFA = True and False
if plot_free_responses_GSFA:
    L = 17
    Ng = 3
    num_samples = L * Ng
    dim = num_samples - 1
    last_slow_signal = 3
    show_center_line = True
    
    samples = numpy.random.normal(size=(num_samples,dim))
    samples = samples-samples.mean(axis=0)
    samples = samples / samples.std(axis=0)

    #TODO:Change sequence to serial 
    training_graphs = ["sequence", "mixed"]    
    for training_graph in training_graphs:
        sfa_node = mdp.nodes.SFANode()
        sfa_node.train(samples, train_mode = training_graph, block_size = Ng)
        sl = sfa_node.execute(samples)
        signs = numpy.sign(sl[0,:])
        sl[:,:] = sl[:,:] * signs * -1

        #TODO:generate function that creates node_weights and edge_weights of training graphs
        #delta = delta_sfa_graph(sl[:,0], node_weights=node_weights, edge_weights=edge_weights)
        #print "Computed delta of slowest feature=", delta
        print "sl[:,0]=", sl[:,0]
        print sl.shape
        f0 = plt.figure()
        plt.suptitle("G-SFA with %s training graph learning from %d random samples of dimension %d (overfitted regime)"%(training_graph, num_samples, dim))
        plt.plot(range(num_samples), sl[:,0:last_slow_signal], "v-")
        plt.xlabel("Sample")
            
        if show_center_line:
            for i in range(last_slow_signal):    
                sample_indices = numpy.arange(L)*Ng+ Ng/2
                sample_x_positions = numpy.arange(L)*Ng+ (Ng-1.0)/2
                plt.plot(sample_x_positions, sl[sample_indices,i], "k-", linewidth=1.5)

        plt.xlim(0-0.5, L*Ng-1+0.5)
        plt.ylim(-1.6, 1.6)    
##    if plot_eigenvectors_regular_sfa:
##        node_weights_reg = numpy.ones(num_samples)*1.0
##        Q_reg = node_weights_reg.sum()
##        Gamma_reg = 1.0*(numpy.diag(numpy.ones(num_samples-1), 1) + numpy.diag(numpy.ones(num_samples-1), -1))
##        R_reg = Gamma_reg.sum()
##        ones_2 = numpy.ones((num_samples,num_samples))
##        A_reg = 4/(R_reg*Q_reg) * numpy.dot(ones_2, Gamma_reg) - 4/R_reg * Gamma_reg
##
##        evals_reg, evects_reg = LA.eig(A_reg)
##        signs_reg = numpy.sign(evects_reg[0,:])
##        evects_reg[:,:] = evects_reg[:,:]*signs_reg*-1
##        order_reg = numpy.argsort(evals_reg)[-1::-1]
##        evects_reg = evects_reg[:,order_reg]
##        evals_reg = evals_reg[order_reg]
##        
##        f4 = plt.figure()
##        plt.suptitle("Eigenvalues of M_reg = Diag(V^-1/2) Gamma_reg Diag(V^-1/2); V = ones")
##        plt.plot(range(num_samples), evals_reg, "*-") 
##        plt.xlabel("Index")
##        plt.ylabel("Eigenvalue")
##        
##        f5 = plt.figure()
##        plt.suptitle("Eigenvectors of M_reg = Diag(V^-1/2) Gamma_reg Diag(V^-1/2); V = ones")
##        plt.plot(range(num_samples), evects_reg[:,0:num_eigenvectors_show], "x-") 
##
##        legend = []
##        for i in range(num_eigenvectors_show):
##            legend.extend(["Eigenvector[%d]"%(i+1)])
##        plt.legend(legend)
##
##        f6 = plt.figure()
##        plt.suptitle("Scaled Eigenvectors of M_reg = Diag(V^-1/2) Gamma_reg Diag(V^-1/2); V = ones")
##        scaled_evects_reg = evects_reg * evals_reg
##        plt.plot(range(num_samples), scaled_evects_reg[:,0:num_eigenvectors_show], "x-") 
##
##        legend = []
##        for i in range(num_eigenvectors_show):
##            legend.extend(["ScaledEigenvector[%d]"%(i+1)])
##        plt.legend(legend)


##        if dg_enable_2_labels:
##            plt.legend(["label1/||label1||", "label2/||label2||", "Eigenvector 1, EV=%f, Lam1=%f, mean=%f"%(evals[0], lll1[0], evects_means[0]), "Eigenvector 2, EV=%f, Lam1=%f, mean=%f"%(evals[1], lll1[1], evects_means[1]), 
##                    "Eigenvector 3, EV=%f, Lam1=%f, mean=%f"%(evals[2], lll1[2], evects_means[2]), "Eigenvector %d, EV=%f, Lam1=%f, mean=%f"%(dim, evals[dim-1], lll1[dim-1], evects_means[dim-1])], loc=4)
##        if 0==1:
##            pass
##        else:
##            plt.legend(["label/||label||", "Eigenvector 1, EV=%f, Lam1=%f, mean=%f"%(evals[0], lll1[0], evects_means[0]), "Eigenvector 2, EV=%f, Lam1=%f, mean=%f"%(evals[1], lll1[1], evects_means[1]), 
##                    "Eigenvector 3, EV=%f, Lam1=%f, mean=%f"%(evals[2], lll1[2], evects_means[2]), "Eigenvector %d, EV=%f, Lam1=%f, mean=%f"%(dim, evals[dim-1], lll1[dim-1], evects_means[dim-1])], loc=4)
##  

plt.show()
quit()
    
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


