#Basic Experiments for GraphBased SFA Training
#By Alberto Escalante. Alberto.Escalante@ini.ruhr-uni-bochum.de First Version 18 March 2010
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott
import numpy
import scipy
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
sys.path.append("/home/escalafl/work3/cuicuilco/src")
import patch_mdp
import sfa_libs 


#First Experiment: Extract slow signals from noise using regular training graph, but with arbitrary edge weights
#Here we approximate different functions by modifying the edge weights following an heuristic
#Signals are extracted from noise, which is possible only because of extreme overfitting
activate_first_experiment = False

if activate_first_experiment:
    num_steps = 500
    num_slow_signals = 10
    
    std_noise = .15
    
    t = numpy.linspace(0, numpy.pi, num_steps)
    
    dim = 499
    noise = numpy.random.normal(size = (num_steps, dim))
    #noise[:,0] += numpy.arange(num_steps)
    noise = noise-noise.mean(axis=0)
    noise = noise / noise.std(axis=0)
    
    #node_weights = numpy.sin(t)
    node_weights = numpy.ones(num_steps)
    
    weight_shapes = ["normal", "sine", "sine**2", "normal_sine", "cos+1", "sin/(t+0.01)"] # , "sin/t**2", "sine+0.5", "normal_sine2", "normal_sine2_rep", sin/(t+0.01)
    
    max_amplitude_sfa = 3.0
    for plt_num, weight_shape in enumerate(weight_shapes):
    #weight_shape = "normal_sine2"
    
        edge_weights={}
        
        for i in range(0, num_steps-1):
            if weight_shape == "normal":
                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = 1.0 
            elif weight_shape == "sine":
                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])+0.0005
            elif weight_shape == "sine+0.5":
                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])+0.5
            elif weight_shape == "normal_sine":
                if i >= num_steps/2:
                    edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])+0.0005
                else:
                    edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = 1.0
            elif weight_shape == "cos+1":
                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.cos(t[i])+1.0005            
            elif weight_shape == "sine**2":
                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])**2+0.0005
            elif weight_shape == "normal_sine**2":
                if i >= num_steps/2:
                    edge_weights[(i+1,i)] =  edge_weights[(i,i+1)] = numpy.sin(t[i])**2+0.0005
                else:
                    edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = 1.0
            elif weight_shape == "normal_sine2_rep":
                if i >= num_steps/2:
                    edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])**2+0.0005
                    edge_weights[(i+1,i-1)] = edge_weights[(i-1,i+1)] = numpy.sin(t[i])**2+0.0005
                else:
                    edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = 1.0
                    edge_weights[(i+2,i)] = edge_weights[(i,i+2)] = 1.0                
            elif weight_shape == "sin/t**2":
                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])/ (2* (i * numpy.pi/num_steps)+0.01)**2 + 0.00025
            elif weight_shape == "sin/(t+0.01)":
                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])/ (t[i] + 0.01) + 0.000025
            elif weight_shape == "sin**0.5":
                edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = numpy.sin(t[i])**0.5 + 0.00025
    
        sfa_node = mdp.nodes.SFANode(output_dim=10)
        sfa_node.train(noise, train_mode = "graph", block_size = 1, node_weights=node_weights, edge_weights=edge_weights)
        
        sl = sfa_node.execute(noise)
        print sl.shape
        sl_signs = numpy.sign(sl[0,:])
        sl = sl * sl_signs * -1
        
        ax = plt.subplot(2,3,plt_num+1)
        plt.plot(t, sl[:,3], "y.")
        plt.plot(t, sl[:,2], "g.")
        plt.plot(t, sl[:,1], "r.")
        plt.plot(t, sl[:,0], "b.")
        plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
        plt.title(weight_shape+", D[0:3]=[%f,%f,%f]"%(sfa_node.d[0], sfa_node.d[1], sfa_node.d[2]))
    #    plt.plot(t, ap1, "r.")
    #    if enable_fit:
    #        plt.plot(t, ap2, "m.")
    #    plt.plot(t, sl2, "g.")
    #    if enable_fit:    
    #        plt.legend(["x(t)=cos(t)+%f*n(t)"%std_noise, "white(|x|^%f)"%exponent, "white(fit(|x|))", "cos(2t)"], loc=4)
    #    else:
    #        plt.legend(["x(t)=cos(t)+%f*n(t)"%std_noise, "white(|x|^%f)"%exponent, "cos(2t)"], loc=4)
    #
    #    error = sl2-ap1
    #    error2 = sl2-ap2
    #    if enable_fit:
    #        plt.xlabel("approximation error white(|x|^%f) and white(fit(|x|)) vs cos(2t): %f and %f"%(exponent, error.std(), error2.std()))
    #    else:
    #        plt.xlabel("approximation error white(|x|^%f) vs cos(2t): %f"%(exponent, error.std()))        

activate_second_experiment=True

if activate_second_experiment:
    #Second Experiment: Extract slow signals from noise using regular training graph, but with arbitrary NODE weights
    #Here we approximate different functions by modifying the NODE weights following an heuristic
    #Signals are extracted from noise, which is possible only because of extreme overfitting
    num_steps = 500
    num_slow_signals = 10
    
    std_noise = .15  
    t = numpy.linspace(0, numpy.pi, num_steps)    
    dim = 499
    noise = numpy.random.normal(size = (num_steps, dim))
    #noise[:,0] += numpy.arange(num_steps)
    noise = noise-noise.mean(axis=0)
    noise = noise / noise.std(axis=0)
    
    #node_weights = numpy.sin(t)
    node_weights = numpy.ones(num_steps)

    edge_weights = {}
    for i in range(0, num_steps-1):
        edge_weights[(i+1,i)] = edge_weights[(i,i+1)] = 1.0
                
    goal = 1.0*t**2+t
    goal = goal-goal.mean(axis=0)
    goal = goal / goal.std(axis=0)
    dgoal = goal[1:]-goal[0:-1]
    dgoal = numpy.concatenate((dgoal, [dgoal[-1]]))

    weight_shapes = ["normal2", "test7", "test8", "test9", "test10","test6"] 
    # , "sin/t**2", "sine+0.5", "normal_sine2", "normal_sine2_rep", sin/(t+0.01)
    
    max_amplitude_sfa = 3.0
    for plt_num, weight_shape in enumerate(weight_shapes):
    #weight_shape = "normal_sine2"
        if weight_shape == "normal":
            node_weights = numpy.ones(num_steps)
        if weight_shape == "normal2":
            node_weights = numpy.ones(num_steps)*2
        elif weight_shape == "(|c(i)|+k)/(|l(i)|+k)":
            node_weights = (numpy.abs(numpy.cos(t))+0.001) / (numpy.abs(goal)+0.001)
        elif weight_shape == "test1":
            node_weights = 1.0 / (numpy.abs(goal)+0.001)
        elif weight_shape == "test2":
            node_weights = 1.0 / ((numpy.abs(goal)**2+0.001))
        elif weight_shape == "test3":
            node_weights = 1.0 / ((numpy.abs(goal)**0.5+0.001))
        elif weight_shape == "test4":
            node_weights = (numpy.abs(numpy.cos(t))+0.0001) / ((numpy.abs(goal)**0.5+0.0001))
        elif weight_shape == "test5":
            node_weights = (numpy.abs(numpy.cos(t))**2+0.0001) / ((numpy.abs(goal)**0.5+0.0001))
        elif weight_shape == "test6":
            node_weights = 1.0 / ((numpy.abs(goal)**0.5+0.0001)*(numpy.abs(numpy.sin(t))+0.001))
        elif weight_shape == "test7":
            node_weights = 1.0 / (numpy.abs(numpy.sin(t))+0.001)
        elif weight_shape == "test8":
            node_weights = 1.0 / (numpy.abs(numpy.sin(t))+0.001)
        elif weight_shape == "test9":
            node_weights = 1.0 / (numpy.abs(numpy.sin(t))**2+0.001)
        elif weight_shape == "test10":
            node_weights = (numpy.abs(numpy.sin(t))**0.01+0.001)

        sfa_node = mdp.nodes.SFANode(output_dim=10)
        sfa_node.train(noise, train_mode = "graph", block_size = 1, node_weights=node_weights, edge_weights=edge_weights)
        
        sl = sfa_node.execute(noise)
        print sl.shape
        sl_signs = numpy.sign(sl[0,:])
        sl = sl * sl_signs * -1
        
        ax = plt.subplot(2,3,plt_num+1)
        plt.plot(t, sl[:,3], "y.")
        plt.plot(t, sl[:,2], "g.")
        plt.plot(t, sl[:,1], "r.")
        plt.plot(t, sl[:,0], "b.")
        plt.plot(t, goal, "k.")
        plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
        plt.title(weight_shape+", D[0:3]=[%f,%f,%f]"%(sfa_node.d[0], sfa_node.d[1], sfa_node.d[2]))

plt.show()

plt.show()

