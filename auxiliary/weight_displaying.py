#! /usr/bin/env python

#General purpose hierarchical network for data processing
#Changes: New more modularized version, with new three structure and node/signal cache (wip) 
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 9 Dec 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import Image
import mdp
import more_nodes
import patch_mdp

import object_cache as cache
import os, sys
import glob
import random
import sfa_libs
from sfa_libs import (scale_to, distance_squared_Euclidean, str3, wider_1Darray, ndarray_to_string, cutoff)
 
import SystemParameters
from imageLoader import *
import classifiers_regressions as classifiers
import network_builder
import time
from matplotlib.ticker import MultipleLocator
import copy
import string
from nonlinear_expansion import *


#TODO: ADD Node within Layer selection mechanism
#Think about trainable GeneralExpansion, why input_dim not stored?
#Think about GeneralExpansion in Layer, why not in CloneLayer

#from mdp import numx
#sys.path.append("/home/escalafl/workspace/hiphi/src/hiphi/utils")
#import misc
import object_cache

normalization_strings = {0:"None", 1:"SQRT(ABS(X))", 2:"SGN*SQRT(ABS(X))"}
    
def mask_pairwise_adjacent_expansion(input_dim, adj, reflexive=True):
    if reflexive is True:
        k=0
    else:
        k=1
#   number of variables, to which the first variable is paired/connected
    mix = adj-k

    v1 = numpy.zeros(mix * (input_dim-adj+1), dtype='int')
    for i in range(input_dim-adj+1):
        v1[i*mix:(i+1)*mix] = i
    v2 = numpy.zeros(mix * (input_dim-adj+1), dtype='int')
    for i in range(input_dim-adj+1):
        v2[i*mix:(i+1)*mix] = range(i+k,i+adj)

    mask = numpy.zeros((input_dim,input_dim), dtype='int' )
    mask[v1, v2] = 1
    return mask > 0.5
#    return zip(v1,v2)
  
def mask_pairwise_expansion(input_dim, reflexive=True):
    if reflexive==True:
        k=0
    else:
        k=1
    mask = numpy.triu(numpy.ones((input_dim,input_dim)), k) > 0.5
    return mask

#m = mask_pairwise_adjacent_expansion(4, 2, reflexive=True)
#print m
#
#m2 = mask_pairwise_expansion(4, reflexive=True)
#print m2          

list_pointwise_ex = [identity, unsigned_11expo, signed_11expo, unsigned_15expo, signed_15expo, unsigned_08expo, 
                     signed_06expo, unsigned_06expo, signed_08expo, signed_sqrt, unsigned_sqrt, signed_sqr]

list_adj_mix_ex = [pair_prod_adj1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex, 
                   pair_prod_adj4_ex, pair_prod_adj5_ex, pair_prod_adj6_ex,
                   pair_prod_mix1_ex, pair_prod_mix2_ex, pair_prod_mix3_ex]

list_adj_ex = [pair_prod_adj1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex,pair_prod_adj4_ex, pair_prod_adj5_ex, pair_prod_adj6_ex]
list_mix_ex = [pair_prod_mix1_ex, pair_prod_mix2_ex, pair_prod_mix3_ex]

def generate_weight_image(w, input_dim, exp_funcs):
    total_disp_cols = 0
    num_cols = []
    num_points = []

    if input_dim == None:
        print "Fixing input_dim=None in generate_weight_image. w.shape is:", w.shape
        input_dim = w.shape[0]
 
    for func in exp_funcs:         
        if func in list_pointwise_ex: #or..... 08
            total_disp_cols += 1
            num_cols.append(1)
            num_points.append(input_dim)
        else:
            total_disp_cols += input_dim
            num_cols.append(input_dim)
            num_points.append(len(func(numpy.zeros((1,input_dim)))[0]))
    
    verbose=False
    if verbose:
        print "input dimension is=", input_dim,
        print "columns for display=", total_disp_cols, 
        print "columns used:", num_cols,
        print "points taken:", num_points,
        print "exp_funcs:", exp_funcs
    
#    total_weights = 0
#    for p in num_points:
#        total_weights += p 
#    w = numpy.ones(total_weights)
    

    current_weight = 0
    current_col = 0
    out_matrix = numpy.zeros((input_dim, total_disp_cols))
    for func in exp_funcs:
        if func in list_pointwise_ex: #or.....
            num_w = input_dim
            out_matrix[:, current_col] = w[current_weight:current_weight+input_dim]
            current_col += 1
            current_weight += num_w
        elif func == pair_prod_ex:
            mask = mask_pairwise_expansion(input_dim, reflexive=True)
            num_w = len(func(numpy.zeros((1,input_dim)))[0])
            out_matrix[:, current_col:current_col+input_dim][mask] = w[current_weight:current_weight+num_w]
            current_col += input_dim
            current_weight += num_w
        elif func in list_adj_mix_ex:
            if func in [pair_prod_adj1_ex]:
                adj=1
            elif func in [pair_prod_adj2_ex, pair_prod_mix1_ex]:
                adj=2
            elif func in [pair_prod_adj3_ex, pair_prod_mix2_ex]:
                adj=3
            elif func in [pair_prod_adj4_ex, pair_prod_mix3_ex]:
                adj=4
            elif func in [pair_prod_adj5_ex]:
                adj=5
            elif func in [pair_prod_adj6_ex]:
                adj=6
            else:
                print "Unknown function in weight displaying A"
    
            if func in list_adj_ex:
                ref = True
            elif func in list_mix_ex:
                ref = False
            else:
                print "Unknown function in weight displaying B"
    
            mask = mask_pairwise_adjacent_expansion(input_dim, adj, reflexive=ref)
            if verbose:
                print "mask used:", mask
            num_w = len(func(numpy.zeros((1,input_dim)))[0])
            if verbose:
                print "num_w=", num_w , current_col, input_dim, mask, current_weight, num_w
            out_matrix[:, current_col:current_col+input_dim][mask] = w[current_weight:current_weight+num_w]
            current_col += input_dim
            current_weight += num_w
        else:
            print "Unknown function in weight displaying C"
    return out_matrix

def redisplay_weights(num_layers, num_nodes, num_weights, out_matrices, all_exp_funcs, wnodes, rec_fields, selected_layer, selected_node, selected_weight, normalization_type, a11, a12, a13, a14, a15, a21):         
    print "selected_layer = " + str(selected_layer),
    print "selected_node = " + str(selected_node),
    print "selected_weight = " + str(selected_weight),
    print "normalization_type = " + str(normalization_type) + normalization_strings[normalization_type]
            
    layer_disp = numpy.linspace(0.0, 0.5, num_layers)
    layer_disp[selected_layer] = 1.0
    layer_disp = layer_disp.reshape((num_layers,1))
    a11.imshow(layer_disp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
    a11.set_xlabel("Act. Layer=%d"%selected_layer)

    node_disp = numpy.linspace(0.0, 0.5, num_nodes[selected_layer])
    node_disp[selected_node] = 1.0
    node_disp = node_disp.reshape((num_nodes[selected_layer],1))
    a12.imshow(node_disp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
    a12.set_xlabel("Act. Node=%d"%selected_node)

    rec_field_disp = rec_fields[(selected_layer, selected_node * num_weights[selected_layer] + selected_weight)]
 
    weight_num_disp = numpy.linspace(0.0, 0.5, num_weights[selected_layer])
    weight_num_disp[selected_weight] = 1.0
    weight_num_disp = weight_num_disp.reshape((num_weights[selected_layer],1))
    a13.imshow(weight_num_disp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)  
    a13.set_xlabel("Act. Weight=%d"%selected_weight )

    if rec_fields != None:
        print "Showing receptive field..."
        rec_field_disp = rec_fields[(selected_layer, selected_node*num_weights[selected_layer]+selected_weight)]
        print "rec_field_disp.shape =", rec_field_disp.shape
        print "rec_field_disp=", rec_field_disp
#        rec_field_disp = numpy.random.normal(size=(64,64))
        a21.imshow(rec_field_disp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
#        a21.set_xlabel("Receptive field of Selected Node")
        
    out_matrix = out_matrices[(selected_layer, selected_node, selected_weight)]
    if normalization_type == 0:
        wdisp = out_matrix + 0.0
    elif normalization_type == 1:
        wdisp = numpy.sqrt(numpy.abs(out_matrix))
    else:
        wdisp = signed_sqrt(out_matrix)

    #wdisp = scale_to(wdisp, wdisp.mean(), wdisp.max()-wdisp.min(), 127.5, 255.0, scale_disp, 'tanh')
    pic = a14.imshow(wdisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.jet)
    label = str(wnodes[selected_layer]) + ". Norm: "+ str(normalization_type) + "=" + normalization_strings[normalization_type] + ". Exp_funcs="
    for fun in all_exp_funcs[selected_layer]:
        label += fun.__name__ + ","
    a14.set_xlabel(label)

    plt.colorbar(pic, a15)  
    #plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp)+str3(sfa_libs.comp_eta(sl_seq)[0:5]))


artificial_weights = False
if artificial_weights:
    exp_funcs = [identity, pair_prod_adj1_ex, pair_prod_ex]
    #, pair_prod_adj2_ex, pair_prod_adj3_ex, pair_prod_ex, pair_prod_mix1_ex, pair_prod_mix2_ex]
    #, identity, pair_prod_adj2_ex, identity, pair_prod_adj1_ex]
    num_samples= 1
    input_dim = 10
    x= numpy.random.normal(size=(num_samples, input_dim))
    y = sfa_libs.apply_funcs_to_signal(exp_funcs, x)
    output_dim = len(y[0])
    
    node_output_dim = 5 
    
    num_layers = 8
 
    print "input_dim = ", input_dim, "output_dim =", output_dim
    
    #Number of weights available at each layer
    num_nodes = {}
    num_weights = {}
    
    #output matrices for each layer, node, and weight
    out_matrices = {}   
    all_exp_funcs = {}

    wnodes = {}

    rec_fields = None

    for num_layer in range(num_layers):
        num_nodes[num_layer] = 2
        num_weights[num_layer] = 5
        all_exp_funcs[num_layer] = exp_funcs

        wnodes[num_layer] = "SimulatedNode"

        for num_node in range (num_nodes[num_layer]):
            for num_weight in range(num_weights[num_layer]):
                w = numpy.random.normal(size=output_dim)
                out_matrix = generate_weight_image(w, input_dim, exp_funcs)
                out_matrices[(num_layer, num_node, num_weight)] = out_matrix
                print "out_matrix.shape=", out_matrix.shape
    
else:
    on_lok21 = os.path.lexists("/local2/tmp/escalafl/")
    on_lok09 = os.path.lexists("/local/escalafl/on_lok09")
    on_lok10 = os.path.lexists("/local/escalafl/on_lok10")
    if on_lok21:
        network_load = object_cache.Cache("/local2/tmp/escalafl/Alberto/SavedNetworks", "")
        network_base_dir = "/local2/tmp/escalafl/Alberto/SavedNetworks"
    elif on_lok09 or on_lok10:
        network_load = object_cache.Cache("/local/escalafl/Alberto/SavedNetworks", "")
        network_base_dir = "/local/escalafl/Alberto/SavedNetworks"
    else:
        network_load = object_cache.Cache("/local/tmp/escalafl/Alberto/SavedNetworks", "")
        network_base_dir = "/local/tmp/escalafl/Alberto/SavedNetworks"
    print "network_load =", network_load
    print "network_base_dir =", network_base_dir

    print "Looking for saved Networks..."        
    network_beginning_filename = "Network"  
    network_filenames = cache.find_filenames_beginning_with(network_base_dir, network_beginning_filename, recursion=False, extension=".pckl")
    if len(network_filenames) < 0:
        print "Aborting computation, no networks found"
        quit()

    print "The following Networks were found:"
    for i, network_filename in enumerate(network_filenames):
        network_filename = string.split(network_filename, sep=".")[0] #Remove extension
        print "%d:"%i, network_filename

    selected_network = None
    while selected_network == None:
        selected_network = int(raw_input("Please select a network: "))
        if selected_network < 0:
            selected_network = None
        elif selected_network >=  len(network_filenames):
            selected_network = None
                
    print "Network %d was selected:"%selected_network, network_filenames[selected_network]
    network_filename = string.split(network_filenames[selected_network], sep=".")[0]
    (flow, layers, benchmark, Network) = network_load.load_obj_from_cache(hash_value=None, base_dir = None, base_filename=network_filename , verbose=True)

        #update_cache([flow, layers, benchmark, Network], None, network_base_dir, "Network"+iTrain.name, overwrite=True, use_hash=network_hash, verbose=True)
    layers = None
        
    #Number of nodes with weights available for display
    num_layers=0
    #node description available at each of those layers
    wnodes = {}
    #Number of nodes and weights available at each of those layers
    num_nodes = {}
    num_weights = {}
    #output matrices for each of those nodes and output signal
    out_matrices = {}
    #expansion functions used for each of those nodes and output signal
    all_exp_funcs = {}

    rec_fields = {}
    temporal_rec_fields = {}

    flow_input_dim = flow[0].input_dim
    image_width = image_height = int(numpy.sqrt(flow_input_dim)+0.5)
    
    for i in range(flow_input_dim):
        rec_field = numpy.zeros((image_width, image_height))
        rec_field[i/image_width, i%image_width] = 1.0
#        rec_fields[(-1, i)] = rec_field
        temporal_rec_fields[(0,i)] = rec_field

    previous_exp_funcs = [identity]
    exp_input_dim = 0

    current_layer = 0
    for i, node in enumerate(flow):
#        if i >= 1:
#            break
        print "L=", i
        if isinstance(node, (mdp.hinet.CloneLayer, mdp.hinet.Layer)):
            layer = node
            if isinstance(node, mdp.hinet.CloneLayer):
                num_nodes[current_layer] = 1
            else:
                num_nodes[current_layer] = len(layer.nodes)

            wnodes[current_layer] = str(layer.nodes[0])   
            num_weights[current_layer] = layer.nodes[0].output_dim
            all_exp_funcs[current_layer] = previous_exp_funcs


            if isinstance(layer[0], more_nodes.GeneralExpansionNode):
                print "Warning, GeneralExpansionNode inside Layer"
                previous_exp_funcs = layer[0].funcs
                exp_input_dim = layer[0].input_dim
                print "GeneralExpansionNode.input_dim is:", exp_input_dim
                if exp_input_dim == None:
                    print "Attemtping to fix this"
                    exp_input_dim = flow[i-1].nodes[0].output_dim
                    print "Now GeneralExpansionNode.input_dim is:", exp_input_dim

                curr_input = 0
                curr_output = 0
                tmp_temporal_rec_fields = {}

                print "layer.input_dim=%d, layer.output_dim=%d, nr_nodes=%d, node.input_dim=%d"%(layer.input_dim, layer.output_dim, len(layer.nodes), layer.nodes[0].input_dim)
                for node_nr, node_in_layer in enumerate(layer.nodes):
                    tmp_rec_field = numpy.zeros((image_width, image_height))
                    print "node_nr=%d, node.input_dim=%d, node.output_dim=%d"%(node_nr, node_in_layer.input_dim, node_in_layer.output_dim)
                    for ii in range(node_in_layer.input_dim):
                        tmp_rec_field += temporal_rec_fields[(current_layer, curr_input+ii)]
                    tmp_rec_field[tmp_rec_field > 0.5] = 1.0 
                    curr_input += node_in_layer.input_dim
    
                    for jj in range(node_in_layer.output_dim):
                        tmp_temporal_rec_fields[(current_layer, curr_output+jj)] = tmp_rec_field 
                    curr_output += node_in_layer.output_dim
    
                for jj in range(layer.output_dim):
                    temporal_rec_fields[(current_layer, jj)] = tmp_temporal_rec_fields[(current_layer, jj)]


            elif isinstance(layer[0], (mdp.nodes.SFANode, mdp.nodes.PCANode, mdp.nodes.WhiteningNode)):
                for current_node in range(num_nodes[current_layer]):
                    for current_weight in range(layer[current_node].output_dim):                
                        if isinstance(layer[current_node], mdp.nodes.SFANode):
                            w = layer[current_node].sf[:,current_weight]
                        elif isinstance(layer[current_node], (mdp.nodes.PCANode, mdp.nodes.WhiteningNode)):
                            w = layer[current_node].v[:,current_weight] 
                        else:
                            raise Exception("This condition should never occur")                        
        
#                        print "w[%d, %d,%d].shape is: "%(current_layer, current_node, current_weight), w.shape,
        
                        if previous_exp_funcs != [identity]:
                            input_dim = exp_input_dim
                        else:
                            input_dim = layer[current_node].input_dim
    
                        if input_dim == None:
                            print "Warning: input_dim of node %d was None, trying to fix this"%current_node 
                            print "layer[%d].node[%d] is"%(current_layer, current_node), layer[current_node]
                            input_dim = flow[i-1].nodes[0].output_dim
                        else:
                            print "OK",
                        out_matrix = generate_weight_image(w, input_dim, previous_exp_funcs)
                        out_matrices[(current_layer, current_node, current_weight)] = out_matrix
 #                       print "out_matrix.shape=", out_matrix.shape
                        out_matrices[(current_layer, current_node, current_weight)]=out_matrix


                curr_input = 0
                curr_output = 0
    
                print "layer.input_dim=%d, layer.output_dim=%d, nr_nodes=%d, node.input_dim=%d"%(layer.input_dim, layer.output_dim, len(layer.nodes), layer.nodes[0].input_dim)
                for node_nr, node_in_layer in enumerate(layer.nodes):
                    tmp_rec_field = numpy.zeros((image_width, image_height))
                    print "node_nr=%d, node.input_dim=%d, node.output_dim=%d"%(node_nr, node_in_layer.input_dim, node_in_layer.output_dim)
                    for ii in range(node_in_layer.input_dim):
                        tmp_rec_field += temporal_rec_fields[(current_layer, curr_input+ii)]
                    curr_input += node_in_layer.input_dim
                    tmp_rec_field[tmp_rec_field > 0.5] = 1.0
                        
                    for jj in range(node_in_layer.output_dim):
                        rec_fields[(current_layer, curr_output+jj)] = tmp_rec_field 
                    curr_output += node_in_layer.output_dim
    
                for jj in range(layer.output_dim):
                    temporal_rec_fields[(current_layer+1, jj)] = rec_fields[(current_layer, jj)]

                    
                current_layer += 1
                num_layers += 1
                previous_exp_funcs = [identity]               
            else:
                print "flow[%d].nodes[%d] of type %s inside CloneLayer was fully ignored"%(i, current_node, str(node[current_node]))       



        elif isinstance(node, (mdp.nodes.SFANode, mdp.nodes.PCANode, mdp.nodes.WhiteningNode)):
            current_node = 0
            all_exp_funcs[current_layer] = previous_exp_funcs
            wnodes[current_layer] = str(node)
            num_nodes[current_layer] = 1
            num_weights[current_layer] = node.output_dim

            tmp_rec_fields = {}
            tmp_rec_fields[(current_layer, 0)] = numpy.zeros((image_width, image_height))
            for ii in range(node.input_dim):
                tmp_rec_fields[(current_layer, 0)] += temporal_rec_fields[(current_layer, ii)]            
            tmp_rec_fields[(current_layer, 0)][tmp_rec_field[(current_layer, 0)] > 0.5] = 1.0
            
            for jj in range(node.output_dim):
                rec_fields[(current_layer, jj)] = tmp_rec_fields[(current_layer, 0)]         
                temporal_rec_fields[(current_layer+1, jj)] = rec_fields[(current_layer, jj)]


            for current_weight in range(node.output_dim):          
                #TODO: Finish this part, include other node types, also in layers!      
                if isinstance(node, mdp.SFANode):
                    w = node.sf[:,current_weight]
                elif isinstance(node, (mdp.nodes.PCANode, mdp.nodes.WhiteningNode)):
                    w = node.v[:,current_weight]
                else:
                    raise Exception("This condition should never ocurr")

#                print "w[%d, %d,%d].shape is: "%(current_layer, 0, current_weight), w.shape,
  #                                  , mdp.PCANode, mdp.WhiteningNode]):
                if previous_exp_funcs != [identity]:
                    input_dim = exp_input_dim
                else:
                    input_dim = node.input_dim
                out_matrix = generate_weight_image(w, input_dim, exp_funcs)
                out_matrices[current_layer].append(out_matrix)
 #               print "out_matrix.shape=", out_matrix.shape

            current_layer += 1
            num_layers += 1
            previous_exp_funcs = [identity]               
        elif isinstance(node, more_nodes.PInvSwitchboard):
            print "Processing PInvSwitchboard Node"       
            print "temporal_rec_fields[(current_layer, 0)][0] was", temporal_rec_fields[(current_layer, 0)][0]
            print "temporal_rec_fields[(current_layer, 0)][1] was", temporal_rec_fields[(current_layer, 0)][1]
            
            tmp_rec_fields = {}

            for i in range(node.input_dim):
                tmp_rec_fields[(current_layer, i)] = numpy.zeros((image_width, image_height))
                
            for i in range(node.input_dim):
#Todo, case where inverse_indices is not exactly one element!!!
#                tmp_rec_fields[(current_layer, node.connections[i])] = temporal_rec_fields[(current_layer, i)]
                tmp_rec_fields[(current_layer, i)] = temporal_rec_fields[(current_layer, node.connections[i])]

            for i in range(node.input_dim):
                temporal_rec_fields[(current_layer, i)] = tmp_rec_fields[(current_layer, i)]

            print "temporal_rec_fields[(current_layer, 0)][0] is", temporal_rec_fields[(current_layer, 0)][0]
            print "temporal_rec_fields[(current_layer, 0)][1] is", temporal_rec_fields[(current_layer, 0)][1]
        else:
            print "flow[%d] of type %s was fully ignored"%(i, str(node))       

   
normalization_type = 0
selected_layer = 0
selected_node = 0
selected_weight = 0
if True:
    print "************ Displaying Training Set SFA and Inverses **************"
    #Create Figure
    f1 = plt.figure()
    plt.suptitle("Weights Displaying System")
          
    #display SFA
    f1a11 = plt.subplot(2,5,1)
    f1a11.set_position([0.05, 0.5, 0.04, 0.4])
    plt.title("Layer Sel.")

    f1a12 = plt.subplot(2,5,2)
    f1a12.set_position([0.15, 0.5, 0.04, 0.4])
    plt.title("Node Sel.")

    
    f1a13 = plt.subplot(2,5,3)
    f1a13.set_position([0.25, 0.5, 0.04, 0.4])
    plt.title("Output Sel.")
      
    f1a15 = plt.subplot(2,5,4)
    f1a15.set_position([0.35, 0.1, 0.04, 0.8])
    plt.title("Colorbar")

    f1a14 = plt.subplot(2,5,5)
    f1a14.set_position([0.45, 0.1, 0.5, 0.8])
    plt.title("Extracted Weights")

    f1a21 = plt.subplot(2,5,6)
    f1a21.set_position([0.05, 0.1, 0.27, 0.32])
    plt.title("Receptive Field of Selected Node")


    
    #Retrieve Image in Sequence
    def on_press(event):
        global plt, f1, f1a11, f1a12, f1a13, f1a14, f1a15, f1a21, selected_layer, selected_node, selected_weight, normalization_type, all_exp_funcs, out_matrices, wnodes, rec_fields
        print 'you pressed:', event, event.inaxes, event.button, event.xdata, event.ydata, event.x, event.y
    
        if event.inaxes == f1a11:
            selected_layer = int(float(event.ydata)+0.5)
            if selected_layer < 0:
                selected_layer = 0
            if selected_layer >= num_layers:
                selected_layer = num_layers -1
            print "selected_layer = " + str(selected_layer)
        elif event.inaxes == f1a12:
            selected_node = int(float(event.ydata)+0.5)
            if selected_node < 0:
                selected_node = 0
            print "selected_node = " + str(selected_node)
        elif event.inaxes == f1a13:
            selected_weight = int(float(event.ydata)+0.5)
            if selected_weight < 0:
                selected_weight = 0
            print "selected_weight = " + str(selected_weight)        
        else:
            normalization_type += 1
            normalization_type %= 3
            print "normalization_type =", normalization_type

        if selected_node >= num_nodes[selected_layer]:
            selected_node = num_nodes[selected_layer] - 1        
        if selected_weight >= num_weights[selected_layer]:
            selected_weight = num_weights[selected_layer] -1
            
        redisplay_weights(num_layers, num_nodes, num_weights, out_matrices, all_exp_funcs, wnodes, rec_fields, selected_layer, selected_node, selected_weight, normalization_type, f1a11, f1a12, f1a13, f1a14, f1a15, f1a21) 
        f1.canvas.draw()
    
    print "num_layers is:", num_layers
    redisplay_weights(num_layers, num_nodes, num_weights, out_matrices, all_exp_funcs, wnodes, rec_fields, selected_layer, selected_node, selected_weight, normalization_type, f1a11, f1a12, f1a13, f1a14, f1a15, f1a21)
    f1.canvas.mpl_connect('button_press_event', on_press)

#f1a11.mpl_connect('button_press_event', on_press)
plt.show()



###Display Original Image
##    subimage_im = subimages[y].reshape((sTrain.subimage_height, sTrain.subimage_width))
##    f1a12.imshow(subimage_im.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
##
##    if show_linear_inv == False:
##        f1.canvas.draw()
##        return
##
###Display Reconstructed Image
##    data_out = sl_seq[y].reshape((1, hierarchy_out_dim))
##    inverted_im = flow.inverse(data_out)
##    inverted_im = inverted_im.reshape((sTrain.subimage_height, sTrain.subimage_width))
##    f1a13.imshow(inverted_im.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)        
###Display Reconstruction Error
##    error_scale_disp=1.5
##    error_im = subimages[y].reshape((sTrain.subimage_height, sTrain.subimage_width)) - inverted_im 
##    error_im_disp = scale_to(error_im, error_im.mean(), error_im.max()-error_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
##    f1a21.imshow(error_im_disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
##    plt.axis = f1a21
##    f1a21.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y=%d" % (error_im.min(), error_im.max(), error_im.std(), error_scale_disp, y))
###Display Differencial change in reconstruction
##    error_scale_disp=1.5
##    if y >= sTrain.num_images - 1:
##        y_next = 0
##    else:
##        y_next = y+1
##    print "y_next=" + str(y_next)
##    data_out2 = sl_seq[y_next].reshape((1, hierarchy_out_dim))
##    inverted_im2 = flow.inverse(data_out2).reshape((sTrain.subimage_height, sTrain.subimage_width))
##    diff_im = inverted_im2 - inverted_im 
##    diff_im_disp = scale_to(diff_im, diff_im.mean(), diff_im.max()-diff_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
##    f1a22.imshow(diff_im_disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
##    plt.axis = f1a22
##    f1a22.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y=%d" % (diff_im.min(), diff_im.max(), diff_im.std(), error_scale_disp, y))
###Display Difference from PINV(y) and PINV(0)
##    error_scale_disp=1.0
##    dif_pinv = inverted_im - pinv_zero 
##    dif_pinv_disp = scale_to(dif_pinv, dif_pinv.mean(), dif_pinv.max()-dif_pinv.min(), 127.5, 255.0, error_scale_disp, 'tanh')
##    f1a23.imshow(dif_pinv.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
##    plt.axis = f1a23
##    f1a23.set_xlabel("PINV(y) - PINV(0): min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y=%d" % (dif_pinv.min(), dif_pinv.max(), dif_pinv.std(), error_scale_disp, y))



#    if network_write:
#        print "Saving flow, layers, benchmark, Network ..."
#        network_write.update_cache([flow, layers, benchmark, Network], None, network_base_dir, "Network"+iTrain.name, overwrite=True, use_hash=network_hash, verbose=True)
#  
