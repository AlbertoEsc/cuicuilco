#Functions for building a hierarchical network, according to the specification of each layer
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 10 Dec 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import mdp
import more_nodes
import lattice
from nonlinear_expansion import identity
from sfa_libs import remove_Nones
import copy
import time
import system_parameters
import numpy

def CreateNetwork(Network, subimage_width, subimage_height, block_size, train_mode, benchmark, in_channel_dim=1, num_features_appended_to_input=0):
    """ This function creates a hierarchical network according to 
    the description stored in the object Network.
    
    
    Network is instantiated from ParamsNetwork() and consists of several layers L0-L10
    """
    print "Using Hierarchical Network: ", Network.name

    if len(Network.layers) > 0:
        layers = []
        for layer in Network.layers:
            if layer != None:
                layers.append(layer)   
    else:
        er = "Obsolete code? Network.layers should have at least one layer!"
        raise Exception(er)

        L0 = copy.deepcopy(Network.L0)
        L1 = copy.deepcopy(Network.L1)
        L2 = copy.deepcopy(Network.L2)
        L3 = copy.deepcopy(Network.L3)
        L4 = copy.deepcopy(Network.L4)
        L5 = copy.deepcopy(Network.L5)
        L6 = copy.deepcopy(Network.L6)
        L7 = copy.deepcopy(Network.L7)
        L8 = copy.deepcopy(Network.L8)
        L9 = copy.deepcopy(Network.L9)
        L10 = copy.deepcopy(Network.L10)
        
        layers = []
        for layer in [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10]:
            if layer != None:
                layers.append(layer)

    layers[0].in_channel_dim = in_channel_dim  #3, warning 1 for L, 3 for RGB
    for i in range(len(layers)):
        if i>0:
            layers[i].in_channel_dim = layers[i-1].sfa_out_dim 
        
    print "Layers: ", layers
    
    for layer in layers:
        print "layer ", layer
        print "here use pca_class to determine if block_size or train_mode is needed!!!"
        if layer.pca_node_class == mdp.nodes.SFANode:
            layer.pca_args["block_size"] = block_size
            layer.pca_args["train_mode"] = train_mode
        if layer.ord_node_class == mdp.nodes.SFANode:
            layer.ord_args["block_size"] = block_size
            layer.ord_args["train_mode"] = train_mode
        if layer.red_node_class == mdp.nodes.SFANode:
            layer.red_args["block_size"] = block_size
            layer.red_args["train_mode"] = train_mode
        if layer.sfa_node_class == mdp.nodes.SFANode:
            layer.sfa_args["block_size"] = block_size
            layer.sfa_args["train_mode"] = train_mode         

    t1 = time.time()

    print "layers =", layers
    node_list = []  
    previous_layer = None
    for i, layer in enumerate(layers):   
        if i==0:
            layer = create_layer(None, layer, i, subimage_height, subimage_width, num_features_appended_to_input)
        else:
            layer = create_layer(previous_layer, layer, i)
        previous_layer = layer
        print "L=", layer
        print "L.node_list=", layer.node_list
        node_list.extend(layer.node_list)
 
    node_list = remove_Nones(node_list)
    print "Flow.node_list=", node_list

    flow = mdp.Flow(node_list, verbose=True)
    t2 = time.time()
    
    print "Finished hierarchy construction, with total time %0.3f ms"% ((t2-t1)*1000.0) 
    benchmark.append(("Hierarchy construction", t2-t1))

    return flow, layers, benchmark


def create_layer(prevLA, LA, num_layer, prevLA_height=None, prevLA_width=None, num_features_appended_to_input=0):
    """Creates a new layer according to the specifications of LA, where
    LA is of type system_parameters.ParamsSFASuperNode or ParamsSFALayer
    it uses prevLA to get info about the previous layer, but it's not 
    necessary for the first layer is prevLA_height and prevLA_width are given
    """
    if LA == None:
        return None
    if isinstance(LA, system_parameters.ParamsSFASuperNode):
        #Note, there is a bug in current MDP SFA Node, in which the output dimension is ignored if the input dimension is unknown. See function "_set_range()".
        print "************ Creating Layer *******"
        print "Creating ParamsSFASuperNode L%d"%num_layer
        if LA.pca_node_class != None:
            print "PCA_node will be created"
            LA.pca_node = LA.pca_node_class(output_dim=LA.pca_out_dim, **LA.pca_args)
        else:
            LA.pca_node = None
            #(input_dim=LA.preserve_mask_sparse.sum(), output_dim=LA.pca_out_dim, **LA.pca_args)
        
        if LA.ord_node_class != None:
            print "Ord_node will be created"
            LA.ord_node = LA.ord_node_class(**LA.ord_args)
        else:
            LA.ord_node = None
        
        #TODO:USE ARGUMENTS EXP_ARGS HERE?
        if LA.exp_funcs != [identity] and LA.exp_funcs != None:
            LA.exp_node = more_nodes.GeneralExpansionNode(LA.exp_funcs, use_hint=LA.inv_use_hint, max_steady_factor=LA.inv_max_steady_factor, \
                                               delta_factor=LA.inv_delta_factor, min_delta=LA.inv_min_delta)
        else:
            LA.exp_node = None
            
        if LA.red_node_class != None:
            LA.red_node = LA.red_node_class(output_dim=LA.red_out_dim, **LA.red_args)
        else:
            LA.red_node = None

        if LA.clip_func != None or LA.clip_inv_func != None:
            LA.clip_node = more_nodes.PointwiseFunctionNode(LA.clip_func, LA.clip_inv_func)
        else:
            LA.clip_node = None

        if LA.sfa_node_class != None:
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "LA.sfa_out_dim= ", LA.sfa_out_dim
    
            LA.sfa_node = LA.sfa_node_class(output_dim=LA.sfa_out_dim, **LA.sfa_args)

            
        LA.node_list = ([LA.pca_node, LA.ord_node, LA.exp_node, LA.red_node, LA.clip_node, LA.sfa_node])    

    elif isinstance(LA, system_parameters.ParamsSFALayer): 
        if prevLA != None:
            previous_layer_height, previous_layer_width, _ = prevLA.lat_mat.shape
        elif prevLA_height != None and prevLA_width != None:
            previous_layer_height = prevLA_height
            previous_layer_width = prevLA_width
        else:
            er = "Error, prevLA, prevLA_height and prevLA_width are None"
            raise Exception(er)

        print "*********************    Creating Layer *************************"
        print "Creating ParamsSFALayer L%d"%num_layer
        
        LA.v1 = [LA.x_field_spacing, 0]
        LA.v2 = [LA.x_field_spacing, LA.y_field_spacing]
        
        LA.preserve_mask, LA.preserve_mask_sparse = lattice.compute_lsrf_preserve_masks(LA.x_field_channels, LA.y_field_channels, LA.nx_value, LA.ny_value, LA.in_channel_dim)
             
        print "About to create (lattice based) intermediate layer width=%d, height=%d"%(LA.x_field_channels, LA.y_field_channels) 
        print "With a spacing of horiz=%d, vert=%d, and %d channels"%(LA.x_field_spacing, LA.y_field_spacing, LA.in_channel_dim) 
        LA.y_in_channels = previous_layer_height 
        LA.x_in_channels = previous_layer_width
        print "LA.x_in_channels, LA.y_in_channels = ", LA.x_in_channels, LA.y_in_channels   
        #switchboard_La = mdp.hinet.Rectangular2dSwitchboard(12, 6, x_field_channels_La,y_field_channels_La,x_field_spacing_La,y_field_spacing_La,in_channel_dim_La)
        (LA.mat_connections, LA.lat_mat) = lattice.compute_lsrf_matrix_connections_with_input_dim(LA.v1, LA.v2, LA.preserve_mask, LA.preserve_mask_sparse, LA.x_in_channels, LA.y_in_channels, LA.in_channel_dim)
        #print "matrix connections La:"
        print LA.mat_connections
        orig_input_dim = LA.x_in_channels * LA.y_in_channels * LA.in_channel_dim
        if num_features_appended_to_input > 0:
            #Assuming the receptive fields have size LA.x_field_channels * LA.y_field_channels * LA.in_channel_dim
            print "specifying %d appended features to the switchboard"
            orig_node_input_dim = LA.x_field_channels * LA.y_field_channels * LA.in_channel_dim
            LA.mat_connections = add_additional_features_to_connections(LA.mat_connections, orig_node_input_dim, orig_input_dim, num_features_appended_to_input)
        LA.switchboard = more_nodes.PInvSwitchboard(orig_input_dim + num_features_appended_to_input, LA.mat_connections)
            
        #LA.switchboard.connections
        LA.num_nodes = LA.lat_mat.size / 2 
            
        if LA.pca_node_class != None:
            if LA.cloneLayer == True:
                print "Layer L%d with "%num_layer, LA.num_nodes, " cloned PCA nodes will be created"
                #print "Warning!!! layer L%d using cloned PCA instead of several independent copies!!!"%num_layer
                LA.pca_node = LA.pca_node_class(input_dim=LA.preserve_mask_sparse.sum()+num_features_appended_to_input, output_dim=LA.pca_out_dim, **LA.pca_args) #input_dim=LA.preserve_mask_sparse.sum()
                #Create array of sfa_nodes (just one node, but cloned)
                LA.pca_layer = mdp.hinet.CloneLayer(LA.pca_node, n_nodes=LA.num_nodes)
            else:
                print "Layer L%d with "%num_layer, LA.num_nodes, " independent PCA nodes will be created, with arguments ", LA.pca_args
                LA.PCA_nodes = range(LA.num_nodes)
                for i in range(LA.num_nodes):
                    LA.PCA_nodes[i] = LA.pca_node_class(input_dim=LA.preserve_mask_sparse.sum(), output_dim=LA.pca_out_dim, **LA.pca_args)
                LA.pca_layer = mdp.hinet.Layer(LA.PCA_nodes, homogeneous = True)
        else:
            LA.pca_layer = None

        if LA.ord_node_class != None:
            if LA.cloneLayer == True:
                print "Ord_node will be created"
                print "Layer L%d with "%num_layer, LA.num_nodes, " cloned ORD nodes will be created"
                #print "Warning!!! layer L%d using cloned ORD instead of several independent copies!!!"%num_layer
                LA.ord_node = LA.ord_node_class(**LA.ord_args)
                #Create array of sfa_nodes (just one node, but cloned)
                LA.ord_layer = mdp.hinet.CloneLayer(LA.ord_node, n_nodes=LA.num_nodes)
            else:
                print "Layer L%d with "%num_layer, LA.num_nodes, " independent ORD nodes will be created"
                LA.ORD_nodes = range(LA.num_nodes)
                for i in range(LA.num_nodes):
                    LA.ORD_nodes[i] = LA.ord_node_class(**LA.ord_args)
                LA.ord_layer = mdp.hinet.Layer(LA.ORD_nodes, homogeneous = True)
        else:
            LA.ord_layer = None

        if LA.exp_funcs != [identity] and LA.exp_funcs != None:
            LA.exp_node = more_nodes.GeneralExpansionNode(LA.exp_funcs, use_hint=True, max_steady_factor=0.05, \
                                               delta_factor=0.6, min_delta=0.0001)
            LA.exp_layer = mdp.hinet.CloneLayer(LA.exp_node, n_nodes=LA.num_nodes)
        else:
            LA.exp_layer = None

        if LA.red_node_class != None:
            if LA.cloneLayer == True: 
                #print "Warning!!! layer L%d using cloned RED instead of several independent copies!!!"%num_layer
                LA.red_node = LA.red_node_class(output_dim=LA.red_out_dim, **LA.red_args)   
                LA.red_layer = mdp.hinet.CloneLayer(LA.red_node, n_nodes=LA.num_nodes)
            else:    
                print "Layer L%d with "%num_layer, LA.num_nodes, " independent RED nodes will be created"
                LA.RED_nodes = range(LA.num_nodes)
                for i in range(LA.num_nodes):
                    LA.RED_nodes[i] = LA.red_node_class(output_dim=LA.red_out_dim, **LA.red_args)
                LA.red_layer = mdp.hinet.Layer(LA.RED_nodes, homogeneous = True)
        else:
            LA.red_layer = None

        if LA.clip_func != None or LA.clip_inv_func != None:
            LA.clip_node = more_nodes.PointwiseFunctionNode(LA.clip_func, LA.clip_inv_func)
        else:
            LA.clip_node = None
        
        if LA.sfa_node_class != None:       
            if LA.cloneLayer == True: 
                #print "Warning!!! layer L%d using cloned SFA instead of several independent copies!!!"%num_layer
                #sfa_node_La = mdp.nodes.SFANode(input_dim=switchboard_LA.out_channel_dim, output_dim=sfa_out_dim_La)
                LA.sfa_node = LA.sfa_node_class(output_dim=LA.sfa_out_dim, **LA.sfa_args)
                #!!! aniadir el atributo output_channels al more_nodes.PInvSwitchboard    
                LA.sfa_layer = mdp.hinet.CloneLayer(LA.sfa_node, n_nodes=LA.num_nodes)
            else:    
                print "Layer L%d with "%num_layer, LA.num_nodes, " independent SFA nodes will be created, with arguments ", LA.sfa_args
                LA.SFA_nodes = range(LA.num_nodes)
                for i in range(LA.num_nodes):
                    LA.SFA_nodes[i] = LA.sfa_node_class(output_dim=LA.sfa_out_dim, **LA.sfa_args)

                LA.sfa_layer = mdp.hinet.Layer(LA.SFA_nodes, homogeneous = True)
        else:
            LA.sfa_layer = None

        LA.node_list = ([LA.switchboard , LA.pca_layer, LA.ord_layer, LA.exp_layer, LA.red_layer, LA.clip_node, LA.sfa_layer])    
    else:
        er = "Unknown Layer type, cannot be created."
        raise Exception(er)
    return LA


def expand_iSeq_sSeq_Layer_to_Network(iSeq_set, sSeq_set, Network):
    iSeq_set_exp = []
    sSeq_set_exp = []
    for i, LA in enumerate(Network.layers):
        num_nodes = 0
        if isinstance(LA, system_parameters.ParamsSFASuperNode):
            if LA.pca_node_class != None:
                num_nodes += 1
            if LA.ord_node_class != None:
                num_nodes += 1            
            if LA.exp_funcs != [identity] and LA.exp_funcs != None:
                num_nodes += 1            
            if LA.red_node_class != None:
                num_nodes += 1
            if LA.clip_func != None or LA.clip_inv_func != None:
                num_nodes += 1
            if LA.sfa_node_class != None:
                num_nodes += 1
        elif isinstance(LA, system_parameters.ParamsSFALayer): 
            num_nodes += 1 #For the switchboard
            if LA.pca_node_class != None:
                num_nodes += 1
            if LA.ord_node_class != None:
                num_nodes += 1            
            if LA.exp_funcs != [identity] and LA.exp_funcs != None:
                num_nodes += 1            
            if LA.red_node_class != None:
                num_nodes += 1
            if LA.clip_func != None or LA.clip_inv_func != None:
                num_nodes += 1
            if LA.sfa_node_class != None:
                num_nodes += 1
        ### Modified to support iSeq_set, sSeqSet not lists
        ##if isinstance(iSeq_set,list):
        j = min(i, len(iSeq_set)-1)
        ###else:
        ###    j = i
        for _ in range(num_nodes):
            iSeq_set_exp.append(iSeq_set[j])
            sSeq_set_exp.append(sSeq_set[j])
        #print iSeq_set_exp
    return iSeq_set_exp, sSeq_set_exp


def add_additional_features_to_connections(connections, components_per_node, original_input_dim, num_features_appended_to_input):
    orig_out_dim = len(connections)
    num_nodes = orig_out_dim / components_per_node
    final_out_dim = orig_out_dim+num_features_appended_to_input
    
    connections_out = numpy.zeros(num_nodes*final_out_dim)
    for i in range(num_nodes):
        connections_out[i*final_out_dim:i*final_out_dim+orig_out_dim] = connections[i*orig_out_dim:(i+1)*orig_out_dim]
    for i in range(num_nodes):
        connections_out[i*final_out_dim+orig_out_dim:(i+1)*final_out_dim] = numpy.arange(original_input_dim, original_input_dim+num_features_appended_to_input)
    return connections_out
