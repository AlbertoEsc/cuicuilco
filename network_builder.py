#Functions for building a hierarchical network, according to the specification of each layer
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 10 Dec 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy
import mdp
import more_nodes
import patch_mdp
import lattice
from nonlinear_expansion import identity
from sfa_libs import remove_Nones
import copy
import time
import SystemParameters

def CreateNetwork(Network, subimage_width, subimage_height, block_size, train_mode, benchmark, in_channel_dim=1):
    """ This function creates a hierarchical network according to 
    the description stored in the object Network.
    
    
    Network is instantiated from ParamsNetwork() and consists of several layers L0-L10
    """
    print "Using Hierarchical Network: ", Network.name
    #TODO: Make this more flexible, no upper bound is needed 
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
    
    ## HIPER WARNING!!!!!!!
    # block_size = train_mode = None
    
    for layer in layers:
        print "layer ", layer
        print "here use pca_class to determine if block_size or train_mode is needed!!!"
        if layer.pca_node_class == mdp.nodes.SFANode:
            #Mega Hiper Turbo warning
#            layer.pca_node_class = None
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
#        if layer.pca_class == mdp.nodes.WhiteningNode or layer.pca_class == mdp.nodes.WhiteningNode

#    previous_layer_height = subimage_height
#    previous_layer_width = subimage_width 
#    print "*********************    Creating Layer L0   *************************"
#    num_layer=0
#    L0.v1 = [L0.x_field_spacing, 0]
#    L0.v2 = [L0.x_field_spacing, L0.y_field_spacing]
#    
#    #preserve_mask_L0_3D = wider(preserve_mask_L0, scale_x=in_channel_dim)
##    if L0.in_channel_dim > 1:
##        L0.preserve_mask = numpy.ones((L0.y_field_channels, L0.x_field_channels, L0.in_channel_dim)) > 0.5
##    else:
##        L0.preserve_mask = numpy.ones((L0.y_field_channels, L0.x_field_channels)) > 0.5
#
#    L0.preserve_mask, L0.preserve_mask_sparse = lattice.compute_lsrf_preserve_masks(L0.x_field_channels, L0.y_field_channels, L0.nx_value, L0.ny_value, L0.in_channel_dim)    
#        
#    print "About to create (Lattice based) intermediate Layer widht=%d, height=%d"%(L0.x_field_channels, L0.y_field_channels) 
#    print "With a spacing of horiz=%d, vert=%d, and %d channels"%(L0.x_field_spacing, L0.y_field_spacing, L0.in_channel_dim) 
#    L0.y_in_channels = previous_layer_height 
#    L0.x_in_channels = previous_layer_width
#    
#    #remember, here tmp is always two!!!
#    #switchboard_L0 = mdp.hinet.RectanguL0r2dSwitchboard(12, 6, x_field_channels_L0,y_field_channels_L0,x_field_spacing_L0,y_field_spacing_L0,in_channel_dim_L0)
##    (L0.mat_connections, L0.lat_mat) = compute_lattice_matrix_connections_with_input_dim(L0.v1, L0.v2, L0.preserve_mask, L0.x_in_channels, L0.y_in_channels, L0.in_channel_dim)
#    (L0.mat_connections, L0.lat_mat) = lattice.compute_lsrf_matrix_connections_with_input_dim(L0.v1, L0.v2, L0.preserve_mask, L0.preserve_mask_sparse, L0.x_in_channels, L0.y_in_channels, L0.in_channel_dim)
#    print "matrix connections L%d:"%num_layer
#    print L0.mat_connections
#    L0.switchboard = more_nodes.PInvSwitchboard(L0.x_in_channels * L0.y_in_channels * L0.in_channel_dim, L0.mat_connections)
#        
#    #L0.switchboard.connections
#        
#    
#    L0.num_nodes = L0.lat_mat.size / 2 
#           
#    if L0.cloneLayer == True:
#        print "Pre Layer L%d with "%num_layer, L0.num_nodes, " cloned ", L0.pca_node_class, " nodes will be created"
#        #print "Warning!!! Layer L%d using cloned PCA instead of several independent copies!!!"%num_layer
#        L0.pca_node = L0.pca_node_class(input_dim=L0.preserve_mask_sparse.sum(), output_dim=L0.pca_out_dim, **L0.pca_args)
#        #Create array of sfa_nodes (just one node, but cloned)
#        L0.pca_layer = mdp.hinet.CloneLayer(L0.pca_node, n_nodes=L0.num_nodes)
#    else:
#        print "Pre Layer L%d with "%num_layer, L0.num_nodes, " independent ", L0.pca_node_class, " nodes will be created"
#        L0.PCA_nodes = range(L0.num_nodes)
#        for i in range(L0.num_nodes):
#            L0.PCA_nodes[i] = L0.pca_node_class(input_dim=L0.preserve_mask_sparse.sum(), output_dim=L0.pca_out_dim, **L0.pca_args)
#        L0.pca_layer = mdp.hinet.Layer(L0.PCA_nodes, homogeneous = True)
#        
#    L0.exp_node = more_nodes.GeneralExpansionNode(L0.exp_funcs, use_hint=True, max_steady_factor=0.05, \
#                                       delta_factor=0.6, min_delta=0.0001)
#    L0.exp_layer = mdp.hinet.CloneLayer(L0.exp_node, n_nodes=L0.num_nodes)
#    
#    if L0.cloneLayer == True: 
#        #print "Warning!!! layer L%d using cloned RED instead of several independent copies!!!"%num_layer
#        L0.red_node = L0.red_node_class(output_dim=L0.red_out_dim, **L0.red_args)   
#        L0.red_layer = mdp.hinet.CloneLayer(L0.red_node, n_nodes=L0.num_nodes)
#    else:    
#        print "Layer L%d with "%num_layer, L0.num_nodes, " independent RED nodes will be created"
#        L0.RED_nodes = range(L0.num_nodes)
#        for i in range(L0.num_nodes):
#            L0.RED_nodes[i] = L0.red_node_class(output_dim=L0.red_out_dim, **L0.red_args)
#        L0.red_layer = mdp.hinet.Layer(L0.RED_nodes, homogeneous = True)
#    
#    L0.clip_node = more_nodes.PointwiseFunctionNode(L0.clip_func, L0.clip_inv_func)
#        
#    if L0.cloneLayer == True: 
#        #print "Warning!!! layer L%d using cloned SFA instead of several independent copies!!!"%num_layer
#        #sfa_node_La = mdp.nodes.SFANode(input_dim=switchboard_L0.out_channel_dim, output_dim=sfa_out_dim_La)
#        L0.sfa_node = L0.sfa_node_class(output_dim=L0.sfa_out_dim, **L0.sfa_args)    
#        #!!!no ma, ya aniadele el atributo output_channels al more_nodes.PInvSwitchboard    
#        L0.sfa_layer = mdp.hinet.CloneLayer(L0.sfa_node, n_nodes=L0.num_nodes)
#    else:    
#        print "Layer L%d with "%num_layer, L0.num_nodes, " independent SFA nodes will be created"
#        L0.SFA_nodes = range(L0.num_nodes)
#        for i in range(L0.num_nodes):
#            L0.SFA_nodes[i] = L0.sfa_node_class(output_dim=L0.sfa_out_dim, **L0.sfa_args)
#        L0.sfa_layer = mdp.hinet.Layer(L0.SFA_nodes, homogeneous = True)
#    L0.node_list = ([L0.switchboard, L0.pca_layer, L0.exp_layer, L0.red_layer, L0.clip_node, L0.sfa_layer])

#    previous_layer_height, previous_layer_width, tmp = L0.lat_mat.shape
#    print "*********************    Creating Layer L1   *************************"
#    num_layer=1
#    L1.v1 = [L1.x_field_spacing, 0]
#    L1.v2 = [L1.x_field_spacing, L1.y_field_spacing]
#    
#    #preserve_mask_La_3D = wider(preserve_mask_La, scale_x=in_channel_dim)
##    if L1.in_channel_dim > 1:
##        L1.preserve_mask = numpy.ones((L1.y_field_channels, L1.x_field_channels, L1.in_channel_dim)) > 0.5
##    else:
##        L1.preserve_mask = numpy.ones((L1.y_field_channels, L1.x_field_channels)) > 0.5
#    print "L1.nx_value & ny_value are ", L1.nx_value, L1.ny_value
#    L1.preserve_mask, L1.preserve_mask_sparse = lattice.compute_lsrf_preserve_masks(L1.x_field_channels, L1.y_field_channels, L1.nx_value, L1.ny_value, L1.in_channel_dim)    
#    print "L1.preserve_mask_sparse is", L1.preserve_mask_sparse
#    print "About to create (lattice based) intermediate layer widht=%d, height=%d"%(L1.x_field_channels, L1.y_field_channels) 
#    print "With a spacing of horiz=%d, vert=%d, and %d channels"%(L1.x_field_spacing, L1.y_field_spacing, L1.in_channel_dim) 
#    L1.y_in_channels = previous_layer_height 
#    L1.x_in_channels = previous_layer_width
#        
#    #remember, here tmp is always two!!!
#    #switchboard_La = mdp.hinet.Rectangular2dSwitchboard(12, 6, x_field_channels_La,y_field_channels_La,x_field_spacing_La,y_field_spacing_La,in_channel_dim_La)
#    (L1.mat_connections, L1.lat_mat) = lattice.compute_lsrf_matrix_connections_with_input_dim(L1.v1, L1.v2, L1.preserve_mask, L1.preserve_mask_sparse, L1.x_in_channels, L1.y_in_channels, L1.in_channel_dim)
#    print "matrix connections La:"
#    print L1.mat_connections
#    L1.switchboard = more_nodes.PInvSwitchboard(L1.x_in_channels * L1.y_in_channels * L1.in_channel_dim, L1.mat_connections)
#        
#    #L1.switchboard.connections
#    L1.num_nodes = L1.lat_mat.size / 2 
#       
#           
#    if L1.cloneLayer == True:
#        print "Layer L%d with "%num_layer, L1.num_nodes, " cloned PCA nodes will be created"
#        #print "Warning!!! layer L%d using cloned PCA instead of several independent copies!!!"%num_layer
#        L1.pca_node = L1.pca_node_class(input_dim=L1.preserve_mask_sparse.sum(), output_dim=L1.pca_out_dim, **L1.pca_args)
#        #Create array of sfa_nodes (just one node, but cloned)
#        L1.pca_layer = mdp.hinet.CloneLayer(L1.pca_node, n_nodes=L1.num_nodes)
#    else:
#        print "Layer L%d with "%num_layer, L1.num_nodes, " independent PCA nodes will be created"
#        L1.PCA_nodes = range(L1.num_nodes)
#        for i in range(L1.num_nodes):
#            L1.PCA_nodes[i] = L1.pca_node_class(input_dim=L1.preserve_mask_sparse.sum(), output_dim=L1.pca_out_dim, **L1.pca_args)
#        L1.pca_layer = mdp.hinet.Layer(L1.PCA_nodes, homogeneous = True)
#        
#    L1.exp_node = more_nodes.GeneralExpansionNode(L1.exp_funcs, use_hint=True, max_steady_factor=0.05, \
#                                       delta_factor=0.6, min_delta=0.0001)
#    L1.exp_layer = mdp.hinet.CloneLayer(L1.exp_node, n_nodes=L1.num_nodes)
#      
#    if L1.cloneLayer == True: 
#        #print "Warning!!! layer L%d using cloned RED instead of several independent copies!!!"%num_layer
#        L1.red_node = L1.red_node_class(output_dim=L1.red_out_dim, **L1.red_args)   
#        L1.red_layer = mdp.hinet.CloneLayer(L1.red_node, n_nodes=L1.num_nodes)
#    else:    
#        print "Layer L%d with "%num_layer, L1.num_nodes, " independent RED nodes will be created"
#        L1.RED_nodes = range(L1.num_nodes)
#        for i in range(L1.num_nodes):
#            L1.RED_nodes[i] = L1.red_node_class(output_dim=L1.red_out_dim, **L1.red_args)
#        L1.red_layer = mdp.hinet.Layer(L1.RED_nodes, homogeneous = True)
#    
#    L1.clip_node = more_nodes.PointwiseFunctionNode(L1.clip_func, L1.clip_inv_func)
#        
#    if L1.cloneLayer == True: 
#        #print "Warning!!! layer L%d using cloned SFA instead of several independent copies!!!"%num_layer
#        #sfa_node_La = mdp.nodes.SFANode(input_dim=switchboard_L1.out_channel_dim, output_dim=sfa_out_dim_La)
#        L1.sfa_node = L1.sfa_node_class(output_dim=L1.sfa_out_dim, **L1.sfa_args)    
#        #!!!no ma, ya aniadele el atributo output_channels al more_nodes.PInvSwitchboard    
#        L1.sfa_layer = mdp.hinet.CloneLayer(L1.sfa_node, n_nodes=L1.num_nodes)
#    else:    
#        print "Layer L%d with "%num_layer, L1.num_nodes, " independent SFA nodes will be created"
#        L1.SFA_nodes = range(L1.num_nodes)
#        for i in range(L1.num_nodes):
#            L1.SFA_nodes[i] = L1.sfa_node_class(output_dim=L1.sfa_out_dim, **L1.sfa_args)
#        L1.sfa_layer = mdp.hinet.Layer(L1.SFA_nodes, homogeneous = True)
#    L1.node_list = ([L1.switchboard, L1.pca_layer, L1.exp_layer, L1.red_layer, L1.clip_node, L1.sfa_layer])
#    
#    
#    previous_layer_height, previous_layer_width, tmp = L1.lat_mat.shape
#    print "*********************    Creating Layer L2   *************************"
#    num_layer=2
#    
#    L2.v1 = [L2.x_field_spacing, 0]
#    L2.v2 = [L2.x_field_spacing, L2.y_field_spacing]
#    
#    #preserve_mask_La_3D = wider(preserve_mask_La, scale_x=in_channel_dim)
##    if L2.in_channel_dim > 1:
##        L2.preserve_mask = numpy.ones((L2.y_field_channels, L2.x_field_channels, L2.in_channel_dim)) > 0.5
##    else:
##        L2.preserve_mask = numpy.ones((L2.y_field_channels, L2.x_field_channels)) > 0.5
#    L2.preserve_mask, L2.preserve_mask_sparse = lattice.compute_lsrf_preserve_masks(L2.x_field_channels, L2.y_field_channels, L2.nx_value, L2.ny_value, L2.in_channel_dim)    
#        
#    print "About to create (lattice based) intermediate layer widht=%d, height=%d"%(L2.x_field_channels, L2.y_field_channels) 
#    print "With a spacing of horiz=%d, vert=%d, and %d channels"%(L2.x_field_spacing, L2.y_field_spacing, L2.in_channel_dim) 
#    L2.y_in_channels = previous_layer_height 
#    L2.x_in_channels = previous_layer_width
#        
#    #remember, here tmp is always two!!!
#    #switchboard_La = mdp.hinet.Rectangular2dSwitchboard(12, 6, x_field_channels_La,y_field_channels_La,x_field_spacing_La,y_field_spacing_La,in_channel_dim_La)
#    (L2.mat_connections, L2.lat_mat) = lattice.compute_lsrf_matrix_connections_with_input_dim(L2.v1, L2.v2, L2.preserve_mask, L2.preserve_mask_sparse, L2.x_in_channels, L2.y_in_channels, L2.in_channel_dim)
#    print "matrix connections La:"
#    print L2.mat_connections
#    
#    L2.switchboard = more_nodes.PInvSwitchboard(L2.x_in_channels * L2.y_in_channels * L2.in_channel_dim, L2.mat_connections)
#        
#    #L2.switchboard.connections
#    L2.num_nodes = L2.lat_mat.size / 2 
#        
#    if L2.pca_node_class != None:
#        if L2.cloneLayer == True:
#            print "Layer L%d with "%num_layer, L2.num_nodes, " cloned PCA nodes will be created"
#            #print "Warning!!! layer L%d using cloned PCA instead of several independent copies!!!"%num_layer
#            L2.pca_node = L2.pca_node_class(input_dim=L2.preserve_mask_sparse.sum(), output_dim=L2.pca_out_dim, **L2.pca_args)
#            #Create array of sfa_nodes (just one node, but cloned)
#            L2.pca_layer = mdp.hinet.CloneLayer(L2.pca_node, n_nodes=L2.num_nodes)
#        else:
#            print "Layer L%d with "%num_layer, L2.num_nodes, " independent PCA nodes will be created"
#            L2.PCA_nodes = range(L2.num_nodes)
#            for i in range(L2.num_nodes):
#                L2.PCA_nodes[i] = L2.pca_node_class(input_dim=L2.preserve_mask_sparse.sum(), output_dim=L2.pca_out_dim, **L2.pca_args)
#            L2.pca_layer = mdp.hinet.Layer(L2.PCA_nodes, homogeneous = True)
#    else:
#        L2.pca_layer = None
#
#    if L2.exp_funcs != [identity]:
#        L2.exp_node = more_nodes.GeneralExpansionNode(L2.exp_funcs, use_hint=True, max_steady_factor=0.05, \
#                                           delta_factor=0.6, min_delta=0.0001)
#        L2.exp_layer = mdp.hinet.CloneLayer(L2.exp_node, n_nodes=L2.num_nodes)
#    else:
#        L2.exp_layer = None
#
#    if L2.red_node_class != None:
#        if L2.cloneLayer == True: 
#            #print "Warning!!! layer L%d using cloned RED instead of several independent copies!!!"%num_layer
#            L2.red_node = L2.red_node_class(output_dim=L2.red_out_dim, **L2.red_args)   
#            L2.red_layer = mdp.hinet.CloneLayer(L2.red_node, n_nodes=L2.num_nodes)
#        else:    
#            print "Layer L%d with "%num_layer, L2.num_nodes, " independent RED nodes will be created"
#            L2.RED_nodes = range(L2.num_nodes)
#            for i in range(L2.num_nodes):
#                L2.RED_nodes[i] = L2.red_node_class(output_dim=L2.red_out_dim, **L2.red_args)
#            L2.red_layer = mdp.hinet.Layer(L2.RED_nodes, homogeneous = True)
#    else:
#        L2.red_layer = None
#
#    if L2.clip_func != None or L2.clip_inv_func != None:
#        L2.clip_node = more_nodes.PointwiseFunctionNode(L2.clip_func, L2.clip_inv_func)
#    else:
#        L2.clip_node = None
#    
#    if L2.sfa_node_class != None:
#        if L2.cloneLayer == True: 
#            #print "Warning!!! layer L%d using cloned SFA instead of several independent copies!!!"%num_layer
#            #sfa_node_La = mdp.nodes.SFANode(input_dim=switchboard_L2.out_channel_dim, output_dim=sfa_out_dim_La)
#            L2.sfa_node = L2.sfa_node_class(output_dim=L2.sfa_out_dim, **L2.sfa_args)    
#            #!!!no ma, ya aniadele el atributo output_channels al more_nodes.PInvSwitchboard    
#            L2.sfa_layer = mdp.hinet.CloneLayer(L2.sfa_node, n_nodes=L2.num_nodes)
#        else:    
#            print "Layer L%d with "%num_layer, L2.num_nodes, " independent SFA nodes will be created"
#            L2.SFA_nodes = range(L2.num_nodes)
#            for i in range(L2.num_nodes):
#                L2.SFA_nodes[i] = L2.sfa_node_class(output_dim=L2.sfa_out_dim, **L2.sfa_args)
#            L2.sfa_layer = mdp.hinet.Layer(L2.SFA_nodes, homogeneous = True)
#    else:
#        L2.sfa_layer = None
#
#    L2.node_list = ([L2.switchboard , L2.pca_layer, L2.exp_layer, L2.red_layer, L2.clip_node, L2.sfa_layer])

    t1 = time.time()
        
    L0 = create_layer(None, L0, 0, subimage_height, subimage_width)
    L1 = create_layer(L0, L1, 1)
    L2 = create_layer(L1, L2, 2)
    L3 = create_layer(L2, L3, 3)
    L4 = create_layer(L3, L4, 4)
    L5 = create_layer(L4, L5, 5)
    L6 = create_layer(L5, L6, 6)
    L7 = create_layer(L6, L7, 7)
    L8 = create_layer(L7, L8, 8)
    L9 = create_layer(L8, L9, 9)
    L10 = create_layer(L9, L10, 10)

#    L4 = L5 = None
#    print "*********************    Creating SFA SuperNode L3   *************************"
#    num_layer=3
#    
#    L3.pca_node = L3.pca_node_class(output_dim=L3.pca_out_dim, **L3.pca_args)
#    L3.exp_node = more_nodes.GeneralExpansionNode(L3.exp_funcs, use_hint=L3.inv_use_hint, max_steady_factor=L3.inv_max_steady_factor, \
#                                       delta_factor=L3.inv_delta_factor, min_delta=L3.inv_min_delta)
#    L3.red_node = L3.red_node_class(output_dim=L3.red_out_dim, **L3.red_args)
#    L3.clip_node = more_nodes.PointwiseFunctionNode(L3.clip_func, L3.clip_inv_func)
#    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#    print "L3.sfa_out_dim= ", L3.sfa_out_dim
#    L3.sfa_node = L3.sfa_node_class(output_dim=L3.sfa_out_dim, **L3.sfa_args)    
#    
#    if L4 != None:
#        print "*********************    Creating SFA SuperNode L4   *************************"
#        num_layer=4
#    
#        L4.pca_node = L4.pca_node_class(output_dim=L4.pca_out_dim, **L4.pca_args)
#        L4.exp_node = more_nodes.GeneralExpansionNode(L4.exp_funcs, use_hint=L4.inv_use_hint, max_steady_factor=L4.inv_max_steady_factor, \
#                                           delta_factor=L4.inv_delta_factor, min_delta=L4.inv_min_delta)
#        L4.red_node = L4.red_node_class(output_dim=L4.red_out_dim, **L4.red_args)
#        L4.clip_node = more_nodes.PointwiseFunctionNode(L4.clip_func, L4.clip_inv_func)
#        L4.sfa_node = L4.sfa_node_class(output_dim=L4.sfa_out_dim, **L4.sfa_args)    
    
    
    #MEGAWARNING!!!!!!!!!!!
    #flow = mdp.Flow([L0.switchboard, L0.pca_layer, L0.exp_layer, L0.red_layer, L0.sfa_layer, L1.switchboard, L1.pca_layer, L1.exp_layer, L1.red_layer, L1.sfa_layer, L2.switchboard, L2.pca_layer], verbose=True)
    node_list = []
    print layers
    for layer in layers:
        print "L=", layer
        print "L.node_list=", layer.node_list
        node_list.extend(layer.node_list)

    node_list = remove_Nones(node_list)
    print "Flow.node_list=", node_list

#    if L4 == None:
#        flow = mdp.Flow([L0.switchboard, L0.pca_layer, L0.exp_layer, L0.red_layer, L0.clip_node, L0.sfa_layer, L1.switchboard, L1.pca_layer, L1.exp_layer, L1.red_layer, L1.clip_node, L1.sfa_layer, L2.switchboard, L2.pca_layer, L2.exp_layer, L2.red_layer, L2.clip_node, L2.sfa_layer, L3.pca_node, L3.exp_node, L3.red_node, L3.clip_node, L3.sfa_node], verbose=True)
#    else:
#        if L5 == None:
#            flow = mdp.Flow([L0.switchboard, L0.pca_layer, L0.exp_layer, L0.red_layer, L0.clip_node, L0.sfa_layer, L1.switchboard, L1.pca_layer, L1.exp_layer, L1.red_layer, L1.clip_node, L1.sfa_layer, L2.switchboard, L2.pca_layer, L2.exp_layer, L2.red_layer, L2.clip_node, L2.sfa_layer, L3.pca_node, L3.exp_node, L3.red_node, L3.clip_node, L3.sfa_node, L4.pca_node, L4.exp_node, L4.red_node, L4.clip_node, L4.sfa_node], verbose=True)
#        else:
#            flow = mdp.Flow([L0.switchboard, L0.pca_layer, L0.exp_layer, L0.red_layer, L0.clip_node, L0.sfa_layer, L1.switchboard, L1.pca_layer, L1.exp_layer, L1.red_layer, L1.clip_node, L1.sfa_layer, L2.switchboard, L2.pca_layer, L2.exp_layer, L2.red_layer, L2.clip_node, L2.sfa_layer, L3.pca_node, L3.exp_node, L3.red_node, L3.clip_node, L3.sfa_node, L4.pca_node, L4.exp_node, L4.red_node, L4.clip_node, L4.sfa_node, L5.pca_node, L5.exp_node, L5.red_node, L5.clip_node, L5.sfa_node], verbose=True)
    flow = mdp.Flow(node_list, verbose=True)
    
    #flow = mdp.Flow([switchboard_L0, pca_layer_L0, sfa_layer_L0, switchboard_L1, pca_layer_L1, sfa_layer_L1, switchboard_L2, pca_layer_L2,  sfa_layer_L2, pca_node_L3,  sfa_node_L3], verbose=True)   
    #flow = mdp.Flow([switchboard_L0, sfa_layer_L0, switchboard_L1, sfa_layer_L1])
    #flow = mdp.Flow([switchboard_L0, pca_layer_L0, exp_layer_L0, red_layer_L0, sfa_layer_L0, switchboard_L1, pca_layer_L1, sfa_layer_L1, switchboard_L2, pca_layer_L2, sfa_layer_L2, pca_node_L3, sfa_node_L3], verbose=True)
    t2 = time.time()
    
    print "Finished hierarchy construction, with total time %0.3f ms"% ((t2-t1)*1000.0) 
    benchmark.append(("Hierarchy construction", t2-t1))

    return flow, layers, benchmark


#    num_layer=2
def create_layer(prevLA, LA, num_layer, prevLA_height=None, prevLA_width=None):
    """Creates a new layer according to the specifications of LA, where
    LA is of type SystemParameters.ParamsSFASuperNode or ParamsSFALayer
    it uses prevLA to get info about the previous layer, but it's not 
    necessary for the first layer is prevLA_height and prevLA_width are given
    """
    if LA == None:
        return None
    if isinstance(LA, SystemParameters.ParamsSFASuperNode):
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
#            er = "Nope, there is a bug in here"
#            print "Warning, there might be a bug in here: ", er
#            raise Exception(er)
        
        if LA.exp_funcs != [identity]:
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

    elif isinstance(LA, SystemParameters.ParamsSFALayer): 
        if prevLA != None:
            previous_layer_height, previous_layer_width, tmp = prevLA.lat_mat.shape
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
        
        #preserve_mask_La_3D = wider(preserve_mask_La, scale_x=in_channel_dim)
#        if LA.in_channel_dim > 1:
#            LA.preserve_mask = numpy.ones((LA.y_field_channels, LA.x_field_channels, LA.in_channel_dim)) > 0.5
#        else:
#            LA.preserve_mask = numpy.ones((LA.y_field_channels, LA.x_field_channels)) > 0.5
        LA.preserve_mask, LA.preserve_mask_sparse = lattice.compute_lsrf_preserve_masks(LA.x_field_channels, LA.y_field_channels, LA.nx_value, LA.ny_value, LA.in_channel_dim)
             
        print "About to create (lattice based) intermediate layer widht=%d, height=%d"%(LA.x_field_channels, LA.y_field_channels) 
        print "With a spacing of horiz=%d, vert=%d, and %d channels"%(LA.x_field_spacing, LA.y_field_spacing, LA.in_channel_dim) 
        LA.y_in_channels = previous_layer_height 
        LA.x_in_channels = previous_layer_width
        print "LA.x_in_channels, LA.y_in_channels = ", LA.x_in_channels, LA.y_in_channels   
        #switchboard_La = mdp.hinet.Rectangular2dSwitchboard(12, 6, x_field_channels_La,y_field_channels_La,x_field_spacing_La,y_field_spacing_La,in_channel_dim_La)
        (LA.mat_connections, LA.lat_mat) = lattice.compute_lsrf_matrix_connections_with_input_dim(LA.v1, LA.v2, LA.preserve_mask, LA.preserve_mask_sparse, LA.x_in_channels, LA.y_in_channels, LA.in_channel_dim)
        #print "matrix connections La:"
        print LA.mat_connections
        
        LA.switchboard = more_nodes.PInvSwitchboard(LA.x_in_channels * LA.y_in_channels * LA.in_channel_dim, LA.mat_connections)
            
        #LA.switchboard.connections
        LA.num_nodes = LA.lat_mat.size / 2 
            
        if LA.pca_node_class != None:
            if LA.cloneLayer == True:
                print "Layer L%d with "%num_layer, LA.num_nodes, " cloned PCA nodes will be created"
                #print "Warning!!! layer L%d using cloned PCA instead of several independent copies!!!"%num_layer
                LA.pca_node = LA.pca_node_class(input_dim=LA.preserve_mask_sparse.sum(), output_dim=LA.pca_out_dim, **LA.pca_args)
                #Create array of sfa_nodes (just one node, but cloned)
                LA.pca_layer = mdp.hinet.CloneLayer(LA.pca_node, n_nodes=LA.num_nodes)
            else:
                print "Layer L%d with "%num_layer, LA.num_nodes, " independent PCA nodes will be created"
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

        if LA.exp_funcs != [identity]:
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
                #!!!no ma, ya aniadele el atributo output_channels al more_nodes.PInvSwitchboard    
                LA.sfa_layer = mdp.hinet.CloneLayer(LA.sfa_node, n_nodes=LA.num_nodes)
            else:    
                print "Layer L%d with "%num_layer, LA.num_nodes, " independent SFA nodes will be created"
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
        if isinstance(LA, SystemParameters.ParamsSFASuperNode):
            if LA.pca_node_class != None:
                num_nodes += 1
            if LA.ord_node_class != None:
                num_nodes += 1            
            if LA.exp_funcs != [identity]:
                num_nodes += 1            
            if LA.red_node_class != None:
                num_nodes += 1
            if LA.clip_func != None or LA.clip_inv_func != None:
                num_nodes += 1
            if LA.sfa_node_class != None:
                num_nodes += 1
        elif isinstance(LA, SystemParameters.ParamsSFALayer): 
            num_nodes += 1 #For the switchboard
            if LA.pca_node_class != None:
                num_nodes += 1
            if LA.ord_node_class != None:
                num_nodes += 1            
            if LA.exp_funcs != [identity]:
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
        for w in range(num_nodes):
            iSeq_set_exp.append(iSeq_set[j])
            sSeq_set_exp.append(sSeq_set[j])
#    print iSeq_set_exp
#    quit()
    return iSeq_set_exp, sSeq_set_exp
