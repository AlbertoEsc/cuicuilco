#####################################################################################################################
# network_builder: This module implements functions useful for building a hierarchical network from an abstract,    #
#                  high-level representation.                                                                       #
#                                                                                                                   #
# One can describe a hierarchical using a system_parameters.ParamsNetwork object, which contains several layers.    #
# A layer can be of type system_parametersParamsSFALayer or system_parametersParamsSFASuperNode. In both            #
# cases, these abstract layers are composed of several MDP layers or nodes.                                         #
# See Escalante-B, A.-N., 2017, "Extensions of Hierarchical Slow Feature Analysis for Efficient                     #
# Classification and Regression on High-Dimensional Data", PhD Thesis, Appendices 3, 4, and 5.                      #
#                                                                                                                   #
# By Alberto Escalante. Alberto.Escalante@ini.rub.de                                                                #
# Ruhr-University Bochum, Institute for Neural Computation, Group of Prof. Dr. Wiskott                              #
#####################################################################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy
import copy
import time

import mdp

from . import more_nodes
from . import lattice
from .nonlinear_expansion import identity
from .sfa_libs import remove_Nones
from . import system_parameters


# def CreateNetwork(Network, subimage_width, subimage_height, block_size, train_mode, benchmark, in_channel_dim=1,
def create_network(network, subimage_width, subimage_height, benchmark, in_channel_dim=1,
                   num_features_appended_to_input=0):
    """ This function creates a hierarchical network according to the description
    stored in the object 'network'.

    The object 'network' is of type system_parameters.ParamsNetwork() and contains
    several layers (either hierarchical or non-hierarchical).
    """
    print("Using hierarchical network: ", network.name)

    if len(network.layers) > 0:
        layers = []
        for layer in network.layers:
            if layer is not None:
                layers.append(layer)
    else:
        er = "Obsolete network description? network.layers should have at least one layer!"
        raise Exception(er)

    layers[0].in_channel_dim = in_channel_dim  # 1 for L, 3 for RGB
    for i in range(len(layers)):
        if i > 0:
            layers[i].in_channel_dim = layers[i - 1].sfa_out_dim

    print("Layers: ", layers)

    t1 = time.time()

    print("layers =", layers)
    node_list = []
    previous_layer = None
    for i, layer in enumerate(layers):
        if i == 0:
            layer = create_layer(None, layer, i, subimage_height, subimage_width, num_features_appended_to_input)
        else:
            layer = create_layer(previous_layer, layer, i)
        previous_layer = layer
        print("L=", layer)
        print("L.node_list=", layer.node_list)
        node_list.extend(layer.node_list)

    node_list = remove_Nones(node_list)
    print("Flow.node_list=", node_list)

    flow = mdp.Flow(node_list, verbose=True)
    t2 = time.time()

    print("Finished hierarchy construction, with total time %0.3f ms" % ((t2 - t1) * 1000.0))
    if benchmark:
        benchmark.append(("Hierarchy construction", t2 - t1))

    return flow, layers, benchmark


def create_layer(prev_layer, layer, num_layer, prev_layer_height=None, prev_layer_width=None,
                 num_features_appended_to_input=0):
    """Creates a new layer according to the specifications of 'layer'.

    'layer' is of type system_parameters.ParamsSFASuperNode or ParamsSFALayer.
    This function uses prev_layer to get info about the previous layer, but it's not
    necessary for the first layer if prev_layer_height and prev_layer_width are given.
    """
    if layer is None:
        return None
    if isinstance(layer, system_parameters.ParamsSFALayer):
        if prev_layer:
            previous_layer_height, previous_layer_width, _ = prev_layer.lat_mat.shape
        elif prev_layer_height and prev_layer_width:
            previous_layer_height = prev_layer_height
            previous_layer_width = prev_layer_width
        else:
            er = "Error, prev_layer, prev_layer_height and prev_layer_width are None"
            raise Exception(er)

        print("*********************    Creating Layer *************************")
        print("Creating ParamsSFALayer L%d" % num_layer)

        layer.v1 = [layer.x_field_spacing, 0]
        layer.v2 = [layer.x_field_spacing, layer.y_field_spacing]

        (layer.preserve_mask, layer.preserve_mask_sparse) = \
            lattice.compute_lsrf_preserve_masks(layer.x_field_channels, layer.y_field_channels,
                                                layer.nx_value, layer.ny_value, layer.in_channel_dim)

        print("About to create (lattice based) intermediate layer width=%d, height=%d" % (layer.x_field_channels,
                                                                                          layer.y_field_channels))
        print("With a spacing of horiz=%d, vert=%d, and %d channels" % (layer.x_field_spacing, layer.y_field_spacing,
                                                                        layer.in_channel_dim))
        layer.y_in_channels = previous_layer_height
        layer.x_in_channels = previous_layer_width
        print("layer.x_in_channels, layer.y_in_channels = ", layer.x_in_channels, layer.y_in_channels)
        # switchboard_La = mdp.hinet.Rectangular2dSwitchboard(12, 6, x_field_channels_La, y_field_channels_La,
        # x_field_spacing_La,y_field_spacing_La,in_channel_dim_La)
        (layer.mat_connections, layer.lat_mat) = \
            lattice.compute_lsrf_matrix_connections_with_input_dim(layer.v1, layer.v2, layer.preserve_mask,
                                                                   layer.preserve_mask_sparse, layer.x_in_channels,
                                                                   layer.y_in_channels, layer.in_channel_dim)
        # print "matrix connections La:"
        print(layer.mat_connections)
        orig_input_dim = layer.x_in_channels * layer.y_in_channels * layer.in_channel_dim
        if num_features_appended_to_input > 0:
            # Assuming the receptive fields have size layer.x_field_channels * layer.y_field_channels *
            # layer.in_channel_dim
            print("specifying %d appended features to the switchboard")
            orig_node_input_dim = layer.x_field_channels * layer.y_field_channels * layer.in_channel_dim
            layer.mat_connections = add_additional_features_to_connections(layer.mat_connections, orig_node_input_dim,
                                                                           orig_input_dim,
                                                                           num_features_appended_to_input)
        layer.switchboard = more_nodes.PInvSwitchboard(orig_input_dim + num_features_appended_to_input,
                                                       layer.mat_connections)

        # layer.switchboard.connections
        layer.num_nodes = layer.lat_mat.size // 2

        if layer.pca_node_class:
            if layer.cloneLayer:
                print("Layer L%d with " % num_layer, layer.num_nodes, " cloned PCA nodes will be created")
                # print "Warning!!! layer L%d using cloned PCA instead of several independent copies!!!"%num_layer
                layer.pca_node = layer.pca_node_class(input_dim=layer.preserve_mask_sparse.sum() +
                                                                num_features_appended_to_input,
                                                      output_dim=layer.pca_out_dim,
                                                      **layer.pca_args)  # input_dim=layer.preserve_mask_sparse.sum()
                # Create array of sfa_nodes (just one node, but cloned)
                layer.pca_layer = mdp.hinet.CloneLayer(layer.pca_node, n_nodes=layer.num_nodes)
            else:
                print("Layer L%d with " % num_layer, layer.num_nodes)
                print(" independent PCA nodes will be created, with arguments ", layer.pca_args)
                layer.PCA_nodes = list(range(layer.num_nodes))
                for i in range(layer.num_nodes):
                    layer.PCA_nodes[i] = layer.pca_node_class(input_dim=layer.preserve_mask_sparse.sum(),
                                                              output_dim=layer.pca_out_dim, **layer.pca_args)
                layer.pca_layer = mdp.hinet.Layer(layer.PCA_nodes, homogeneous=True)
        else:
            layer.pca_layer = None

        if layer.ord_node_class:
            if layer.cloneLayer:
                print("Ord_node will be created")
                print("Layer L%d with " % num_layer, layer.num_nodes, " cloned ORD nodes will be created")
                # print "Warning!!! layer L%d using cloned ORD instead of several independent copies!!!"%num_layer
                layer.ord_node = layer.ord_node_class(**layer.ord_args)
                # Create array of sfa_nodes (just one node, but cloned)
                layer.ord_layer = mdp.hinet.CloneLayer(layer.ord_node, n_nodes=layer.num_nodes)
            else:
                print("Layer L%d with " % num_layer, layer.num_nodes, " independent ORD nodes will be created")
                layer.ORD_nodes = list(range(layer.num_nodes))
                for i in range(layer.num_nodes):
                    layer.ORD_nodes[i] = layer.ord_node_class(**layer.ord_args)
                layer.ord_layer = mdp.hinet.Layer(layer.ORD_nodes, homogeneous=True)
        else:
            layer.ord_layer = None

        if layer.exp_funcs != [identity] and layer.exp_funcs:
            layer.exp_node = more_nodes.GeneralExpansionNode(layer.exp_funcs, use_hint=True, max_steady_factor=0.05,
                                                             delta_factor=0.6, min_delta=0.0001)
            layer.exp_layer = mdp.hinet.CloneLayer(layer.exp_node, n_nodes=layer.num_nodes)
        else:
            layer.exp_layer = None

        if layer.red_node_class:
            if layer.cloneLayer:
                # print "Warning!!! layer L%d using cloned RED instead of several independent copies!!!"%num_layer
                layer.red_node = layer.red_node_class(output_dim=layer.red_out_dim, **layer.red_args)
                layer.red_layer = mdp.hinet.CloneLayer(layer.red_node, n_nodes=layer.num_nodes)
            else:
                print("Layer L%d with " % num_layer, layer.num_nodes, " independent RED nodes will be created")
                layer.RED_nodes = list(range(layer.num_nodes))
                for i in range(layer.num_nodes):
                    layer.RED_nodes[i] = layer.red_node_class(output_dim=layer.red_out_dim, **layer.red_args)
                layer.red_layer = mdp.hinet.Layer(layer.RED_nodes, homogeneous=True)
        else:
            layer.red_layer = None

        if layer.clip_func or layer.clip_inv_func:
            layer.clip_node = more_nodes.PointwiseFunctionNode(layer.clip_func, layer.clip_inv_func)
        else:
            layer.clip_node = None

        if layer.sfa_node_class:
            if layer.cloneLayer:
                # print "Warning!!! layer L%d using cloned SFA instead of several independent copies!!!"%num_layer
                # sfa_node_La = mdp.nodes.SFANode(input_dim=switchboard_layer.out_channel_dim,
                # output_dim=sfa_out_dim_La)
                layer.sfa_node = layer.sfa_node_class(output_dim=layer.sfa_out_dim, **layer.sfa_args)
                # !!! aniadir el atributo output_channels al more_nodes.PInvSwitchboard
                layer.sfa_layer = mdp.hinet.CloneLayer(layer.sfa_node, n_nodes=layer.num_nodes)
            else:
                print("Layer L%d with " % num_layer, layer.num_nodes)
                print(" independent SFA nodes will be created, with arguments ", layer.sfa_args)
                layer.SFA_nodes = list(range(layer.num_nodes))
                for i in range(layer.num_nodes):
                    layer.SFA_nodes[i] = layer.sfa_node_class(output_dim=layer.sfa_out_dim, **layer.sfa_args)

                layer.sfa_layer = mdp.hinet.Layer(layer.SFA_nodes, homogeneous=True)
        else:
            layer.sfa_layer = None

        layer.node_list = ([layer.switchboard, layer.pca_layer, layer.ord_layer, layer.exp_layer, layer.red_layer,
                            layer.clip_node, layer.sfa_layer])
        
    elif isinstance(layer, system_parameters.ParamsSFASuperNode):
        # Note, there was a bug in MDP SFA Node, in which the output dimension is ignored if the input dimension
        # is unknown. See function "_set_range()".
        print("************ Creating Layer *******")
        print("Creating ParamsSFASuperNode L%d" % num_layer)
        if layer.pca_node_class:
            print("PCA_node will be created")
            layer.pca_node = layer.pca_node_class(output_dim=layer.pca_out_dim, **layer.pca_args)
        else:
            layer.pca_node = None
            # (input_dim=layer.preserve_mask_sparse.sum(), output_dim=layer.pca_out_dim, **layer.pca_args)

        if layer.ord_node_class:
            print("Ord_node will be created")
            layer.ord_node = layer.ord_node_class(**layer.ord_args)
        else:
            layer.ord_node = None

        # TODO:USE ARGUMENTS EXP_ARGS HERE?
        if layer.exp_funcs != [identity] and layer.exp_funcs:
            layer.exp_node = more_nodes.GeneralExpansionNode(layer.exp_funcs, use_hint=layer.inv_use_hint,
                                                             max_steady_factor=layer.inv_max_steady_factor,
                                                             delta_factor=layer.inv_delta_factor,
                                                             min_delta=layer.inv_min_delta)
        else:
            layer.exp_node = None

        if layer.red_node_class:
            layer.red_node = layer.red_node_class(output_dim=layer.red_out_dim, **layer.red_args)
        else:
            layer.red_node = None

        if layer.clip_func or layer.clip_inv_func:
            layer.clip_node = more_nodes.PointwiseFunctionNode(layer.clip_func, layer.clip_inv_func)
        else:
            layer.clip_node = None

        if layer.sfa_node_class:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("layer.sfa_out_dim= ", layer.sfa_out_dim)

            layer.sfa_node = layer.sfa_node_class(output_dim=layer.sfa_out_dim, **layer.sfa_args)
            print("layer.sfa_node= ", layer.sfa_node)

        layer.node_list = ([layer.pca_node, layer.ord_node, layer.exp_node, layer.red_node, layer.clip_node,
                            layer.sfa_node])

    else:
        er = "Unknown Layer type, cannot be created."
        raise Exception(er)
    return layer


def expand_iSeq_sSeq_Layer_to_Network(iSeq_set, sSeq_set, network):
    """Creates two nested lists  (of depth = 2) containing the elements in iSeq_set or sSeq_set.

    The computed lists contain the abstract training data explicitly used to train each mdp node in the described
    network. iSeq_set and sSeq_set must have the same shape. If len(iSeq_set) < num_nodes of network, iSeq_set is
    extended.
    """
    iSeq_set_exp = []
    sSeq_set_exp = []
    for i, layer in enumerate(network.layers):
        num_nodes = 0
        if isinstance(layer, system_parameters.ParamsSFASuperNode):
            if layer.pca_node_class:
                num_nodes += 1
            if layer.ord_node_class:
                num_nodes += 1
            if layer.exp_funcs != [identity] and layer.exp_funcs:
                num_nodes += 1
            if layer.red_node_class:
                num_nodes += 1
            if layer.clip_func or layer.clip_inv_func:
                num_nodes += 1
            if layer.sfa_node_class:
                num_nodes += 1
        elif isinstance(layer, system_parameters.ParamsSFALayer):
            num_nodes += 1  # For the switchboard
            if layer.pca_node_class:
                num_nodes += 1
            if layer.ord_node_class:
                num_nodes += 1
            if layer.exp_funcs != [identity] and layer.exp_funcs:
                num_nodes += 1
            if layer.red_node_class:
                num_nodes += 1
            if layer.clip_func or layer.clip_inv_func:
                num_nodes += 1
            if layer.sfa_node_class:
                num_nodes += 1
        # ## Modified to support iSeq_set, sSeqSet not lists
        # #if isinstance(iSeq_set,list):
        j = min(i, len(iSeq_set) - 1)
        # ##else:
        # ##    j = i
        for _ in range(num_nodes):
            iSeq_set_exp.append(iSeq_set[j])
            sSeq_set_exp.append(sSeq_set[j])
            # print iSeq_set_exp
    return iSeq_set_exp, sSeq_set_exp


def add_additional_features_to_connections(connections, components_per_node, original_input_dim,
                                           num_features_appended_to_input):
    """ Computes a new connections matrix, which extends a regular connection matrix by providing connections to
    an auxiliary input of size num_features_appended_to_input.

    This function also supports switchboard connections,
    where the next layer has nodes with the same input dimensionality 'components_per_node'.
    """
    orig_out_dim = len(connections)
    num_nodes = orig_out_dim // components_per_node
    final_out_dim = orig_out_dim + num_features_appended_to_input

    connections_out = numpy.zeros(num_nodes * final_out_dim)
    for i in range(num_nodes):
        connections_out[i * final_out_dim:i * final_out_dim + orig_out_dim] = connections[
                                                                              i * orig_out_dim:(i + 1) * orig_out_dim]
    for i in range(num_nodes):
        connections_out[i * final_out_dim + orig_out_dim:(i + 1) * final_out_dim] = \
            numpy.arange(original_input_dim, original_input_dim + num_features_appended_to_input)
    return connections_out
