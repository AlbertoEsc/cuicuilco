
import numpy
import scipy
import scipy.optimize
#import scipy.optimize
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import PIL
#import Image
import mdp
#import os
#import glob
#import random
##For patching MDP
#from mdp import numx
#from mdp.utils import (mult, pinv, symeig, CovarianceMatrix, QuadraticForm,
#                       SymeigException)
from mdp.utils import (mult, pinv, symeig)
#from mdp.utils import (mult, pinv, symeig, CovarianceMatrix, SymeigException)

import copy
#import time
#import SystemParameters
#import hashlib
#
import sys
##sys.path.append("/home/escalafl/workspace/hiphi/src/hiphi/utils")
##import misc
#import object_cache
#import more_nodes
from sfa_libs import select_rows_from_matrix
from inversion import invert_exp_funcs2

#using the provided average and standard deviation
def gauss_noise(x, avg, std):
    return numpy.random.normal(avg, std, x.shape)

#Zero centered
def additive_gauss_noise(x, avg, std):
    return x + numpy.random.normal(0, std, x.shape)

class RandomizedMaskNode(mdp.Node):
    """Selectively mask some components of a random variable by 
    hiding them with arbitrary noise or by removing them
    Original code contributed by Alberto Escalante, inspired by NoiseNode
    """    
    def __init__(self, remove_mask=None, noise_amount_mask=None, noise_func = gauss_noise, noise_args = (0, 1),
                 noise_mix_func = None, input_dim = None, dtype = None):
        self.remove_mask = remove_mask
 
        self.noise_amount_mask = noise_amount_mask
        self.noise_func = noise_func
        self.noise_args = noise_args
        self.noise_mix_func = noise_mix_func 
        self.seen_samples = 0
        self.x_avg = None
        self.x_std = None
        self.type=dtype

        if remove_mask != None and input_dim == None:
            input_dim = remove_mask.size
        elif remove_mask == None and input_dim != None: 
            remove_mask = numpy.zeros(input_dim) > 0.5
        elif remove_mask != None and input_dim != None:
            if remove_mask.size != input_dim:
                err = "size of remove_mask and input_dim not compatible"
                raise Exception(err)
        else:
            err = "At least one of input_dim or remove_mask should be specified"
            raise Exception(err)
        
        if noise_amount_mask is None:
            print "Signal will be only the computed noise"
            self.noise_amount_mask = numpy.ones(input_dim)
        else:
            self.noise_amount_mask = noise_amount_mask 
            
        output_dim = remove_mask.size - remove_mask.sum()
        print "Output_dim should be:", output_dim 
        super(RandomizedMaskNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
                   
    def is_trainable(self):
        return True

    def _train(self, x):
        if self.x_avg == None:
            self.x_avg = numpy.zeros(self.input_dim, dtype=self.type)
            self.x_std = numpy.zeros(self.input_dim, dtype=self.type)
        new_samples = x.shape[0]
        self.x_avg = (self.x_avg * self.seen_samples + x.sum(axis=0)) / (self.seen_samples + new_samples)
        self.x_std = (self.x_std * self.seen_samples + x.std(axis=0)*new_samples) / (self.seen_samples + new_samples)
        self.seen_samples = self.seen_samples + new_samples
        
    def is_invertible(self):
        return False

    def _execute(self, x):
        vec_noise_func = numpy.vectorize(self.noise_func)

        print "computed X_avg=", self.x_avg
        print "computed X_std=", self.x_std
        noise_mat = self.noise_func(x, self.x_avg, self.x_std)
#        noise_mat = self._refcast(self.noise_func(*self.noise_args,
#                                                  **{'size': x.shape}))
        print "Noise_amount_mask:", self.noise_amount_mask
        print "Noise_mat:", noise_mat
        noisy_signal = (1.0 - self.noise_amount_mask) * x + self.noise_amount_mask * noise_mat
        preserve_mask = self.remove_mask == False
        return noisy_signal[:, preserve_mask]

   
    
class GeneralExpansionNode(mdp.Node):
    def __init__(self, funcs, input_dim = None, dtype = None, \
                 use_pseudoinverse=True, use_hint=False, max_steady_factor=1.5, \
                 delta_factor=0.6, min_delta=0.00001):
        self.funcs = funcs
        self.use_pseudoinverse = use_pseudoinverse
        self.use_hint = use_hint
        self.max_steady_factor = max_steady_factor
        self.delta_factor = delta_factor
        self.min_delta = min_delta
        super(GeneralExpansionNode, self).__init__(input_dim, dtype)
    def expanded_dim(self, n):
        exp_dim=0
        x = numpy.zeros((1,n))
        for func in self.funcs:
            outx = func(x)
            exp_dim += outx.shape[1]
        return exp_dim
    def output_sizes(self, n):
        exp_dim=0
        sizes = numpy.zeros(len(self.funcs))
        x = numpy.zeros((1,n))
        for i, func in enumerate(self.funcs):
            outx = func(x)
            sizes[i] = outx.shape[1]
        return sizes
    def is_trainable(self):
        return False
    def _train(self, x):
        if self.input_dim is None:
            self.set_input_dim(x.shape[1])
    def is_invertible(self):
        return self.use_pseudoinverse
    def inverse(self, x, use_hint=None, max_steady_factor=None, \
                 delta_factor=None, min_delta=None):
        if self.use_pseudoinverse is False:
            ex = "Inversion not activated"
            raise Exception(ex)
        if use_hint is None:
            use_hint = self.use_hint
        if max_steady_factor is None:
            max_steady_factor = self.max_steady_factor
        if delta_factor is None:
            delta_factor  = self.delta_factor    
        if min_delta is None:
            min_delta = self.min_delta

#        print "Noisy pre = ", x, "****************************************************"
        app_x_2, app_ex_x_2 = invert_exp_funcs2(x, self.input_dim, self.funcs, use_hint=use_hint, max_steady_factor=max_steady_factor, delta_factor=delta_factor, min_delta=min_delta)
#        print "Noisy post = ", x, "****************************************************"
        return app_x_2
    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = self.expanded_dim(n)

    def _execute(self, x):
        if self.input_dim is None:
            self.set_input_dim(x.shape[1]) 

        num_samples = x.shape[0]
#        output_dim = expanded_dim(self.input_dim)
        sizes = self.output_sizes(self.input_dim)

        out = numpy.zeros((num_samples, self.output_dim))

        current_pos = 0
        for i, func in enumerate(self.funcs):
            out[:,current_pos:current_pos+sizes[i]] = func(x)
            current_pos += sizes[i]
        return out
    
    
#applies a function to the whole input, possible with an inverse
class PointwiseFunctionNode(mdp.Node):
    def __init__(self, func, inv_func, input_dim = None, dtype = None):
        self.func = func
        self.inv_func = inv_func
#        self.row_func = lambda x: row_func(func, x)
#        self.row_inv_func = lambda x: row_func(inv_func, x)  
#        self.max_in_amplitude = max_in_amplitude
#        self.max_out_amplitude = max_out_amplitude
        super(PointwiseFunctionNode, self).__init__(input_dim, dtype)
        
    def is_trainable(self):
        return False

#MEGAWARNING!!!!!!!!!!!!!!!!!
    def is_invertible(self):
        if self.inv_func == None:
            return True
        else:
            return True

    def inverse(self, x):
        if self.inv_func != None:
            return self.inv_func(x)
        else:
            return x
#        return numpy.array(map(self.row_inv_func, x))

    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = n

    def _execute(self, x):
        if self.input_dim is None:
            self.set_input_dim(x.shape[1]) 
        if self.func != None:
            return self.func(x)
        else:
            return x
#        return numpy.array(map(self.row_func, x))

    
    
    
#Note: generalize with other functions!!!
#Careful with the sign used!!!
class BinomialAbsoluteExpansionNode(mdp.Node):
#        mask_x_coordinates = mask_indices % x_in_channels
    def expanded_dim(self, n):
        return n + n*(n+1)/2
    def is_trainable(self):
        return False
    def is_invertible(self, x):
        return False
    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = self.expanded_dim(n)
    def _execute(self, x):
        out = numpy.concatenate((x, pairwise_expansion(x, abs_sum)), axis=1)
        return out



class PInvSwitchboard(mdp.hinet.Switchboard):
    def __init__(self, input_dim, connections, slow_inv=False, type_inverse="average"):
        super(PInvSwitchboard, self).__init__(input_dim=input_dim, connections=connections)
        self.pinv = None
        self.mat2 = None
        self.slow_inv = slow_inv
        self.type_inverse = type_inverse
        self.output_dim = connections.size
        if self.inverse_connections == None and self.slow_inv == False:
            index_array = numpy.argsort(connections)
            value_array = connections[index_array]
        
            value_range = numpy.zeros((input_dim, 2))
            self.inverse_indices = range(input_dim)
            for i in range(input_dim):
                value_range[i] = numpy.searchsorted(value_array, [i-0.5, i+0.5])
                if value_range[i][1] == value_range[i][0]:
                    self.inverse_indices[i]=[]
                else:
                    self.inverse_indices[i]= index_array[ value_range[i][0]: value_range[i][1] ]
        elif self.inverse_connections == None and self.slow_inv == True:
            print "warning using slow inversion in PInvSwitchboard!!!"
#find input variables not used by connections:
            used_inputs = numpy.unique(connections)
            used_inputs_set = set(used_inputs)
            all_inputs_set = set(range(input_dim))
            unused_inputs_set = all_inputs_set - all_inputs_set.intersection(used_inputs_set)
            unused_inputs = list(unused_inputs_set)
            self.num_unused_inputs = len(unused_inputs)
#extend connections array
            ext_connections = numpy.concatenate((connections, unused_inputs))
#create connections matrix
            mat_height = len(connections) + len(unused_inputs)
            mat_width =  input_dim
            mat = numpy.zeros((mat_height,mat_width)) 
#fill connections matrix
            for i in range(len(connections)):
                mat[i,connections[i]] = 1
#            
            for i in range(len(unused_inputs)):
                mat[i+len(connections), unused_inputs[i]] = 1
#    
            print "extended matrix is:"
            print mat
#compute pseudoinverse
            mat2 = numpy.matrix(mat)
            self.mat2 = mat2
            self.pinv = (mat2.T * mat2).I * mat2.T
        else:
            print "Inverse connections already given, in PInvSwitchboard"
            
#
#PInvSwitchboard is always invertible                
    def is_invertible(self):
        return True
#
#If true inverse is present, just use it, otherwise compute it by means of the pseudoinverse                    
    def _inverse(self, x):
        if self.inverse_connections == None and self.slow_inv == False:
            height_x = x.shape[0]
            mat2 = numpy.zeros((height_x, self.input_dim))
            for row in range(height_x):
                x_row = x[row]
                for i in range(self.input_dim):
                    elements = x_row[self.inverse_indices[i]] 
                    if self.type_inverse == "average":
                        if elements.size > 0:
                            mat2[row][i] = elements.mean()
                    else:
                        err = "self.type_inverse not supported: " + self.type_inverse
                        raise Exception(err)
            return mat2
        elif self.inverse_connections == None and self.slow_inv == True:
            print "x=", x
            height_x = x.shape[0]
            full_x = numpy.concatenate((x, 255*numpy.ones((height_x, self.num_unused_inputs))), axis=1)
            data2 = numpy.matrix(full_x)
            print "data2=", data2
            print "PINV=", self.pinv
            return (self.pinv * data2.T).T
        else:
#            return apply_permutation_to_signal(x, self.inverse_connections, self.input_dim)
            return select_rows_from_matrix(x, self.inverse_connections)
        
    
#Random Permutation Node
#Randomly permutes the components of the input signal in a constant way after initialization
class RandomPermutationNode(mdp.Node):
    def __init__(self, input_dim = None, output_dim=None, dtype = None, verbose=False):
        super(RandomPermutationNode, self).__init__(input_dim, output_dim, dtype)
        self.permutation = None
        self.inv_permutation = None
        self.dummy = 5 #without it the hash fails!!!!! neiiinnn!!! wie kann das sein!!!!! verbessert die hash du!!!!
        
    def is_trainable(self):
        return True

    def is_invertible(self):
        return True
 
    def inverse(self, x):
        return select_rows_from_matrix(x, self.inv_permutation)

#    def localized_inverse(self, xf, yf, y):
#        return y[:, self.inv_permutation]

    def _set_input_dim(self, n, verbose=False):
        if verbose:
            print "!!!!!!!!!!!!!! RandomPermutationNode: Setting input_dim to ", n
        self._input_dim = n
        self._output_dim = n

    def _train(self, x, verbose=True):
        n=x.shape[1]

        if self.input_dim == None:
            self.set_input_dim(n)
        
        if self.input_dim == None:
            print "!!!!!!!!!!! *******Really Setting input_dim to ", n           
            self.input_dim = n

        if self.output_dim == None:
            print "*******Really Setting output_dim to ", n           
            self.output_dim = n

        if self.permutation == None:
            if verbose:
                print "Creating new random permutation"
                print "Permutation=", self.permutation
                print "x=", x, "with shape", x.shape
                print "Input dim is: ", self.input_dim
            self.permutation = numpy.random.permutation(range(self.input_dim))
            self.inv_permutation = numpy.zeros(self.input_dim, dtype="int")
            self.inv_permutation[self.permutation] = numpy.arange(self.input_dim)
            if verbose:
                print "Permutation=", self.permutation
                print "Output dim is: ", self.output_dim

    def _execute(self, x, verbose=False):
#        print "RandomPermutationNode: About to excecute, with input x= ", x
        y = select_rows_from_matrix(x, self.permutation)
        if verbose:
            print "Output shape is = ", y.shape, 
        return y



def sfa_pretty_coefficients(sfa_node, transf_training):
    count = 0
    for i in range(transf_training.shape[1]):
        if transf_training[0,i] + transf_training[1,i] + transf_training[2,i] + transf_training[3,i] + \
           transf_training[4,i] + transf_training[5,i] + transf_training[6,i] + transf_training[7,i] + \
           transf_training[8,i] + transf_training[9,i] + transf_training[10,i] + transf_training[11,i] > 0:
            sfa_node.sf[:,i] = (sfa_node.sf[:,i] * -1)
            transf_training[:,i] = (transf_training[:,i] * -1)
            count +=1
    print "Polarization of %d SFA Signals Corrected!!!\n"%count,
    sfa_node._bias = mdp.utils.mult(sfa_node.avg, sfa_node.sf)
    print "Bias updated"
    return transf_training



def describe_flow(flow):
    length = len(flow)
    
    total_size = 0
    print "Flow has %d nodes"%length
    for i in range(length):
        node = flow[i]
        node_size = compute_node_size(node)
        total_size += node_size
        print i
        print str(node)
        print node.input_dim
        print node.output_dim
        print node_size

        print "Node[%d] is %s, has input_dim=%d, output_dim=%d and size=%d"%(i, str(node), node.input_dim, node.output_dim, node_size)
        if isinstance(node, mdp.hinet.CloneLayer):
            print "   contains %d cloned nodes of type %s, each with input_dim=%d, output_dim=%d"%(len(node.nodes), str(node.nodes[0]), node.nodes[0].input_dim, node.nodes[0].output_dim)
            
        elif isinstance(node, mdp.hinet.Layer):
            print "   contains %d nodes of type %s, each with input_dim=%d, output_dim=%d"%(len(node.nodes), str(node.nodes[0]), node.nodes[0].input_dim, node.nodes[0].output_dim)
    print "Total flow size: %d"%total_size
    print "Largest node size: %d"%compute_largest_node_size(flow)


#modes: "FirstNodeInLayer", "Average", "All"
def display_eigenvalues(flow, mode="All"):
    length = len(flow)
    
    print "Displaying eigenvalues of SFA Nodes in flow of length",length

    for i in range(length):
        node = flow[i]
        if isinstance(node, mdp.hinet.CloneLayer):
            if isinstance(node.nodes[0], mdp.nodes.SFANode):
                print "Node %d is a CloneLayer that contains an SFANode with d="%i, node.nodes[0].d
            elif isinstance(node.nodes[0], mdp.nodes.IEVMNode):
                if node.nodes[0].use_sfa:
                    print "Node %d is a CloneLayer that contains an IEVMNode containing an SFA node with num_sfa_features_preserved=%d and d="%(i, node.nodes[0].num_sfa_features_preserved), node.nodes[0].sfa_node.d
            elif isinstance(node.nodes[0], mdp.nodes.IEVMLRecNode):
                print "Node %d is a CloneLayer that contains an IEVMLRecNode containing an SFA node with num_sfa_features_preserved=%d and d="%(i, node.nodes[0].num_sfa_features_preserved), node.nodes[0].sfa_node.d

        elif isinstance(node, mdp.hinet.Layer):
            if isinstance(node.nodes[0], mdp.nodes.SFANode):
                if mode == "Average":
                    out = 0.0
                    for n in node.nodes:
                        out += n.d
                    print "Node %d is a Layer that contains SFANodes with avg(d)="%i, out / len(node.nodes)
                elif mode == "All":
                    for n in node.nodes:
                        print "Node %d is a Layer that contains an SFANode with d="%i, n.d
                elif mode == "FirstNodeInLayer":
                    print "Node %d is a Layer, and its first SFANode has d="%i, node.nodes[0].d
                else:
                    er = 'Unknown mode in display_eigenvalues, try "FirstNodeInLayer", "Average" or "All"'
                    raise Exception(er)
            elif isinstance(node.nodes[0], mdp.nodes.IEVMNode) and (node.nodes[0].use_sfa or node.nodes[0].out_sfa_filter):
                use_sfa = node.nodes[0].use_sfa
                out_sfa_filter = node.nodes[0].out_sfa_filter
                if mode == "Average":
                    out = 0.0
                    num_sfa_features = 0.0
                    filter_out = 0.0
                    for n in node.nodes:
                        if use_sfa:
                            out += n.sfa_node.d
                            num_sfa_features += n.num_sfa_features_preserved
                        if out_sfa_filter:
                            filter_out += n.out_sfa_node.d
                    print "Node %d is a Layer that contains IEVMNodes containing SFANodes with avg(num_sfa_features_preserved)=%f and avg(d)="%(i, node.nodes[0].num_sfa_features_preserved),  out / len(node.nodes)
                    if out_sfa_filter:
                        print "and output SFA filter with avg(out_sfa_node.d)=", filter_out / len(node.nodes)

                elif mode == "All":
                    for n in node.nodes:
                        print "Node %d is a Layer that contains an IEVMNode"%i,
                        if use_sfa: 
                            print "containing an SFANode with num_sfa_features_preserved=%f and d="%n.num_sfa_features_preserved, n.sfa_node.d
                        if out_sfa_filter:
                            print "and output SFA filter with out_sfa_node.d=", n.out_sfa_node.d

                elif mode == "FirstNodeInLayer":
                    print "Node %d is a Layer, and its first IEVMNode"%i
                    if use_sfa:
                        print "contains an SFANode with num_sfa_features_preserved)=%f and d="%node.nodes[0].num_sfa_features_preserved, node.nodes[0].sfa_node.d
                    if out_sfa_filter:
                        print "and Output SFA filter with out_sfa_node.d=", node.nodes[0].out_sfa_node.d   
                else:
                    er = 'Unknown mode in display_eigenvalues, try "FirstNodeInLayer", "Average" or "All"'
                    raise Exception(er)
            elif isinstance(node.nodes[0], mdp.nodes.IEVMLRecNode):
                if mode == "Average":
                    evar_avg = 0.0
                    d_avg = 0.0
                    avg_num_sfa_features = 0.0
                    for n in node.nodes:
                        d_avg += n.sfa_node.d
                        evar_avg += n.evar 
                        avg_num_sfa_features += n.num_sfa_features_preserved
                    d_avg /=  len(node.nodes)
                    evar_avg /=  len(node.nodes)
                    avg_num_sfa_features /=  len(node.nodes)
                    print "Node %d is a Layer that contains IEVMLRecNodes containing SFANodes with avg(num_sfa_features_preserved)=%f and avg(d)=%s and avg(evar)=%f"%(i,avg_num_sfa_features, str(d_avg), evar_avg)
                elif mode == "All":
                    print "Node %d is a Layer that contains IEVMLRecNodes:"%i
                    for n in node.nodes:
                        print "  IEVMLRecNode containing an SFANode with num_sfa_features_preserved=%f, d=%s and evar=%f"%(n.num_sfa_features_preserved, str(n.sfa_node.d), n.evar)
                elif mode == "FirstNodeInLayer":
                    print "Node %d is a Layer, and its first IEVMLRecNode"%i
                    print "contains an SFANode with num_sfa_features_preserved)=%f, d=%s and evar=%f"%(node.nodes[0].num_sfa_features_preserved, str(node.nodes[0].sfa_node.d), node.nodes[0].evar)
                else:
                    er = 'Unknown mode in display_eigenvalues, try "FirstNodeInLayer", "Average" or "All"'
                    raise Exception(er)
            elif isinstance(node.nodes[0], mdp.nodes.SFAAdaptiveNLNode):
                if mode == "Average":
                    out = 0.0
                    for n in node.nodes:
                        out += n.sfa_node.d
                    print "Node %d is a Layer that contains SFAAdaptiveNLNodes containing SFANodes with avg(d)="%i,  out / len(node.nodes)
                elif mode == "All":
                    for n in node.nodes:
                        print "Node %d is a Layer that contains an SFAAdaptiveNLNode"%i,
                        print "containing an SFANode with d=", n.sfa_node.d
                elif mode == "FirstNodeInLayer":
                    print "Node %d is a Layer, and its first SFAAdaptiveNLNode"%i
                    print "contains an SFANode with d=", node.nodes[0].sfa_node.d
                else:
                    er = 'Unknown mode in display_eigenvalues, try "FirstNodeInLayer", "Average" or "All"'
                    raise Exception(er)                    


def compute_node_size(node):
    """ Computes the number of weights learned by node after training.
    Note: Means are not counted, only weights.
    The following nodes are supported up to now:
    SFANode,PCANode,WhitheningNode,CloneLayer,Layer
    """
    if isinstance(node, (mdp.nodes.SFANode, mdp.nodes.PCANode, mdp.nodes.WhiteningNode)) and node.input_dim != None and node.output_dim != None:
        if isinstance(node, mdp.nodes.SFANode):
            return node.input_dim * node.output_dim
        elif isinstance(node, mdp.nodes.PCANode):
            return node.input_dim * node.output_dim
        elif isinstance(node, mdp.nodes.WhiteningNode):
            return node.input_dim * node.output_dim
        else:
            er = "Unknown Node class"
            raise Exception(er)
    elif isinstance(node, mdp.hinet.CloneLayer):
            return compute_node_size(node.nodes[0])
    elif isinstance(node, mdp.hinet.Layer):
            size = 0
            for nodechild in node.nodes:
                size += compute_node_size(nodechild)
            return size
    else:
        return 0

def compute_flow_size(flow):
    """ Computes the number of weights learned by the whole flow after training.
    See compute_node_size for more details on the counting procedure
    """
    flow_size = 0
    for node in flow:
        flow_size += compute_node_size(node)
    return flow_size

def compute_largest_node_size(flow):
    """ Computes the larger number of weights learned by a node after training.
    See compute_node_size for more details on the counting procedure
    """
    largest_size = 0
    for node in flow:
        if (isinstance(node, mdp.nodes.SFANode) or isinstance(node, mdp.nodes.PCANode) or
                isinstance(node, mdp.nodes.WhiteningNode)):
            current_size = compute_node_size(node)
        elif isinstance(node, mdp.hinet.CloneLayer):
            current_size = compute_node_size(node.nodes[0])
        elif isinstance(node, mdp.hinet.Layer):
            current_size = 0
            for nodechild in node.nodes:
                tmp_size = compute_node_size(nodechild)
                if tmp_size > current_size:
                    current_size = tmp_size
        else:
            current_size = 0
        if current_size > largest_size:
            largest_size = current_size
        
    return largest_size

    
#Used to compare the effectiveness of several PCA Networks
def estimated_explained_variance(images, flow, sl_images, num_considered_images=100, verbose=True):
    exp_var=0
    num_images = images.shape[0]
    im_numbers = numpy.random.randint(num_images, size=num_considered_images)

    avg_image = images[im_numbers].mean(axis = 0)
    
    selected_images = images[im_numbers]
    ori_differences = selected_images - avg_image
    ori_energies = ori_differences ** 2
    ori_energy = ori_energies.sum()

    sl_selected_images = sl_images[im_numbers]
    print "sl_selected_images.shape=", sl_selected_images.shape
    inverses = flow.inverse(sl_selected_images)
    rec_differences = inverses - avg_image
    rec_energies = rec_differences ** 2
    rec_energy = rec_energies.sum()

    rec_errors = selected_images - inverses
    rec_error_energies = rec_errors ** 2
    rec_error_energy = rec_error_energies.sum()
    
    if verbose:
        explained_individual =  rec_energies.sum(axis=1)/ori_energies.sum(axis=1)
        print "Individual explained variances: ", explained_individual
        print "Which, itself has standar deviation: ", explained_individual.std()
        print "Therefore, estimated explained variance has std of about: ", explained_individual.std()/numpy.sqrt(num_considered_images)
        print "rec_error_energy/ori_energy=", rec_error_energy/ori_energy
        print "Thus explained variance about:", 1-rec_error_energy/ori_energy
    return rec_energy/ori_energy
    




class HeadNode(mdp.Node):
    """Preserve only the first k dimensions from the data
    """    
    def __init__(self, input_dim = None, output_dim=None, dtype = None):
        self.type=dtype
        super(HeadNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
                   
    def is_trainable(self):
        return True

    def _train(self, x):
        pass
    
    def _is_invertible(self):
        return True

    def _execute(self, x):
        if self.output_dim == None:
            er = "Warning 12345..."
            raise Exception(er)
        return x[:, 0:self.output_dim]
    
    def _stop_training(self):
        pass 

    def _inverse(self, y):
        num_samples, out_dim = y.shape[0], y.shape[1]
        zz = numpy.zeros((num_samples, self.input_dim - out_dim))
        return numpy.concatenate((y, zz), axis=1)



class SFAPCANode(mdp.Node):
    """Node that extracts slow features unless their delta value is too high. In such a case PCA features are extracted.
    """    
    def __init__(self, input_dim = None, output_dim=None, max_delta=1.95, sfa_args={}, pca_args={}, **argv ):
        super(SFAPCANode, self).__init__(input_dim =input_dim, output_dim=output_dim, **argv) 
        self.sfa_node = mdp.nodes.SFANode(**sfa_args)
        self.max_delta = max_delta #max delta value allowed for a slow feature, otherwise a principal component is extracted
        self.avg = None #input average
        self.W = None #weights for complete transformation 
        self.pinv = None #weights for pseudoinverse of complete transformation
    
    def is_trainable(self):
        return True

    def _train(self, x, **argv):
        self.sfa_node.train(x, **argv)
        
    def _is_invertible(self):
        return True

    def _execute(self, x):
        W = self.W
        avg = self.avg
        return numpy.dot(x-avg, W) 
    
    def _stop_training(self, **argv):
        #New GraphSFA node
        if "_covdcovmtx" in dir(self.sfa_node): 
            #Warning, fix is computed twice. TODO: avoid double computation
            C, self.avg, CD  = self.sfa_node._covdcovmtx.fix()      
        else:
            #Old fix destroys data... so we copy the matrices first.
            cov_mtx = copy.deepcopy(self.sfa_node._cov_mtx)
            dcov_mtx = copy.deepcopy(self.sfa_node._dcov_mtx)

            C, self.avg, tlen = cov_mtx.fix()
            DC, davg, dtlen = dcov_mtx.fix()
            
        dim = C.shape[0]
        type_ = C.dtype
        self.sfa_node.stop_training()
        d = self.sfa_node.d
        sfa_output_dim = len(d[d<=self.max_delta])
        sfa_output_dim = min(sfa_output_dim, self.output_dim)
        print "sfa_output_dim=", sfa_output_dim

        Wsfa = self.sfa_node.sf[:,0:sfa_output_dim]
        print "Wsfa.shape=", Wsfa.shape
        if Wsfa.shape[1] == 0: #No slow components will be used
            print "No Psfa created"
            PS = numpy.zeros((dim,dim), dtype=type_)
        else:
            Psfa = pinv(Wsfa)
            print "Psfa.shape=", Psfa.shape
            PS = numpy.dot(Wsfa, Psfa)
        
        print "PS.shape=", PS.shape
        Cproy =  numpy.dot(PS, numpy.dot(C, PS.T))
        Cpca = C - Cproy

        if self.output_dim == None:
            self.output_dim = dim
            
        pca_output_dim = self.output_dim - sfa_output_dim
        print "PCA output_dim=", pca_output_dim
        if pca_output_dim > 0:
            pca_node = mdp.nodes.PCANode(output_dim = pca_output_dim) #WARNING: WhiteningNode should be used here
            pca_node._cov_mtx._dtype = type_ 
            pca_node._cov_mtx._input_dim = dim
            pca_node._cov_mtx._avg = numpy.zeros(dim, type_)
            pca_node._cov_mtx.bias = True
            pca_node._cov_mtx._tlen = 1 #WARNING!!! 1
            pca_node._cov_mtx._cov_mtx = Cpca
            pca_node._input_dim = dim
            pca_node._train_phase_started = True
            pca_node.stop_training()
            print "pca_node.d=", pca_node.d
            print "1000000 * pca_node.d[0]=", 1000000*pca_node.d[0]
            #quit()
            Wpca = pca_node.v
            Ppca = pca_node.v.T
        else:
            Wpca = numpy.array([]).reshape((dim,0))
            Ppca = numpy.array([]).reshape((0,dim))
            
        print "Wpca.shape=", Wpca.shape
        print "Ppca.shape=", Ppca.shape
        
        self.W = numpy.concatenate((Wsfa, Wpca),axis=1)
        self.pinv = None # WARNING, why this does not work correctly: numpy.concatenate((Psfa, Ppca),axis=0) ?????       
#        print "Pinv 1=", self.pinv
#        print "Pinv 2-Pinv1=", pinv(self.W)-self.pinv
        print "W.shape=", self.W.shape
#        print "pinv.shape=", self.pinv.shape
        print "avg.shape=", self.avg.shape
    def _inverse(self, y):
        if self.pinv is None:
            print "Computing PINV",
            self.pinv = pinv(self.W)
        return numpy.dot(y, self.pinv) + self.avg

#Computes the variance of some MDP data array
def data_variance(x):
    return ((x-x.mean(axis=0))**2).sum(axis=1).mean()

def estimated_explained_var_linearly(x, y, x_test, y_test):
    x_test_app = approximate_linearly(x, y, y_test)
    
    explained_variance = compute_explained_var(x_test, x_test_app)

    x_variance = data_variance(x_test)
    print "x_variance=",x_variance,", explained_variance=",  explained_variance
    return explained_variance/x_variance

def approximate_linearly(x, y, y_test):
    lr_node = mdp.nodes.LinearRegressionNode()
    lr_node.train(y,x)
    lr_node.stop_training()
   
    x_test_app = lr_node.execute(y_test)
    return x_test_app

#Approximates x from y, and computes how sensitive the estimation is to changes in y
def sensivity_of_linearly_approximation(x, y):
    lr_node = mdp.nodes.LinearRegressionNode()
    lr_node.train(y,x)
    lr_node.stop_training()
    beta = lr_node.beta[1:,:] # bias is used by default, we do not need to consider it

    print "beta.shape=", beta.shape
    sens = (beta**2).sum(axis=1)
    return sens

def estimated_explained_var_with_kNN(x, y, max_num_samples_for_ev = None, max_test_samples_for_ev=None, k=1, ignore_closest_match = False, operation="average"):
    num_samples = x.shape[0]
    indices_all_x = numpy.arange(x.shape[0])

    if max_num_samples_for_ev != None: #use all samples for reconstruction
        max_num_samples_for_ev = min(max_num_samples_for_ev, num_samples)
        indices_all_x_selection = indices_all_x +0
        numpy.random.shuffle(indices_all_x_selection)
        indices_all_x_selection = indices_all_x_selection[0:max_num_samples_for_ev]
        x_sel = x[indices_all_x_selection]
        y_sel = y[indices_all_x_selection]
    else:
        x_sel = x
        y_sel = y

    if max_test_samples_for_ev != None: #use all samples for reconstruction
        max_test_samples_for_ev = min(max_test_samples_for_ev, num_samples)
        indices_all_x_selection = indices_all_x +0
        numpy.random.shuffle(indices_all_x_selection)
        indices_all_x_selection = indices_all_x_selection[0:max_test_samples_for_ev]
        x_test = x[indices_all_x_selection]
        y_test = y[indices_all_x_selection]
    else:
        x_test = x
        y_test = y

    x_app_test = approximate_kNN_op(x_sel, y_sel, y_test,k, ignore_closest_match, operation=operation)
    print "x_test=",x_test
    print "x_app_test=",x_app_test
    
    explained_variance = compute_explained_var(x_test, x_app_test)
    test_variance = data_variance(x_test)

    print "explained_variance=", explained_variance
    print "test_variance=", test_variance

    return explained_variance / test_variance


def random_subindices(num_indices, size_selection):
    if size_selection > num_indices:
        ex = "Error, size_selection is larger than num_indices! (", size_selection, ">", num_indices, ")"
        raise Exception(ex)
    all_indices = numpy.arange(num_indices)
    numpy.random.shuffle(all_indices)
    return all_indices[0:size_selection] + 0

#Function that computes how much variance is explained linearly from a global mapping
#Linear regression is trained with sl_seq_training and subimages_train
#Estimation is done on subset of size number_samples_EV_linear_global from training and test data
#For training data evaluation is done on the same data used to train LR, and on new random subset of data.
#For test data all samples are used.
def estimate_explained_var_linear_global(subimages_train, sl_seq_training, subimages_newid, sl_seq_newid, reg_num_signals, number_samples_EV_linear_global):
    indices_all_train1 = random_subindices(subimages_train.shape[0], number_samples_EV_linear_global)
    indices_all_train2 = random_subindices(subimages_train.shape[0], number_samples_EV_linear_global)
    indices_all_newid = numpy.arange(subimages_newid.shape[0])
    
    lr_node = mdp.nodes.LinearRegressionNode()
    sl_seq_training_sel1 = sl_seq_training[indices_all_train1, 0:reg_num_signals]
    subimages_train_sel1 = subimages_train[indices_all_train1]
    lr_node.train(sl_seq_training_sel1, subimages_train_sel1) #Notice that the input "x"=n_sfa_x and the output to learn is "y" = x_pca
    lr_node.stop_training()  

    subimages_train_app1 = lr_node.execute(sl_seq_training_sel1)
    EVLinGlobal_train1 = compute_explained_var(subimages_train_sel1, subimages_train_app1)
    data_variance_train1 = data_variance(subimages_train_sel1)

    sl_seq_training_sel2 = sl_seq_training[indices_all_train2, 0:reg_num_signals]
    subimages_train_sel2 = subimages_train[indices_all_train2]
    subimages_train_app2 = lr_node.execute(sl_seq_training_sel2)
    EVLinGlobal_train2 = compute_explained_var(subimages_train_sel2, subimages_train_app2)
    data_variance_train2 = data_variance(subimages_train_sel2)

    sl_seq_newid_sel = sl_seq_newid[indices_all_newid, 0:reg_num_signals]
    subimages_newid_sel = subimages_newid[indices_all_newid]
    subimages_newid_app = lr_node.execute(sl_seq_newid_sel)
    EVLinGlobal_newid = compute_explained_var(subimages_newid_sel, subimages_newid_app)
    data_variance_newid = data_variance(subimages_newid_sel)

    print "Data variances=", data_variance_train1, data_variance_train2, data_variance_newid
    print "EVLinGlobal=", EVLinGlobal_train1, EVLinGlobal_train2, EVLinGlobal_newid
    return EVLinGlobal_train1/data_variance_train1, EVLinGlobal_train2/data_variance_train2, EVLinGlobal_newid/data_variance_newid

# Computes the explained variance provided by the approximation to some data, with respect to the true data
# Additionally, the original data variance is provided
# app = true + error
# exp_var ~ energy(true) - energy(error)
# (OLD) Returns also the true energy of the samples
def compute_explained_var(true_samples, approximated_samples):
    #print true_samples.shape
    #print approximated_samples.shape
    #print "MMM1="
    #print approximated_samples
    error =  (approximated_samples - true_samples)
    error_energy = (error**2.0).sum(axis=1).mean() #average squared error per sample
    true_energy =  data_variance(true_samples) # (true_samples-true_samples.mean(axis=0)).var()
#    approximation_energy =  data_variance(approximated_samples) #(approximated_samples-approximated_samples.mean(axis=0)).var()
    explained_var = true_energy - error_energy
    #print "Infor:", error_energy, true_energy
    return explained_var

#Approximates a signal y given its expansion y_exp. The method is kNN with training data given by x, x_exp
#If label_avg=True, the inputs of the k closest expansions are averaged, otherwise the most frequent among k-closest is returned
#When label_avg=True, one can also specify to ignore the best match (useful if y_exp = x_exp)
def approximate_kNN_op(x, x_exp, y_exp,k=1, ignore_closest_match = False, operation=None):   
    n = mdp.nodes.KNNClassifier(k=k, execute_method="label")   
    n.train(x_exp, range(len(x_exp)))

    if operation=="average":
        n.stop_training()
        ii = n.klabels(y_exp)
        #print "ii=",ii
        #print type(ii)
        if ignore_closest_match and k==1:
            ex = "Error, k==1 but ignoring closest match!"
            raise Exception(ex)
        elif ignore_closest_match:
            ii = ii[:,1:]
        y = x[ii].mean(axis=1)

        y_exp_app = x_exp[ii].mean(axis=1)
        #print "Error for y_exp is:", ((y_exp_app - y_exp)**2).sum(axis=1).mean()

        #print "y=",y
        return y #x[ii].mean(axis=1)
    elif operation=="lin_app":
        n.stop_training()
        ii = n.klabels(y_exp)
#        print "ii=",ii
#        print type(ii)
        if ignore_closest_match and k==1:
            ex = "Error, k==1 but ignoring closest match!"
            raise Exception(ex)
        elif ignore_closest_match:
            ii = ii[:,1:]

        x_dim = x.shape[1]
        x_exp_dim = x_exp.shape[1]

        x_mean = x.mean(axis=0)
        x = x-x_mean
    
        nk = ii.shape[1]
        y = numpy.zeros((len(y_exp), x_dim))
        y_exp_app = numpy.zeros((len(y_exp), x_exp_dim))
        x_ind = x[ii]        
        x_exp_ind = x_exp[ii]        
        y_expit = numpy.zeros((x_exp_dim+1, 1))
        k = 1.0e10 #make larger to force sum closer to one?!
        y_expit[x_exp_dim,0] =  0.0 * 1.0 * k
        x_expit = numpy.zeros((x_exp_dim+1, nk))
        x_expit[x_exp_dim,:] = 1.0 * k#
        zero_threshold = -40.0500 #-0.004
        max_zero_weights = nk/5
        
        w_0 = numpy.ones((nk, 1)) * 1.0 / nk
        #print "w_0", w_0
        for i in range(len(y_exp)):            
            negative_weights = 0
            iterate = True
            #print "Iteration: ", i,
            x_expit[0:x_exp_dim,:] = x_exp_ind[i].T
            y_0 = numpy.dot(x_exp_ind[i].T, w_0)
            
            fixing_zero_threshold = zero_threshold * 500
            while iterate:
                #print x_exp_ind[i].T.shape
                #print x_expit[0:x_exp_dim,:].shape                
                x_pinv = numpy.linalg.pinv(x_expit)
                #print y_0.shape, y_exp[i].shape
                y_expit[0:x_exp_dim,0] = y_exp[i]-y_0.flatten()
                w_i = numpy.dot(x_pinv,y_expit)+w_0
                
                iterate = False
                if (w_i< zero_threshold).any():
                    #print "w_i[:,0] =", w_i[:,0]
                    #print "x_expit = ", x_expit
                    negative_weights += (w_i<fixing_zero_threshold).sum()
                    negative_elements = numpy.arange(nk)[w_i[:,0]<fixing_zero_threshold]
                    numpy.random.shuffle(negative_elements)
                    for nn in negative_elements:
                        #print "nn=", nn
                        x_expit[0:x_exp_dim+1, nn] = 0.0
                    #print "negative_elements", negative_elements
                    iterate = True
                fixing_zero_threshold /= 2 
                if negative_weights >= max_zero_weights:
                    iterate = False 

                #FORCE SUM WEIGHTS=1: 
                #print "w_i[:,0] =", w_i[:,0]
                #print "weight sum=",w_i.sum(),"min_weight=",w_i.min(),"max_weight=",w_i.max(), "negative weights=", negative_weights
                w_i /= w_i.sum()
     #           print "y[i].shape", y[i].shape
     #           print "as.shape", numpy.dot(x_ind[i].T, w_i).T.shape
                y[i] = numpy.dot(x_ind[i].T, w_i).T + x_mean# numpy.dot(w_i, x_ind[i]).T
                y_exp_app[i] = numpy.dot(x_exp_ind[i].T, w_i).T
            
            if w_i.min() < zero_threshold: #0.1: #negative_weights >= max_zero_weights:
                #quit()max_zero_weights
                print "Warning smallest weight is", w_i.min(), "thus replacing with simple average"
                #print "Warning, at least %d all weights turned out to be negative! (%d)"%(max_zero_weights, negative_weights)
                #print x_ind[i]
                #print x_ind[i].shape
                y[i] = x_ind[i].mean(axis=0)
        print ".",
        #print "Error for y_exp is:", ((y_exp_app - y_exp)**2).sum(axis=1).mean()
        #print "y=",y
        return y #x[ii].mean(axis=1)
    elif operation=="plainKNN":
        ii = n.execute(y_exp)
        ret = x[ii]
        #print "len(ii)=", len(ii)
        return ret
    else:
        er = "operation unknown:", operation
        raise Exception(er)
    
def approximate_kNN(x, x_exp, y_exp,k=1, ignore_closest_match = False, label_avg=True):
    n = mdp.nodes.KNNClassifier(k=k, execute_method="label")   
    n.train(x_exp, range(len(x_exp)))

    if label_avg:
        n.stop_training()
        ii = n.klabels(y_exp)
        #print "ii=",ii
        if ignore_closest_match and k==1:
            ex = "Error, k==1 but ignoring closest match!"
            raise Exception(ex)
        elif ignore_closest_match:
            ii = ii[:,1:]
        y = x[ii].mean(axis=1)
        #print "y=",y
        return y #x[ii].mean(axis=1)
    else:
        ii = n.execute(y_exp)
        ret = x[ii]
    #print "len(ii)=", len(ii)
    return ret

#Third ranking method more robust and closer to max EV(x; y_i + Y)-EV(x;Y) for all Y, EV computed linearly.  
#Ordering and scoring of signals respects principle of best incremental feature selection
#Computes a scores vector that measures the importance of each expanded component at reconstructing a signal 
#x, x_exp are training data, y and y_exp are test data
#At most max_comp are evaluated exhaustively, the rest is set equal to the remaining     
def rank_expanded_signals_max_linearly(x, x_exp, y, y_exp, max_comp=10, max_num_samples_for_ev = None, max_test_samples_for_ev=None, verbose=False):
    dim_out = x_exp.shape[1]
    all_indices = numpy.arange(dim_out)

    indices_all_x = numpy.arange(x.shape[0])
    indices_all_y = numpy.arange(y.shape[0])
    
    max_scores = numpy.zeros(dim_out)
    available_mask = numpy.zeros(dim_out) >= 0 #boolean mask that indicates which elements are not yet scored
    taken = [] #list with the same elements.

#   Compute maximum explainable variance (taking all components)
    total_variance  = data_variance(y)

    last_explained_var = 0.0
    last_score = 0.0
    for iteration in range(min(max_comp, dim_out)):
        #find individual contribution to expl var, from not taken
        indices_available = all_indices[available_mask] #mapping from index_short to index_long
        temp_explained_vars = numpy.zeros(dim_out-iteration) #s_like(indices_available, dtype=") #explained variances for each available index

        #On each iteration, the subset of samples used for testing and samples for reconstruction are kept fixed
        if max_num_samples_for_ev != None and max_num_samples_for_ev < x.shape[0]:
            indices_all_x_selection = indices_all_x + 0
            numpy.random.shuffle(indices_all_x_selection)
            indices_all_x_selection = indices_all_x_selection[0:max_num_samples_for_ev]
            x_sel = x[indices_all_x_selection]
            x_exp_sel = x_exp[indices_all_x_selection]
        else:
            x_sel = x
            x_exp_sel = x_exp

        if max_test_samples_for_ev != None and max_test_samples_for_ev < x.shape[0]:
            indices_all_y_selection = indices_all_y + 0
            numpy.random.shuffle(indices_all_y_selection)
            indices_all_y_selection = indices_all_y_selection[0:max_test_samples_for_ev]
            y_sel = y[indices_all_y_selection]
            y_exp_sel = y_exp[indices_all_y_selection]
        else:
            y_sel = y
            y_exp_sel = y_exp

        if verbose:
            print "indices available=", indices_available
        for index_short, index_long in enumerate(indices_available):    
            taken_tmp = list(taken)            #Copy the taken list
            taken_tmp.append(index_long)       #Add index_long to it
            x_exp_tmp_sel = x_exp_sel[:,taken_tmp]     #Select the variables
            y_exp_tmp_sel = y_exp_sel[:,taken_tmp]


            y_app_sel = approximate_linearly(x_sel, x_exp_tmp_sel, y_exp_tmp_sel)

            #print "QQQ=", compute_explained_var(y_sel, y_app_sel)
            temp_explained_vars[index_short] = compute_explained_var(y_sel, y_app_sel) #compute explained var
            if verbose:
                print "taken_tmp=", taken_tmp, "temp_explained_vars[%d (long = %d) ]=%f"%(index_short, index_long, temp_explained_vars[index_short])

        #Update scores
        max_scores[indices_available] = numpy.maximum(max_scores[indices_available], temp_explained_vars - last_explained_var) 

        #select maximum
        #print "temp_explained_vars=", temp_explained_vars
        max_explained_var_index_short = temp_explained_vars.argmax()
        #print "max_explained_var_index_short=", max_explained_var_index_short
        #print "indices_available=",indices_available
        max_explained_var_index_long = indices_available[max_explained_var_index_short]   
        if verbose:
            print "Selecting index short:", max_explained_var_index_short, " and index_ long:", max_explained_var_index_long
       
        #mark as taken and update temporal variables
        taken.append(max_explained_var_index_long)
        available_mask[max_explained_var_index_long] = False 
#        last_score = scores[max_explained_var_index_long]
        last_explained_var = temp_explained_vars[max_explained_var_index_short] 

    print "brute max_scores = ", max_scores
    print "brute taken = ", taken
   
    #Find ordering of variables not yet taken
    if max_comp < dim_out:
        max_explained_var_indices_short = temp_explained_vars.argsort()[::-1][1:] #In increasing order, then remove first element, which was already added to taken
        
        for max_explained_var_index_short in max_explained_var_indices_short:
            taken.append(indices_available[max_explained_var_index_short])

    print "final taken = ", taken

    #Make scoring decreasing in ordering stored in taken
    last_explained_var = max(last_explained_var, 0.01) #For numerical reasons
    last_max_score = -numpy.inf
    sum_max_scores = 0.0
    for i, long_index in enumerate(taken):
        current_max_score = max_scores[long_index]
        sum_max_scores += current_max_score
        if current_max_score > last_max_score and i>0:
            max_scores[long_index] = last_max_score
            tmp_sum_max_scores = max_scores[taken[0:i+1]].sum()
            max_scores[taken[0:i+1]] += (sum_max_scores - tmp_sum_max_scores)/(i+1)
        last_max_score = max_scores[long_index]
        #print "iteration max_scores = ", max_scores

    print "preeliminar max_scores = ", max_scores
    
#    max_scores *= (last_explained_var / max_scores.sum())**0.5
#NOTE: last_explained_var is not the data variance. 
#Here it is the variance up to max_comp components
#3 options: all features, first max_comp features, output_dim features  
    max_scores *= (last_explained_var / max_scores.sum())**0.5

    print "final max_scores = ", max_scores

    if (max_scores==0.0).any():
        print "WARNING, removing 0.0 max_scores!"
        max_score_min =  (max_scores[max_scores>0.0]).min()
        max_scores += max_score_min * 0.001 #TODO:Find reasonable way to fix this, is this causing the distorted reconstructions???
#         max_scores += (max_scores[max_scores>0.0])
    return max_scores 










#Second ranking method more robust and closer to max I(x; y_i + Y)-I(x;Y) for all Y.  
#Ordering and scoring of signals respects principle of best incremental feature selection
#Computes a scores vector that measures the importance of each expanded component at reconstructing a signal 
#x, x_exp are training data, y and y_exp are test data
#At most max_comp are evaluated exhaustively, the rest is set equal to the remaining     
def rank_expanded_signals_max(x, x_exp, y, y_exp, max_comp=10, k=1, operation="average", max_num_samples_for_ev = None, max_test_samples_for_ev=None, offsetting_mode = "max_comp features", verbose=False):
    dim_out = x_exp.shape[1]
    all_indices = numpy.arange(dim_out)

    indices_all_x = numpy.arange(x.shape[0])
    indices_all_y = numpy.arange(y.shape[0])
    
    max_scores = numpy.zeros(dim_out)
    available_mask = numpy.zeros(dim_out) >= 0 #boolean mask that indicates which elements are not yet scored
    taken = [] #list with the same elements.

#   Compute maximum explainable variance (taking all components)
    total_variance  = data_variance(y)

    last_explained_var = 0.0
    last_score = 0.0
    for iteration in range(min(max_comp, dim_out)):
        #find individual contribution to expl var, from not taken
        indices_available = all_indices[available_mask] #mapping from index_short to index_long
        temp_explained_vars = numpy.zeros(dim_out-iteration) #s_like(indices_available, dtype=") #explained variances for each available index

        #On each iteration, the subset of samples used for testing and samples for reconstruction are kept fixed
        if max_num_samples_for_ev != None and max_num_samples_for_ev < x.shape[0]:
            indices_all_x_selection = indices_all_x + 0
            numpy.random.shuffle(indices_all_x_selection)
            indices_all_x_selection = indices_all_x_selection[0:max_num_samples_for_ev]
            x_sel = x[indices_all_x_selection]
            x_exp_sel = x_exp[indices_all_x_selection]
        else:
            x_sel = x
            x_exp_sel = x_exp

        if max_test_samples_for_ev != None and max_test_samples_for_ev < x.shape[0]:
            indices_all_y_selection = indices_all_y + 0
            numpy.random.shuffle(indices_all_y_selection)
            indices_all_y_selection = indices_all_y_selection[0:max_test_samples_for_ev]
            y_sel = y[indices_all_y_selection]
            y_exp_sel = y_exp[indices_all_y_selection]
        else:
            y_sel = y
            y_exp_sel = y_exp

        if verbose:
            print "indices available=", indices_available
        for index_short, index_long in enumerate(indices_available):    
            taken_tmp = list(taken)            #Copy the taken list
            taken_tmp.append(index_long)       #Add index_long to it
            x_exp_tmp_sel = x_exp_sel[:,taken_tmp]     #Select the variables
            y_exp_tmp_sel = y_exp_sel[:,taken_tmp]


            if operation == "linear_rec":
                y_app_sel = approximate_linearly(x_sel, x_exp_tmp_sel, y_exp_tmp_sel)
            else:           
                y_app_sel = approximate_kNN_op(x_sel, x_exp_tmp_sel, y_exp_tmp_sel, k=k, ignore_closest_match = True, operation=operation)    #invert from taken variables


            #print "QQQ=", compute_explained_var(y_sel, y_app_sel)
            temp_explained_vars[index_short] = compute_explained_var(y_sel, y_app_sel) #compute explained var
            if verbose:
                print "taken_tmp=", taken_tmp, "temp_explained_vars[%d (long = %d) ]=%f"%(index_short, index_long, temp_explained_vars[index_short])

        #Update scores
        max_scores[indices_available] = numpy.maximum(max_scores[indices_available], temp_explained_vars - last_explained_var) 

        #select maximum
        #print "temp_explained_vars=", temp_explained_vars
        max_explained_var_index_short = temp_explained_vars.argmax()
        #print "max_explained_var_index_short=", max_explained_var_index_short
        #print "indices_available=",indices_available
        max_explained_var_index_long = indices_available[max_explained_var_index_short]   
        if verbose:
            print "Selecting index short:", max_explained_var_index_short, " and index_ long:", max_explained_var_index_long
       
        #mark as taken and update temporal variables
        taken.append(max_explained_var_index_long)
        available_mask[max_explained_var_index_long] = False 
#        last_score = scores[max_explained_var_index_long]
        last_explained_var = temp_explained_vars[max_explained_var_index_short] 

    print "brute max_scores = ", max_scores
    print "brute taken = ", taken
   
    #Find ordering of variables not yet taken
    if max_comp < dim_out:
        max_explained_var_indices_short = temp_explained_vars.argsort()[::-1][1:] #In increasing order, then remove first element, which was already added to taken
        
        for max_explained_var_index_short in max_explained_var_indices_short:
            taken.append(indices_available[max_explained_var_index_short])

    print "final taken = ", taken

    #Make scoring decreasing in ordering stored in taken
    last_explained_var = max(last_explained_var, 0.01) #For numerical reasons
    last_max_score = -numpy.inf
    sum_max_scores = 0.0
    for i, long_index in enumerate(taken):
        current_max_score = max_scores[long_index]
        sum_max_scores += current_max_score
        if current_max_score > last_max_score and i>0:
            max_scores[long_index] = last_max_score
            tmp_sum_max_scores = max_scores[taken[0:i+1]].sum()
            max_scores[taken[0:i+1]] += (sum_max_scores - tmp_sum_max_scores)/(i+1)
        last_max_score = max_scores[long_index]
        #print "iteration max_scores = ", max_scores

    print "preeliminar max_scores = ", max_scores
    
    #Compute explained variance with all features
    indices_all_x_selection = random_subindices(x.shape[0], max_num_samples_for_ev)
    x_sel = x[indices_all_x_selection]
    x_exp_sel = x_exp[indices_all_x_selection]
    indices_all_y_selection = random_subindices(y.shape[0], max_test_samples_for_ev)
    y_sel = y[indices_all_y_selection]
    y_exp_sel = y_exp[indices_all_y_selection]
    if operation == "linear_rec":
        y_app_sel = approximate_linearly(x_sel, x_exp_sel, y_exp_sel)
    else:           
        y_app_sel = approximate_kNN_op(x_sel, x_exp_sel, y_exp_sel, k=k, ignore_closest_match = True, operation=operation)    #invert from taken variables
    explained_var_all_feats = compute_explained_var(y_sel, y_app_sel)
    
    print "last_explained_var =", last_explained_var
    print "explained_var_all_feats=", explained_var_all_feats, "total input variance:", total_variance

#    max_scores *= (last_explained_var / max_scores.sum())**0.5
#NOTE: last_explained_var is not the data variance. It is the variance up to max_comp components
#3 options: all scores, max_comp scores, output_dim scores (usually all scores)
    if offsetting_mode == "max_comp features":
        max_scores *= (last_explained_var / max_scores.sum())
    elif offsetting_mode == "all features":
        print "explained_var_all_feats=", explained_var_all_feats, "total input variance:", total_variance
        max_scores *= (explained_var_all_feats / max_scores.sum())
    elif offsetting_mode == "all features smart":
        max_scores *= (last_explained_var / max_scores.sum())
        print "scaled max_scores=", max_scores
        max_scores += (explained_var_all_feats-last_explained_var)/max_scores.shape[0]
        print "offsetted max_scores=", max_scores
    elif offsetting_mode == "democratic":
        max_scores = numpy.ones_like(max_scores) * explained_var_all_feats / max_scores.shape[0]
        print "democractic max_scores=", max_scores
    elif offsetting_mode == "linear":
        #Code fixed!!!
        max_scores = numpy.arange(dim_out, 0, -1) * explained_var_all_feats / (dim_out * (dim_out + 1) /2)
        print "linear max_scores=", max_scores
    elif offsetting_mode == "sensitivity_based":
        sens = sensivity_of_linearly_approximation(x_sel, x_exp_sel)
        max_scores = sens * explained_var_all_feats / sens.sum() 
        print "sensitivity_based max_scores=", max_scores
    else:
        ex = "offsetting_mode unknown", offsetting_mode
        raise Exception(ex)
    print "final max_scores = ", max_scores

    if (max_scores==0.0).any():
        print "WARNING, removing 0.0 max_scores!"
        max_score_min =  (max_scores[max_scores>0.0]).min()
        max_scores += max_score_min * 0.001 #TODO:Find reasonable way to fix this, is this causing the distorted reconstructions???
#         max_scores += (max_scores[max_scores>0.0])
    return max_scores 





#TODO: Improve: if max_comp < output_dim choose remaining features from the last evaluation of explained variances.
#Computes a scores vector that measures the importance of each expanded component at reconstructing a signal 
#x, x_exp are training data, y and y_exp are test data
#At most max_comp are evaluated exhaustively, the rest is set equal to the remaining     
def rank_expanded_signals(x, x_exp, y, y_exp, max_comp=10, k=1, linear=False, max_num_samples_for_ev = None, max_test_samples_for_ev=None, verbose=False):
    dim_out = x_exp.shape[1]
    all_indices = numpy.arange(dim_out)

    indices_all_x = numpy.arange(x.shape[0])
    indices_all_y = numpy.arange(y.shape[0])
    
    scores = numpy.zeros(dim_out)
    available_mask = numpy.zeros(dim_out) >= 0 #boolean mask that indicates which elements are not yet scored
    taken = [] #list with the same elements.

#   Compute maximum explainable variance (taking all components)
    total_variance  = data_variance(y)


    last_explained_var = 0.0
    last_score = 0.0
    for iteration in range(min(max_comp, dim_out)):
        #find individual contribution to expl var, from not taken
        indices_available = all_indices[available_mask] #mapping from index_short to index_long
        temp_explained_vars = numpy.zeros(dim_out-iteration) #s_like(indices_available, dtype=") #explained variances for each available index

        #On each iteration, the subset of samples used for testing and samples for reconstruction are kept fixed
        if max_num_samples_for_ev != None and max_num_samples_for_ev < x.shape[0]:
            indices_all_x_selection = indices_all_x + 0
            numpy.random.shuffle(indices_all_x_selection)
            indices_all_x_selection = indices_all_x_selection[0:max_num_samples_for_ev]
            x_sel = x[indices_all_x_selection]
            x_exp_sel = x_exp[indices_all_x_selection]
        else:
            x_sel = x
            x_exp_sel = x_exp

        if max_test_samples_for_ev != None and max_test_samples_for_ev < x.shape[0]:
            indices_all_y_selection = indices_all_y + 0
            numpy.random.shuffle(indices_all_y_selection)
            indices_all_y_selection = indices_all_y_selection[0:max_test_samples_for_ev]
            y_sel = y[indices_all_y_selection]
            y_exp_sel = y_exp[indices_all_y_selection]
        else:
            y_sel = y
            y_exp_sel = y_exp

        if verbose:
            print "indices available=", indices_available
        for index_short, index_long in enumerate(indices_available):    
            taken_tmp = list(taken)            #Copy the taken list
            taken_tmp.append(index_long)       #Add index_long to it
            x_exp_tmp_sel = x_exp_sel[:,taken_tmp]     #Select the variables
            y_exp_tmp_sel = y_exp_sel[:,taken_tmp]

            y_app_sel = approximate_kNN(x_sel, x_exp_tmp_sel, y_exp_tmp_sel, k=k, ignore_closest_match = True, label_avg=True)    #invert from taken variables

            #print "QQQ=", compute_explained_var(y_sel, y_app_sel)
            temp_explained_vars[index_short] = compute_explained_var(y_sel, y_app_sel) #compute explained var
            if verbose:
                print "taken_tmp=", taken_tmp, "temp_explained_vars[%d (long = %d) ]=%f"%(index_short, index_long, temp_explained_vars[index_short])
        #select maximum

        #print "temp_explained_vars=", temp_explained_vars
        max_explained_var_index_short = temp_explained_vars.argmax()
        #print "max_explained_var_index_short=", max_explained_var_index_short
        #print "indices_available=",indices_available
        max_explained_var_index_long = indices_available[max_explained_var_index_short]   
        if verbose:
            print "Selecting index short:", max_explained_var_index_short, " and index_ long:", max_explained_var_index_long
       
        #update total explained var & scores
        #Add logic to robustly handle strange contributions: 3, 2, 1, 4 =>  5, 2.5, 1.25, 1.25 ?
        #TODO:FIX NORMALIZATION WHEN FIRST SCORES ARE ZERO OR NEGATIVE!
        #TODO:NORMALIZATION SHOULD BE OPTIONAL, SINCE IT WEAKENS THE INTERPRETATION OF THE SCORES
        explained_var = max(temp_explained_vars[max_explained_var_index_short], 0.0)
        new_score = explained_var - last_explained_var
        if verbose:
            print "new_score raw = ", new_score
        new_score = max(new_score, 0.0)
        if new_score > last_score and iteration > 0:
            new_score = last_score #Here some options are available to favour components taken first
        scores[max_explained_var_index_long] = new_score     
        
        if verbose:
            print "tmp scores = ", scores
        #normalize scores, so that they sume up to explained_var
        sum_scores = scores.sum()
        residual = max(explained_var,0.0) - sum_scores 
        if residual > 0.0:
            correction = residual / (iteration+1)
            scores[taken] += correction
            scores[max_explained_var_index_long] += correction
                      
#        scores = scores * explained_var / (sum_scores+1e-6) #TODO:CORRECT THIS; INSTEAD OF FACTOR USE ADDITIVE TERM
        if verbose:
            print "normalized scores = ", scores, "sum to:", scores.sum(), "explained_var =", explained_var

        #mark as taken and update temporal variables
        taken.append(max_explained_var_index_long)
        available_mask[max_explained_var_index_long] = False 
        last_score = scores[max_explained_var_index_long]
        last_explained_var = explained_var

    #handle variables not used, assign equal scores to all of them
    preserve_last_evaluation = True
    if preserve_last_evaluation and max_comp < dim_out:
        #The score of the last feature found will be modified, as well as of not yet found features
        #TODO: Take care of negative values
        if (last_score <= 0.0):
            last_score = 0.01 #Just some value is needed here
        remaining_output_features = len(temp_explained_vars) #including feature already processed
        remaining_ordered_explained_variances_short_index = numpy.argsort(temp_explained_vars)[::-1]
        remaining_ordered_explained_variances_long_index = indices_available[remaining_ordered_explained_variances_short_index]
        remaining_ordered_explained_variances = temp_explained_vars[remaining_ordered_explained_variances_short_index]+0.0
        remaining_total_contribution = last_score
        print "last_score=", last_score
            
        beta = 0.95
        remaining_ordered_explained_variances[remaining_ordered_explained_variances<=0.0] = 0.0001 #To avoid division over zero, numerical hack
        # numpy.clip(remaining_ordered_explained_variances, 0.0, None) fails here!!!!  
        print "remaining_ordered_explained_variances=",remaining_ordered_explained_variances
        minimum = remaining_ordered_explained_variances.min() # first element
        ev_sum = remaining_ordered_explained_variances.sum()
        normalized_scores = (remaining_total_contribution / (ev_sum - remaining_output_features * minimum) * beta) * (remaining_ordered_explained_variances - minimum) + ((1.0-beta)/ remaining_output_features)* remaining_total_contribution  
        print "normalized_scores=", normalized_scores
        print "remaining_ordered_explained_variances_long_index=", remaining_ordered_explained_variances_long_index
        print scores.dtype
        print normalized_scores.dtype

        scores[remaining_ordered_explained_variances_long_index] = normalized_scores                       
    else:
        #rest_explained_variance = total_variance-last_explained_var
        sum_scores = scores.sum()
        rest_explained_variance = total_variance-sum_scores
        if verbose:
            print "rest_explained_variance=", rest_explained_variance
        correction = rest_explained_variance / dim_out
        scores += correction
    
    if (scores==0.0).any():
        print "WARNING, removing 0.0 scores!"
        scores += 0.0001

#    num_unused = dim_out - max_comp
#    scores[available_mask] = min(rest_explained_variance / num_unused, last_score) 
#    sum_scores = scores.sum()
#    scores = scores * explained_var / (sum_scores+1e-6)

    if verbose:
        print "final scores: ", scores
    
    if verbose and linear and False:
        for i in indices_available:
            taken.append(i)
        scores[taken] = numpy.arange(dim_out-1, -1, -1) #**2 #WARNING!!! QUADRATIC SCORES!!!
        scores = scores * total_variance / scores.sum()  
        print "Overriding with linear scores:", scores

    return scores #**0.5 #WARNING!!!!!



class IEVMNode(mdp.Node):
#     """Node implementing simple Incremental Explained Variance Maximization. 
#     Extracted features are moderately useful for reconstruction, although this node does 
#     itself provide reconstruction.
#    The expansion function is optional, as well as performing PCA on the scores.
#    The added variance of the first k-outputs is equal to the explained variance of such k-outputs. 
#    """    
    def __init__(self, input_dim = None, output_dim=None, expansion_funcs=None, k=5, max_comp = None, max_num_samples_for_ev = None, max_test_samples_for_ev=None, use_pca=False, use_sfa=False, max_preserved_sfa=2.0, second_weighting=False, operation="average", out_sfa_filter=False, **argv ):
        super(IEVMNode, self).__init__(input_dim =input_dim, output_dim=output_dim, **argv) 
        if expansion_funcs != None:
            self.exp_node = GeneralExpansionNode(funcs=expansion_funcs)
        else:
            self.exp_node = None
        self.sfa_node = None
        self.second_weighting = second_weighting
        self.use_pca = use_pca
        self.use_sfa = use_sfa
        if use_sfa==True and use_pca==False:
            er = "Combination of use_sfa and use_pca not considered. Please activate use_pca or deactivate use_sfa"
            raise Exception(er)
        self.k = k
        self.max_comp = max_comp
        self.max_num_samples_for_ev = max_num_samples_for_ev
        self.max_test_samples_for_ev = max_test_samples_for_ev
        self.feature_scaling_factor = 0.5 #Factor that prevents amplitudes of feautes growing acrosss the networks
        self.exponent_variance = 0.5
        self.operation=operation
        self.max_preserved_sfa=max_preserved_sfa
        self.out_sfa_filter = out_sfa_filter
        
#        self.avg = None #input average
#        self.W = None #weights for complete transformation        
    def is_trainable(self):
        return True

    #sfa_block_size, sfa_train_mode, etc. would be preferred
    # max_preserved_sfa=1.995
    def _train(self, x, block_size=None, train_mode = None, node_weights=None, edge_weights=None, scheduler = None, n_parallel=None, **argv):
        num_samples, self.input_dim = x.shape
       
        if self.output_dim == None:
            self.output_dim = self.input_dim

        if self.max_comp == None:
            self.max_comp = min(self.input_dim, self.output_dim)
        else:
            self.max_comp = min(self.max_comp, self.input_dim, self.output_dim)

        print "Training IEVMNode..."

        self.x_mean = x.mean(axis=0) #Remove mean before expansion
        x=x-self.x_mean
        
        if self.exp_node != None: #Expand data
            print "expanding x..."
            exp_x = self.exp_node.execute(x)
        else:
            exp_x = x

        self.expanded_dim = exp_x.shape[1]
        self.exp_x_mean = exp_x.mean(axis=0)
        self.exp_x_std = exp_x.std(axis=0)
            
        print "self.exp_x_mean=", self.exp_x_mean
        print "self.exp_x_std=", self.exp_x_std
        if (self.exp_x_std==0).any():
            er = "zero-component detected"
            raise Exception(er)
        
        n_exp_x = (exp_x - self.exp_x_mean) / self.exp_x_std #Remove media and variance from expansion
        
        print "ranking n_exp_x ..."
        rankings = rank_expanded_signals_max(x, n_exp_x, x, n_exp_x, max_comp=self.max_comp, k=self.k, operation=self.operation, max_num_samples_for_ev = self.max_num_samples_for_ev, max_test_samples_for_ev=self.max_test_samples_for_ev, verbose=True)
        rankings *= self.feature_scaling_factor
        print "rankings=",rankings
        if (rankings==0).any():
            er = "zero-component detected"
            raise Exception(er)
        
        self.perm1 = numpy.argsort(rankings)[::-1] #Sort in decreasing ranking
        self.magn1 = rankings
        print "self.perm1=",  self.perm1

        s_x_1 = n_exp_x * self.magn1 ** self.exponent_variance     #Scale according to ranking
        s_x_1 = s_x_1[:,self.perm1]       #Permute with most important signal first


        if self.second_weighting:
            print "ranking s_x_1 ..."
            rankings_B = rank_expanded_signals_max(x, s_x_1, x, s_x_1, max_comp=self.max_comp, k=self.k, operation=self.operation, max_num_samples_for_ev = self.max_num_samples_for_ev, max_test_samples_for_ev=self.max_test_samples_for_ev, verbose=False)
            print "rankings_B=",rankings_B
            if (rankings_B==0).any():
                er = "zero-component detected"
                raise Exception(er)
            
            self.perm1_B = numpy.argsort(rankings_B)[::-1] #Sort in decreasing ranking
            self.magn1_B = rankings_B
            print "self.perm1_B=",  self.perm1_B
    
            #WARNING, this only works for normalized s_x_1
            s_x_1B = s_x_1 * self.magn1_B ** self.exponent_variance     #Scale according to ranking
            s_x_1B = s_x_1B[:,self.perm1_B]       #Permute with most important signal first
        else:
            s_x_1B = s_x_1


        if self.use_sfa:
            self.sfa_node = mdp.nodes.SFANode()
            #TODO: Preserve amplitude
            self.sfa_node.train(s_x_1B, block_size=block_size, train_mode = train_mode)#, node_weights=None, edge_weights=None, scheduler = None, n_parallel=None)
            self.sfa_node.stop_training()

            print "self.sfa_node.d", self.sfa_node.d
            
            #Adaptive mechanism based on delta values
            if isinstance(self.max_preserved_sfa, float):
                self.num_sfa_features_preserved = (self.sfa_node.d <= self.max_preserved_sfa).sum()
            elif isinstance(self.max_preserved_sfa, int):
                self.num_sfa_features_preserved = self.max_preserved_sfa
            else:
                ex = "Cannot handle type of self.max_preserved_sfa"
                print ex
                raise Exception(ex)

            #self.num_sfa_features_preserved = 10
            sfa_x = self.sfa_node.execute(s_x_1B)
          
            
            #TODO: Change internal variables of SFANode, so that we do not need to zero some components
            #TODO: Is this equivalent to truncation of the matrices??? PERHAPS IT IS NOT !!!
            sfa_x[:,self.num_sfa_features_preserved:] = 0.0

            proj_sfa_x =  self.sfa_node.inverse(sfa_x)

            sfa_x = sfa_x[:,0:self.num_sfa_features_preserved]
            #Notice that sfa_x has WEIGHTED zero-mean, thus we correct this here?
            self.sfa_x_mean = sfa_x.mean(axis=0)
            self.sfa_x_std = sfa_x.std(axis=0)
            
            print "self.sfa_x_mean=", self.sfa_x_mean
            print "self.sfa_x_std=", self.sfa_x_std
            sfa_x -= self.sfa_x_mean
            
            sfa_removed_x = s_x_1B - proj_sfa_x #Remove sfa projection of data
            

        else:
            self.num_sfa_features_preserved = 0
            sfa_x = numpy.ones((num_samples,0))
            sfa_removed_x = s_x_1B
            
        pca_out_dim = self.expanded_dim - self.num_sfa_features_preserved
        if self.use_pca and pca_out_dim > 0:            
            self.pca_node = mdp.nodes.PCANode(output_dim = pca_out_dim)
            self.pca_node.train(sfa_removed_x)
            
            #TODO:check that pca_out_dim > 0
            pca_x = self.pca_node.execute(sfa_removed_x)
        
            self.pca_x_mean = pca_x.mean(axis=0)
            self.pca_x_std = pca_x.std(axis=0)
            
            print "self.pca_x_std=",self.pca_x_std
            if (self.pca_x_std==0).any():
                er = "zero-component detected"
                raise Exception(er)
            #TODO: Is this step needed? if heuristic works well this weakens algorithm
            n_pca_x = (pca_x - self.pca_x_mean) / self.pca_x_std
        else:
            n_pca_x = sfa_removed_x[:,0:pca_out_dim]


#            pca_rankings = rank_expanded_signals_max(x, n_pca_x, x, n_pca_x, max_comp=self.max_comp, k=self.k, operation=self.operation, max_num_samples_for_ev = self.max_num_samples_for_ev, max_test_samples_for_ev=self.max_test_samples_for_ev, verbose=False)
#            pca_rankings *= self.feature_scaling_factor
#            print "pca_rankings=", pca_rankings
#            if (pca_rankings==0).any():
#                er = "zero-component detected"
#                raise Exception(er)
#            
#            self.perm2 = numpy.argsort(pca_rankings)[::-1] #Sort in decreasing ranking
#            self.magn2 = pca_rankings
#
#            s_x_2 = n_pca_x * self.magn2 ** self.exponent_variance #Scale according to ranking
#            s_x_2 = s_x_2[:,self.perm2]       #Permute with most important signal first

        #Concatenate SFA and PCA signals and rank them preserving SFA components in ordering
        if self.use_pca or self.use_sfa:
            #TODO: Either both signals conserve magnitudes or they are both normalized
            sfa_pca_x = numpy.concatenate((sfa_x, n_pca_x), axis=1)
            
            sfa_pca_rankings = rank_expanded_signals_max(x, sfa_pca_x, x, sfa_pca_x, max_comp=self.max_comp, k=self.k, operation=self.operation, 
                                                         max_num_samples_for_ev = self.max_num_samples_for_ev, max_test_samples_for_ev=self.max_test_samples_for_ev, 
                                                         verbose=False)
            sfa_pca_rankings *= self.feature_scaling_factor #Only one magnitude normalization by node, but where should it be done? I guess after last transformation
            print "sfa_pca_rankings=", sfa_pca_rankings
            if (sfa_pca_rankings==0).any():
                er = "zero-component detected"
                raise Exception(er)
            
            
            
            self.magn2 = sfa_pca_rankings
            perm2a = numpy.arange(self.num_sfa_features_preserved, dtype="int")
            perm2b = numpy.argsort(sfa_pca_rankings[self.num_sfa_features_preserved:])[::-1]
            self.perm2 = numpy.concatenate((perm2a, perm2b+self.num_sfa_features_preserved))
            print "second permutation=", self.perm2          

            #WARNING, this only works for normalized sfa_pca_x
            s_x_2 = sfa_pca_x * self.magn2 ** self.exponent_variance #Scale according to ranking
            s_x_2 = s_x_2[:,self.perm2]       #Permute with slow features first, and then most important signal first
        else:
            s_x_2 = n_pca_x

        #Tuncating output_dim components
        s_x_2_truncated =  s_x_2[:,0:self.output_dim]
        
        #Filtering output through SFA
        if self.out_sfa_filter:
            self.out_sfa_node = mdp.nodes.SFANode()
            self.out_sfa_node.train(s_x_2_truncated, block_size=block_size, train_mode = train_mode)
            self.out_sfa_node.stop_training()
            sfa_filtered = self.out_sfa_node.execute(s_x_2_truncated)
        else:
            sfa_filtered = s_x_2_truncated

        self.stop_training()

#    def __init__(self, funcs, input_dim = None, dtype = None, \
#                 use_pseudoinverse=True, use_hint=False, max_steady_factor=1.5, \
#                 delta_factor=0.6, min_delta=0.00001):    
#            
#        
#        
#        self.sfa_node.train(x, **argv)
        
    def _is_invertible(self):
        return True

    def _execute(self, x):
        x_orig = x +0.0
        num_samples = x.shape[0]
        zm_x = x - self.x_mean
        
        if self.exp_node != None:
            exp_x = self.exp_node.execute(zm_x)
        else:
            exp_x = zm_x
                    
        n_exp_x = (exp_x - self.exp_x_mean) / self.exp_x_std
        if numpy.isnan(n_exp_x).any() or numpy.isinf(n_exp_x).any():
            print "n_exp_x=", n_exp_x
            quit()
        n_exp_x[numpy.isnan(n_exp_x)] = 0.0

        if numpy.isnan(self.magn1).any():
            print "self.magn1=", self.magn1
            quit()
            
        s_x_1 = n_exp_x * self.magn1 ** self.exponent_variance #Scale according to ranking
        s_x_1 = s_x_1[:,self.perm1]       #Permute with most important signal first

        if self.second_weighting:
            s_x_1B = s_x_1 * self.magn1_B ** self.exponent_variance #Scale according to ranking_B
            s_x_1B = s_x_1B[:,self.perm1_B]       #Permute with most important signal first
        else:
            s_x_1B = s_x_1
            
        if numpy.isnan(s_x_1B).any():
            print "s_x_1B=", s_x_1B
            quit()

        if self.use_sfa:
            sfa_x = self.sfa_node.execute(s_x_1B)

            #TODO: Change internal variables of SFANode, so that we do not need to zero some components
            sfa_x[:,self.num_sfa_features_preserved:] = 0.0

            proj_sfa_x =  self.sfa_node.inverse(sfa_x)

            sfa_x = sfa_x[:,0:self.num_sfa_features_preserved]
            sfa_x -= self.sfa_x_mean
            
            sfa_removed_x = s_x_1B - proj_sfa_x           
        else:
            sfa_x = numpy.ones((num_samples,0))
            sfa_removed_x = s_x_1B

        pca_out_dim = self.expanded_dim - self.num_sfa_features_preserved
        if self.use_pca and pca_out_dim>0:
            pca_x = self.pca_node.execute(sfa_removed_x)
             
            n_pca_x = (pca_x - self.pca_x_mean) / self.pca_x_std
        else:            
            n_pca_x = sfa_removed_x[:,0:pca_out_dim]
    
        if self.use_pca or self.use_sfa:
            sfa_pca_x = numpy.concatenate((sfa_x, n_pca_x), axis=1)
            
            s_x_2 = sfa_pca_x * self.magn2 ** self.exponent_variance #Scale according to ranking
            s_x_2 = s_x_2[:,self.perm2]       #Permute with most important signal first            
        else:
            s_x_2 = n_pca_x
                
        if numpy.isnan(s_x_2).any():
            print "s_x_2=", s_x_2
            quit()

        #Tuncating output_dim components
        s_x_2_truncated =  s_x_2[:,0:self.output_dim]
        
        #Filtering output through SFA
        if self.out_sfa_filter:
            sfa_filtered = self.out_sfa_node.execute(s_x_2_truncated)
        else:
            sfa_filtered = s_x_2_truncated

        verbose=False
        if verbose:
            print "x[0]=",x_orig[0]
            print "x_zm[0]=", x[0]
            print "exp_x[0]=", exp_x[0]
            print "s_x_1[0]=", s_x_1[0]
            print "sfa_removed_x[0]=", sfa_removed_x[0]
            print "proj_sfa_x[0]=", proj_sfa_x[0]
            print "pca_x[0]=", pca_x[0]
            print "n_pca_x[0]=", n_pca_x[0]        
            print "sfa_x[0]=", sfa_x[0] + self.sfa_x_mean
            print "s_x_2_truncated[0]=", s_x_2_truncated[0]
            print "sfa_filtered[0]=", sfa_filtered[0]

        return sfa_filtered
  
    #TODO:Code inverse with SFA
    def _inverse(self, y):
        num_samples = y.shape[0]
        if y.shape[1] != self.output_dim:
            er = "Serious dimensionality inconsistency:", y.shape[0], self.output_dim
            raise Exception(er)
#        input_dim = self.input_dim
        
        #De-Filtering output through SFA
        sfa_filtered=y
        if self.out_sfa_filter:
            s_x_2_truncated = self.out_sfa_node.inverse(sfa_filtered)
        else:
            s_x_2_truncated = sfa_filtered

        #De-Tuncating output_dim components
        s_x_2_full = numpy.zeros((num_samples,self.expanded_dim))
        s_x_2_full[:,0:self.output_dim] = s_x_2_truncated
        

        if self.use_pca or self.use_sfa:                      
            perm_2_inv = numpy.zeros(self.expanded_dim, dtype="int")
#            print "input_dim", input_dim
#            print "self.perm2", self.perm2
#            print "len(self.perm2)", len(self.perm2)
            perm_2_inv[self.perm2] = numpy.arange(self.expanded_dim, dtype="int")
            #print perm_2_inv
            sfa_pca_x =  s_x_2_full[:, perm_2_inv]
            sfa_pca_x /= self.magn2 ** self.exponent_variance            

            sfa_x = sfa_pca_x[:,0:self.num_sfa_features_preserved]
            n_pca_x = sfa_pca_x[:,self.num_sfa_features_preserved:]
        else:
            #sfa_x = ...?
            n_pca_x = s_x_2_full
                
        pca_out_dim = self.expanded_dim - self.num_sfa_features_preserved
        if self.use_pca and pca_out_dim>0:
            pca_x = n_pca_x * self.pca_x_std + self.pca_x_mean
            sfa_removed_x = self.pca_node.inverse(pca_x)
        else:
            sfa_removed_x = n_pca_x


        if self.use_sfa:
            sfa_x += self.sfa_x_mean
            sfa_x_full = numpy.zeros((num_samples, self.expanded_dim))
            sfa_x_full[:,0:self.num_sfa_features_preserved] = sfa_x
            proj_sfa_x =  self.sfa_node.inverse(sfa_x_full)
            s_x_1B = sfa_removed_x + proj_sfa_x 
        else:
            s_x_1B = sfa_removed_x 

        if self.second_weighting:
            perm_1B_inv = numpy.zeros(self.expanded_dim, dtype="int")
            perm_1B_inv[self.perm1_B] = numpy.arange(self.expanded_dim, dtype="int")
            s_x_1 =  s_x_1B[:, perm_1B_inv]
            s_x_1 /= self.magn1_B ** self.exponent_variance
        else:
            s_x_1 =  s_x_1B

        perm_1_inv = numpy.zeros(self.expanded_dim, dtype="int")
        perm_1_inv[self.perm1] = numpy.arange(self.expanded_dim, dtype="int")
        n_exp_x =  s_x_1[:, perm_1_inv]
        n_exp_x /= self.magn1 ** self.exponent_variance

        exp_x = n_exp_x * self.exp_x_std + self.exp_x_mean
        if self.exp_node != None:
            zm_x = self.exp_node.inverse(exp_x)
        else:
            zm_x = exp_x
        x = zm_x + self.x_mean

        verbose=False
        if verbose:
            print "x[0]=",x[0]
            print "zm_x[0]=", zm_x[0]
            print "exp_x[0]=", exp_x[0]
            print "s_x_1[0]=", s_x_1[0]
            print "proj_sfa_x[0]=", proj_sfa_x[0]
            print "sfa_removed_x[0]=", sfa_removed_x[0]
            print "pca_x[0]=", pca_x[0]
            print "n_pca_x[0]=", n_pca_x[0]        
            print "sfa_x[0]=", sfa_x[0]
        
        return x

#Computes output errors dimension by dimension for a single sample: y - node.execute(x_app)
#The library fails when dim(x_app) > dim(y), thus filling of x_app with zeros is recommended
def f_residual(x_app_i, node, y_i):
    #print "%",
    #print "f_residual: x_appi_i=", x_app_i, "node.execute=", node.execute, "y_i=", y_i
    res_long = numpy.zeros_like(y_i)
    y_i = y_i.reshape((1,-1))
    y_i_short = y_i[:,0:node.output_dim]
#    x_app_i = x_app_i[0:node.input_dim]
    res = (y_i_short - node.execute(x_app_i.reshape((1,-1)))).flatten()
    #print "res_long=", res_long, "y_i=", y_i, "res", res
    res_long[0:len(res)]=res
#    res = (y_i - node.execute(x_app_i))
    #print "returning resudial res=", res
    return res_long

class IEVMLRecNode(mdp.Node):
#     """Node implementing simple Incremental Explained Variance Maximization. 
#     Extracted features are moderately useful for reconstruction, although this node does 
#     itself provide reconstruction.
#    The expansion function is optional, as well as performing PCA on the scores.
#    The added variance of the first k-outputs is equal to the explained variance of such k-outputs. 
#    """    
    def __init__(self, input_dim = None, output_dim=None, pre_expansion_node_class = None, expansion_funcs=None, max_comp = None, max_num_samples_for_ev = None, max_test_samples_for_ev=None, offsetting_mode = "all features", max_preserved_sfa=1.999999, reconstruct_with_sfa = True,  out_sfa_filter=False, **argv ):
        super(IEVMLRecNode, self).__init__(input_dim =input_dim, output_dim=output_dim, **argv)
        self.pre_expansion_node_class = pre_expansion_node_class
        self.pre_expansion_node = None
        if expansion_funcs != None:
            self.exp_node = GeneralExpansionNode(funcs=expansion_funcs)
        else:
            self.exp_node = None
        self.sfa_node = None
        self.max_comp = max_comp
        
        self.max_num_samples_for_ev = max_num_samples_for_ev
        self.max_test_samples_for_ev = max_test_samples_for_ev
        self.feature_scaling_factor = 0.5 #Factor that prevents amplitudes of features growing acrosss the networks
        self.exponent_variance = 0.5
        self.max_preserved_sfa=max_preserved_sfa
        self.reconstruct_with_sfa = reconstruct_with_sfa #Indicates whether (nonlinear) SFA components are used for reconstruction
        self.out_sfa_filter = out_sfa_filter
        self.compress_input_with_pca = False #True
        self.compression_out_dim = 0.99 #0.99 #0.95 #98
        self.offsetting_mode = offsetting_mode
        
    def is_trainable(self):
        return True

    #sfa_block_size, sfa_train_mode, etc. would be preferred
    # max_preserved_sfa=1.995
    def _train(self, x, block_size=None, train_mode = None, node_weights=None, edge_weights=None, scheduler = None, n_parallel=None, **argv):
        self.input_dim = x.shape[1]
       
        if self.output_dim == None:
            self.output_dim = self.input_dim

        if self.max_comp == None:
            self.max_comp = min(self.input_dim, self.output_dim)
        else:
            self.max_comp = min(self.max_comp, self.input_dim, self.output_dim)

        print "Training IEVMNodeLRec..."

        #Remove mean before expansion
        self.x_mean = x.mean(axis=0) 
        x_zm=x-self.x_mean
        
        #Reorder/preprocess data before expansion, but only if there is an expansion.
        if self.pre_expansion_node_class != None and self.exp_node != None:
            self.pre_expansion_node = mdp.nodes.SFANode()
            self.pre_expansion_node.train(x_zm, block_size=block_size, train_mode = train_mode)
            self.pre_expansion_node.stop_training()
            x_pre_exp = self.pre_expansion_node.execute(x_zm)
        else:
            x_pre_exp = x_zm

        #Expand data
        if self.exp_node != None: 
            print "expanding x..."
            exp_x = self.exp_node.execute(x_pre_exp) #x_zm
        else:
            exp_x = x_pre_exp

        self.expanded_dim = exp_x.shape[1]
        
        #Apply SFA to it
        self.sfa_node = mdp.nodes.SFANode()
        self.sfa_node.train(exp_x, block_size=block_size, train_mode = train_mode)#, node_weights=None, edge_weights=None, scheduler = None, n_parallel=None)
        self.sfa_node.stop_training()
        print "self.sfa_node.d", self.sfa_node.d
            
        #Adaptive mechanism based on delta values
        if isinstance(self.max_preserved_sfa, float):
            self.num_sfa_features_preserved = (self.sfa_node.d <= self.max_preserved_sfa).sum()
        elif isinstance(self.max_preserved_sfa, int):
            self.num_sfa_features_preserved = self.max_preserved_sfa
        else:
            ex = "Cannot handle type of self.max_preserved_sfa"
            print ex
            raise Exception(ex)

        if self.num_sfa_features_preserved > self.output_dim:
            self.num_sfa_features_preserved = self.output_dim
            
        #self.num_sfa_features_preserved = 10
        sfa_x = self.sfa_node.execute(exp_x)
          
        #Truncate leaving only slowest features             
        sfa_x = sfa_x[:,0:self.num_sfa_features_preserved]
        
        #normalize sfa_x                    
        self.sfa_x_mean = sfa_x.mean(axis=0)
        self.sfa_x_std = sfa_x.std(axis=0)            
        print "self.sfa_x_mean=", self.sfa_x_mean
        print "self.sfa_x_std=", self.sfa_x_std
        if (self.sfa_x_std==0).any():
            er = "zero-component detected"
            raise Exception(er)
        n_sfa_x = (sfa_x - self.sfa_x_mean) / self.sfa_x_std 

        if self.reconstruct_with_sfa:      
            #Compress input
            if self.compress_input_with_pca:
                self.compress_node = mdp.nodes.PCANode(output_dim=self.compression_out_dim)
                self.compress_node.train(x_zm)
                x_pca = self.compress_node.execute(x_zm)
                print "compress: %d components out of %d sufficed for the desired compression_out_dim"%(x_pca.shape[1], x_zm.shape[1]), self.compression_out_dim
            else:
                x_pca = x_zm
    
            #approximate input linearly, done inline to preserve node
            self.lr_node = mdp.nodes.LinearRegressionNode()
            self.lr_node.train(n_sfa_x, x_pca) #Notice that the input "x"=n_sfa_x and the output to learn is "y" = x_pca
            self.lr_node.stop_training()  
            x_pca_app = self.lr_node.execute(n_sfa_x)
    
            if self.compress_input_with_pca:
                x_app = self.compress_node.inverse(x_pca_app)
            else:
                x_app = x_pca_app
        else:
            x_app = numpy.zeros_like(x_zm)
            
        #Remove linear approximation
        sfa_removed_x = x_zm - x_app
        
        #print "Data_variance(x_zm)=", data_variance(x_zm)
        #print "Data_variance(x_app)=", data_variance(x_app)
        #print "Data_variance(sfa_removed_x)=", data_variance(sfa_removed_x)
        #print "x_app.mean(axis=0)=", x_app
        #TODO:Compute variance removed by linear approximation
        if self.reconstruct_with_sfa and self.offsetting_mode not in ["QR_decomposition", "sensitivity_based_pure", "sensitivity_based_normalized"]:
            #Compute scaling and permutation of sfa signals. Note that permutation is not needed actually
            #TODO: what to use x, x_zm, x_pca???
            n_sfa_rankings = rank_expanded_signals_max(x_zm, n_sfa_x, x_zm, n_sfa_x, max_comp=self.max_comp, operation="linear_rec", 
                                                             max_num_samples_for_ev = self.max_num_samples_for_ev, max_test_samples_for_ev=self.max_test_samples_for_ev, 
                                                             offsetting_mode=self.offsetting_mode, verbose=True)
            print "n_sfa_rankings=", n_sfa_rankings, "n_sfa_rankings.sum()=", n_sfa_rankings.sum()
            #TODO: scale rankings to be equal to true explained variance from linear approximation
            self.magn_n_sfa_x = n_sfa_rankings
    #        self.perm_n_sfa_x = numpy.argsort(n_sfa_rankings[self.num_sfa_features_preserved:])[::-1]
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x ** self.exponent_variance #Scale according to ranking
        elif self.reconstruct_with_sfa and self.offsetting_mode == "QR_decomposition": #AKA Laurenz method for feature scaling( +rotation)
            M = self.lr_node.beta[1:,:].T # bias is used by default, we do not need to consider it
            Q,R = numpy.linalg.qr(M)
            self.Q = Q
            self.R = R
            self.Rpinv = pinv(R)
            s_n_sfa_x = numpy.dot(n_sfa_x, R.T)
        elif self.reconstruct_with_sfa and self.offsetting_mode == "sensitivity_based_pure": #AKA Laurenz method for feature scaling( +rotation)
            beta = self.lr_node.beta[1:,:] # bias is used by default, we do not need to consider it
            sens = (beta**2).sum(axis=1)
            self.magn_n_sfa_x = sens
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x ** self.exponent_variance       
        elif self.reconstruct_with_sfa and self.offsetting_mode == "sensitivity_based_normalized": #AKA Laurenz method for feature scaling( +rotation)
            beta = self.lr_node.beta[1:,:] # bias is used by default, we do not need to consider it
            sens = (beta**2).sum(axis=1)
            self.magn_n_sfa_x = sens * ((x_pca_app**2).sum(axis=1).mean() / sens.sum())
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x ** self.exponent_variance    
        else:
            self.magn_n_sfa_x = 0.01 * numpy.min(x_zm.var(axis=0)) # SFA components have a variance of 1/10000 smallest data variance
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x ** self.exponent_variance #Scale according to ranking

#        s_n_sfa_x = s_n_sfa_x[:,self.perm_n_sfa_x]       #Permute with slow features first, and then most important signal first
 
        #Apply PCA to sfa removed data             
#        pca_out_dim = self.expanded_dim - self.num_sfa_features_preserved

        self.pca_node = mdp.nodes.PCANode(reduce=True) #output_dim = pca_out_dim)
        self.pca_node.train(sfa_removed_x)
            
        #TODO:check that pca_out_dim > 0
        pca_x = self.pca_node.execute(sfa_removed_x)
        
        if self.pca_node.output_dim + self.num_sfa_features_preserved < self.output_dim:
            er = "Error, the number of features computed is SMALLER than the output dimensionality of the node" + \
            "self.pca_node.output_dim=", self.pca_node.output_dim, "self.num_sfa_features_preserved=", self.num_sfa_features_preserved, "self.output_dim=", self.output_dim
            raise Exception(er)

        #Finally output is the concatenation of scaled slow features and remaining pca components
        sfa_pca_x = numpy.concatenate((s_n_sfa_x, pca_x), axis=1)
        
        sfa_pca_x_truncated =  sfa_pca_x[:,0:self.output_dim]
        
        #Compute explained variance from amplitudes of output compared to amplitudes of input
        #Only works because amplitudes of SFA are scaled to be equal to explained variance, and because PCA is a rotation
        #And because data has zero mean
        self.evar =  (sfa_pca_x_truncated**2).sum() / (x_zm**2).sum()
        print "Explained variance (linearly)=", self.evar

        self.stop_training()

    def _is_invertible(self):
        return True

    def _execute(self, x):
        num_samples = x.shape[0]

        x_zm = x - self.x_mean

        if self.pre_expansion_node != None:
            x_pre_exp = self.pre_expansion_node.execute(x_zm)
        else:
            x_pre_exp = x_zm

        if self.exp_node != None: 
            exp_x = self.exp_node.execute(x_pre_exp)
        else:
            exp_x = x_pre_exp

        sfa_x = self.sfa_node.execute(exp_x)
        sfa_x = sfa_x[:,0:self.num_sfa_features_preserved]
        
        n_sfa_x = (sfa_x - self.sfa_x_mean) / self.sfa_x_std
            
#        if self.compress_input_with_pca:
#            x_pca = self.compress_node.execute(x_zm)
#        else:
#            x_pca = x_zm
        if self.reconstruct_with_sfa:
            #approximate input linearly, done inline to preserve node
            x_pca_app = self.lr_node.execute(n_sfa_x)
    
            if self.compress_input_with_pca:
                x_app = self.compress_node.inverse(x_pca_app)
            else:
                x_app = x_pca_app
        else:
            x_app = numpy.zeros_like(x_zm)

        #Remove linear approximation
        sfa_removed_x = x_zm - x_app
        
        if self.reconstruct_with_sfa and self.offsetting_mode == "QR_decomposition": #AKA Laurenz method for feature scaling( +rotation)
            s_n_sfa_x = numpy.dot(n_sfa_x, self.R.T)
        else:
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x ** self.exponent_variance #Scale according to ranking

        #Apply PCA to sfa removed data             
        pca_x = self.pca_node.execute(sfa_removed_x)
        
        #Finally output is the concatenation of scaled slow features and remaining pca components
        sfa_pca_x = numpy.concatenate((s_n_sfa_x, pca_x), axis=1)
        
        sfa_pca_x_truncated =  sfa_pca_x[:,0:self.output_dim]

#        print "Data_variance(x_zm)=", data_variance(x_zm)
#        print "Data_variance(x_app)=", data_variance(x_app)
#        print "Data_variance(sfa_removed_x)=", data_variance(sfa_removed_x)
#        print "x_app.mean(axis=0)=", x_app

        return sfa_pca_x_truncated

#      
#        verbose=False
#        if verbose:
#            print "x[0]=",x_orig[0]
#            print "x_zm[0]=", x[0]
#            print "exp_x[0]=", exp_x[0]
#            print "s_x_1[0]=", s_x_1[0]
#            print "sfa_removed_x[0]=", sfa_removed_x[0]
#            print "proj_sfa_x[0]=", proj_sfa_x[0]
#            print "pca_x[0]=", pca_x[0]
#            print "n_pca_x[0]=", n_pca_x[0]        
#            print "sfa_x[0]=", sfa_x[0] + self.sfa_x_mean
#            print "s_x_2_truncated[0]=", s_x_2_truncated[0]
#            print "sfa_filtered[0]=", sfa_filtered[0]

    def _inverse(self, y, linear_inverse=True):
        if linear_inverse:
            return self.linear_inverse(y)
        else:
            return self.non_linear_inverse(y)

    def non_linear_inverse(self, y, verbose=False):
        x_lin = self.linear_inverse(y)
        rmse_lin = ((y - self.execute(x_lin))**2).sum(axis=1).mean()**0.5 
#        scipy.optimize.leastsq(func, x0, args=(), Dfun=None, full_output=0, col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=0.0, factor=100, diag=None)
        x_nl = numpy.zeros_like(x_lin)
        y_dim = y.shape[1]
        x_dim = x_lin.shape[1]
        if y_dim < x_dim:
            num_zeros_filling = x_dim - y_dim
        else:
            num_zeros_filling = 0
        if verbose:
            print "x_dim=", x_dim, "y_dim=", y_dim, "num_zeros_filling=", num_zeros_filling
        y_long = numpy.zeros(y_dim+num_zeros_filling)

        for i, y_i in enumerate(y):
            y_long[0:y_dim]= y_i
            if verbose:
                print "x_0=", x_lin[i], 
                print "y_long=", y_long            
            plsq = scipy.optimize.leastsq(func=f_residual, x0=x_lin[i], args=(self, y_long), full_output=False)
            x_nl_i = plsq[0]
            if verbose:
                print "x_nl_i=", x_nl_i, "plsq[1]=", plsq[1]
            if plsq[1] != 2:
                print "Quitting: plsq[1]=", plsq[1]
                #quit()
            x_nl[i] = x_nl_i
            print "|E_lin(%d)|="%i, ((y_i - self.execute(x_lin[i].reshape((1,-1))))**2).sum()**0.5, 
            print "|E_nl(%d)|="%i, ((y_i - self.execute(x_nl_i.reshape((1,-1))))**2).sum()**0.5
        rmse_nl = ((y - self.execute(x_nl))**2).sum(axis=1).mean()**0.5
        print "rmse_lin(all samples)=", rmse_lin, "rmse_nl(all samples)=", rmse_nl
        return x_nl

    def linear_inverse(self, y):
        num_samples = y.shape[0]
        if y.shape[1] != self.output_dim:
            er = "Serious dimensionality inconsistency:", y.shape[0], self.output_dim
            raise Exception(er)

        sfa_pca_x_full = numpy.zeros((num_samples, self.pca_node.output_dim+self.num_sfa_features_preserved)) #self.input_dim
        #print "self.output_dim=", self.output_dim, "y.shape=", y.shape, "sfa_pca_x_full.shape=", sfa_pca_x_full.shape, "self.num_sfa_features_preserved=",self.num_sfa_features_preserved 
        sfa_pca_x_full[:,0:self.output_dim] = y
        
        s_n_sfa_x = sfa_pca_x_full[:, 0:self.num_sfa_features_preserved]
        pca_x = sfa_pca_x_full[:, self.num_sfa_features_preserved:]
        
        if pca_x.shape[1]>0:
            #print "self.pca_node.input_dim=",self.pca_node.input_dim, "self.pca_node.output_dim=",self.pca_node.output_dim
            sfa_removed_x = self.pca_node.inverse(pca_x)
        else:
            sfa_removed_x = numpy.zeros((num_samples, self.input_dim))

        #print "s_n_sfa_x.shape=", s_n_sfa_x.shape, "magn_n_sfa_x.shape=", self.magn_n_sfa_x.shape       
        if self.reconstruct_with_sfa and self.offsetting_mode == "QR_decomposition": #AKA Laurenz method for feature scaling( +rotation)          
            n_sfa_x = numpy.dot(s_n_sfa_x, self.Rpinv.T)
        else:
            n_sfa_x = s_n_sfa_x / self.magn_n_sfa_x ** self.exponent_variance

        #sfa_x = n_sfa_x * self.sfa_x_std + self.sfa_x_mean 
        if self.reconstruct_with_sfa:
            x_pca_app = self.lr_node.execute(n_sfa_x)
        
            if self.compress_input_with_pca:
                x_app = self.compress_node.inverse(x_pca_app)
            else:
                x_app = x_pca_app        
        else:
            x_app = numpy.zeros_like(sfa_removed_x)

        x_zm = sfa_removed_x + x_app
        
        x = x_zm + self.x_mean

        #print "Data_variance(x_zm)=", data_variance(x_zm)
        #print "Data_variance(x_app)=", data_variance(x_app)
        #print "Data_variance(sfa_removed_x)=", data_variance(sfa_removed_x)
        #print "x_app.mean(axis=0)=", x_app

#        verbose=False
#        if verbose:
#            print "x[0]=",x[0]
#            print "zm_x[0]=", zm_x[0]
#            print "exp_x[0]=", exp_x[0]
#            print "s_x_1[0]=", s_x_1[0]
#            print "proj_sfa_x[0]=", proj_sfa_x[0]
#            print "sfa_removed_x[0]=", sfa_removed_x[0]
#            print "pca_x[0]=", pca_x[0]
#            print "n_pca_x[0]=", n_pca_x[0]        
#            print "sfa_x[0]=", sfa_x[0]
#        
        return x

def export_to_libsvm(labels_classes, features, filename):
    dim_features = features.shape[1]
    file = open(filename, "wb")
    if len(features) != len(labels_classes):
        er="number of labels_classes %d does not match number of samples %d!"%(len(labels_classes), len(features))
        raise Exception(er)
    for i in range(len(features)):
        file.write("%d"%labels_classes[i])
        for j in range(dim_features):
            file.write(" %d:%f"%(j+1, features[i,j]))
        file.write("\n")
    file.close()

def is_monotonic_increasing(x):
    prev = x[0]
    for curr in x[1:]:
        if curr <= prev:
            return False
        prev = curr
    return True

def compute_average_labels_for_each_class(classes, labels):
    all_classes = numpy.unique(classes)
    avg_labels = numpy.zeros(len(all_classes))
    for i, cl in enumerate(all_classes):
        avg_label = labels[classes == cl].mean()
        avg_labels[i] = avg_label
    return avg_labels

def map_class_numbers_to_avg_label(all_classes, avg_labels, class_numbers):
    if not(is_monotonic_increasing(all_classes)):
        er = "Array of class numbers should be monotonically increasing:", all_classes
        raise Exception(er)
    if not(is_monotonic_increasing(avg_labels)):
        er = "Array of labels should be monotonically increasing:", avg_labels
        raise Exception(er)
    if len(all_classes) != len(avg_labels):
        er = "Array of classes should have the same lenght as the array of labels:", len(all_classes), " vs. ", len(avg_labels)

    indices = numpy.searchsorted(all_classes, class_numbers)
    return avg_labels[indices]

def map_labels_to_class_number(all_classes, avg_labels, labels):
    if not(is_monotonic_increasing(all_classes)):
        er = "Array of class numbers should be monotonically increasing:", all_classes
        raise Exception(er)
    if not(is_monotonic_increasing(avg_labels)):
        er = "Array of labels should be monotonically increasing:", avg_labels
        raise Exception(er)
    if len(all_classes) != len(avg_labels):
        er = "Array of classes should have the same lenght as the array of labels:", len(all_classes), " vs. ", len(avg_labels)

    interval_midpoints = (avg_labels[1:]+avg_labels[:-1])/2.0

    indices = numpy.searchsorted(interval_midpoints, labels)
    return all_classes[indices]

def random_boolean_array(size):
    return numpy.random.randint(2, size=size)==1


def generate_random_sigmoid_weights(input_dim, num_features):
    c = numpy.random.normal(loc=0.0, scale=1.0, size=(input_dim, num_features))
    l = numpy.random.normal(loc=0.0, scale=1.0, size=num_features)
    return (c,l)

def extract_sigmoid_features(x, c1, l1, scale=1.0, offset=0.0, smart_fix=False):
    if x.shape[1] != c1.shape[0] or c1.shape[1] != len(l1):
        er = "Array dimensions mismatch: x.shape =" + str(x.shape) + ", c1.shape =" + str(c1.shape) + ", l1.shape=" + str(l1.shape)
        print er
        raise Exception(er)   
    s = numpy.dot(x,c1)+l1
    f = numpy.tanh(s)
    if smart_fix: 
        #replace features with l1 = -1.0 to x^T * c1[i]
        #replace features with l1 =  0.8 to 0.8 expo(x^T * c1[i])
        print "f.shape=", f.shape
        print "numpy.dot(x,c1[:,0]).shape=", numpy.dot(x,c1[:,0]).shape
        fixed=0
        for i in range(c1.shape[1]):
            if l1[i] == 0.8:
                f[:,i] = numpy.abs(numpy.dot(x,c1[:,i]))**0.8
                fixed += 1
            elif l1[i] == -1.0:
                f[:,i] = numpy.dot(x,c1[:,i])
                fixed += 1
    print "Number of features apted to either identity or 08Expo:", fixed
    return f * scale + offset

#sf_matrix has shape input_dim x output_dim
def evaluate_coefficients(sf_matrix):
    #Exponentially decaying weights
#    weighting = numpy.e ** -numpy.arange(sf_matrix.shape[1])
    weighting = 2.0 ** -numpy.arange(sf_matrix.shape[1])
    weighted_relevances = numpy.abs(sf_matrix) * weighting
    relevances = weighted_relevances.sum(axis=1)
    return relevances


class SFAAdaptiveNLNode(mdp.Node):
#     """Node implementing SFA with adaptive non-linearity learning.
#    """    
    def __init__(self, input_dim = None, output_dim=None, pre_expansion_node_class = None, final_expanded_dim = None, 
                 initial_expansion_size = None, starting_point=None, expansion_size_decrement = None, expansion_size_increment = None, 
                 number_iterations = 2, **argv ):
        super(SFAAdaptiveNLNode, self).__init__(input_dim =input_dim, output_dim=output_dim, **argv)
        self.pre_expansion_node_class = pre_expansion_node_class
        self.pre_expansion_node = None

        self.final_expanded_dim = final_expanded_dim
        self.initial_expansion_size =initial_expansion_size
        self.starting_point = starting_point
        self.expansion_size_decrement =expansion_size_decrement
        self.expansion_size_increment =expansion_size_increment
        self.number_iterations = number_iterations
        self.sfa_node = None
        self.f1_mean = None
        self.f1_std = None
        
    def is_trainable(self):
        return True

    #sfa_block_size, sfa_train_mode, etc. would be preferred
    # max_preserved_sfa=1.995
    def _train(self, x, block_size=None, train_mode = None, node_weights=None, edge_weights=None, scheduler = None, n_parallel=None, **argv):
        self.input_dim = x.shape[1]
       
        if self.output_dim == None:
            self.output_dim = self.input_dim

        print "Training SFAAdaptiveNLNode..."
        print "block_size =", block_size, ", train_mode =", train_mode
        print "x.shape=", x.shape, "self.starting_point=", self.starting_point  

        #Remove mean before expansion
#        self.x_mean = x.mean(axis=0) 
#        x_zm=x-self.x_mean

#TODO:Make this code more pretty (refactoring)
        if self.starting_point == "Identity":
            print "wrong1"
            c0 = numpy.identity(self.input_dim)
            l0 = numpy.ones(self.input_dim)* -1.0 #Code identity
        elif self.starting_point == "08Exp":
            print "good 1"
            c0 = numpy.concatenate((numpy.identity(self.input_dim), numpy.identity(self.input_dim)),axis=1)
            l0 = numpy.concatenate((numpy.ones(self.input_dim)*-1.0, numpy.ones(self.input_dim)* 0.8),axis=0)
            
        if self.starting_point == "Identity" or self.starting_point == "08Exp":
            print "good 2"
            remaining_feats = self.initial_expansion_size - c0.shape[1]
            print "remaining_feats =", remaining_feats
            if remaining_feats < 0:
                er = "Error, features needed for identity or 08Exp exceeds number of features availabe"+ \
                "remaining_feats=%d < 0"%remaining_feats+". self.initial_expansion_size=%d"%self.initial_expansion_size + \
                "c0.shape[1]%d"%c0.shape[1]
                raise Exception(er)
            c2, l2 = generate_random_sigmoid_weights(self.input_dim, remaining_feats)
            c1 = numpy.concatenate((c0, c2), axis=1)
            l1 = numpy.concatenate((l0, l2), axis=0)
        else:
            print "wrong wrong"
            c1, l1 = generate_random_sigmoid_weights(self.input_dim, self.initial_expansion_size-self.expansion_size_increment)
                   
        for num_iter in range(self.number_iterations):    
            print "**************** Iteration %d of %d ********************"%(num_iter, self.number_iterations)
            if num_iter > 0: #Only add additional features after first iteration
                cp, lp = generate_random_sigmoid_weights(self.input_dim, self.expansion_size_increment)
                c1 = numpy.append(c1, cp, axis=1)
                l1 = numpy.append(l1, lp, axis=0)
            #print "c1=", c1
            #print "l1=", l1
            f1 = extract_sigmoid_features(x, c1, l1, smart_fix=True)
            f1_mean = f1.mean(axis=0)
            f1 = f1 - f1_mean
            f1_std = f1.std(axis=0) 
            f1 = f1 / f1_std
            #print "Initial features f1=", f1
            print "f1.shape=", f1.shape
            print "f1[0]=", f1[0]
            print "f1[-1]=", f1[-1]
            sfa_node = mdp.nodes.SFANode(output_dim=self.output_dim)
            sfa_node.train(f1, block_size=block_size, train_mode = train_mode, node_weights=node_weights, edge_weights=edge_weights, scheduler = scheduler, n_parallel=n_parallel)
            sfa_node.stop_training()
            print "self.sfa_node.d (full expanded) =", sfa_node.d
            
            #Evaluate features based on sfa coefficient
            coeffs = evaluate_coefficients(sfa_node.sf)
            print "Scores of each feature from SFA coefficients:", coeffs 
            #find indices of best features. Largest scores first
            best_feat_indices = coeffs.argsort()[::-1] 
            print "indices of best features:", best_feat_indices
            
            #remove worst expansion_size_decrement features
            if num_iter < self.number_iterations - 1: #Except during last iteration
                best_feat_indices = best_feat_indices[:-self.expansion_size_decrement]
            
            c1 = c1[:,best_feat_indices]
            l1 = l1[best_feat_indices]
            #print "cc=", cc
            #print "ll=", ll

        if c1.shape[1] > self.final_expanded_dim:
            c1 = c1[:,:self.final_expanded_dim]
            l1 = l1[:self.final_expanded_dim]

        self.c1 = c1
        self.l1 = l1
        print "self.c1.shape=,",self.c1.shape,"self.l1.shape=,",self.l1.shape
        print "Learning of non-linear features finished"
        f1 = extract_sigmoid_features(x, self.c1, self.l1, smart_fix=True)
        self.f1_mean = f1.mean(axis=0)
        f1 -= self.f1_mean
        self.f1_std = f1.std(axis=0) 
        f1 /= self.f1_std
        self.sfa_node = mdp.nodes.SFANode(output_dim=self.output_dim)
        self.sfa_node.train(f1, block_size=block_size, train_mode = train_mode, node_weights=node_weights, edge_weights=edge_weights, scheduler = scheduler, n_parallel=n_parallel)
        self.sfa_node.stop_training()
        print "self.sfa_node.d (final features) =", self.sfa_node.d
        #Evaluate features based on sfa coefficient
        coeffs = evaluate_coefficients(self.sfa_node.sf)
        print "evaluation of each features from SFA coefficients: ", coeffs 
        #find indices of best features. Largest scores first
        best_feat_indices = coeffs.argsort()[::-1] 
        print "indices of best features:", best_feat_indices
        
        print "f1.shape=", f1.shape
        #Train linear regression node for a linear approximation to inversion
        self.lr_node = mdp.nodes.LinearRegressionNode()
        y = self.sfa_node.execute(f1)
        self.lr_node.train(y, x)
        self.lr_node.stop_training()
        x_app = self.lr_node.execute(y)
        ev_linear_inverse = compute_explained_var(x, x_app) / data_variance(x)
        print "EV_linear_inverse (train)=", ev_linear_inverse
        self.stop_training()

    def _is_invertible(self):
        return True

    def _execute(self, x):
        num_samples = x.shape[0]

        f1 = extract_sigmoid_features(x, self.c1, self.l1, smart_fix=True)
        f1 -= self.f1_mean
        f1 /= self.f1_std
        return self.sfa_node.execute(f1)
      
    def _inverse(self, y, linear_inverse=True):
        x_app = self.lr_node.execute(y)
        return x_app

#TODO:Finish this and correct it
def indices_training_graph_split(num_samples, train_mode="regular", block_size=None, num_parts=1):
    if train_mode == "regular":
        indices = numpy.arange(num_samples)
        block_assignment = (indices*1.0*num_parts/num_samples).astype(int)
        numpy.random.shuffle(block_assignment)
        part_indices = []
        for num_part in range(num_parts):
            part = indices[block_assignment==num_part]
            part_indices.append(part)

    elif train_mode in ["serial", "sequence"]:
        if isinstance(block_size, int):
            shuffled_indices = numpy.zeros(num_samples)
            for block in range(num_samples / block_size):
                shuffled_indices[block*block_size:(block+1)*block_size] = (numpy.arange(block_size)*1.0*num_parts/block_size).astype(int) 
            
            for block in range(num_samples / block_size):
                shuffled_indices = (numpy.arange(block_size)*1.0*num_parts/block_size).astype(int)
            
                numpy.random.shuffle(shuffled_indices[block*block_size:(block+1)*block_size])
            part_indices = []
            for num_part in range(num_parts):
                part = indices[block_assignment==num_part]
                part_indices.append(part)            
        else:
            er = "Inhomogeneous block sizes not supported for now"
            raise Exception(er)
    
    elif train_mode == "complete":
        print "Mode unuported for now... FIX this!!!"

#Cumulative score metric
def cumulative_score(ground_truth, estimation, largest_error, integer_rounding=True):
    if len(ground_truth) != len(estimation):
        er = "ground_truth and estimation have different number of elements"
        raise Exception(er)

    if integer_rounding:
        _estimation = numpy.rint(estimation)
    else:
        _estimation = estimation

    N_e_le_j = (numpy.absolute(_estimation-ground_truth) <= largest_error).sum()
    return N_e_le_j * 1.0 / len(ground_truth)

##################### GSFA functions/classes ################

#Special purpose object to compute the covariance matrices used by SFA.
#It efficiently supports special training methods: clustered, serial, mixed
#TODO: Remove falsely global unneeded variables
#TODO: 
class CovDCovMatrix(object):
    def __init__(self, block_size):
        self.block_size = block_size
        self.sum_x = None
        self.sum_prod_x = None
        self.num_samples = 0
        self.sum_diffs = None 
        self.sum_prod_diffs = None  
        self.num_diffs = 0
        self.last_block = None
#Permanent Results Storage
        self.cov_mtx = None
        self.avg = None
        self.dcov_mtx = None
        self.tlen = 0            

    def AddSamples(self, sum_prod_x, sum_x, num_samples, weight=1.0):
        weighted_sum_x = sum_x * weight
        weighted_sum_prod_x = sum_prod_x * weight
        weighted_num_samples = num_samples * weight
        
        if self.sum_prod_x is None:
            self.sum_prod_x = weighted_sum_prod_x
            self.sum_x = weighted_sum_x
        else:
            self.sum_prod_x = self.sum_prod_x + weighted_sum_prod_x
            self.sum_x = self.sum_x + weighted_sum_x

        self.num_samples = self.num_samples + weighted_num_samples

    def AddDiffs(self, sum_prod_diffs, num_diffs, weight=1.0):
        #weighted_sum_diffs = sum_diffs * weight
        weighted_sum_prod_diffs = sum_prod_diffs * weight
        weighted_num_diffs = num_diffs * weight
        
        if self.sum_prod_diffs is None:
            self.sum_prod_diffs = weighted_sum_prod_diffs
#            self.sum_diffs = weighted_sum_diffs
        else:
            self.sum_prod_diffs = self.sum_prod_diffs + weighted_sum_prod_diffs
#            self.sum_diffs = self.sum_diffs + weighted_sum_diffs

        self.num_diffs = self.num_diffs + weighted_num_diffs   
       
    #TODO:Add option to skip last sample from Cov part.
    #Add unlabeled samples to Cov matrix (DCov remains unmodified)
    def updateUnlabeled(self, x, weight=1.0):
        num_samples, dim = x.shape
                       
        sum_x = x.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)
    
    #TODO:Add option to skip last sample from Cov part.
    def updateRegular(self, x, weight=1.0):
        num_samples, dim = x.shape
        
        #Update Cov Matrix
        sum_x = x.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x.T, x)            
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)

        #Update DCov Matrix
        diffs = x[1:,:]-x[:-1,:]
        num_diffs = num_samples - 1
#        sum_diffs = diffs.sum(axis=0)
        sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
        self.AddDiffs(sum_prod_diffs, num_diffs, weight)

    #Usually: sum(node_weights)=num_samples 
    def updateGraph(self, x, node_weights=None, edge_weights=None, ignore_edge_avg=True):
        num_samples, dim = x.shape

        if node_weights==None:
            node_weights = numpy.ones(num_samples)

        if len(node_weights) != num_samples:
            er = "Node weights should be the same length as the number of samples"
            raise Exception(er)

        if edge_weights==None:
            er = "edge_weights should be a dictionary with entries: d[(i,j)] = w_{i,j}"
            raise Exception(er)

        node_weights_column = node_weights.reshape((num_samples, 1))                
        #Update Cov Matrix
        weighted_x = x * node_weights_column
    
        weighted_sum_x = weighted_x.sum(axis=0) 
        weighted_sum_prod_x = mdp.utils.mult(x.T, weighted_x)
        weighted_num_samples = node_weights.sum()            
        self.AddSamples(weighted_sum_prod_x, weighted_sum_x, weighted_num_samples)

        #Update DCov Matrix
        num_diffs = len(edge_weights)
        diffs = numpy.zeros((num_diffs, dim))
        weighted_diffs = numpy.zeros((num_diffs, dim))
        weighted_num_diffs = 0
        for ii, (i,j) in enumerate(edge_weights.keys()):
            diff = x[j,:]-x[i,:]
            diffs[ii] = diff
            w_ij = edge_weights[(i,j)]
            weighted_diff = diff * w_ij
            weighted_diffs[ii] = weighted_diff
            weighted_num_diffs += w_ij

        weighted_sum_diffs = weighted_diffs.sum(axis=0)
        weighted_sum_prod_diffs = mdp.utils.mult(diffs.T, weighted_diffs)
        self.AddDiffs(weighted_sum_prod_diffs, weighted_num_diffs)

    #Note, this method makes sense from consistency constraints for "larger" windows. 
    def updateMirroringSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth #window_halfwidth is way too long to write it

        #Update Cov Matrix
        #All samples have same weight
        sum_x = x.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x.T, x)       
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)
        
        #Update DCov Matrix
        #First mirror the borders
        x_mirror = numpy.zeros((num_samples+2*width, dim))
        x_mirror[width:-width] = x #center part
        x_mirror[0:width, :] = x[0:width, :][::-1,:] #first end
        x_mirror[-width:, :] = x[-width:, :][::-1,:] #second end

        #Center part
        x_full = x
        sum_prod_x_full = mdp.utils.mult(x_full.T, x_full)
#        print "sum_prod_x_full[0]=", sum_prod_x_full[0]
#        print "(2*width+1)*sum_prod_x_full=", (2*width+1)*sum_prod_x_full[0:3,0:3]

        Aacc123 = numpy.zeros((dim, dim))
        for i in range(0, 2*width): # [0, 2*width-1]
            Aacc123 += (i+1)*mdp.utils.mult(x_mirror[i:i+1,:].T, x_mirror[i:i+1,:]) #(i+1)=1,2,3..., 2*width

        for i in range(num_samples, num_samples+2*width): # [num_samples-width, num_samples-1]
            Aacc123 += (num_samples+2*width-i)*mdp.utils.mult(x_mirror[i:i+1,:].T, x_mirror[i:i+1,:])
#            print (num_samples+2*width-i)
        #Warning [-1,0] does not work!!!    
        #for i in range(0, 2*width): # [0, 2*width-1]
        #    Aacc123 += (i+1)*mdp.utils.mult(x_mirror[-(i+1):-i,:].T, x_mirror[-(i+1):-i,:]) #(i+1)=1,2,3..., 2*width
        x_middle = x_mirror[2*width:-2*width,:] #intermediate values of x, which are connected 2*width+1 times
        Aacc123 += (2*width+1)*mdp.utils.mult(x_middle.T, x_middle)
#        print "Aacc123[0:3,0:3]=", Aacc123[0:3,0:3]

        b = numpy.zeros((num_samples+1+2*width, dim))
        b[1:] = x_mirror.cumsum(axis=0)
        B = b[2*width+1:]-b[0:-2*width-1]
        Bprod = mdp.utils.mult(x_full.T, B)
#        print "Bprod[0:3,0:3]=", Bprod[0:3,0:3]
         
        sum_prod_diffs_full = (2*width+1)*sum_prod_x_full + (Aacc123) - Bprod - Bprod.T
        num_diffs = num_samples*(2*width) # removed zero differences
#         print "N3=", num_diffs
        sum_diffs = numpy.zeros(dim)
        self.AddDiffs(sum_prod_diffs_full, num_diffs, weight)        
    
    #Note, this method makes sense from consistency constraints for "larger" windows. 
    def updateSlowMirroringSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth #window_halfwidth is way too long to write it
      
        #Update Cov Matrix
        #All samples have same weight
        sum_x = x.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x.T, x)       
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)
                
        #Update DCov Matrix
        #window = numpy.ones(2*width+1) # Rectangular window
        x_mirror = numpy.zeros((num_samples+2*width, dim))
        x_mirror[width:-width] = x #center part
        x_mirror[0:width, :] = x[0:width, :][::-1,:] #first end
        x_mirror[-width:, :] = x[-width:, :][::-1,:] #second end
                
        diffs = numpy.zeros((num_samples, dim))
        #print "width=", width
        for offset in range(-width, width+1):
            if offset == 0:
                pass
            else:
                diffs = x_mirror[offset+width:offset+width+num_samples,:]-x
                                 
                sum_diffs = diffs.sum(axis=0) 
                sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
                num_diffs = len(diffs)
# WARNING!
#                self.AddDiffs(sum_prod_diffs, sum_diffs, num_diffs, weight * window[width+offset])
                self.AddDiffs(sum_prod_diffs, sum_diffs, num_diffs, weight)               
    
    def updateFastSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth

#MOST CORRECT VERSION
        x_sel = x+0.0
        w_up = numpy.arange(width, 2*width) / (2.0*width)
        w_up = w_up.reshape((width, 1))
        w_down = numpy.arange(2*width-1,width-1,-1) / (2.0*width)
        w_down = w_down.reshape((width, 1))
        x_sel[0:width, :] =  x_sel[0:width, :] * w_up
        x_sel[-width:, :] =  x_sel[-width:, :] * w_down
        
        sum_x = x_sel.sum(axis=0) 
        sum_prod_x = mdp.utils.mult(x_sel.T, x_sel)       #There is also a bug here!!!! fix me!!!!
        self.AddSamples(sum_prod_x, sum_x, num_samples - (0.5 *window_halfwidth-0.5), weight)
        
        #Update DCov Matrix
        #First we compute the borders.
        #Left border
        for i in range(0, width): # [0, width -1]
            diffs = x[0:width+i+1,:]-x[i,:]
            sum_diffs = diffs.sum(axis=0) 
            sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
            num_diffs = len(diffs) - 1 # removed zero differences
#            print "N1=", num_diffs
#            print "sum_prod_diffs[0]=", sum_prod_diffs[0]
            self.AddDiffs(sum_prod_diffs, num_diffs, weight)
        #Right border
        for i in range(num_samples-width, num_samples): # [num_samples-width, num_samples-1]
            diffs = x[i-width:num_samples,:]-x[i,:]                                 
            sum_diffs = diffs.sum(axis=0) 
            sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
            num_diffs = len(diffs) - 1 # removed zero differences
#            print "N2=", num_diffs
#            print "sum_prod_diffs[0]=", sum_prod_diffs[0]
            self.AddDiffs(sum_prod_diffs, num_diffs, weight)


        #Center part
        x_full = x[width:num_samples-width,:]
        sum_prod_x_full = mdp.utils.mult(x_full.T, x_full)
#        print "sum_prod_x_full[0]=", sum_prod_x_full[0]

        Aacc123 = numpy.zeros((dim, dim))
        for i in range(0, 2*width): # [0, 2*width-1]
            Aacc123 += (i+1)*mdp.utils.mult(x[i:i+1,:].T, x[i:i+1,:]) #(i+1)=1,2,3..., 2*width

        for i in range(num_samples-2*width, num_samples): # [num_samples-width, num_samples-1]
            Aacc123 += (num_samples-i)*mdp.utils.mult(x[i:i+1,:].T, x[i:i+1,:]) #(num_samples-1)=2*width,...,3,2,1

        x_middle = x[2*width:num_samples-2*width,:] #intermediate values of x, which are connected 2*width+1 times

        Aacc123 += (2*width+1)*mdp.utils.mult(x_middle.T, x_middle)

##        a = numpy.zeros((num_samples+1, dim, dim))
##        x1 = numpy.reshape(x, (num_samples, dim, 1))
##        x2 = numpy.reshape(x, (num_samples, 1, dim))
##        pp =numpy.multiply(x1, x2)
##        #print "pp[0]", pp[0], pp.shape
##        #for i in range(1,num_samples+1):
##        #    #reimplement this using sumcum?
##        #    a[i] = a[i-1] + mdp.utils.mult(x[i-1:i,:].T, x[i-1:i,:])
##        #print "a[1]", a[1]
##        #
##        #a2 = a + 0.0
##        a[1:] = pp.cumsum(axis=0)
        #print "a[-1]", a[-1]
        #print "a2[-1]", a2[-1]
            #print "a[i]", a[i]
#        print "a[0:2]", a[0:2]
        b = numpy.zeros((num_samples+1, dim))
        b[1:] = x.cumsum(axis=0)
#        for i in range(1,num_samples+1):
#            b[i] = b[i-1] + x[i-1,:]
#        A = a[2*width+1:]-a[0:-2*width-1]
        B = b[2*width+1:]-b[0:-2*width-1]
#        Aacc = A.sum(axis=0)
#        print "Aacc[0]=", Aacc[0]
        Bprod = mdp.utils.mult(x_full.T, B)
#        print "Bprod[0]=", Bprod[0]
#        print sum_prod_x_full.shape, Aacc.shape, Bprod.shape
        sum_prod_diffs_full = (2*width+1)*sum_prod_x_full + (Aacc123) - Bprod - Bprod.T
        num_diffs = (num_samples-2*width)*(2*width) # removed zero differences
#         print "N3=", num_diffs
        sum_diffs = numpy.zeros(dim)
        self.AddDiffs(sum_prod_diffs_full, num_diffs, weight)        
    #Add sliding window data to covariance matrices
    def updateSlidingWindow(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth

        #Update Cov Matrix
        #Warning, truncating samples to avoid edge problems
#TRUNCATED VERSION
#        x_sel = x[window_halfwidth:-window_halfwidth,:]
#        sum_x = x_sel.sum(axis=0) 
#        sum_prod_x = mdp.utils.mult(x_sel.T, x_sel)       
#        self.AddSamples(sum_prod_x, sum_x, num_samples-2*window_halfwidth, weight)

#MOST CORRECT VERSION
        x_sel = x+0.0
        w_up = numpy.arange(width, 2*width) / (2.0 * width)
        w_up = w_up.reshape((width, 1))
        w_down = numpy.arange(2*width-1,width-1,-1) / (2.0 * width)
        w_down = w_down.reshape((width, 1))
        x_sel[0:width, :] =  x_sel[0:width, :] * w_up
        x_sel[-width:, :] =  x_sel[-width:, :] * w_down
        
        sum_x = x_sel.sum(axis=0)
        print "F:)",
        sum_prod_x = mdp.utils.mult(x_sel.T, x)       #Bug fixed!!! computing w * X^T * X, with X=(x1,..xN)^T
        self.AddSamples(sum_prod_x, sum_x, num_samples - (0.5 *window_halfwidth-0.5), weight)
        
        #Update DCov Matrix
        window = numpy.ones(2*width+1) # Rectangular window
        
        diffs = numpy.zeros((num_samples-2*width, dim))
        #print "width=", width
        for offset in range(-width, width+1):
            if offset == 0:
                pass
            else:
                if offset > 0:
                    diffs = x[offset:,:]-x[0:num_samples-offset,:]
                else: #offset < 0, only makes sense if window asymetric! (be careful about mean, however)
                    abs_offset = -1 * offset
                    diffs = x[0:num_samples-abs_offset,:] - x[abs_offset:,:]
                                 
                sum_diffs = diffs.sum(axis=0) 
                sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
                num_diffs = len(diffs)
# WARNING!
#                self.AddDiffs(sum_prod_diffs, sum_diffs, num_diffs, weight * window[width+offset])
                self.AddDiffs(sum_prod_diffs, num_diffs, weight)

# ERRONEOUS VERSION USED FOR PERFORMANCE REPORT:
#                diffs = x[width+offset:num_samples-width+offset,:]-x[width:num_samples-width,:]
#                                 
#                sum_diffs = diffs.sum(axis=0) 
#                sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
#                num_diffs = len(diffs)
## WARNING!
##                self.AddDiffs(sum_prod_diffs, sum_diffs, num_diffs, weight * window[width+offset])
#                self.AddDiffs(sum_prod_diffs, sum_diffs, num_diffs, weight)

#            def AddDiffs(self, sum_prod_diffs, sum_diffs, num_diffs, weight=1.0):
    #Add samples belonging to a serial training graph
    #TODO:Remove torify option 
    #NOTE: include_last_block not needed
    #Torify: copy first block to the end of the serial
    def updateSerial(self, x, block_size, Torify=False, weight=1.0, include_last_block=True):
        num_samples, dim = x.shape
        if block_size == None:
            er = "block_size must be specified"
            raise Exception(er)
            block_size = self.block_size
        if isinstance(block_size, (numpy.ndarray)):
            err = "Inhomogeneous block sizes not yet supported in updateSerial"
            raise Exception(err)
                    
        if Torify is True:    
            x2 = numpy.zeros((num_samples+block_size, dim))
            x2[0:num_samples] = x
            x2[num_samples:] = x[0:block_size]
            x = x2
            num_samples = num_samples + block_size
                       
        if num_samples % block_size > 0:
            err = "Consistency error: num_samples is not a multiple of block_size"
            raise Exception(err)
        num_blocks = num_samples / block_size

        #warning, plenty of dtype missing!!!!!!!!
        #Optimize computation of x.T ???
        #Warning, remove last element of x (incremental computation)!!!

        #Correlation Matrix. Computing sum of outer products (the easy part)
        xp = x[block_size:num_samples-block_size]
        x_b_ini = x[0:block_size]
        x_b_end =  x[num_samples-block_size:]
        sum_x = x_b_ini.sum(axis=0) + 2 * xp.sum(axis=0) + x_b_end.sum(axis=0)

#            print "Sum_x[0:3] is ", sum_x[0:3]
        sum_prod_x = mdp.utils.mult(x_b_ini.T, x_b_ini) + 2 *  mdp.utils.mult(xp.T, xp) + mdp.utils.mult(x_b_end.T, x_b_end)        
        num_samples = 2 * block_size + 2 * (num_samples- 2 * block_size)
        
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)
        
        self.last_block = x[num_samples-block_size:]

        #DCorrelation Matrix. Compute medias signal
        media = numpy.zeros((num_blocks, dim))
        for i in range(num_blocks):
            media[i] = x[i*block_size:(i+1)*block_size].sum(axis=0) * (1.0 / block_size)

        media_a = media[0:-1]
        media_b = media[1:]
        sum_prod_mixed_meds = (mdp.utils.mult(media_a.T, media_b) + mdp.utils.mult(media_b.T, media_a))
#        prod_first_media = numpy.outer(media[0], media[0]) * block_size
#        prod_last_media = numpy.outer(media[num_blocks-1], media[num_blocks-1]) * block_size
        prod_first_block = mdp.utils.mult(x[0:block_size].T, x[0:block_size])
        prod_last_block = mdp.utils.mult(x[num_samples-block_size:].T, x[num_samples-block_size:])

        #next line causes an exception index out of bounds: n_threads=10
#        sum_diffs = (media[num_blocks-1] - media[0]) * (block_size * block_size)  * (1.0 / block_size)
#        num_diffs = (block_size * block_size) * (num_blocks - 1)
#       WARNING? why did I remove one factor block_size?
        num_diffs = block_size * (num_blocks - 1)

#warning with factors!!!! they should be 2.0, 1.0, -1.0
#        sum_prod_diffs = block_size * (2.0 * sum_prod_x + 2.0 * prod_last_block - 0.0 *prod_first_block) - (block_size * block_size) * sum_prod_mixed_meds
        sum_prod_diffs = (block_size * (sum_prod_x) - (block_size * block_size) * sum_prod_mixed_meds) * (1.0 / block_size)
#        sum_prod_diffs = block_size * (2.0 * sum_prod_x + 2.0 * prod_last_block + 0.0 *prod_first_block) - (block_size * block_size) * sum_prod_mixed_meds
        self.AddDiffs(sum_prod_diffs, num_diffs, weight)

#Here block_size must be an array or list
#why block_sizes is not a parameter???
#This will be changed to updateClustered
#Weight should refer to node weights
    def update_clustered(self, x, block_sizes = None, weight=1.0):       
        num_samples, dim = x.shape

        if isinstance(block_sizes, (int)):
            return self.updateClustered_homogeneous_blocks(x, weight=weight, block_sizes=block_sizes)
        
        if block_sizes == None:
            er = "error, block_size not specified!!!!"
            raise Exception(er)
            block_sizes = self.block_size

        if num_samples != block_sizes.sum():
            err = "Inconsistency error: num_samples (%d) is not equal to sum of block_sizes:"%num_samples, block_sizes
            raise Exception(err)

#        num_blocks = len(block_sizes)
        
        counter_sample=0
        for block_size in block_sizes:
            #Note, here a sqrt might be useful to compensate for very unbalanced datasets
            #normalized_weight = weight * block_size * 1.0 / num_samples
            #Warning Warning
#            normalized_weight = weight / block_size #Warning! is this necessary!!??? I do sample balancing, in general what should be done???
            normalized_weight = weight
            self.update_clustered_homogeneous_block_sizes(x[counter_sample:counter_sample+block_size], weight=normalized_weight, block_size=block_size)
            counter_sample += block_size

# If the input is an array, the inhomogeneous function is used
# Create true updateClustered
# Change to updateClustered
#TODO: For consisency with paper: make sure edge weights are = 1/Ns, and that self-loops are not counted, so divide by Ns * (Ns-1)
  
    def update_clustered_homogeneous_block_sizes(self, x, weight=1.0, block_size=None):
        if block_size == None:
            er = "error, block_size not specified!!!!"
            raise Exception(er)
            block_size = self.block_size

        if isinstance(block_size, (numpy.ndarray)):
            er = "Error: inhomogeneous block sizes not supported by this function"
            raise Exception(er)
#            return self.updateClustered_inhomogeneous_blocks(x, weight=weight, block_sizes=block_size)
        
        #Assuming block_size is an integer:
        num_samples, dim = x.shape
        if num_samples % block_size > 0:
            err = "Inconsistency error: num_samples (%d) is not a multiple of block_size (%d)"%(num_samples, block_size)
            raise Exception(err)
        num_blocks = num_samples / block_size
        num_neighbours = block_size-1

        #warning, plenty of dtype missing!!!!!!!!

        #Optimize computation of x.T ???
        #Correlation Matrix. Computing sum of outer products, all elements are equally likely
        sum_x = x.sum(axis=0)
#            print "Sum_x[0:3] is ", sum_x[0:3]
        sum_prod_x = mdp.utils.mult(x.T, x)
        
        #Note that each node is in 2 * (N-1) links, half of it beginning the link, half ending it.
        #the number of nodes is N * B
        #WARNING THEORY; IT WAS:
        #weighted_sum_x = (2 * num_neighbours) * sum_x * weight
        #weighted_sum_prod_x = (2 * num_neighbours) * sum_prod_x * weight
        self.AddSamples(sum_prod_x, sum_x, num_samples, weight)

        self.last_block = None
        #DCorrelation Matrix. Compute medias signal
        media = numpy.zeros((num_blocks, dim))
        for i in range(num_blocks):
            media[i] = x[i*block_size:(i+1)*block_size].sum(axis=0) * (1.0 / block_size)
    
        sum_prod_meds = mdp.utils.mult(media.T, media)
        sum_diffs = numpy.zeros((1,dim))
        #note there are N * (N-1) * B links
        #WARNING!
        #num_diffs = (block_size * (block_size-1)) * num_blocks
        #WARNING!!!!!
#        num_diffs = (block_size-1) * num_blocks
#TODO: why such factor 0.5???
        num_diffs = block_size * 0.5 * num_blocks

        #WARNING!
        #sum_prod_diffs = (2 * block_size) * sum_prod_x - 2 * (block_size * block_size) * sum_prod_meds
#TODO: why the extra factor block_size in both terms, why divide here by num_neighbors??? both terms almost cancel.
        sum_prod_diffs = ((2 * block_size) * sum_prod_x - 2 * (block_size * block_size) * sum_prod_meds)/num_neighbours
        
        self.AddDiffs(sum_prod_diffs, num_diffs, weight)
        
    def addCovDCovMatrix(self, cov_dcov_mat, adding_weight=1.0, own_weight=1.0):
        if self.sum_prod_x is None:
            self.sum_prod_x = cov_dcov_mat.sum_prod_x * adding_weight
            self.sum_x = cov_dcov_mat.sum_x * adding_weight
        else:
            self.sum_prod_x = self.sum_prod_x * own_weight + cov_dcov_mat.sum_prod_x * adding_weight
            self.sum_x = self.sum_x * own_weight + cov_dcov_mat.sum_x * adding_weight
        self.num_samples = self.num_samples * own_weight + cov_dcov_mat.num_samples * adding_weight
        if self.sum_prod_diffs is None:
            self.sum_diffs = cov_dcov_mat.sum_diffs * adding_weight
            self.sum_prod_diffs = cov_dcov_mat.sum_prod_diffs * adding_weight
        else:
            self.sum_diffs = self.sum_diffs * own_weight + cov_dcov_mat.sum_diffs * adding_weight
            self.sum_prod_diffs = self.sum_prod_diffs * own_weight + cov_dcov_mat.sum_prod_diffs * adding_weight
        self.num_diffs = self.num_diffs * own_weight + cov_dcov_mat.num_diffs * adding_weight

    #This is not a good place to select whether to include the last sample or not... move it somewhere or remove it if not needed
    def fix(self, divide_by_num_samples_or_differences=True, include_tail=False, verbose=False, center_dcov=False):
        if verbose:
            print "Fixing CovDCovMatrix, with block_size=", self.block_size
  
        #Finalize covariance matrix of x
        if include_tail is True:
            print "Including data tail into computation of covariance matrix of shape:" + str(self.last_block.shape)
            self.sum_x = self.sum_x + self.last_block.sum(axis=0)
            self.sum_prod_x = self.sum_prod_x + mdp.utils.mult(self.last_block.T, self.last_block )
            self.num_samples = self.num_samples + self.block_size
            
        avg_x = self.sum_x * (1.0 / self.num_samples)
#        avg_x = avg_x.reshape((1,self.input_dim))

        #TEORY; This computation has a bias            
        #exp_prod_x = self.sum_prod_x * (1.0 / self.num_samples) 
        #prod_avg_x = numpy.outer(avg_x, avg_x)
        #cov_x = exp_prod_x - prod_avg_x
        prod_avg_x = numpy.outer(avg_x, avg_x)
        if divide_by_num_samples_or_differences: # as specified by the theory on Training Graphs
            cov_x = (self.sum_prod_x - self.num_samples * prod_avg_x) / (1.0 * self.num_samples)
        else: # standard unbiased estimation
            cov_x = (self.sum_prod_x - self.num_samples * prod_avg_x) / (self.num_samples-1.0)
        
        #Finalize covariance matrix of dx
        if divide_by_num_samples_or_differences:
            cov_dx = self.sum_prod_diffs / (1.0 * self.num_diffs)
        else:
            cov_dx = self.sum_prod_diffs / (self.num_diffs-1.0)
            
        self.cov_mtx = cov_x
        self.avg = avg_x
        self.tlen = self.num_samples            
        self.dcov_mtx = cov_dx
        
#        print "cov_mtx[0]", self.cov_mtx[0]
#        print "avg", self.avg
#        print "dcov_mtx[0]", self.dcov_mtx[0]
#        quit()

#Safely decomment for debugging
#        print "Finishing training CovDcovMtx: ",  self.num_samples, " num_samples, and ", self.num_diffs, " num_diffs"
#        print "Avg[0:3] is", self.avg[0:4]
#        print "Prod_avg_x[0:3,0:3] is", prod_avg_x[0:3,0:3]
#        print "Cov[0:3,0:3] is", self.cov_mtx[0:3,0:3]
#        print "DCov[0:3,0:3] is", self.dcov_mtx[0:3,0:3]
#        print "AvgDiff[0:4] is", avg_diff[0:4]
#        print "Prod_avg_diff[0:3,0:3] is", prod_avg_diff[0:3,0:3]
#        print "Sum_prod_diffs[0:3,0:3] is", self.sum_prod_diffs[0:3,0:3]
#        print "exp_prod_diffs[0:3,0:3] is", exp_prod_diffs[0:3,0:3]
      
        return self.cov_mtx, self.avg, self.dcov_mtx 

######## Helper functions for parallel processing and CovDcovMatrices #########
def ComputeCovMatrix(x, verbose=False):
    print "PCov",
    if verbose:
        print "Computation Began!!! **********************************************************8"
        sys.stdout.flush()
    covmtx = CovarianceMatrix(bias=True)
    covmtx.update(x)
    if verbose:
        print "Computation Ended!!! **********************************************************8"
        sys.stdout.flush()
    return covmtx

def ComputeCovDcovMatrixClustered(params, verbose=False):
    print "PComp",
    if verbose:
        print "Computation Began!!! **********************************************************8"
        sys.stdout.flush()
    x, block_size, weight = params
    covdcovmtx = CovDCovMatrix(block_size)
    covdcovmtx.update_clustered_homogeneous_block_sizes(x, block_size=block_size, weight=weight)
    if verbose:
        print "Computation Ended!!! **********************************************************8"
        sys.stdout.flush()
    return covdcovmtx

def ComputeCovDcovMatrixSerial(params, verbose=False):
    print "PSeq",
    if verbose:
        print "Computation Began!!! **********************************************************8"
        sys.stdout.flush()
    x, block_size = params
    Torify= False
    covdcovmtx = CovDCovMatrix(block_size)
    covdcovmtx.updateSerial(x, block_size = block_size, Torify=Torify)
    if verbose:
        print "Computation Ended!!! **********************************************************8"
        sys.stdout.flush()
    return covdcovmtx

def ComputeCovDcovMatrixMixed(params, verbose=False):
    print "PMixed",
    if verbose:
        print "Computation Began!!! **********************************************************8"
        sys.stdout.flush()
    x, block_size = params
    bs=block_size
    covdcovmtx = CovDCovMatrix(block_size)
    covdcovmtx.update_clustered_homogeneous_block_sizes(x[0:bs], block_size=block_size, weight=0.5)
    covdcovmtx.update_clustered_homogeneous_block_sizes(x[bs:-bs], block_size=block_size, weight=1.0)
    covdcovmtx.update_clustered_homogeneous_block_sizes(x[-bs:], block_size=block_size, weight=0.5)
    covdcovmtx.updateSerial(x, block_size = block_size, Torify=False)            
    if verbose:
        print "Computation Ended!!! **********************************************************8"
        sys.stdout.flush()
    return covdcovmtx

class GSFANode(mdp.Node):
    def __init__(self, input_dim=None, output_dim=None, dtype=None, block_size=None, train_mode=None, sfa_expo=None, pca_expo=None, magnitude_sfa_biasing=None):
        super(mdp.nodes.SFANode, self).__init__(input_dim, output_dim, dtype)
        #Warning! bias activated, "courtesy" of Alberto
        # init two covariance matrices
        # one for the input data
        self._cov_mtx = CovarianceMatrix(dtype, bias=True)
        # one for the derivatives
        self._dcov_mtx = CovarianceMatrix(dtype, bias=True)
    
        self.pinv = None
        self._symeig = symeig
        self.block_size= block_size
        self.train_mode = train_mode
    
        self.sum_prod_x = None
        self.sum_x = None
        self.num_samples = 0
        self.sum_diff = None 
        self.sum_prod_diff = None  
        self.num_diffs = 0
        
        self.sfa_expo=sfa_expo # 1.2: 1=Regular SFA directions, >1: schrink derivative in the direction of the principal components
        self.pca_expo=pca_expo # 0.25: 0=No magnitude reduction 1=Whitening
        self.magnitude_sfa_biasing = magnitude_sfa_biasing
        self._myvar = None
        self._covdcovmtx = CovDCovMatrix(block_size)
        self.list_train_params = ["scheduler", "n_parallel", "train_mode", "block_size"] #Parameters accepted during training
        
    def _inverse(self, y):
        #code for storing pseudoinverse courtesy of Alberto Escalante
        if self.pinv is None:
            self.pinv = pinv(self.sf)
    #        print "SFA.pinv = ", self.pinv
    #        print "Shape of SFA.pinv = ", self.pinv.shape
    #        print "SFA.sf = ", self.sf
    #        print "Shape of sf = ", self.sf.shape        
    #        sf_t = self.sf.T
    #        print "sf_t= ", sf_t
    #        
    #        m2 = mult(sf_t, self.sf)
    #        print "m2 = ", m2
    #        print "For orthonormal sf, m2 is the identity"
    #
    #        m3 = mult(self.sf, sf_t)
    #        print "m3 = ", m3
    #        print "just curiosity"
    #        
    #        s_mod = (self.sf * self.sf).sum(axis=0)
    #        print "s_mod = ", s_mod
    #        print "(sf/s_mod).T= ", (self.sf / s_mod).T
        return mult(y, self.pinv)+self.avg

    def train_with_scheduler(self, x, block_size=None, train_mode = None, node_weights=None, edge_weights=None, scheduler = None, n_parallel=None):      
        self._train_phase_started = True
        if train_mode == None:
            train_mode = self.train_mode
        if block_size == None:
            block_size = self.block_size
        if scheduler == None or n_parallel == None or train_mode == None:
    #        print "NO parallel sfa done...  scheduler=", ,uler, " n_parallel=", n_parallel
            return SFANode_train(self, x, block_size=block_size, train_mode=train_mode, node_weights=node_weights, edge_weights=edge_weights)
        else:
    #        self._covdcovmtx.update_clustered_homogeneous_block_sizes(x, weight=1.0)
    
    #        chunk_size=None 
    #        num_chunks = n_parallel
            num_chunks = min(n_parallel, x.shape[0]/block_size)
            #here chunk_size is given in blocks!!!
    #        chunk_size = int(numpy.ceil((x.shape[0]/block_size)*1.0/num_chunks))
            chunk_size = int((x.shape[0]/block_size)/num_chunks)
            
            #Notice that parallel training doesn't work with clustered mode and inhomogeneous blocks
            #TODO:Fix this
            print "%d chunks, of size %d blocks, last chunk contains %d blocks"%(num_chunks, chunk_size, (x.shape[0]/block_size)%chunk_size)
            if train_mode == 'clustered':
                #TODO: Implement this
                for i in range(num_chunks):
                    print "Adding scheduler task //////////////////////"
                    sys.stdout.flush()
                    if i < num_chunks - 1:
                        scheduler.add_task((x[i*block_size*chunk_size:(i+1)*block_size*chunk_size], block_size, 1.0), ComputeCovDcovMatrixClustered)
                    else: # i == num_chunks - 1
                        scheduler.add_task((x[i*block_size*chunk_size:], block_size, 1.0), ComputeCovDcovMatrixClustered)
                    print "Done Adding scheduler task ///////////////////"
                    sys.stdout.flush()
            elif train_mode == 'serial':
                for i in range(num_chunks):
                    if i==0:
                        print "adding task %d from sample "%i, i*block_size*chunk_size, "to", (i+1)*block_size*chunk_size
                        scheduler.add_task((x[i*block_size*chunk_size:(i+1)*block_size*chunk_size], block_size), ComputeCovDcovMatrixSerial)
                    elif i==num_chunks-1: #Add one previous block to the chunk
                        print "adding task %d from sample "%i, i*block_size*chunk_size-block_size, "to the end"
                        scheduler.add_task((x[i*block_size*chunk_size-block_size:], block_size), ComputeCovDcovMatrixSerial)
                    else:
                        print "adding task %d from sample "%i, i*block_size*chunk_size-block_size, "to", (i+1)*block_size*chunk_size
                        scheduler.add_task((x[i*block_size*chunk_size-block_size:(i+1)*block_size*chunk_size], block_size), ComputeCovDcovMatrixSerial)
                        
            elif train_mode == 'mixed':
                bs = block_size
                self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[0:bs], weight=0.5)
    
    #           xxxx self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[bs:-bs], weight=1.0)
                x2 = x[bs:-bs]
    #            num_chunks2 = int(numpy.ceil((x2.shape[0]/block_size-2)*1.0/chunk_size))
                num_chunks2 = int((x2.shape[0]/block_size-2)/chunk_size)
                for i in range(num_chunks2):
                    if i < num_chunks2-1:
                        scheduler.add_task((x2[i*block_size*chunk_size:(i+1)*block_size*chunk_size], block_size, 1.0), ComputeCovDcovMatrixClustered)
                    else:
                        scheduler.add_task((x2[i*block_size*chunk_size:], block_size, 1.0), ComputeCovDcovMatrixClustered)
                        
                self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[-bs:], weight=0.5)
    
    #           xxxx self._covdcovmtx.updateSerial(x, Torify=False)            
                for i in range(num_chunks):
                    if i==0:
                        scheduler.add_task((x[i*block_size*chunk_size:(i+1)*block_size*chunk_size], block_size), ComputeCovDcovMatrixSerial)
                    elif i==num_chunks-1: #Add one previous block to the chunk
                        scheduler.add_task((x[i*block_size*chunk_size-block_size:], block_size), ComputeCovDcovMatrixSerial)
                    else: #Add one previous block to the chunk
                        scheduler.add_task((x[i*block_size*chunk_size-block_size:(i+1)*block_size*chunk_size], block_size), ComputeCovDcovMatrixSerial)
            elif train_mode == 'unlabeled':
                #TODO: IMPLEMENT THIS
                for i in range(num_chunks):
                    print "Adding scheduler task //////////////////////"
                    sys.stdout.flush()
                    scheduler.add_task((x[i*block_size*chunk_size:(i+1)*block_size*chunk_size], block_size, 1.0), ComputeCovDcovMatrixClustered)
                    print "Done Adding scheduler task ///////////////////"
                    sys.stdout.flush()
            else:
                ex = "Unknown training method:", self.train_mode
                raise Exception(ex)
            
            print "Getting results"
            sys.stdout.flush()
    
            results = scheduler.get_results()
    #        print "Shutting down scheduler"
            sys.stdout.flush()
    
            for covdcovmtx in results:
                self._covdcovmtx.addCovDCovMatrix(covdcovmtx)

    def _train(self, x, block_size=None, train_mode = None, node_weights=None, edge_weights=None):
        if train_mode == None:
            train_mode = self.train_mode
        if block_size == None:
            er="SFA no block_size"
            raise Exception(er)
            block_size = self.block_size
        
        self._myvar=1
        self.set_input_dim(x.shape[1])
    
        ## update the covariance matrices
        # cut the final point to avoid a trivial solution in special cases
        # WARNING: Force artificial training
        print "train_mode=", train_mode
        if train_mode == 'unlabeled':
            print "updateUnlabeled"
            self._covdcovmtx.updateUnlabeled(x, weight=0.00015) #Warning, set this weight appropiately!
        elif train_mode == "regular":
            print "updateRegular"
            self._covdcovmtx.updateRegular(x, weight=1.0)
        elif train_mode == 'clustered':
            print "update_clustered_homogeneous_block_sizes"
            self._covdcovmtx.update_clustered_homogeneous_block_sizes(x, weight=1.0, block_size=block_size)
        elif train_mode == 'serial':
            print "updateSerial"
            self._covdcovmtx.updateSerial(x, Torify=False, block_size=block_size)
        elif train_mode == 'mixed':
            print "update mixed"
            bs = block_size
    # WARNING: THIS Generates degenerated solutions!!! Check code!! it should actually work fine??!!
            self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[0:bs], weight=2.0, block_size=block_size)
            self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[bs:-bs], weight=1.0, block_size=block_size)
            self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[-bs:], weight=2.0, block_size=block_size)
    #        self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[0:bs], weight=0.5, block_size=block_size)
    #        self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[bs:-bs], weight=1.0, block_size=block_size)
    #        self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[-bs:], weight=0.5, block_size=block_size)
            self._covdcovmtx.updateSerial(x, Torify=False, block_size=block_size)            
        elif train_mode[0:6] == 'window':
            window_halfwidth = int(train_mode[6:])
            print "Window (%d)"%window_halfwidth
            self._covdcovmtx.updateSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
        elif train_mode[0:7] == 'fwindow':
            window_halfwidth = int(train_mode[7:])
            print "Fast Window (%d)"%window_halfwidth
            self._covdcovmtx.updateFastSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
        elif train_mode[0:13] == 'mirror_window':
            window_halfwidth = int(train_mode[13:])
            print "Mirroring Window (%d)"%window_halfwidth
            self._covdcovmtx.updateMirroringSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
        elif train_mode[0:14] == 'smirror_window':
            window_halfwidth = int(train_mode[14:])
            print "Slow Mirroring Window (%d)"%window_halfwidth
            self._covdcovmtx.updateSlowMirroringSlidingWindow(x, weight=1.0, window_halfwidth=window_halfwidth)
        elif train_mode == 'graph':
            print "updateGraph"
            self._covdcovmtx.updateGraph(x, weight=1.0, node_weights=node_weights, edge_weights=edge_weights)
        else:
            ex = "Unknown training method"
            raise Exception(ex)

    def _stop_training(self, debug=False, verbose=False, pca_term = 0.995, pca_exp=2.0):
    #    self._myvar = self._myvar * 10
    #    Warning, changed block_size to artificial_training
    #    if (self.block_size == None) or (isinstance(self.block_size, int) and self.block_size == 1):
    #   WARNING!!! WARNING!!!
    #TODO: Define a proper way to fix training matrices... add training mode "regular" to SFA???
        if ((self.block_size == None) or (isinstance(self.block_size, int) and self.block_size == 1)) and False:
            ##### request the covariance matrices and clean up
            self.cov_mtx, self.avg, self.tlen = self._cov_mtx.fix()
            del self._cov_mtx
            self.dcov_mtx, davg, dtlen = self._dcov_mtx.fix()
            del self._dcov_mtx
            print "Finishing training (regular2222). %d tlen, %d tlen of diff"%(self.tlen, dtlen)
            print "Avg[0:3] is", self.avg[0:3]
            print "Cov[0:3,0:3] is", self.cov_mtx[0:3,0:3]
            print "DCov[0:3,0:3] is", self.dcov_mtx[0:3,0:3]
        else:
            if verbose or True:
                print "stop_training: Warning, using experimental SFA training method, with self.block_size=", self.block_size
            print "self._covdcovmtx.num_samples = ", self._covdcovmtx.num_samples 
            print "self._covdcovmtx.num_diffs= ", self._covdcovmtx.num_diffs
            self.cov_mtx, self.avg, self.dcov_mtx = self._covdcovmtx.fix()
                   
            print "Finishing SFA training: ",  self.num_samples, " num_samples, and ", self.num_diffs, " num_diffs"
    #        print "Avg[0:3] is", self.avg[0:4]
    #        print "Prod_avg_x[0:3,0:3] is", prod_avg_x[0:3,0:3]
    #        print "Cov[0:3,0:3] is", self.cov_mtx[0:3,0:3]
            print "DCov[0:3,0:3] is", self.dcov_mtx[0:3,0:3]
    #        quit()
    
        if pca_term != 0.0 and False:
            signs = numpy.sign(self.dcov_mtx)
            self.dcov_mtx = ((1.0-pca_term) * signs * numpy.abs(self.dcov_mtx) ** (1.0/pca_exp) + pca_term * numpy.identity(self.dcov_mtx.shape[0])) ** pca_exp          
            self.dcov_mtx = numpy.identity(self.dcov_mtx.shape[0])
        rng = self._set_range()
        
            
        #### solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        #SUPERMEGAWARNING, moved dcov to the second argument!!!
        #self.d, self.sf = self._symeig(self.dcov_mtx, self.cov_mtx, range=rng, overwrite=(not debug))
        try:
            print "***Range used=", rng
            if self.sfa_expo != None and self.pca_expo!=None:
                self.d, self.sf = _symeig_fake_regularized(self.dcov_mtx, self.cov_mtx, range=rng, overwrite=(not debug), sfa_expo=self.sfa_expo, pca_expo=self.pca_expo, magnitude_sfa_biasing=self.magnitude_sfa_biasing)
            else:
                self.d, self.sf = self._symeig(self.dcov_mtx, self.cov_mtx, range=rng, overwrite=(not debug))
            d = self.d
            # check that we get only *positive* eigenvalues
            if d.min() < 0:
                raise SymeigException("Got negative eigenvalues: %s." % str(d))
        except SymeigException, exception:
            errstr = str(exception)+"\n Covariance matrices may be singular."
            raise Exception(errstr)
    
        # delete covariance matrix if no exception occurred
        del self.cov_mtx
        del self.dcov_mtx
        del self._covdcovmtx
        # store bias
        self._bias = mult(self.avg, self.sf)
        print "shape of SFANode.sf is=", self.sf.shape


