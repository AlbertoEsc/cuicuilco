#####################################################################################################################
# more_nodes: This module implements several new nodes and helper functions. It is part of the Cuicuilco framework. #
#                                                                                                                   #
# These nodes include: BasicAdaptiveCutoffNode, SFA_GaussianClassifier, RandomizedMaskNode, GeneralExpansionNode,   #
# PointwiseFunctionNode, RandomPermutationNode                                                                      #
#                                                                                                                   #
# By Alberto Escalante. Alberto.Escalante@ini.ruhr-uni-bochum.de                                                    #
# Ruhr-University-Bochum, Institute for Neural Computation, Group of Prof. Dr. Wiskott                              #
#####################################################################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy
import scipy
import scipy.optimize
import scipy.stats
from scipy.stats import ortho_group
import copy
import sys
import inspect

import mdp
from mdp.utils import (mult, pinv, symeig, CovarianceMatrix, SymeigException)

from . import sfa_libs
from .sfa_libs import select_rows_from_matrix, distance_squared_Euclidean
# from . import inversion
from .histogram_equalization import *


def add_corrections(initial_corrections, added_corrections):
    if initial_corrections is None:
        return added_corrections
    elif added_corrections is None:
        return initial_corrections
    else:
        return initial_corrections * added_corrections


def combine_correction_factors(flow_or_node, average_over_layers = True, average_inside_layers=False):
    """This function takes into account all corrections performed by the BasicAdaptiveCutoffNodes of
    a flow (possibly a hierarchical network) and combines them into a single vector. The function also
    works on standard nodes.

    average_over_layers: if True, the combined corrections are the average of the corrections of each
        node in the flow, otherwise they are multiplied (omitting nodes without corrections)
    average_inside_layers: if True, the combined corrections of Layers are computed as the average of
        the corrections of each node in the layer, otherwise they are multiplied

    The combined correction factor of each sample estimates the probability that it is not an anomaly. That is,
    correction=1.0 implies "not anomaly", and smaller values increase the rareness of the sample.
    """
    final_corrections = None
    final_gauss_corrections = None
    if isinstance(flow_or_node, mdp.Flow):
        flow = flow_or_node
        if average_over_layers:
            corrections = []
            gauss_corrections = []
            for node in flow:
                another_node_corrections, another_node_gauss_corrections = combine_correction_factors(node, average_over_layers)
                if another_node_corrections is not None:
                    corrections.append(another_node_corrections)
                if another_node_gauss_corrections is not None:
                    gauss_corrections.append(another_node_gauss_corrections)
            if len(corrections) > 0:
                corrections = numpy.stack(corrections, axis=1)
                final_corrections = corrections.mean(axis=1)
                gauss_corrections = numpy.stack(gauss_corrections, axis=1)
                final_gauss_corrections = gauss_corrections.mean(axis=1)

            else:
                final_corrections = None
                final_gauss_corrections = None
        else:
            for node in flow:
                another_node_corrections, another_node_gauss_corrections = combine_correction_factors(node)
                final_corrections = add_corrections(final_corrections, another_node_corrections)
                final_gauss_corrections = add_corrections(final_gauss_corrections, another_node_gauss_corrections)
    elif isinstance(flow_or_node, mdp.Node):
        node = flow_or_node
        if isinstance(node, mdp.hinet.CloneLayer):
            err = "CloneLayers not yet supported when computing/storing correction factors"
            print(err)
            final_corrections = None
            final_gauss_corrections = None
            # raise Exception(err)
        elif isinstance(node, mdp.hinet.Layer):
            if average_inside_layers:
                corrections = []
                gauss_corrections = []
                for another_node in node.nodes:
                    another_node_corrections, another_node_gauss_corrections = combine_correction_factors(another_node)
                    corrections.append(another_node_corrections)
                    gauss_corrections.append(another_node_gauss_corrections)
                if len(corrections) > 0:
                    corrections = numpy.stack(corrections, axis=1)
                    final_corrections = corrections.mean(axis=1)
                    gauss_corrections = numpy.stack(gauss_corrections, axis=1)
                    final_gauss_corrections = gauss_corrections.mean(axis=1)
                else:
                    final_corrections = None
                    final_gauss_corrections = None
            else:
                for another_node in node.nodes:
                    another_node_corrections, another_node_gauss_corrections = combine_correction_factors(another_node)
                    final_corrections = add_corrections(final_corrections, another_node_corrections)
                    final_gauss_corrections = add_corrections(final_gauss_corrections, another_node_gauss_corrections)
        elif isinstance(node, BasicAdaptiveCutoffNode):
            final_corrections = add_corrections(final_corrections, node.corrections)
            final_gauss_corrections = add_corrections(final_gauss_corrections, node.gauss_corrections)
    return final_corrections, final_gauss_corrections


class BasicAdaptiveCutoffNode(mdp.PreserveDimNode):
    """Node that allows to "cut off" values at bounds derived from the training data.

    This node is similar to CutoffNode, but the bounds are computed based on the training data. And it is
    also similar to AdaptiveCutoffNode, but no histograms are stored and the limits are hard.

    This node does not have any have no effect on training data but it corrects atypical variances in test data
    and may improve generalization.
    """

    def __init__(self, input_dim=None, output_dim=None, num_rotations=1, measure_corrections=False,
                 only_measure=False, verbose=True, dtype=None):
        """Initialize node. """
        super(BasicAdaptiveCutoffNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.lower_bounds = None
        self.upper_bounds = None
        self.rotation_matrices = None
        self.num_rotations = num_rotations
        self.measure_corrections = measure_corrections
        self.corrections = None
        self.gauss_corrections = None
        self.only_measure = only_measure
        self.verbose = verbose

        self._avg_x = None
        self._avg_x_squared = None
        self._num_samples = 0
        self._std_x = None

        if self.verbose:
            print("num_rotations:", num_rotations, "measure_corrections:", measure_corrections,
                  "only_measure:", only_measure, "verbose:", verbose)

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def is_invertible():
        return True

    @staticmethod
    def _get_supported_dtypes():
        return (mdp.utils.get_dtypes('Float'))

    def _train(self, x):
        # initialize rotations and arrays that store the bounds
        dim = x.shape[1]
        if self.rotation_matrices is None:
            self.rotation_matrices = [None] * self.num_rotations
            self.lower_bounds = [None] * self.num_rotations
            self.upper_bounds = [None] * self.num_rotations
            if self.num_rotations >= 1:
                self.rotation_matrices[0] = numpy.eye(dim)
            for i in range(1, self.num_rotations):
                self.rotation_matrices[i] = ortho_group.rvs(dim=dim)

        # The training method updates the lower and upper bounds
        for i in range(self.num_rotations):
            rotated_data = numpy.dot(x, self.rotation_matrices[i])
            if self.lower_bounds[i] is None:
                self.lower_bounds[i] = rotated_data.min(axis=0)
            else:
                self.lower_bounds[i] = numpy.minimum(self.lower_bounds[i], rotated_data.min(axis=0))

            if self.upper_bounds[i] is None:
                self.upper_bounds[i] = rotated_data.max(axis=0)
            else:
                self.upper_bounds[i] = numpy.maximum(self.upper_bounds[i], rotated_data.max(axis=0))

        if self._avg_x is None:
            self._avg_x = x.sum(axis=0)
            self._avg_x_squared = (x**2).sum(axis=0)
        else:
            self._avg_x += x.sum(axis=0)
            self._avg_x_squared += (x ** 2).sum(axis=0)
        self._num_samples += x.shape[0]

    def _stop_training(self):
        self._avg_x /= self._num_samples
        self._avg_x_squared /= self._num_samples
        self._std_x = (self._avg_x_squared - self._avg_x **2) ** 0.5
        if self.verbose:
            print("self._avg_x", self._avg_x)
            print("self._avg_x_squared", self._avg_x_squared)
            print("self._std_x", self._std_x)

    def _execute(self, x):
        """Return the clipped data."""
        num_samples = x.shape[0]
        self.corrections = numpy.ones(num_samples)
        self.gauss_corrections = numpy.ones(num_samples)

        if self.only_measure:
            x_copy = x.copy()

        for i in range(self.num_rotations):
            data_rotated = numpy.dot(x, self.rotation_matrices[i])
            data_rotated_clipped = numpy.clip(data_rotated, self.lower_bounds[i], self.upper_bounds[i])
            if self.measure_corrections:
                interval = numpy.abs(self.upper_bounds[i] - self.lower_bounds[i])
                delta = numpy.abs(data_rotated_clipped - data_rotated)
                # factors = interval ** 2 / (delta + interval) ** 2
                norm_delta = delta / interval
                factors = 1.0 - (norm_delta / (norm_delta + 0.15)) ** 2
                self.corrections *= factors.prod(axis=1)  # consider using here and below the mean instead of the product
                if self.verbose:
                    print("Factors of BasicAdaptiveCutoffNode:", factors)

                # Computation of Gaussian probabilities
                factors = scipy.stats.norm.pdf(x, loc=self._avg_x, scale=4*self._std_x)
                if self.verbose:
                    print("Factors of BasicAdaptiveCutoffNode (gauss):", factors)
                    print("x.mean(axis=0):", x.mean(axis=0))
                    print("x.std(axis=0):", x.std(axis=0))
                self.gauss_corrections *= factors.prod(axis=1)

            x = numpy.dot(data_rotated_clipped, self.rotation_matrices[i].T)  # Project back to original coordinates

        if self.verbose:
            print("Corrections of BasicAdaptiveCutoffNode:", self.corrections)
            print("20 worst final corrections at indices:", numpy.argsort(self.corrections)[0:20])
            print("20 worst final corrections:", self.corrections[numpy.argsort(self.corrections)[0:20]])

            print("Gaussian corrections of BasicAdaptiveCutoffNode:", self.gauss_corrections)
            print("20 worst final Gaussian corrections at indices:", numpy.argsort(self.gauss_corrections)[0:20])
            print("20 worst final Gaussian corrections:",
                  self.corrections[numpy.argsort(self.gauss_corrections)[0:20]])

        if self.only_measure:
            return x_copy
        else:
            return x

    def _inverse(self, x):
        """An approximate inverse applies the same clipping. """
        return self.execute(x)


class SFA_GaussianClassifier(mdp.ClassifierNode):
    """ This node is a simple extension of the GaussianClassifier node, where SFA is applied before the classifier.

    The labels are important, since they are used to order the data samples before SFA.
    """

    def __init__(self, reduced_dim=None, verbose=False, **argv):
        super(SFA_GaussianClassifier, self).__init__(**argv)
        self.gc_node = mdp.nodes.GaussianClassifier()
        self.reduced_dim = reduced_dim
        if self.reduced_dim > 0:
            self.sfa_node = mdp.nodes.SFANode(output_dim=self.reduced_dim)
        else:
            self.sfa_node = mdp.nodes.IdentityNode()
        self.verbose = verbose

    def _train(self, x, labels=None):
        if self.reduced_dim > 0:
            ordering = numpy.argsort(labels)
            x_ordered = x[ordering, :]
            self.sfa_node.train(x_ordered)
            self.sfa_node.stop_training()
            if self.verbose:
                print("SFA_GaussianClassifier: sfa_node.d = ", self.sfa_node.d)
        else:  # sfa_node is the identity node
            pass
        y = self.sfa_node.execute(x)
        self.gc_node.train(y, labels=labels)
        self.gc_node.stop_training()

    def _label(self, x):
        y = self.sfa_node.execute(x)
        return self.gc_node.label(y)

    def regression(self, x, avg_labels, estimate_std=False):
        y = self.sfa_node.execute(x)
        return self.gc_node.regression(y, avg_labels, estimate_std)

    def regressionMAE(self, x, avg_labels):
        y = self.sfa_node.execute(x)
        return self.gc_node.regressionMAE(y, avg_labels)

    def softCR(self, x, true_classes):
        y = self.sfa_node.execute(x)
        return self.gc_node.softCR(y, true_classes)

    def class_probabilities(self, x):
        y = self.sfa_node.execute(x)
        return self.gc_node.class_probabilities(y)

    @staticmethod
    def is_trainable():
        return True


# using the provided average and standard deviation
def gauss_noise(x, avg, std):
    return numpy.random.normal(avg, std, x.shape)


# Zero centered
def additive_gauss_noise(x, std):
    return x + numpy.random.normal(0, std, x.shape)


class RandomizedMaskNode(mdp.Node):
    """Selectively mask some components of a random variable by 
    hiding them with arbitrary noise or by removing them from the feature vector.

    This code has been inspired by NoiseNode
    """
    def __init__(self, remove_mask=None, noise_amount_mask=None, noise_func=gauss_noise, noise_args=(0, 1),
                 noise_mix_func=None, input_dim=None, dtype=None):
        self.remove_mask = remove_mask

        self.noise_amount_mask = noise_amount_mask
        self.noise_func = noise_func
        self.noise_args = noise_args
        self.noise_mix_func = noise_mix_func
        self.seen_samples = 0
        self.x_avg = None
        self.x_std = None
        self.type = dtype

        if remove_mask is not None and input_dim is None:
            input_dim = remove_mask.size
        elif remove_mask is None and input_dim is not None:
            remove_mask = numpy.zeros(input_dim) > 0.5
        elif remove_mask and input_dim is not None:
            if remove_mask.size != input_dim:
                err = "size of remove_mask and input_dim not compatible"
                raise Exception(err)
        else:
            err = "At least one of input_dim or remove_mask should be specified"
            raise Exception(err)

        if noise_amount_mask is None:
            print ("Signal will be only the computed noise")
            self.noise_amount_mask = numpy.ones(input_dim)
        else:
            self.noise_amount_mask = noise_amount_mask

        output_dim = remove_mask.size - remove_mask.sum()
        print ("Output_dim should be:", output_dim)
        super(RandomizedMaskNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)

    @staticmethod
    def is_trainable():
        return True

    def _train(self, x):
        if self.x_avg is None:
            self.x_avg = numpy.zeros(self.input_dim, dtype=self.type)
            self.x_std = numpy.zeros(self.input_dim, dtype=self.type)
        new_samples = x.shape[0]
        self.x_avg = (self.x_avg * self.seen_samples + x.sum(axis=0)) / (self.seen_samples + new_samples)
        self.x_std = (self.x_std * self.seen_samples + x.std(axis=0) * new_samples) / (self.seen_samples + new_samples)
        self.seen_samples = self.seen_samples + new_samples

    @staticmethod
    def is_invertible():
        return False

    def _execute(self, x):
        print ("computed X_avg=", self.x_avg)
        print ("computed X_std=", self.x_std)
        noise_mat = self.noise_func(x, self.x_avg, self.x_std)
        #        noise_mat = self._refcast(self.noise_func(*self.noise_args,
        #                                                  **{'size': x.shape}))
        print ("Noise_amount_mask:", self.noise_amount_mask)
        print ("Noise_mat:", noise_mat)
        noisy_signal = (1.0 - self.noise_amount_mask) * x + self.noise_amount_mask * noise_mat
        preserve_mask = (self.remove_mask == False)
        return noisy_signal[:, preserve_mask]


class GeneralExpansionNode(mdp.Node):
    def __init__(self, funcs, input_dim=None, dtype=None, \
                 use_pseudoinverse=True, use_hint=False, output_dim=None, starting_point=None, use_special_features=False, max_steady_factor=1.5,
                 delta_factor=0.6, min_delta=0.00001, verbose=False):
        self.funcs = funcs
        self.exp_output_dim = output_dim
        self.expanded_dims = None
        self.starting_point = starting_point
        self.use_special_features = use_special_features
        if self.funcs == "RandomSigmoids" and self.exp_output_dim <= 0:
            er = "output_dim in GeneralExpansion node with RandomSigmoids should be at least 1, but is" + \
                 str(self.exp_output_dim)
            raise Exception(er)
        self.use_pseudoinverse = use_pseudoinverse
        self.use_hint = use_hint
        self.max_steady_factor = max_steady_factor
        self.delta_factor = delta_factor
        self.min_delta = min_delta
        self.verbose = verbose

        if self.verbose:
            print("GeneralExpansionNode with expansion functions:", funcs)

        self.rs_coefficients = None
        self.rs_offsets = None
        self.rs_data_training_std = None
        self.rs_data_training_mean = None
        self.normalization_constant = None
        super(GeneralExpansionNode, self).__init__(input_dim, dtype)

    def expanded_dim(self, n):
        exp_dim = 0
        x = numpy.zeros((1, n))
        for func in self.funcs:
            outx = func(x)
            # print "outx= ", outx
            exp_dim += outx.shape[1]
        return exp_dim

    def output_sizes(self, n):
        if self.funcs == "RandomSigmoids":
            sizes = [self.exp_output_dim]
        else:
            sizes = numpy.zeros(len(self.funcs), dtype=int)
            x = numpy.zeros((1, n))
            for i, func in enumerate(self.funcs):
                outx = func(x)
                sizes[i] = outx.shape[1]
                print ("S", end="")
        return sizes

    def is_trainable(self):
        if self.funcs == "RandomSigmoids":
            return True
        else:
            return False

    def _train(self, x, verbose=None):
        if verbose is None:
            verbose = self.verbose

        if self.input_dim is None:
            self.set_input_dim(x.shape[1])

        input_dim = self.input_dim

        # Generate functions used for regression
        self.rs_data_training_mean = x.mean(axis=0)
        self.rs_data_training_std = x.std(axis=0)

        if verbose:
            print ("GeneralExpansionNode: output_dim=", self.output_dim, end="")
        starting_point = self.starting_point
        c1, l1 = generate_random_sigmoid_weights(self.input_dim, self.output_dim)
        if starting_point == "Identity":
            if verbose:
                print ("starting_point: adding (encoded) identity coefficients to expansion")
            c1[0:input_dim, 0:input_dim] = numpy.identity(input_dim)
            l1[0:input_dim] = numpy.ones(input_dim) * 1.0  # Code identity
        elif starting_point == "Sigmoids":
            if verbose:
                print ("starting_point: adding sigmoid of coefficients to expansion")
            c1[0:input_dim, 0:input_dim] = 4.0 * numpy.identity(input_dim)
            l1[0:input_dim] = numpy.ones(input_dim) * 0.0
        elif starting_point == "08Exp":
            if verbose:
                print ("starting_point: adding (encoded) 08Exp coefficients to expansion")
            c1[0:input_dim, 0:input_dim] = numpy.identity(input_dim)
            c1[0:input_dim, input_dim:2 * input_dim] = numpy.identity(input_dim)

            l1[0:input_dim] = numpy.ones(input_dim) * 1.0  # Code identity
            l1[input_dim:2 * input_dim] = numpy.ones(input_dim) * 0.8  # Code abs(x)**0.8
        elif starting_point == "Pseudo-Identity":
            if verbose:
                print ("starting_point: adding pseudo-identity coefficients to expansion")
            c1[0:input_dim, 0:input_dim] = 0.1 * numpy.identity(input_dim)
            l1[0:input_dim] = numpy.zeros(input_dim)  # nothig is encoded
        elif starting_point is None:
            if verbose:
                print ("starting_point: no starting point")
        else:
            er = "Unknown starting_point", starting_point
            raise Exception(er)
        self.rs_coefficients = c1
        self.rs_offsets = l1
        # 4.0 was working fine, 2.0 was apparently better. This also depends on how many features are computed!!!
        self.normalization_constant = (2.0 / self.input_dim) ** 0.5

    def is_invertible(self):
        return self.use_pseudoinverse

    def inverse(self, x, use_hint=None, max_steady_factor=None, delta_factor=None, min_delta=None):
        if self.use_pseudoinverse is False:
            ex = "Inversion not activated"
            raise Exception(ex)
        if use_hint is None:
            use_hint = self.use_hint
        if max_steady_factor is None:
            max_steady_factor = self.max_steady_factor
        if delta_factor is None:
            delta_factor = self.delta_factor
        if min_delta is None:
            min_delta = self.min_delta

        # print "Noisy pre = ", x, "****************************************************"
        app_x_2, app_ex_x_2 = invert_exp_funcs2(x, self.input_dim, self.funcs, use_hint=use_hint,
                                                max_steady_factor=max_steady_factor, delta_factor=delta_factor,
                                                min_delta=min_delta)
        # print "Noisy post = ", x, "****************************************************"
        return app_x_2

    def _set_input_dim(self, n):
        self._input_dim = n

        if self.funcs == "RandomSigmoids":
            self._output_dim = self.exp_output_dim
        else:
            self._output_dim = self.expanded_dim(n)
        self.expanded_dims = self.output_sizes(n)

    def _execute(self, x):
        if self.input_dim is None:
            self.set_input_dim(x.shape[1])

        if "expanded_dims" not in self.__dict__:
            self.expanded_dims = self.output_sizes(self.input_dim)

        if self.funcs != "RandomSigmoids":
            num_samples = x.shape[0]
            #        output_dim = expanded_dim(self.input_dim)
            # self.expanded_dims = self.output_sizes(self.input_dim)
            out = numpy.zeros((num_samples, self.output_dim))

            current_pos = 0
            for i, func in enumerate(self.funcs):
                out[:, current_pos:current_pos + self.expanded_dims[i]] = func(x)
                current_pos += self.expanded_dims[i]
        else:
            data_norm = self.normalization_constant * (x - self.rs_data_training_mean) / self.rs_data_training_std
            # A variation of He random weight initialization
            out = extract_sigmoid_features(data_norm, self.rs_coefficients, self.rs_offsets, scale=1.0, offset=0.0,
                                           use_special_features=self.use_special_features)
        return out


class PointwiseFunctionNode(mdp.Node):
    """"This node applies a function to the whole input.
    
    It also supports a given 'inverse' function.
    """
    def __init__(self, func, inv_func, input_dim=None, dtype=None):
        self.func = func
        self.inv_func = inv_func
        super(PointwiseFunctionNode, self).__init__(input_dim, dtype)

    @staticmethod
    def is_trainable():
        return False

    def is_invertible(self):
        if self.inv_func is None:
            return True
        else:
            return False

    def inverse(self, x):
        if self.inv_func:
            return self.inv_func(x)
        else:
            return x

    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = n

    def _execute(self, x):
        if self.input_dim is None:
            self.set_input_dim(x.shape[1])
        if self.func:
            return self.func(x)
        else:
            return x


class PairwiseAbsoluteExpansionNode(mdp.Node):
    def expanded_dim(self, n):
        return n + n * (n + 1) // 2

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = self.expanded_dim(n)

    def _execute(self, x):
        out = numpy.concatenate((x, pairwise_expansion(x, abs_sum)), axis=1)
        return out


# TODO:ADD inverse type sum, suitable for when output_scaling is True
class PInvSwitchboard(mdp.hinet.Switchboard):
    """This node is a variation of the RectangularSwitchboard that facilitates (approximate) inverse operations. """
    def __init__(self, input_dim, connections, slow_inv=False, type_inverse="average", output_scaling=True,
                 additive_noise_std=0.00004, verbose=False):
        super(PInvSwitchboard, self).__init__(input_dim=input_dim, connections=connections)
        self.pinv = None
        self.mat2 = None
        self.slow_inv = slow_inv
        self.type_inverse = type_inverse
        self.output_dim = len(connections)
        self.output_scales = None
        self.additive_noise_std = additive_noise_std
        self.verbose = verbose

        if verbose:
            print ("self.inverse_connections=", self.inverse_connections, "self.slow_inv=", self.slow_inv)
        # WARNING! IF/ELIF doesn't make any sense! what are the semantics of inverse_connections
        if self.inverse_connections is None:
            if verbose:
                print ("type(connections)", type(connections))
            all_outputs = numpy.arange(self.output_dim)

            self.inverse_indices = [[]] * self.input_dim
            for i in range(self.input_dim):
                self.inverse_indices[i] = all_outputs[connections == i]
                # print "inverse_indices[%d]="%i, self.inverse_indices[i]
                # print "inverse_indices =", self.inverse_indices
        elif self.inverse_connections is None and not self.slow_inv:
            index_array = numpy.argsort(connections)
            value_array = connections[index_array]

            value_range = numpy.zeros((input_dim, 2))
            self.inverse_indices = range(input_dim)
            for i in range(input_dim):
                value_range[i] = numpy.searchsorted(value_array, [i - 0.5, i + 0.5])
                if value_range[i][1] == value_range[i][0]:
                    self.inverse_indices[i] = []
                else:
                    self.inverse_indices[i] = index_array[value_range[i][0]: value_range[i][1]]
            if verbose:
                print ("inverse_indices computed in PINVSB")

        elif self.inverse_connections is None and self.slow_inv:
            if verbose:
                print ("warning using slow inversion in PInvSwitchboard!!!")
            # find input variables not used by connections:
            used_inputs = numpy.unique(connections)
            used_inputs_set = set(used_inputs)
            all_inputs_set = set(range(input_dim))
            unused_inputs_set = all_inputs_set - all_inputs_set.intersection(used_inputs_set)
            unused_inputs = list(unused_inputs_set)
            self.num_unused_inputs = len(unused_inputs)
            # extend connections array
            # ext_connections = numpy.concatenate((connections, unused_inputs))
            # create connections matrix
            mat_height = len(connections) + len(unused_inputs)
            mat_width = input_dim
            mat = numpy.zeros((mat_height, mat_width))
            # fill connections matrix
            for i in range(len(connections)):
                mat[i, connections[i]] = 1
            #
            for i in range(len(unused_inputs)):
                mat[i + len(connections), unused_inputs[i]] = 1
            #
            if verbose:
                    print ("extended matrix is:", mat)
            # compute pseudoinverse
            mat2 = numpy.matrix(mat)
            self.mat2 = mat2
            self.pinv = (mat2.T * mat2).I * mat2.T
        else:
            if verbose:
                print ("Inverse connections already given, in PInvSwitchboard")

        if output_scaling:
            if self.inverse_connections is None and not self.slow_inv:
                if verbose:
                    print ("**A", end="")
                if self.type_inverse != "average":
                    err = "self.type_inverse not supported " + self.type_inverse
                    raise Exception(err)
                self.output_scales = numpy.zeros(self.output_dim)
                tt = 0
                for i in range(self.input_dim):
                    output_indices = self.inverse_indices[i]
                    multiplicity = len(output_indices)
                    for j in output_indices:
                        self.output_scales[j] = (1.0 / multiplicity) ** 0.5
                        tt += 1
                if verbose:
                    print ("connections in switchboard considered: ", tt, "output dimension=", self.output_dim)
            elif self.inverse_connections is None and self.slow_inv:
                if verbose:
                    print ("**B", end="")
                err = "use of self.slow_inv = True is obsolete"
                raise Exception(err)
            else:  # inverse connections are unique, mapping bijective
                if verbose:
                    print ("**C", end="")
                self.output_scales = numpy.ones(self.output_dim)
        else:
            if verbose:
                print ("**D", end="")
            self.output_scales = numpy.ones(self.output_dim)
        if verbose:
            print ("PINVSB output_scales =", self.output_scales)
            print ("SUM output_scales/len(output_scales)=", self.output_scales.sum() / len(self.output_scales))
            print ("output_scales.min()", self.output_scales.min())

    # PInvSwitchboard is always invertible
    def is_invertible(self):
        return True

    def _execute(self, x):
        force_float32_type = False  # Experimental variation, ignore
        if force_float32_type:
            x = x.astype("float32")
        use_fortran_ordering = False  # Experimental variation, ignore
        if use_fortran_ordering:
            x = numpy.array(x, order="FORTRAN")
        y = super(PInvSwitchboard, self)._execute(x)
        # print "y computed"
        # print "y.shape", y.shape
        # print "output_scales ", self.output_scales
        y *= self.output_scales  

        if self.additive_noise_std > 0.0:
            n, dim = y.shape
            steps = int(n / 9000 + 1)
            if self.verbose:
                print ("PInvSwitchboard is adding noise to the output features with std", self.additive_noise_std,
                       end="")
                print (" computation in %d steps" % steps)
            step_size = int(n / steps)
            for s in range(steps):
                y[step_size * s:step_size * (s + 1)] += numpy.random.uniform(low=-(3 ** 0.5) * self.additive_noise_std,
                                                                             high=(3 ** 0.5) * self.additive_noise_std,
                                                                             size=(step_size, dim))
                if self.verbose:
                    print ("noise block %d added" % s)
            if step_size * steps < n:
                rest = n - step_size * steps
                y[step_size * steps:step_size * steps + rest] += numpy.random.uniform(
                    low=-(3 ** 0.5) * self.additive_noise_std, high=(3 ** 0.5) * self.additive_noise_std,
                    size=(rest, dim))
                if self.verbose:
                    print ("remaining noise block added")
        return y

    # If true inverse is present, just use it, otherwise compute it by means of the pseudoinverse
    def _inverse(self, x):
        x = x * (1.0 / self.output_scales)
        if self.inverse_connections is None and not self.slow_inv:
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
            output = mat2
        elif self.inverse_connections is None and self.slow_inv:
            height_x = x.shape[0]
            full_x = numpy.concatenate((x, 255 * numpy.ones((height_x, self.num_unused_inputs))), axis=1)
            data2 = numpy.matrix(full_x)
            if self.verbose:
                print ("x=", x)
                print ("data2=", data2)
                print ("PINV=", self.pinv)
            output = (self.pinv * data2.T).T
        else:
            if self.verbose:
                print ("using inverse_connections in PInvSwitchboard")
            # return apply_permutation_to_signal(x, self.inverse_connections, self.input_dim)
            output = select_rows_from_matrix(x, self.inverse_connections)
        return output


class RandomPermutationNode(mdp.Node):
    """This node randomly permutes the components of the input signal in a consistent way.
    
    The concrete permuntation is fixed during the training procedure.
    """
    def __init__(self, input_dim=None, output_dim=None, dtype=None, verbose=False):
        super(RandomPermutationNode, self).__init__(input_dim, output_dim, dtype)
        self.permutation = None
        self.inv_permutation = None
        self.dummy = 5  # without it the hash fails!!!!! 

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
            print ("RandomPermutationNode: Setting input_dim to ", n)
        self._input_dim = n
        self._output_dim = n

    def _train(self, x, verbose=True):
        n = x.shape[1]

        if self.input_dim is None:
            self.set_input_dim(n)

        if self.input_dim is None:
            print ("*******Really Setting input_dim to ", n)
            self.input_dim = n

        if self.output_dim is None:
            print ("*******Really Setting output_dim to ", n)
            self.output_dim = n

        if self.permutation is None:
            if verbose:
                print ("Creating new random permutation")
                print ("Permutation=", self.permutation)
                print ("x=", x, "with shape", x.shape)
                print ("Input dim is: ", self.input_dim())
            self.permutation = numpy.random.permutation(range(self.input_dim))
            self.inv_permutation = numpy.zeros(self.input_dim, dtype="int")
            self.inv_permutation[self.permutation] = numpy.arange(self.input_dim)
            if verbose:
                print ("Permutation=", self.permutation)
                print ("Output dim is: ", self.output_dim)

    def _execute(self, x, verbose=False):
        # print "RandomPermutationNode: About to excecute, with input x= ", x
        y = select_rows_from_matrix(x, self.permutation)
        if verbose:
            print ("Output shape is = ", y.shape, end="")
        return y


def sfa_pretty_coefficients(sfa_node, transf_training, start_negative=True):
    count = 0

    for i in range(sfa_node.output_dim):
        sum_firsts = transf_training[0, i] + transf_training[1, i] + transf_training[2, i] + transf_training[3, i] + \
                     transf_training[4, i] + transf_training[5, i] + transf_training[6, i] + transf_training[7, i] + \
                     transf_training[8, i] + transf_training[9, i] + transf_training[10, i] + transf_training[11, i]

        if (sum_firsts > 0 and start_negative) or (sum_firsts < 0 and not start_negative):
            sfa_node.sf[:, i] = (sfa_node.sf[:, i] * -1)
            transf_training[:, i] = (transf_training[:, i] * -1)
            count += 1

    print ("Polarization of %d SFA Signals Corrected!!!\n" % count, end="")
    sfa_node._bias = mdp.utils.mult(sfa_node.avg, sfa_node.sf)
    print ("Bias updated")
    return transf_training


def describe_flow(flow):
    length = len(flow)

    total_size = 0
    print ("Flow has %d nodes:" % length)
    for i in range(length):
        node = flow[i]
        node_size = compute_node_size(node)
        total_size += node_size

        print ("Node[%d] is %s, has input_dim=%d, output_dim=%d and size=%d" % (i, str(node), node.input_dim,
                                                                                node.output_dim, node_size))
        if isinstance(node, mdp.hinet.CloneLayer):
            print ("   contains %d cloned nodes of type %s, each with input_dim=%d, output_dim=%d" %
                   (len(node.nodes), str(node.nodes[0]), node.nodes[0].input_dim, node.nodes[0].output_dim))

        elif isinstance(node, mdp.hinet.Layer):
            print ("   contains %d nodes of type %s, each with input_dim=%d, output_dim=%d" %
                   (len(node.nodes), str(node.nodes[0]), node.nodes[0].input_dim, node.nodes[0].output_dim))
    print ("Total flow size: %d" % total_size)
    print ("Largest node size: %d" % compute_largest_node_size(flow))


def display_node_eigenvalues(node, i, mode="All"):
    if isinstance(node, mdp.hinet.CloneLayer):
        if isinstance(node.nodes[0], mdp.nodes.SFANode):
            print ("Node %d is a CloneLayer that contains an SFANode with d=" % i, node.nodes[0].d)
        # elif isinstance(node.nodes[0], mdp.nodes.IEVMNode):
        #     if node.nodes[0].use_sfa:
        #         print ("Node %d is a CloneLayer that contains an IEVMNode containing an SFA node with" % i, end="")
        #         print ("num_sfa_features_preserved=%d" % node.nodes[0].num_sfa_features_preserved, end="")
        #         print ("and d=", node.nodes[0].sfa_node.d)
        elif isinstance(node.nodes[0], mdp.nodes.iGSFANode):
            print ("Node %d is a CloneLayer that contains an iGSFANode containing an SFA node with " % i, end="")
            print ("num_sfa_features_preserved=%d " % node.nodes[0].num_sfa_features_preserved, end="")
            print ("and d=", node.nodes[0].sfa_node.d, end=" ")
            print ("and evar=", node.nodes[0].evar)
        elif isinstance(node.nodes[0], mdp.nodes.PCANode):
            print ("Node %d is a CloneLayer that contains a PCANode with d=" % i, node.nodes[0].d, end=" ")
            print ("and evar=", node.nodes[0].explained_variance)

    elif isinstance(node, mdp.hinet.Layer):
        if isinstance(node.nodes[0], mdp.nodes.SFANode):
            if mode == "Average":
                out = 0.0
                for n in node.nodes:
                    out += n.d
                print ("Node %d is a Layer that contains %d SFANodes with avg(d)= " % (i, len(node.nodes)), out / len(node.nodes))
            elif mode == "All":
                for n in node.nodes:
                    print ("Node %d is a Layer that contains an SFANode with d= " % i, n.d)
            elif mode == "FirstNodeInLayer":
                print ("Node %d is a Layer, and its first SFANode has d= " % i, node.nodes[0].d)
            else:
                er = 'Unknown mode in display_eigenvalues, try "FirstNodeInLayer", "Average" or "All"'
                raise Exception(er)
        elif isinstance(node.nodes[0], mdp.nodes.iGSFANode):
            if mode == "Average":
                evar_avg = 0.0
                d_avg = 0.0
                avg_num_sfa_features = 0.0
                min_num_sfa_features_preserved = min([n.num_sfa_features_preserved for n in node.nodes])
                for n in node.nodes:
                    d_avg += n.sfa_node.d[:min_num_sfa_features_preserved]
                    evar_avg += n.evar
                    avg_num_sfa_features += n.num_sfa_features_preserved
                d_avg /= len(node.nodes)
                evar_avg /= len(node.nodes)
                avg_num_sfa_features /= len(node.nodes)
                print ("Node %d" % i, "is a Layer that contains", len(node.nodes), "iGSFANodes containing SFANodes with " +
                       "avg(num_sfa_features_preserved)=%f " % avg_num_sfa_features, "and avg(d)=%s" % str(d_avg) +
                       "and avg(evar)=%f" % evar_avg)
            elif mode == "All":
                print ("Node %d is a Layer that contains iGSFANodeRecNodes:" % i)
                for n in node.nodes:
                    print ("  iGSFANode containing an SFANode with num_sfa_features_preserved=%f, d=%s and evar=%f" %
                           (n.num_sfa_features_preserved, str(n.sfa_node.d), n.evar))
            elif mode == "FirstNodeInLayer":
                print ("Node %d is a Layer, and its first iGSFANode " % i, end="")
                print ("contains an SFANode with num_sfa_features_preserved)=%f, d=%s and evar=%f" %
                       (node.nodes[0].num_sfa_features_preserved, str(node.nodes[0].sfa_node.d), node.nodes[0].evar))
            else:
                er = 'Unknown mode in display_eigenvalues, try "FirstNodeInLayer", "Average" or "All"'
                raise Exception(er)
        elif isinstance(node.nodes[0], mdp.nodes.SFAAdaptiveNLNode):
            if mode == "Average":
                out = 0.0
                for n in node.nodes:
                    out += n.sfa_node.d
                print ("Node %d is a Layer that contains SFAAdaptiveNLNodes containing SFANodes with", end="")
                print ("avg(d)=" % i, out / len(node.nodes))
            elif mode == "All":
                for n in node.nodes:
                    print ("Node %d is a Layer that contains an SFAAdaptiveNLNode" % i, end="")
                    print ("containing an SFANode with d=", n.sfa_node.d)
            elif mode == "FirstNodeInLayer":
                print ("Node %d is a Layer, and its first SFAAdaptiveNLNode" % i)
                print ("contains an SFANode with d=", node.nodes[0].sfa_node.d)
            else:
                er = 'Unknown mode in display_eigenvalues, try "FirstNodeInLayer", "Average" or "All"'
                raise Exception(er)
        elif isinstance(node.nodes[0], mdp.nodes.PCANode):
            if mode == "Average":
                d_avg = 0.0
                evar_avg = 0.0
                min_num_pca_features_preserved = min([n.output_dim for n in node.nodes])
                for n in node.nodes:
                    d_avg += n.d[:min_num_pca_features_preserved]
                    evar_avg += n.explained_variance
                d_avg /= len(node.nodes)
                evar_avg /= len(node.nodes)
                print ("Node %d is a Layer that contains PCA nodes with avg(d)=%s and avg(evar)=%f" % (
                    i, str(d_avg), evar_avg))
            elif mode == "All":
                print ("Node %d is a Layer that contains PCA nodes:" % i)
                for n in node.nodes:
                    print ("  PCANode with d=%s and evar=%f" % (str(n.d), n.explained_variance))
            elif mode == "FirstNodeInLayer":
                print ("Node %d is a Layer, and its first PCANode" % i, "has d=%s and evar=%f" % (
                    str(node.nodes[0].sfa_node.d), node.nodes[0].explained_variance))
            else:
                er = 'Unknown mode in display_eigenvalues, try "FirstNodeInLayer", "Average" or "All"'
                raise Exception(er)
    elif isinstance(node, mdp.nodes.iGSFANode):
        print ("Node %d is an iGSFANode containing an SFA node with num_sfa_features_preserved=%d" %
               (i, node.num_sfa_features_preserved), end="")
        print ("and d=", node.sfa_node.d)
    elif isinstance(node, mdp.nodes.SFANode):
        print ("Node %d is an SFANode with d=" % i, node.d)
    elif isinstance(node, mdp.nodes.PCANode):
        print ("Node %d is a PCANode with d=%s and evar=%f" % (i, str(node.d), node.explained_variance))
    else:
        print ("Cannot display eigenvalues of Node %d" % i, node)


def display_eigenvalues(flow, mode="All"):
    """This function displays the learned eigenvalues of different nodes in a trained Flow object.
    
    Three mode parameter can take three values and it specifies what to do when a layer is found: 
        "FirstNodeInLayer": the eigenvalues of the first node in the layer are displayed
        "Average": the average eigenvalues of all nodes in a layer are displayed (bounded to the smallest length).
        "All": the eigenvalues of all nodes in the layer are displayed.
    """
    length = len(flow)
    print ("Displaying eigenvalues of SFA Nodes in flow of length", length)

    for i in range(length):
        node = flow[i]
        display_node_eigenvalues(node, i, mode)


def compute_node_size(node, verbose=False):
    """ Computes the number of parameters (weights) that have been learned by node.

    Note: Means and offsets are not counted, only (multiplicative) weights. The node must have been already trained.
    The following nodes are supported currently:
    SFANode, PCANode, WhitheningNode, CloneLayer, Layer, GSFANode, iGSFANode, LinearRegressionNode
    """
    if isinstance(node, mdp.nodes.iGSFANode):
        return compute_node_size(node.sfa_node) + compute_node_size(node.pca_node) + compute_node_size(node.lr_node)
    elif isinstance(node, (mdp.nodes.SFANode, mdp.nodes.PCANode, mdp.nodes.GSFANode, mdp.nodes.LinearRegressionNode,
                    mdp.nodes.WhiteningNode)) and node.input_dim is not None and node.output_dim is not None:
        return node.input_dim * node.output_dim
    elif isinstance(node, mdp.hinet.CloneLayer):
        return compute_node_size(node.nodes[0])
    elif isinstance(node, mdp.hinet.Layer):
        size = 0
        for node_child in node.nodes:
            size += compute_node_size(node_child)
        return size
    else:
        if verbose:
            print ("compute_node_size not implemented for nodes of type:", type(node), "or training has not finished")
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


# Used to compare the effectiveness of several PCA Networks
def estimate_explained_variance(images, flow, sl_images, num_considered_images=100, verbose=True):
    # Here explained variance is defined as 1 - normalized reconstruction error
    num_images = images.shape[0]
    im_numbers = numpy.random.randint(num_images, size=num_considered_images)

    avg_image = images[im_numbers].mean(axis=0)

    selected_images = images[im_numbers]
    ori_differences = selected_images - avg_image
    ori_energies = ori_differences ** 2
    ori_energy = ori_energies.sum()

    sl_selected_images = sl_images[im_numbers]
    print ("sl_selected_images.shape=", sl_selected_images.shape)
    inverses = flow.inverse(sl_selected_images)
    rec_differences = inverses - avg_image
    rec_energies = rec_differences ** 2
    rec_energy = rec_energies.sum()

    rec_errors = selected_images - inverses
    rec_error_energies = rec_errors ** 2
    rec_error_energy = rec_error_energies.sum()

    if verbose:
        explained_individual = rec_energies.sum(axis=1) / ori_energies.sum(axis=1)
        print ("Individual explained variances: ", explained_individual)
        print ("Which, itself has standar deviation: ", explained_individual.std())
        print ("Therefore, estimated explained variance has std of about: ", explained_individual.std() / numpy.sqrt(
            num_considered_images))
        print ("Dumb reconstruction_energy/original_energy=", rec_energy / ori_energy)
        print ("rec_error_energy/ori_energy=", rec_error_energy / ori_energy)
        print ("Thus explained variance about:", 1 - rec_error_energy / ori_energy)
    return 1 - rec_error_energy / ori_energy  # rec_energy/ori_energy


class HeadNode(mdp.Node):
    """Preserve only the first k dimensions from the data
    """

    def __init__(self, input_dim=None, output_dim=None, dtype=None):
        self.type = dtype
        super(HeadNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)

    def is_trainable(self):
        return True

    def _train(self, x):
        pass

    def _is_invertible(self):
        return True

    def _execute(self, x):
        if self.output_dim is None:
            er = "Warning 12345..."
            raise Exception(er)
        return x[:, 0:self.output_dim]

    def _stop_training(self):
        pass

    def _inverse(self, y):
        num_samples, out_dim = y.shape[0], y.shape[1]
        zz = numpy.zeros((num_samples, self.input_dim - out_dim))
        return numpy.concatenate((y, zz), axis=1)


# # This code is obsolete.
# class SFAPCANode(mdp.Node):
#     """Node that extracts slow features unless their delta value is too high. In such a case PCA features are extracted.
#     """
#
#     def __init__(self, input_dim=None, output_dim=None, max_delta=1.95, sfa_args={}, pca_args={}, **argv):
#         super(SFAPCANode, self).__init__(input_dim=input_dim, output_dim=output_dim, **argv)
#         self.sfa_node = mdp.nodes.SFANode(**sfa_args)
#         # max delta value allowed for a slow feature, otherwise a principal component is extracted
#         self.max_delta = max_delta
#         self.avg = None  # input average
#         self.W = None  # weights for complete transformation
#         self.pinv = None  # weights for pseudoinverse of complete transformation
#
#     def is_trainable(self):
#         return True
#
#     def _train(self, x, **argv):
#         self.sfa_node.train(x, **argv)
#
#     @staticmethod
#     def _is_invertible():
#         return True
#
#     def _execute(self, x):
#         W = self.W
#         avg = self.avg
#         return numpy.dot(x - avg, W)
#
#     def _stop_training(self, **argv):
#         # New GraphSFA node
#         if "_covdcovmtx" in dir(self.sfa_node):
#             # Warning, fix is computed twice. TODO: avoid double computation
#             C, self.avg, CD = self.sfa_node._covdcovmtx.fix()
#         else:
#             # Old fix destroys data... so we copy the matrices first.
#             cov_mtx = copy.deepcopy(self.sfa_node._cov_mtx)
#             dcov_mtx = copy.deepcopy(self.sfa_node._dcov_mtx)
#
#             C, self.avg, tlen = cov_mtx.fix()
#             DC, davg, dtlen = dcov_mtx.fix()
#
#         dim = C.shape[0]
#         type_ = C.dtype
#         self.sfa_node.stop_training()
#         d = self.sfa_node.d
#         sfa_output_dim = len(d[d <= self.max_delta])
#         sfa_output_dim = min(sfa_output_dim, self.output_dim)
#         print ("sfa_output_dim=", sfa_output_dim)
#
#         Wsfa = self.sfa_node.sf[:, 0:sfa_output_dim]
#         print ("Wsfa.shape=", Wsfa.shape)
#         if Wsfa.shape[1] == 0:  # No slow components will be used
#             print ("No Psfa created")
#             PS = numpy.zeros((dim, dim), dtype=type_)
#         else:
#             Psfa = pinv(Wsfa)
#             print ("Psfa.shape=", Psfa.shape)
#             PS = numpy.dot(Wsfa, Psfa)
#
#         print ("PS.shape=", PS.shape)
#         Cproy = numpy.dot(PS, numpy.dot(C, PS.T))
#         Cpca = C - Cproy
#
#         if self.output_dim is None:
#             self.output_dim = dim
#
#         pca_output_dim = self.output_dim - sfa_output_dim
#         print ("PCA output_dim=", pca_output_dim)
#         if pca_output_dim > 0:
#             pca_node = mdp.nodes.PCANode(output_dim=pca_output_dim)  # WARNING: WhiteningNode should be used here
#             pca_node._cov_mtx._dtype = type_
#             pca_node._cov_mtx._input_dim = dim
#             pca_node._cov_mtx._avg = numpy.zeros(dim, type_)
#             pca_node._cov_mtx.bias = True
#             pca_node._cov_mtx._tlen = 1  # WARNING!!! 1
#             pca_node._cov_mtx._cov_mtx = Cpca
#             pca_node._input_dim = dim
#             pca_node._train_phase_started = True
#             pca_node.stop_training()
#             print ("pca_node.d=", pca_node.d)
#             print ("1000000 * pca_node.d[0]=", 1000000 * pca_node.d[0])
#
#             Wpca = pca_node.v
#             Ppca = pca_node.v.T
#         else:
#             Wpca = numpy.array([]).reshape((dim, 0))
#             Ppca = numpy.array([]).reshape((0, dim))
#
#         print ("Wpca.shape=", Wpca.shape)
#         print ("Ppca.shape=", Ppca.shape)
#
#         self.W = numpy.concatenate((Wsfa, Wpca), axis=1)
#         self.pinv = None  # WARNING, why this does not work correctly: numpy.concatenate((Psfa, Ppca),axis=0) ?????
#         #        print "Pinv 1=", self.pinv
#         #        print "Pinv 2-Pinv1=", pinv(self.W)-self.pinv
#         print ("W.shape=", self.W.shape)
#         #        print "pinv.shape=", self.pinv.shape
#         print ("avg.shape=", self.avg.shape)
#
#     def _inverse(self, y):
#         if self.pinv is None:
#             print ("Computing PINV", end="")
#             self.pinv = pinv(self.W)
#         return numpy.dot(y, self.pinv) + self.avg


# Computes the variance of some MDP data array
def data_variance(x):
    return ((x - x.mean(axis=0)) ** 2).sum(axis=1).mean()


def estimate_explained_var_linearly(x, y, x_test, y_test):
    x_test_app = approximate_linearly(x, y, y_test)

    explained_variance = compute_explained_var(x_test, x_test_app)

    x_variance = data_variance(x_test)
    print ("x_variance=", x_variance, ", explained_variance=", explained_variance)
    return explained_variance / x_variance


def approximate_linearly(x, y, y_test):
    lr_node = mdp.nodes.LinearRegressionNode(use_pseudoinverse=True)
    lr_node.train(y, x)
    lr_node.stop_training()

    x_test_app = lr_node.execute(y_test)
    return x_test_app


# Approximates x from y, and computes how sensitive the estimation is to changes in y
def sensivity_of_linearly_approximation(x, y):
    lr_node = mdp.nodes.LinearRegressionNode(use_pseudoinverse=True)
    lr_node.train(y, x)
    lr_node.stop_training()
    beta = lr_node.beta[1:, :]  # bias is used by default, we do not need to consider it

    print ("beta.shape=", beta.shape)
    sens = (beta ** 2).sum(axis=1)
    return sens


def estimate_explained_var_with_kNN(x, y, max_num_samples_for_ev=None, max_test_samples_for_ev=None, k=1,
                                    ignore_closest_match=False, operation="average"):
    num_samples = x.shape[0]
    indices_all_x = numpy.arange(x.shape[0])

    if max_num_samples_for_ev is not None:  # use all samples for reconstruction
        max_num_samples_for_ev = min(max_num_samples_for_ev, num_samples)
        indices_all_x_selection = indices_all_x + 0
        numpy.random.shuffle(indices_all_x_selection)
        indices_all_x_selection = indices_all_x_selection[0:max_num_samples_for_ev]
        x_sel = x[indices_all_x_selection]
        y_sel = y[indices_all_x_selection]
    else:
        x_sel = x
        y_sel = y

    if max_test_samples_for_ev is not None:  # use all samples for reconstruction
        max_test_samples_for_ev = min(max_test_samples_for_ev, num_samples)
        indices_all_x_selection = indices_all_x + 0
        numpy.random.shuffle(indices_all_x_selection)
        indices_all_x_selection = indices_all_x_selection[0:max_test_samples_for_ev]
        x_test = x[indices_all_x_selection]
        y_test = y[indices_all_x_selection]
    else:
        x_test = x
        y_test = y

    x_app_test = approximate_kNN_op(x_sel, y_sel, y_test, k, ignore_closest_match, operation=operation)
    print ("x_test=", x_test)
    print ("x_app_test=", x_app_test)

    explained_variance = compute_explained_var(x_test, x_app_test)
    test_variance = data_variance(x_test)

    print ("explained_variance=", explained_variance)
    print ("test_variance=", test_variance)

    return explained_variance / test_variance


def random_subindices(num_indices, size_selection):
    if size_selection > num_indices:
        ex = "Error, size_selection is larger than num_indices! (", size_selection, ">", num_indices, ")"
        raise Exception(ex)
    all_indices = numpy.arange(num_indices)
    numpy.random.shuffle(all_indices)
    return all_indices[0:size_selection] + 0


def estimate_explained_var_linear_global(subimages_train, sl_seq_training, subimages_newid, sl_seq_newid,
                                         reg_num_signals, number_samples_EV_linear_global):
    """Function that computes how much variance is explained linearly from a global mapping.
    
    It works as follows: 1) Linear regression is trained with sl_seq_training and subimages_train.
    2) Estimation is done on subset of size number_samples_EV_linear_global from training and test data
    3) For training data evaluation is done on the same data used to train LR, and on new random subset of data.
    4) For test data all samples are used.
    """
    indices_all_train1 = random_subindices(subimages_train.shape[0], number_samples_EV_linear_global)
    indices_all_train2 = random_subindices(subimages_train.shape[0], number_samples_EV_linear_global)
    indices_all_newid = numpy.arange(subimages_newid.shape[0])

    lr_node = mdp.nodes.LinearRegressionNode()
    sl_seq_training_sel1 = sl_seq_training[indices_all_train1, 0:reg_num_signals]
    subimages_train_sel1 = subimages_train[indices_all_train1]
    lr_node.train(sl_seq_training_sel1,
                  subimages_train_sel1)  # Notice that the input "x"=n_sfa_x and the output to learn is "y" = x_pca
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

    print ("Data variances=", data_variance_train1, data_variance_train2, data_variance_newid)
    print ("EVLinGlobal=", EVLinGlobal_train1, EVLinGlobal_train2, EVLinGlobal_newid)
    return EVLinGlobal_train1 / data_variance_train1, EVLinGlobal_train2 / data_variance_train2, \
           EVLinGlobal_newid / data_variance_newid


def compute_explained_var(true_samples, approximated_samples):
    """Computes the explained variance provided by the approximation to some data, with respect to the true data.
    
    Additionally, the original data variance is provided:
    app = true_samples + error
    exp_var ~ energy(true_samples) - energy(error)
    """
    error = (approximated_samples - true_samples)
    error_energy = (error ** 2.0).sum(axis=1).mean()  # average squared error per sample
    true_energy = data_variance(true_samples)  # (true_samples-true_samples.mean(axis=0)).var()

    explained_var = true_energy - error_energy
    # print "Debug information:", error_energy, true_energy
    return explained_var


def approximate_kNN_op(x, x_exp, y_exp, k=1, ignore_closest_match=False, operation=None):
    """ Approximates a signal y given its expansion y_exp. The method is kNN with training data given by x, x_exp
   
    If label_avg=True, the inputs of the k closest expansions are averaged, otherwise the most frequent
    among k-closest is returned.
    When label_avg=True, one can also specify to ignore the best match (useful if y_exp = x_exp)
    """
    n = mdp.nodes.KNNClassifier(k=k, execute_method="label")
    n.train(x_exp, range(len(x_exp)))

    if operation == "average":
        n.stop_training()
        ii = n.klabels(y_exp)
        if ignore_closest_match and k == 1:
            ex = "Error, k==1 but ignoring closest match!"
            raise Exception(ex)
        elif ignore_closest_match:
            ii = ii[:, 1:]
        y = x[ii].mean(axis=1)

        # y_exp_app = x_exp[ii].mean(axis=1)
        # print "Error for y_exp is:", ((y_exp_app - y_exp)**2).sum(axis=1).mean()

        # print "y=",y
        return y  # x[ii].mean(axis=1)
    elif operation == "lin_app":
        n.stop_training()
        ii = n.klabels(y_exp)

        if ignore_closest_match and k == 1:
            ex = "Error, k==1 but ignoring closest match!"
            raise Exception(ex)
        elif ignore_closest_match:
            ii = ii[:, 1:]

        x_dim = x.shape[1]
        x_exp_dim = x_exp.shape[1]

        x_mean = x.mean(axis=0)
        x = x - x_mean

        nk = ii.shape[1]
        y = numpy.zeros((len(y_exp), x_dim))
        y_exp_app = numpy.zeros((len(y_exp), x_exp_dim))
        x_ind = x[ii]
        x_exp_ind = x_exp[ii]
        y_expit = numpy.zeros((x_exp_dim + 1, 1))
        k = 1.0e10  # make larger to force sum closer to one?!
        y_expit[x_exp_dim, 0] = 0.0 * 1.0 * k
        x_expit = numpy.zeros((x_exp_dim + 1, nk))
        x_expit[x_exp_dim, :] = 1.0 * k  #
        zero_threshold = -40.0500  # -0.004
        max_zero_weights = nk // 5

        w_0 = numpy.ones((nk, 1)) * 1.0 / nk
        # print "w_0", w_0
        for i in range(len(y_exp)):
            negative_weights = 0
            iterate = True
            # print "Iteration: ", i,
            x_expit[0:x_exp_dim, :] = x_exp_ind[i].T
            y_0 = numpy.dot(x_exp_ind[i].T, w_0)

            fixing_zero_threshold = zero_threshold * 500
            while iterate:
                # print x_exp_ind[i].T.shape
                # print x_expit[0:x_exp_dim,:].shape
                x_pinv = numpy.linalg.pinv(x_expit)
                # print y_0.shape, y_exp[i].shape
                y_expit[0:x_exp_dim, 0] = y_exp[i] - y_0.flatten()
                w_i = numpy.dot(x_pinv, y_expit) + w_0

                iterate = False
                if (w_i < zero_threshold).any():
                    # print "w_i[:,0] =", w_i[:,0]
                    # print "x_expit = ", x_expit
                    negative_weights += (w_i < fixing_zero_threshold).sum()
                    negative_elements = numpy.arange(nk)[w_i[:, 0] < fixing_zero_threshold]
                    numpy.random.shuffle(negative_elements)
                    for nn in negative_elements:
                        # print "nn=", nn
                        x_expit[0:x_exp_dim + 1, nn] = 0.0
                    # print "negative_elements", negative_elements
                    iterate = True
                fixing_zero_threshold /= 2
                if negative_weights >= max_zero_weights:
                    iterate = False

                    # FORCE SUM WEIGHTS=1:
                # print "w_i[:,0] =", w_i[:,0]
                # print "weight sum=",w_i.sum(),"min_weight=",w_i.min(),"max_weight=",w_i.max(),
                # "negative weights=", negative_weights
                w_i /= w_i.sum()
                # print "y[i].shape", y[i].shape
                # print "as.shape", numpy.dot(x_ind[i].T, w_i).T.shape
                y[i] = numpy.dot(x_ind[i].T, w_i).T + x_mean  # numpy.dot(w_i, x_ind[i]).T
                y_exp_app[i] = numpy.dot(x_exp_ind[i].T, w_i).T

            if w_i.min() < zero_threshold:  # 0.1: #negative_weights >= max_zero_weights:
                # quit()max_zero_weights
                print ("Warning smallest weight is", w_i.min(), "thus replacing with simple average")
                # print "Warning, at least %d all weights turned out to be negative! (%d)"%(max_zero_weights,
                # negative_weights)
                # print x_ind[i]
                # print x_ind[i].shape
                y[i] = x_ind[i].mean(axis=0)
        print (".", end="")
        # print "Error for y_exp is:", ((y_exp_app - y_exp)**2).sum(axis=1).mean()
        # print "y=",y
        return y  # x[ii].mean(axis=1)
    elif operation == "plainKNN":
        ii = n.execute(y_exp)
        ret = x[ii]
        return ret
    else:
        er = "operation unknown:", operation
        raise Exception(er)


def approximate_kNN(x, x_exp, y_exp, k=1, ignore_closest_match=False, label_avg=True):
    n = mdp.nodes.KNNClassifier(k=k, execute_method="label")
    n.train(x_exp, range(len(x_exp)))

    if label_avg:
        n.stop_training()
        ii = n.klabels(y_exp)
        if ignore_closest_match and k == 1:
            ex = "Error, k==1 but ignoring closest match!"
            raise Exception(ex)
        elif ignore_closest_match:
            ii = ii[:, 1:]
        y = x[ii].mean(axis=1)
        return y  # x[ii].mean(axis=1)
    else:
        ii = n.execute(y_exp)
        ret = x[ii]
    return ret


def rank_expanded_signals_max_linearly(x, x_exp, y, y_exp, max_comp=10, max_num_samples_for_ev=None,
                                       max_test_samples_for_ev=None, verbose=False):
    """ Third ranking method. More robust and closer to max EV(x; y_i + Y)-EV(x;Y) for all Y, EV computed linearly.

        Ordering and scoring of signals respects principle of best incremental feature selection
        Computes a scores vector that measures the importance of each expanded component at reconstructing a signal
        x, x_exp are training data, y and y_exp are test data
        At most max_comp are evaluated exhaustively, the rest is set equal to the remaining
    """
    dim_out = x_exp.shape[1]
    all_indices = numpy.arange(dim_out)

    indices_all_x = numpy.arange(x.shape[0])
    indices_all_y = numpy.arange(y.shape[0])

    max_scores = numpy.zeros(dim_out)
    available_mask = numpy.zeros(dim_out) >= 0  # boolean mask that indicates which elements are not yet scored
    taken = []  # list with the same elements.

    #   Compute maximum explainable variance (taking all components)
    total_variance = data_variance(y)

    last_explained_var = 0.0
    last_score = 0.0
    for iteration in range(min(max_comp, dim_out)):
        # find individual contribution to expl var, from not taken
        indices_available = all_indices[available_mask]  # mapping from index_short to index_long
        temp_explained_vars = numpy.zeros(
            dim_out - iteration)  # s_like(indices_available, dtype=") #explained variances for each available index

        # On each iteration, the subset of samples used for testing and samples for reconstruction are kept fixed
        if max_num_samples_for_ev is not None and max_num_samples_for_ev < x.shape[0]:
            indices_all_x_selection = indices_all_x + 0
            numpy.random.shuffle(indices_all_x_selection)
            indices_all_x_selection = indices_all_x_selection[0:max_num_samples_for_ev]
            x_sel = x[indices_all_x_selection]
            x_exp_sel = x_exp[indices_all_x_selection]
        else:
            x_sel = x
            x_exp_sel = x_exp

        if max_test_samples_for_ev is not None and max_test_samples_for_ev < x.shape[0]:
            indices_all_y_selection = indices_all_y + 0
            numpy.random.shuffle(indices_all_y_selection)
            indices_all_y_selection = indices_all_y_selection[0:max_test_samples_for_ev]
            y_sel = y[indices_all_y_selection]
            y_exp_sel = y_exp[indices_all_y_selection]
        else:
            y_sel = y
            y_exp_sel = y_exp

        if verbose:
            print ("indices available=", indices_available)
        for index_short, index_long in enumerate(indices_available):
            taken_tmp = list(taken)  # Copy the taken list
            taken_tmp.append(index_long)  # Add index_long to it
            x_exp_tmp_sel = x_exp_sel[:, taken_tmp]  # Select the variables
            y_exp_tmp_sel = y_exp_sel[:, taken_tmp]

            y_app_sel = approximate_linearly(x_sel, x_exp_tmp_sel, y_exp_tmp_sel)

            # print "QQQ=", compute_explained_var(y_sel, y_app_sel)
            temp_explained_vars[index_short] = compute_explained_var(y_sel, y_app_sel)  # compute explained var
            if verbose:
                print ("taken_tmp=", taken_tmp, "temp_explained_vars[%d (long = %d) ]=%f" %
                       (index_short, index_long, temp_explained_vars[index_short]))

        # Update scores
        max_scores[indices_available] = numpy.maximum(max_scores[indices_available],
                                                      temp_explained_vars - last_explained_var)

        # select maximum
        # print "temp_explained_vars=", temp_explained_vars
        max_explained_var_index_short = temp_explained_vars.argmax()
        # print "max_explained_var_index_short=", max_explained_var_index_short
        # print "indices_available=",indices_available
        max_explained_var_index_long = indices_available[max_explained_var_index_short]
        if verbose:
            print ("Selecting index short:", max_explained_var_index_short, end="")
            print (" and index_ long:", max_explained_var_index_long)

        # mark as taken and update temporal variables
        taken.append(max_explained_var_index_long)
        available_mask[max_explained_var_index_long] = False
        #        last_score = scores[max_explained_var_index_long]
        last_explained_var = temp_explained_vars[max_explained_var_index_short]

    print ("brute max_scores = ", max_scores)
    print ("brute taken = ", taken)

    # Find ordering of variables not yet taken
    if max_comp < dim_out:
        max_explained_var_indices_short = temp_explained_vars.argsort()[::-1][1:]
        # In increasing order, then remove first element, which was already added to taken

        for max_explained_var_index_short in max_explained_var_indices_short:
            taken.append(indices_available[max_explained_var_index_short])

    print ("final taken = ", taken)

    # Make scoring decreasing in ordering stored in taken
    last_explained_var = max(last_explained_var, 0.01)  # For numerical reasons
    last_max_score = -numpy.inf
    sum_max_scores = 0.0
    for i, long_index in enumerate(taken):
        current_max_score = max_scores[long_index]
        sum_max_scores += current_max_score
        if current_max_score > last_max_score and i > 0:
            max_scores[long_index] = last_max_score
            tmp_sum_max_scores = max_scores[taken[0:i + 1]].sum()
            max_scores[taken[0:i + 1]] += (sum_max_scores - tmp_sum_max_scores) / (i + 1)
        last_max_score = max_scores[long_index]
        # print "iteration max_scores = ", max_scores

    print ("preeliminar max_scores = ", max_scores)

    #    max_scores *= (last_explained_var / max_scores.sum())**0.5
    # NOTE: last_explained_var is not the data variance.
    # Here it is the variance up to max_comp components
    # 3 options: all features, first max_comp features, output_dim features
    max_scores *= (last_explained_var / max_scores.sum()) ** 0.5

    print ("final max_scores = ", max_scores)

    if (max_scores == 0.0).any():
        print ("WARNING, removing 0.0 max_scores!")
        max_score_min = (max_scores[max_scores > 0.0]).min()
        # TODO:Find reasonable way to fix this, is this causing the distorted reconstructions???
        max_scores += max_score_min * 0.001
    #         max_scores += (max_scores[max_scores>0.0])
    return max_scores


def rank_expanded_signals_max(x, x_exp, y, y_exp, max_comp=10, k=1, operation="average", max_num_samples_for_ev=None,
                              max_test_samples_for_ev=None, offsetting_mode="max_comp features", verbose=False):
    """ This Second ranking method more robust and closer to max I(x; y_i + Y)-I(x;Y) for all Y.
        
        Ordering and scoring of signals respects principle of best incremental feature selection
        Computes a scores vector that measures the importance of each expanded component at reconstructing a signal
        x, x_exp are training data, y and y_exp are test data
        At most max_comp are evaluated exhaustively, the rest is set equal to the remaining
    """
    dim_out = x_exp.shape[1]
    all_indices = numpy.arange(dim_out)

    indices_all_x = numpy.arange(x.shape[0])
    indices_all_y = numpy.arange(y.shape[0])

    max_scores = numpy.zeros(dim_out)
    available_mask = numpy.zeros(dim_out) >= 0  # boolean mask that indicates which elements are not yet scored
    taken = []  # list with the same elements.

    #   Compute maximum explainable variance (taking all components)
    total_variance = data_variance(y)

    last_explained_var = 0.0
    last_score = 0.0
    for iteration in range(min(max_comp, dim_out)):
        # find individual contribution to expl var, from not taken
        indices_available = all_indices[available_mask]  # mapping from index_short to index_long
        temp_explained_vars = numpy.zeros(
            dim_out - iteration)  # s_like(indices_available, dtype=") #explained variances for each available index

        # On each iteration, the subset of samples used for testing and samples for reconstruction are kept fixed
        if max_num_samples_for_ev is not None and max_num_samples_for_ev < x.shape[0]:
            indices_all_x_selection = indices_all_x + 0
            numpy.random.shuffle(indices_all_x_selection)
            indices_all_x_selection = indices_all_x_selection[0:max_num_samples_for_ev]
            x_sel = x[indices_all_x_selection]
            x_exp_sel = x_exp[indices_all_x_selection]
        else:
            x_sel = x
            x_exp_sel = x_exp

        if max_test_samples_for_ev is notNone and max_test_samples_for_ev < x.shape[0]:
            indices_all_y_selection = indices_all_y + 0
            numpy.random.shuffle(indices_all_y_selection)
            indices_all_y_selection = indices_all_y_selection[0:max_test_samples_for_ev]
            y_sel = y[indices_all_y_selection]
            y_exp_sel = y_exp[indices_all_y_selection]
        else:
            y_sel = y
            y_exp_sel = y_exp

        if verbose:
            print ("indices available=", indices_available)
        for index_short, index_long in enumerate(indices_available):
            taken_tmp = list(taken)  # Copy the taken list
            taken_tmp.append(index_long)  # Add index_long to it
            x_exp_tmp_sel = x_exp_sel[:, taken_tmp]  # Select the variables
            y_exp_tmp_sel = y_exp_sel[:, taken_tmp]

            if operation == "linear_rec":
                y_app_sel = approximate_linearly(x_sel, x_exp_tmp_sel, y_exp_tmp_sel)
            else:
                y_app_sel = approximate_kNN_op(x_sel, x_exp_tmp_sel, y_exp_tmp_sel, k=k, ignore_closest_match=True,
                                               operation=operation)  # invert from taken variables

            # print "QQQ=", compute_explained_var(y_sel, y_app_sel)
            temp_explained_vars[index_short] = compute_explained_var(y_sel, y_app_sel)  # compute explained var
            if verbose:
                print ("taken_tmp=", taken_tmp, "temp_explained_vars[%d (long = %d) ]=%f" % (
                    index_short, index_long, temp_explained_vars[index_short]))

        # Update scores
        max_scores[indices_available] = numpy.maximum(max_scores[indices_available],
                                                      temp_explained_vars - last_explained_var)

        # select maximum
        # print "temp_explained_vars=", temp_explained_vars
        max_explained_var_index_short = temp_explained_vars.argmax()
        # print "max_explained_var_index_short=", max_explained_var_index_short
        # print "indices_available=",indices_available
        max_explained_var_index_long = indices_available[max_explained_var_index_short]
        if verbose:
            print("Selecting index short:", max_explained_var_index_short,
                  " and index_ long:", max_explained_var_index_long)

        # mark as taken and update temporal variables
        taken.append(max_explained_var_index_long)
        available_mask[max_explained_var_index_long] = False
        #        last_score = scores[max_explained_var_index_long]
        last_explained_var = temp_explained_vars[max_explained_var_index_short]

    print("brute max_scores = ", max_scores)
    print("brute taken = ", taken)

    # Find ordering of variables not yet taken
    if max_comp < dim_out:
        max_explained_var_indices_short = \
            temp_explained_vars.argsort()[::-1][1:]  
        # In increasing order, then remove first element, which was already added to taken

        for max_explained_var_index_short in max_explained_var_indices_short:
            taken.append(indices_available[max_explained_var_index_short])

    print("final taken = ", taken)

    # Make scoring decreasing in ordering stored in taken
    last_explained_var = max(last_explained_var, 0.01)  # For numerical reasons
    last_max_score = -numpy.inf
    sum_max_scores = 0.0
    for i, long_index in enumerate(taken):
        current_max_score = max_scores[long_index]
        sum_max_scores += current_max_score
        if current_max_score > last_max_score and i > 0:
            max_scores[long_index] = last_max_score
            tmp_sum_max_scores = max_scores[taken[0:i + 1]].sum()
            max_scores[taken[0:i + 1]] += (sum_max_scores - tmp_sum_max_scores) / (i + 1)
        last_max_score = max_scores[long_index]
        # print "iteration max_scores = ", max_scores

    print("preeliminar max_scores = ", max_scores)

    # Compute explained variance with all features
    indices_all_x_selection = random_subindices(x.shape[0], max_num_samples_for_ev)
    x_sel = x[indices_all_x_selection]
    x_exp_sel = x_exp[indices_all_x_selection]
    indices_all_y_selection = random_subindices(y.shape[0], max_test_samples_for_ev)
    y_sel = y[indices_all_y_selection]
    y_exp_sel = y_exp[indices_all_y_selection]
    if operation == "linear_rec":
        y_app_sel = approximate_linearly(x_sel, x_exp_sel, y_exp_sel)
    else:
        y_app_sel = approximate_kNN_op(x_sel, x_exp_sel, y_exp_sel, k=k, ignore_closest_match=True,
                                       operation=operation)  # invert from taken variables
    explained_var_all_feats = compute_explained_var(y_sel, y_app_sel)

    print("last_explained_var =", last_explained_var)
    print("explained_var_all_feats=", explained_var_all_feats, "total input variance:", total_variance)

    #    max_scores *= (last_explained_var / max_scores.sum())**0.5
    # NOTE: last_explained_var is not the data variance. It is the variance up to max_comp components
    # 3 options: all scores, max_comp scores, output_dim scores (usually all scores)
    if offsetting_mode == "max_comp features":
        max_scores *= (last_explained_var / max_scores.sum())
    elif offsetting_mode == "all features":
        print("explained_var_all_feats=", explained_var_all_feats, "total input variance:", total_variance)
        max_scores *= (explained_var_all_feats / max_scores.sum())
    elif offsetting_mode == "all features smart":
        max_scores *= (last_explained_var / max_scores.sum())
        print("scaled max_scores=", max_scores)
        max_scores += (explained_var_all_feats - last_explained_var) / max_scores.shape[0]
        print("offsetted max_scores=", max_scores)
    elif offsetting_mode == "democratic":
        max_scores = numpy.ones_like(max_scores) * explained_var_all_feats / max_scores.shape[0]
        print("democractic max_scores=", max_scores)
    elif offsetting_mode == "linear":
        # Code fixed!!!
        max_scores = numpy.arange(dim_out, 0, -1) * explained_var_all_feats / (dim_out * (dim_out + 1) / 2)
        print("linear max_scores=", max_scores)
    elif offsetting_mode == "sensitivity_based":
        sens = sensivity_of_linearly_approximation(x_sel, x_exp_sel)
        max_scores = sens * explained_var_all_feats / sens.sum()
        print("sensitivity_based max_scores=", max_scores)
    else:
        ex = "offsetting_mode unknown", offsetting_mode
        raise Exception(ex)
    print("final max_scores = ", max_scores)

    if (max_scores == 0.0).any():
        print("WARNING, removing 0.0 max_scores!")
        max_score_min = (max_scores[max_scores > 0.0]).min()
        max_scores += max_score_min * 0.001
        # TODO:Find reasonable way to fix this, is this causing the distorted reconstructions???
    #         max_scores += (max_scores[max_scores>0.0])
    return max_scores


# TODO: Improve: if max_comp < output_dim choose remaining features from the last evaluation of explained variances.
def rank_expanded_signals(x, x_exp, y, y_exp, max_comp=10, k=1, linear=False, max_num_samples_for_ev=None,
                          max_test_samples_for_ev=None, verbose=False):
    """ Computes a scores vector that measures the importance of each expanded component at reconstructing a signal
        x, x_exp are training data, y and y_exp are test data
        At most max_comp are evaluated exhaustively, the rest is set equal to the remaining
    """
    dim_out = x_exp.shape[1]
    all_indices = numpy.arange(dim_out)

    indices_all_x = numpy.arange(x.shape[0])
    indices_all_y = numpy.arange(y.shape[0])

    scores = numpy.zeros(dim_out)
    available_mask = numpy.zeros(dim_out) >= 0  # boolean mask that indicates which elements are not yet scored
    taken = []  # list with the same elements.

    #   Compute maximum explainable variance (taking all components)
    total_variance = data_variance(y)

    last_explained_var = 0.0
    last_score = 0.0
    for iteration in range(min(max_comp, dim_out)):
        # find individual contribution to expl var, from not taken
        indices_available = all_indices[available_mask]  # mapping from index_short to index_long
        temp_explained_vars = numpy.zeros(
            dim_out - iteration)  # s_like(indices_available, dtype=") #explained variances for each available index

        # On each iteration, the subset of samples used for testing and samples for reconstruction are kept fixed
        if max_num_samples_for_ev is not None and max_num_samples_for_ev < x.shape[0]:
            indices_all_x_selection = indices_all_x + 0
            numpy.random.shuffle(indices_all_x_selection)
            indices_all_x_selection = indices_all_x_selection[0:max_num_samples_for_ev]
            x_sel = x[indices_all_x_selection]
            x_exp_sel = x_exp[indices_all_x_selection]
        else:
            x_sel = x
            x_exp_sel = x_exp

        if max_test_samples_for_ev is not None and max_test_samples_for_ev < x.shape[0]:
            indices_all_y_selection = indices_all_y + 0
            numpy.random.shuffle(indices_all_y_selection)
            indices_all_y_selection = indices_all_y_selection[0:max_test_samples_for_ev]
            y_sel = y[indices_all_y_selection]
            y_exp_sel = y_exp[indices_all_y_selection]
        else:
            y_sel = y
            y_exp_sel = y_exp

        if verbose:
            print("indices available=", indices_available)
        for index_short, index_long in enumerate(indices_available):
            taken_tmp = list(taken)  # Copy the taken list
            taken_tmp.append(index_long)  # Add index_long to it
            x_exp_tmp_sel = x_exp_sel[:, taken_tmp]  # Select the variables
            y_exp_tmp_sel = y_exp_sel[:, taken_tmp]

            y_app_sel = approximate_kNN(x_sel, x_exp_tmp_sel, y_exp_tmp_sel, k=k, ignore_closest_match=True,
                                        label_avg=True)  # invert from taken variables

            # print "QQQ=", compute_explained_var(y_sel, y_app_sel)
            temp_explained_vars[index_short] = compute_explained_var(y_sel, y_app_sel)  # compute explained var
            if verbose:
                print("taken_tmp=", taken_tmp, "temp_explained_vars[%d (long = %d) ]=%f" % (
                    index_short, index_long, temp_explained_vars[index_short]))
        # select maximum

        # print "temp_explained_vars=", temp_explained_vars
        max_explained_var_index_short = temp_explained_vars.argmax()
        # print "max_explained_var_index_short=", max_explained_var_index_short
        # print "indices_available=",indices_available
        max_explained_var_index_long = indices_available[max_explained_var_index_short]
        if verbose:
            print("Selecting index short:", max_explained_var_index_short)
            print(" and index_ long:", max_explained_var_index_long)

        # update total explained var & scores
        # Add logic to robustly handle strange contributions: 3, 2, 1, 4 =>  5, 2.5, 1.25, 1.25 ?
        # TODO:FIX NORMALIZATION WHEN FIRST SCORES ARE ZERO OR NEGATIVE!
        # TODO:NORMALIZATION SHOULD BE OPTIONAL, SINCE IT WEAKENS THE INTERPRETATION OF THE SCORES
        explained_var = max(temp_explained_vars[max_explained_var_index_short], 0.0)
        new_score = explained_var - last_explained_var
        if verbose:
            print("new_score raw = ", new_score)
        new_score = max(new_score, 0.0)
        if new_score > last_score and iteration > 0:
            new_score = last_score  # Here some options are available to favour components taken first
        scores[max_explained_var_index_long] = new_score

        if verbose:
            print("tmp scores = ", scores)
        # normalize scores, so that they sume up to explained_var
        sum_scores = scores.sum()
        residual = max(explained_var, 0.0) - sum_scores
        if residual > 0.0:
            correction = residual / (iteration + 1)
            scores[taken] += correction
            scores[max_explained_var_index_long] += correction

        # scores = scores * explained_var / (sum_scores+1e-6) #TODO:CORRECT THIS; INSTEAD OF FACTOR USE ADDITIVE TERM
        if verbose:
            print("normalized scores = ", scores, "sum to:", scores.sum(), "explained_var =", explained_var)

        # mark as taken and update temporal variables
        taken.append(max_explained_var_index_long)
        available_mask[max_explained_var_index_long] = False
        last_score = scores[max_explained_var_index_long]
        last_explained_var = explained_var

    # handle variables not used, assign equal scores to all of them
    preserve_last_evaluation = True
    if preserve_last_evaluation and max_comp < dim_out:
        # The score of the last feature found will be modified, as well as of not yet found features
        # TODO: Take care of negative values
        if last_score <= 0.0:
            last_score = 0.01  # Just some value is needed here
        remaining_output_features = len(temp_explained_vars)  # including feature already processed
        remaining_ordered_explained_variances_short_index = numpy.argsort(temp_explained_vars)[::-1]
        remaining_ordered_explained_variances_long_index = indices_available[
            remaining_ordered_explained_variances_short_index]
        remaining_ordered_explained_variances = temp_explained_vars[
                                                    remaining_ordered_explained_variances_short_index] + 0.0
        remaining_total_contribution = last_score
        print("last_score=", last_score)

        beta = 0.95
        remaining_ordered_explained_variances[
            remaining_ordered_explained_variances <= 0.0] = 0.0001  # To avoid division over zero, numerical hack
        # numpy.clip(remaining_ordered_explained_variances, 0.0, None) fails here!!!!  
        print("remaining_ordered_explained_variances=", remaining_ordered_explained_variances)
        minimum = remaining_ordered_explained_variances.min()  # first element
        ev_sum = remaining_ordered_explained_variances.sum()
        normalized_scores = (remaining_total_contribution / (ev_sum - remaining_output_features * minimum) * beta) * \
                            (remaining_ordered_explained_variances - minimum) + \
                            ((1.0 - beta) / remaining_output_features) * remaining_total_contribution
        print("normalized_scores=", normalized_scores)
        print("remaining_ordered_explained_variances_long_index=", remaining_ordered_explained_variances_long_index)
        print(scores.dtype)
        print(normalized_scores.dtype)

        scores[remaining_ordered_explained_variances_long_index] = normalized_scores
    else:
        # rest_explained_variance = total_variance-last_explained_var
        sum_scores = scores.sum()
        rest_explained_variance = total_variance - sum_scores
        if verbose:
            print("rest_explained_variance=", rest_explained_variance)
        correction = rest_explained_variance / dim_out
        scores += correction

    if (scores == 0.0).any():
        print("WARNING, removing 0.0 scores!")
        scores += 0.0001

    #    num_unused = dim_out - max_comp
    #    scores[available_mask] = min(rest_explained_variance / num_unused, last_score)
    #    sum_scores = scores.sum()
    #    scores = scores * explained_var / (sum_scores+1e-6)

    if verbose:
        print("final scores: ", scores)

    if verbose and linear and False:
        for i in indices_available:
            taken.append(i)
        scores[taken] = numpy.arange(dim_out - 1, -1, -1)  # **2 #WARNING!!! QUADRATIC SCORES!!!
        scores = scores * total_variance / scores.sum()
        print("Overriding with linear scores:", scores)

    return scores


# TODO: Remove this node, it is now obsolete
class IEVMNode(mdp.Node):
    """ Node implementing simple Incremental Explained Variance Maximization.
        
        Extracted features are moderately useful for reconstruction, although this node does
        itself provide reconstruction.
        The expansion function is optional, as well as performing PCA on the scores.
        The added variance of the first k-outputs is equal to the explained variance of such k-outputs.
    """
    def __init__(self, input_dim=None, output_dim=None, expansion_funcs=None, k=5, max_comp=None,
                 max_num_samples_for_ev=None, max_test_samples_for_ev=None, use_pca=False, use_sfa=False,
                 max_preserved_sfa=2.0, second_weighting=False, operation="average", out_sfa_filter=False, **argv):
        super(IEVMNode, self).__init__(input_dim=input_dim, output_dim=output_dim, **argv)
        if expansion_funcs is not None:
            self.exp_node = GeneralExpansionNode(funcs=expansion_funcs)
        else:
            self.exp_node = None
        self.sfa_node = None
        self.second_weighting = second_weighting
        self.use_pca = use_pca
        self.use_sfa = use_sfa
        if use_sfa and not use_pca:
            er = "Combination of use_sfa and use_pca not considered. Please activate use_pca or deactivate use_sfa"
            raise Exception(er)
        self.k = k
        self.max_comp = max_comp
        self.max_num_samples_for_ev = max_num_samples_for_ev
        self.max_test_samples_for_ev = max_test_samples_for_ev
        self.feature_scaling_factor = 0.5  # Factor that prevents amplitudes of features from growing across the network
        self.exponent_variance = 0.5
        self.operation = operation
        self.max_preserved_sfa = max_preserved_sfa
        self.out_sfa_filter = out_sfa_filter

    @staticmethod
    def is_trainable():
        return True

    def _train(self, x, block_size=None, train_mode=None, node_weights=None, edge_weights=None, scheduler=None,
               n_parallel=None, **argv):
        num_samples, self.input_dim = x.shape

        if self.output_dim is None:
            self.output_dim = self.input_dim

        if self.max_comp is None:
            self.max_comp = min(self.input_dim, self.output_dim)
        else:
            self.max_comp = min(self.max_comp, self.input_dim, self.output_dim)

        print("Training IEVMNode...")

        self.x_mean = x.mean(axis=0)  # Remove mean before expansion
        x = x - self.x_mean

        if self.exp_node is not None:  # Expand data
            print("expanding x...")
            exp_x = self.exp_node.execute(x)
        else:
            exp_x = x

        self.expanded_dim = exp_x.shape[1]
        self.exp_x_mean = exp_x.mean(axis=0)
        self.exp_x_std = exp_x.std(axis=0)

        print("self.exp_x_mean=", self.exp_x_mean)
        print("self.exp_x_std=", self.exp_x_std)
        if (self.exp_x_std == 0).any():
            er = "zero-component detected"
            raise Exception(er)

        n_exp_x = (exp_x - self.exp_x_mean) / self.exp_x_std  # Remove media and variance from expansion

        print("ranking n_exp_x ...")
        rankings = rank_expanded_signals_max(x, n_exp_x, x, n_exp_x, max_comp=self.max_comp, k=self.k,
                                             operation=self.operation,
                                             max_num_samples_for_ev=self.max_num_samples_for_ev,
                                             max_test_samples_for_ev=self.max_test_samples_for_ev, verbose=True)
        rankings *= self.feature_scaling_factor
        print("rankings=", rankings)
        if (rankings == 0).any():
            er = "zero-component detected"
            raise Exception(er)

        self.perm1 = numpy.argsort(rankings)[::-1]  # Sort in decreasing ranking
        self.magn1 = rankings
        print("self.perm1=", self.perm1)

        s_x_1 = n_exp_x * self.magn1 ** self.exponent_variance  # Scale according to ranking
        s_x_1 = s_x_1[:, self.perm1]  # Permute with most important signal first

        if self.second_weighting:
            print("ranking s_x_1 ...")
            rankings_B = rank_expanded_signals_max(x, s_x_1, x, s_x_1, max_comp=self.max_comp, k=self.k,
                                                   operation=self.operation,
                                                   max_num_samples_for_ev=self.max_num_samples_for_ev,
                                                   max_test_samples_for_ev=self.max_test_samples_for_ev, verbose=False)
            print("rankings_B=", rankings_B)
            if (rankings_B == 0).any():
                er = "zero-component detected"
                raise Exception(er)

            self.perm1_B = numpy.argsort(rankings_B)[::-1]  # Sort in decreasing ranking
            self.magn1_B = rankings_B
            print("self.perm1_B=", self.perm1_B)

            # WARNING, this only works for normalized s_x_1
            s_x_1B = s_x_1 * self.magn1_B ** self.exponent_variance  # Scale according to ranking
            s_x_1B = s_x_1B[:, self.perm1_B]  # Permute with most important signal first
        else:
            s_x_1B = s_x_1

        if self.use_sfa:
            self.sfa_node = mdp.nodes.SFANode()
            # TODO: Preserve amplitude
            self.sfa_node.train(s_x_1B, block_size=block_size, train_mode=train_mode)  
            # , node_weights=None, edge_weights=None, scheduler = None, n_parallel=None)
            self.sfa_node.stop_training()

            print("self.sfa_node.d", self.sfa_node.d)

            # Adaptive mechanism based on delta values
            if isinstance(self.max_preserved_sfa, float):
                self.num_sfa_features_preserved = (self.sfa_node.d <= self.max_preserved_sfa).sum()
            elif isinstance(self.max_preserved_sfa, int):
                self.num_sfa_features_preserved = self.max_preserved_sfa
            else:
                ex = "Cannot handle type of self.max_preserved_sfa"
                print(ex)
                raise Exception(ex)

            # self.num_sfa_features_preserved = 10
            sfa_x = self.sfa_node.execute(s_x_1B)

            # TODO: Change internal variables of SFANode, so that we do not need to zero some components
            # TODO: Is this equivalent to truncation of the matrices??? PERHAPS IT IS NOT !!!
            sfa_x[:, self.num_sfa_features_preserved:] = 0.0

            proj_sfa_x = self.sfa_node.inverse(sfa_x)

            sfa_x = sfa_x[:, 0:self.num_sfa_features_preserved]
            # Notice that sfa_x has WEIGHTED zero-mean, thus we correct this here?
            self.sfa_x_mean = sfa_x.mean(axis=0)
            self.sfa_x_std = sfa_x.std(axis=0)

            print("self.sfa_x_mean=", self.sfa_x_mean)
            print("self.sfa_x_std=", self.sfa_x_std)
            sfa_x -= self.sfa_x_mean

            sfa_removed_x = s_x_1B - proj_sfa_x  # Remove sfa projection of data

        else:
            self.num_sfa_features_preserved = 0
            sfa_x = numpy.ones((num_samples, 0))
            sfa_removed_x = s_x_1B

        pca_out_dim = self.expanded_dim - self.num_sfa_features_preserved
        if self.use_pca and pca_out_dim > 0:
            self.pca_node = mdp.nodes.PCANode(output_dim=pca_out_dim)
            self.pca_node.train(sfa_removed_x)

            # TODO:check that pca_out_dim > 0
            pca_x = self.pca_node.execute(sfa_removed_x)

            self.pca_x_mean = pca_x.mean(axis=0)
            self.pca_x_std = pca_x.std(axis=0)

            print("self.pca_x_std=", self.pca_x_std)
            if (self.pca_x_std == 0).any():
                er = "zero-component detected"
                raise Exception(er)
            # TODO: Is this step needed? if heuristic works well this weakens algorithm
            n_pca_x = (pca_x - self.pca_x_mean) / self.pca_x_std
        else:
            n_pca_x = sfa_removed_x[:, 0:pca_out_dim]

        # Concatenate SFA and PCA signals and rank them preserving SFA components in ordering
        if self.use_pca or self.use_sfa:
            # TODO: Either both signals conserve magnitudes or they are both normalized
            sfa_pca_x = numpy.concatenate((sfa_x, n_pca_x), axis=1)

            sfa_pca_rankings = rank_expanded_signals_max(x, sfa_pca_x, x, sfa_pca_x, max_comp=self.max_comp, k=self.k,
                                                         operation=self.operation,
                                                         max_num_samples_for_ev=self.max_num_samples_for_ev,
                                                         max_test_samples_for_ev=self.max_test_samples_for_ev,
                                                         verbose=False)
            sfa_pca_rankings *= self.feature_scaling_factor
            # Only one magnitude normalization by node, but where should it be done? I guess after last transformation

            print("sfa_pca_rankings=", sfa_pca_rankings)
            if (sfa_pca_rankings == 0).any():
                er = "zero-component detected"
                raise Exception(er)

            self.magn2 = sfa_pca_rankings
            perm2a = numpy.arange(self.num_sfa_features_preserved, dtype="int")
            perm2b = numpy.argsort(sfa_pca_rankings[self.num_sfa_features_preserved:])[::-1]
            self.perm2 = numpy.concatenate((perm2a, perm2b + self.num_sfa_features_preserved))
            print("second permutation=", self.perm2)

            # WARNING, this only works for normalized sfa_pca_x
            s_x_2 = sfa_pca_x * self.magn2 ** self.exponent_variance  # Scale according to ranking
            s_x_2 = s_x_2[:, self.perm2]  # Permute with slow features first, and then most important signal first
        else:
            s_x_2 = n_pca_x

        # Tuncating output_dim components
        s_x_2_truncated = s_x_2[:, 0:self.output_dim]

        # Filtering output through SFA
        if self.out_sfa_filter:
            self.out_sfa_node = mdp.nodes.SFANode()
            self.out_sfa_node.train(s_x_2_truncated, block_size=block_size, train_mode=train_mode)
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
        x_orig = x + 0.0
        num_samples = x.shape[0]
        zm_x = x - self.x_mean

        if self.exp_node:
            exp_x = self.exp_node.execute(zm_x)
        else:
            exp_x = zm_x

        n_exp_x = (exp_x - self.exp_x_mean) / self.exp_x_std
        if numpy.isnan(n_exp_x).any() or numpy.isinf(n_exp_x).any():
            print("n_exp_x=", n_exp_x)
            quit()
        n_exp_x[numpy.isnan(n_exp_x)] = 0.0

        if numpy.isnan(self.magn1).any():
            print("self.magn1=", self.magn1)
            quit()

        s_x_1 = n_exp_x * self.magn1 ** self.exponent_variance  # Scale according to ranking
        s_x_1 = s_x_1[:, self.perm1]  # Permute with most important signal first

        if self.second_weighting:
            s_x_1B = s_x_1 * self.magn1_B ** self.exponent_variance  # Scale according to ranking_B
            s_x_1B = s_x_1B[:, self.perm1_B]  # Permute with most important signal first
        else:
            s_x_1B = s_x_1

        if numpy.isnan(s_x_1B).any():
            print("s_x_1B=", s_x_1B)
            quit()

        if self.use_sfa:
            sfa_x = self.sfa_node.execute(s_x_1B)

            # TODO: Change internal variables of SFANode, so that we do not need to zero some components
            sfa_x[:, self.num_sfa_features_preserved:] = 0.0

            proj_sfa_x = self.sfa_node.inverse(sfa_x)

            sfa_x = sfa_x[:, 0:self.num_sfa_features_preserved]
            sfa_x -= self.sfa_x_mean

            sfa_removed_x = s_x_1B - proj_sfa_x
        else:
            sfa_x = numpy.ones((num_samples, 0))
            sfa_removed_x = s_x_1B

        pca_out_dim = self.expanded_dim - self.num_sfa_features_preserved
        if self.use_pca and pca_out_dim > 0:
            pca_x = self.pca_node.execute(sfa_removed_x)

            n_pca_x = (pca_x - self.pca_x_mean) / self.pca_x_std
        else:
            n_pca_x = sfa_removed_x[:, 0:pca_out_dim]

        if self.use_pca or self.use_sfa:
            sfa_pca_x = numpy.concatenate((sfa_x, n_pca_x), axis=1)

            s_x_2 = sfa_pca_x * self.magn2 ** self.exponent_variance  # Scale according to ranking
            s_x_2 = s_x_2[:, self.perm2]  # Permute with most important signal first
        else:
            s_x_2 = n_pca_x

        if numpy.isnan(s_x_2).any():
            print("s_x_2=", s_x_2)
            quit()

        # Tuncating output_dim components
        s_x_2_truncated = s_x_2[:, 0:self.output_dim]

        # Filtering output through SFA
        if self.out_sfa_filter:
            sfa_filtered = self.out_sfa_node.execute(s_x_2_truncated)
        else:
            sfa_filtered = s_x_2_truncated

        verbose = False
        if verbose:
            print("x[0]=", x_orig[0])
            print("x_zm[0]=", x[0])
            print("exp_x[0]=", exp_x[0])
            print("s_x_1[0]=", s_x_1[0])
            print("sfa_removed_x[0]=", sfa_removed_x[0])
            print("proj_sfa_x[0]=", proj_sfa_x[0])
            print("pca_x[0]=", pca_x[0])
            print("n_pca_x[0]=", n_pca_x[0])
            print("sfa_x[0]=", sfa_x[0] + self.sfa_x_mean)
            print("s_x_2_truncated[0]=", s_x_2_truncated[0])
            print("sfa_filtered[0]=", sfa_filtered[0])

        return sfa_filtered

    # TODO:Code inverse with SFA
    def _inverse(self, y):
        num_samples = y.shape[0]
        if y.shape[1] != self.output_dim:
            er = "Serious dimensionality inconsistency:", y.shape[0], self.output_dim
            raise Exception(er)
        #        input_dim = self.input_dim

        # De-Filtering output through SFA
        sfa_filtered = y
        if self.out_sfa_filter:
            s_x_2_truncated = self.out_sfa_node.inverse(sfa_filtered)
        else:
            s_x_2_truncated = sfa_filtered

        # De-Tuncating output_dim components
        s_x_2_full = numpy.zeros((num_samples, self.expanded_dim))
        s_x_2_full[:, 0:self.output_dim] = s_x_2_truncated

        if self.use_pca or self.use_sfa:
            perm_2_inv = numpy.zeros(self.expanded_dim, dtype="int")
            #            print "input_dim", input_dim
            #            print "self.perm2", self.perm2
            #            print "len(self.perm2)", len(self.perm2)
            perm_2_inv[self.perm2] = numpy.arange(self.expanded_dim, dtype="int")
            # print perm_2_inv
            sfa_pca_x = s_x_2_full[:, perm_2_inv]
            sfa_pca_x /= self.magn2 ** self.exponent_variance

            sfa_x = sfa_pca_x[:, 0:self.num_sfa_features_preserved]
            n_pca_x = sfa_pca_x[:, self.num_sfa_features_preserved:]
        else:
            # sfa_x = ...?
            n_pca_x = s_x_2_full

        pca_out_dim = self.expanded_dim - self.num_sfa_features_preserved
        if self.use_pca and pca_out_dim > 0:
            pca_x = n_pca_x * self.pca_x_std + self.pca_x_mean
            sfa_removed_x = self.pca_node.inverse(pca_x)
        else:
            sfa_removed_x = n_pca_x

        if self.use_sfa:
            sfa_x += self.sfa_x_mean
            sfa_x_full = numpy.zeros((num_samples, self.expanded_dim))
            sfa_x_full[:, 0:self.num_sfa_features_preserved] = sfa_x
            proj_sfa_x = self.sfa_node.inverse(sfa_x_full)
            s_x_1B = sfa_removed_x + proj_sfa_x
        else:
            s_x_1B = sfa_removed_x

        if self.second_weighting:
            perm_1B_inv = numpy.zeros(self.expanded_dim, dtype="int")
            perm_1B_inv[self.perm1_B] = numpy.arange(self.expanded_dim, dtype="int")
            s_x_1 = s_x_1B[:, perm_1B_inv]
            s_x_1 /= self.magn1_B ** self.exponent_variance
        else:
            s_x_1 = s_x_1B

        perm_1_inv = numpy.zeros(self.expanded_dim, dtype="int")
        perm_1_inv[self.perm1] = numpy.arange(self.expanded_dim, dtype="int")
        n_exp_x = s_x_1[:, perm_1_inv]
        n_exp_x /= self.magn1 ** self.exponent_variance

        exp_x = n_exp_x * self.exp_x_std + self.exp_x_mean
        if self.exp_node:
            zm_x = self.exp_node.inverse(exp_x)
        else:
            zm_x = exp_x
        x = zm_x + self.x_mean

        verbose = False
        if verbose:
            print("x[0]=", x[0])
            print("zm_x[0]=", zm_x[0])
            print("exp_x[0]=", exp_x[0])
            print("s_x_1[0]=", s_x_1[0])
            print("proj_sfa_x[0]=", proj_sfa_x[0])
            print("sfa_removed_x[0]=", sfa_removed_x[0])
            print("pca_x[0]=", pca_x[0])
            print("n_pca_x[0]=", n_pca_x[0])
            print("sfa_x[0]=", sfa_x[0])

        return x


def export_to_libsvm(labels_classes, features, filename):
    dim_features = features.shape[1]
    filehandle = open(filename, "wb")
    if len(features) != len(labels_classes):
        er = "number of labels_classes %d does not match number of samples %d!" % (len(labels_classes), len(features))
        raise Exception(er)
    for i in range(len(features)):
        filehandle.write("%d" % labels_classes[i])
        for j in range(dim_features):
            filehandle.write(" %d:%f" % (j + 1, features[i, j]))
        filehandle.write("\n")
    filehandle.close()


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
    if not (is_monotonic_increasing(all_classes)):
        er = "Array of class numbers should be monotonically increasing:" + str(all_classes)
        raise Exception(er)
    if not (is_monotonic_increasing(avg_labels)):
        er = "SEVERE WARNING! Array of labels should be monotonically increasing:" + str(avg_labels)
        raise Exception(er)
    if len(all_classes) != len(avg_labels):
        er = "SEVERE WARNING! Array of classes should have the same length as the array of labels: %d vs. %d" % \
            (len(all_classes), len(avg_labels))
        raise Exception(er)
    indices = numpy.searchsorted(all_classes, class_numbers)
    return avg_labels[indices]


def map_labels_to_class_number(all_classes, avg_labels, labels):
    if not (is_monotonic_increasing(all_classes)):
        er = "Array of class numbers should be monotonically increasing:", all_classes
        raise Exception(er)
    if not (is_monotonic_increasing(avg_labels)):
        er = "Array of labels should be monotonically increasing:", avg_labels
        raise Exception(er)
    if len(all_classes) != len(avg_labels):
        er = "Array of classes should have the same length as the array of labels:" + str(len(all_classes)) + \
             " vs. " + str(len(avg_labels))
        raise Exception(er)
    interval_midpoints = (avg_labels[1:] + avg_labels[:-1]) / 2.0

    indices = numpy.searchsorted(interval_midpoints, labels)
    return all_classes[indices]


def random_boolean_array(size):
    return numpy.random.randint(2, size=size) == 1


def generate_random_sigmoid_weights(input_dim, num_features):
    #    scale_factor = 8.0 / numpy.sqrt(input_dim)
    scale_factor = 1.0
    c = numpy.random.normal(loc=0.0, scale=scale_factor, size=(input_dim, num_features))
    c2 = (numpy.abs(c) ** 1.5)
    #    print "c2=", c2
    #    print "c2[0]=", c2[0]
    c = 4.0 * numpy.sign(c) * c2 / c2.max()
    #    print "c=", c
    #    print "c[0]=", c[0]
    l = numpy.random.normal(loc=0.0, scale=1.0, size=num_features)
    return c, l


def extract_sigmoid_features(x, c1, l1, scale=1.0, offset=0.0, use_special_features=False):
    if x.shape[1] != c1.shape[0] or c1.shape[1] != len(l1):
        er = "Array dimensions mismatch: x.shape =" + str(x.shape) + ", c1.shape =" + str(
            c1.shape) + ", l1.shape=" + str(l1.shape)
        print(er)
        raise Exception(er)
    s = numpy.dot(x, c1) + l1
    f = numpy.tanh(s)
    if use_special_features:
        # replace features with l1 = -1.0 to x^T * c1[i]
        # replace features with l1 =  0.8 to 0.8 expo(x^T * c1[i])
        #        print "f.shape=", f.shape
        #        print "numpy.dot(x,c1[:,0]).shape=", numpy.dot(x,c1[:,0]).shape
        fixed = 0
        for i in range(c1.shape[1]):
            if l1[i] == 0.8:
                f[:, i] = numpy.abs(numpy.dot(x, c1[:, i])) ** 0.8
                fixed += 1
            elif l1[i] == 1.0:  # identity
                f[:, i] = numpy.dot(x, c1[:, i])
                fixed += 1
        print("Number of features adapted to either identity or 08Expo:", fixed)
    return f * scale + offset


# sf_matrix has shape input_dim x output_dim
def evaluate_coefficients(sf_matrix):
    # Exponentially decaying weights
    #    weighting = numpy.e ** -numpy.arange(sf_matrix.shape[1])
    weighting = 2.0 ** -numpy.arange(sf_matrix.shape[1])
    weighted_relevances = numpy.abs(sf_matrix) * weighting
    relevances = weighted_relevances.sum(axis=1)
    return relevances


class SFAAdaptiveNLNode(mdp.Node):
    """Node that implements SFA with an adaptive non-linearity.
    """
    def __init__(self, input_dim=None, output_dim=None, pre_expansion_node_class=None, final_expanded_dim=None,
                 initial_expansion_size=None, starting_point=None, expansion_size_decrement=None,
                 expansion_size_increment=None,
                 number_iterations=2, **argv):
        super(SFAAdaptiveNLNode, self).__init__(input_dim=input_dim, output_dim=output_dim, **argv)
        self.pre_expansion_node_class = pre_expansion_node_class
        self.pre_expansion_node = None

        self.final_expanded_dim = final_expanded_dim
        self.initial_expansion_size = initial_expansion_size
        self.starting_point = starting_point
        self.expansion_size_decrement = expansion_size_decrement
        self.expansion_size_increment = expansion_size_increment
        self.number_iterations = number_iterations
        self.sfa_node = None
        self.f1_mean = None
        self.f1_std = None

    @staticmethod
    def is_trainable():
        return True

    # sfa_block_size, sfa_train_mode, etc. would be preferred
    # max_preserved_sfa=1.995
    def _train(self, x, block_size=None, train_mode=None, node_weights=None, edge_weights=None, scheduler=None,
               n_parallel=None, **argv):
        self.input_dim = x.shape[1]

        if self.output_dim is None:
            self.output_dim = self.input_dim

        print("Training SFAAdaptiveNLNode...")
        print("block_size =", block_size, ", train_mode =", train_mode)
        print("x.shape=", x.shape, "self.starting_point=", self.starting_point)

        # TODO: Remove mean and normalize variance before expansion
        #        self.x_mean = x.mean(axis=0)
        #        x_zm=x-self.x_mean

        # TODO:Make this code more pretty (refactoring)
        if self.starting_point == "Identity":
            print("wrong1")
            c0 = numpy.identity(self.input_dim)
            l0 = numpy.ones(self.input_dim) * -1.0  # Code identity
        elif self.starting_point == "08Exp":
            print("good 1")
            c0 = numpy.concatenate((numpy.identity(self.input_dim), numpy.identity(self.input_dim)), axis=1)
            l0 = numpy.concatenate((numpy.ones(self.input_dim) * 1.0, numpy.ones(self.input_dim) * 0.8), axis=0)

        if self.starting_point == "Identity" or self.starting_point == "08Exp":
            print("good 2")
            remaining_feats = self.initial_expansion_size - c0.shape[1]
            print("remaining_feats =", remaining_feats)
            if remaining_feats < 0:
                er = "Error, features needed for identity or 08Exp exceeds number of features availabe" + \
                     "remaining_feats=%d < 0" % remaining_feats + \
                     ". self.initial_expansion_size=%d" % self.initial_expansion_size + \
                     "c0.shape[1]%d" % c0.shape[1]
                raise Exception(er)
            c2, l2 = generate_random_sigmoid_weights(self.input_dim, remaining_feats)
            c1 = numpy.concatenate((c0, c2), axis=1)
            l1 = numpy.concatenate((l0, l2), axis=0)
        else:
            print("wrong wrong")
            c1, l1 = generate_random_sigmoid_weights(self.input_dim,
                                                     self.initial_expansion_size - self.expansion_size_increment)

        for num_iter in range(self.number_iterations):
            print("**************** Iteration %d of %d ********************" % (num_iter, self.number_iterations))
            if num_iter > 0:  # Only add additional features after first iteration
                cp, lp = generate_random_sigmoid_weights(self.input_dim, self.expansion_size_increment)
                c1 = numpy.append(c1, cp, axis=1)
                l1 = numpy.append(l1, lp, axis=0)
            # print "c1=", c1
            # print "l1=", l1
            f1 = extract_sigmoid_features(x, c1, l1, use_special_features=True)
            f1_mean = f1.mean(axis=0)
            f1 = f1 - f1_mean
            f1_std = f1.std(axis=0)
            f1 = f1 / f1_std
            # print "Initial features f1=", f1
            print("f1.shape=", f1.shape)
            print("f1[0]=", f1[0])
            print("f1[-1]=", f1[-1])
            sfa_node = mdp.nodes.SFANode(output_dim=self.output_dim)
            sfa_node.train(f1, block_size=block_size, train_mode=train_mode, node_weights=node_weights,
                           edge_weights=edge_weights, scheduler=scheduler, n_parallel=n_parallel)
            sfa_node.stop_training()
            print("self.sfa_node.d (full expanded) =", sfa_node.d)

            # Evaluate features based on sfa coefficient
            coeffs = evaluate_coefficients(sfa_node.sf)
            print("Scores of each feature from SFA coefficients:", coeffs)
            # find indices of best features. Largest scores first
            best_feat_indices = coeffs.argsort()[::-1]
            print("indices of best features:", best_feat_indices)

            # remove worst expansion_size_decrement features
            if num_iter < self.number_iterations - 1:  # Except during last iteration
                best_feat_indices = best_feat_indices[:-self.expansion_size_decrement]

            c1 = c1[:, best_feat_indices]
            l1 = l1[best_feat_indices]
            # print "cc=", cc
            # print "ll=", ll

        if c1.shape[1] > self.final_expanded_dim:
            c1 = c1[:, :self.final_expanded_dim]
            l1 = l1[:self.final_expanded_dim]

        self.c1 = c1
        self.l1 = l1
        print("self.c1.shape=,", self.c1.shape, "self.l1.shape=,", self.l1.shape)
        print("Learning of non-linear features finished")
        f1 = extract_sigmoid_features(x, self.c1, self.l1, use_special_features=True)
        self.f1_mean = f1.mean(axis=0)
        f1 -= self.f1_mean
        self.f1_std = f1.std(axis=0)
        f1 /= self.f1_std
        self.sfa_node = mdp.nodes.SFANode(output_dim=self.output_dim)
        self.sfa_node.train(f1, block_size=block_size, train_mode=train_mode, node_weights=node_weights,
                            edge_weights=edge_weights, scheduler=scheduler, n_parallel=n_parallel)
        self.sfa_node.stop_training()
        print("self.sfa_node.d (final features) =", self.sfa_node.d)
        # Evaluate features based on sfa coefficient
        coeffs = evaluate_coefficients(self.sfa_node.sf)
        print("evaluation of each features from SFA coefficients: ", coeffs)
        # find indices of best features. Largest scores first
        best_feat_indices = coeffs.argsort()[::-1]
        print("indices of best features:", best_feat_indices)

        print("f1.shape=", f1.shape)
        # Train linear regression node for a linear approximation to inversion
        self.lr_node = mdp.nodes.LinearRegressionNode()
        y = self.sfa_node.execute(f1)
        self.lr_node.train(y, x)
        self.lr_node.stop_training()
        x_app = self.lr_node.execute(y)
        ev_linear_inverse = compute_explained_var(x, x_app) / data_variance(x)
        print("EV_linear_inverse (train)=", ev_linear_inverse)
        self.stop_training()

    def _is_invertible(self):
        return True

    def _execute(self, x):
        num_samples = x.shape[0]

        f1 = extract_sigmoid_features(x, self.c1, self.l1, use_special_features=True)
        f1 -= self.f1_mean
        f1 /= self.f1_std
        return self.sfa_node.execute(f1)

    def _inverse(self, y, linear_inverse=True):
        x_app = self.lr_node.execute(y)
        return x_app


# TODO:Finish this and correct it
def indices_training_graph_split(num_samples, train_mode="regular", block_size=None, num_parts=1):
    if train_mode == "regular":
        indices = numpy.arange(num_samples)
        block_assignment = (indices * 1.0 * num_parts / num_samples).astype(int)
        numpy.random.shuffle(block_assignment)
        part_indices = []
        for num_part in range(num_parts):
            part = indices[block_assignment == num_part]
            part_indices.append(part)

    elif train_mode in ["serial", "sequence"]:
        if isinstance(block_size, int):
            shuffled_indices = numpy.zeros(num_samples)
            for block in range(num_samples // block_size):
                shuffled_indices[block * block_size:(block + 1) * block_size] = \
                    (numpy.arange(block_size) * 1.0 * num_parts / block_size).astype(int)

            for block in range(num_samples // block_size):
                shuffled_indices = (numpy.arange(block_size) * 1.0 * num_parts / block_size).astype(int)

                numpy.random.shuffle(shuffled_indices[block * block_size:(block + 1) * block_size])
            part_indices = []
            for num_part in range(num_parts):
                part = indices[block_assignment == num_part]
                part_indices.append(part)
        else:
            er = "Inhomogeneous block sizes not supported for now"
            raise Exception(er)

    elif train_mode == "clustered":
        print("Mode unuported for now... FIX this!!!")


# Cumulative score metric
def cumulative_score(ground_truth, estimation, largest_error, integer_rounding=True):
    if len(ground_truth) != len(estimation):
        er = "ground_truth and estimation have different number of elements"
        raise Exception(er)

    if integer_rounding:
        _estimation = numpy.rint(estimation)
    else:
        _estimation = estimation

    N_e_le_j = (numpy.absolute(_estimation - ground_truth) <= largest_error).sum()
    return N_e_le_j * 1.0 / len(ground_truth)


def compute_regression_performance(data_training, correct_labels_training, data_test, correct_labels_test,
                                   size_feature_space, starting_point=None):
    input_dim = data_training.shape[1]

    # Generate functions used for regression
    data_training_mean = data_training.mean(axis=0)
    data_training_std = data_training.std(axis=0)

    data_training_norm = (data_training - data_training_mean) / data_training_std
    data_test_norm = (data_test - data_training_mean) / data_training_std

    c1, l1 = generate_random_sigmoid_weights(input_dim, size_feature_space)
    if starting_point == "Identity":
        # print "adding identity coefficients to expansion"
        c1[0:input_dim, 0:input_dim] = numpy.identity(input_dim)
        l1[0:input_dim] = numpy.ones(input_dim) * 1.0  # Code identity
    elif starting_point == "Sigmoids":
        print("Sigmoid starting point enabled")
        # print "adding identity coefficients to expansion"
        c1[0:input_dim, 0:input_dim] = numpy.identity(input_dim)
        l1[0:input_dim] = numpy.ones(input_dim) * 0.0  # Sigmoids of each component will be computed later
    elif starting_point == "08Exp":  # identity included
        # print "adding 08Exp coefficients to expansion"
        c1[0:input_dim, 0:input_dim] = numpy.identity(input_dim)
        c1[0:input_dim, input_dim:2 * input_dim] = numpy.identity(input_dim)

        l1[0:input_dim] = numpy.ones(input_dim) * 1.0  # Code identity
        l1[input_dim:2 * input_dim] = numpy.ones(input_dim) * 0.8  # Code 08exp
    else:
        er = "starting_point unknown:", starting_point
        raise Exception(er)

    expanded_sl_training = extract_sigmoid_features(data_training_norm, c1, l1, scale=1.0, offset=0.0,
                                                    use_special_features=True)
    expanded_sl_test = extract_sigmoid_features(data_test_norm, c1, l1, scale=1.0, offset=0.0,
                                                use_special_features=True)

    lr_node = mdp.nodes.LinearRegressionNode()
    lr_node.train(expanded_sl_training, correct_labels_training.reshape((-1, 1)))
    lr_node.stop_training()

    estimated_labels_training = lr_node.execute(expanded_sl_training).flatten()
    estimated_labels_test = lr_node.execute(expanded_sl_test).flatten()
    # print "estimated_labels_training.shape=", estimated_labels_training.shape
    # print "diff estimated vs training for training:", estimated_labels_training - correct_labels_training

    RMSE_expansion_training = (distance_squared_Euclidean(correct_labels_training, estimated_labels_training) / len(
        correct_labels_training)) ** 0.5
    RMSE_expansion_test = (distance_squared_Euclidean(correct_labels_test, estimated_labels_test) / len(
        correct_labels_test)) ** 0.5

    return RMSE_expansion_training, RMSE_expansion_test


def correct_classif_rateC(ground_truth, classified, verbose=False):
    """ Computes a float value indicating the classification rate
    
        output = number of success classifications / total number of samples
        both input arrays must have the same length and integer values
    """
    num = len(ground_truth)
    if len(ground_truth) != len(classified):
        ex = "ERROR in class sizes, in correct_classif_rate: len(ground_truth)=%d != len(classified)=%d" % \
             (len(ground_truth), len(classified))
        print(ex)
        raise Exception(ex)

    d1 = numpy.array(ground_truth, dtype="int")
    d2 = numpy.array(classified, dtype="int")
    if verbose:
        print("ground_truth=", d1)
        print("classified=", d2)
    return (d1 == d2).sum() * 1.0 / num


def compute_classification_performance(data_training, correct_classes_training, data_test, correct_classes_test,
                                       size_feature_space, starting_point=None):
    input_dim = data_training.shape[1]

    # Generate functions used for regression
    data_training_mean = data_training.mean(axis=0)
    data_training_std = data_training.std(axis=0)

    data_training_norm = (data_training - data_training_mean) / data_training_std
    data_test_norm = (data_test - data_training_mean) / data_training_std

    c1, l1 = generate_random_sigmoid_weights(input_dim, size_feature_space)
    if starting_point == "Identity":
        print("adding identity coefficients to expansion")
        c1[0:input_dim, 0:input_dim] = numpy.identity(input_dim)
        l1[0:input_dim] = numpy.ones(input_dim) * 1.0  # Code identity
    elif starting_point == "Sigmoids":
        print("adding sigmoid of coefficients to expansion")
        c1[0:input_dim, 0:input_dim] = 8.0 * numpy.identity(input_dim)
        l1[0:input_dim] = numpy.ones(input_dim) * 0.0  # Just set as
    elif starting_point == "08Exp":
        print("adding 08Exp coefficients to expansion")
        c1[0:input_dim, 0:input_dim] = numpy.identity(input_dim)
        c1[0:input_dim, input_dim:2 * input_dim] = numpy.identity(input_dim)

        l1[0:input_dim] = numpy.ones(input_dim) * 1.0  # Code identity
        l1[input_dim:2 * input_dim] = numpy.ones(input_dim) * 0.8  # Code abs(x)**0.8
    else:
        er = "Unknown starting_point", starting_point
        print(er)
        raise Exception(er)
    expanded_data_training = extract_sigmoid_features(data_training_norm, c1, l1, scale=1.0, offset=0.0,
                                                      use_special_features=False)
    expanded_data_test = extract_sigmoid_features(data_test_norm, c1, l1, scale=1.0, offset=0.0,
                                                  use_special_features=False)

    GC_node = mdp.nodes.GaussianClassifier()
    GC_node.train(x=expanded_data_training,
                  labels=correct_classes_training)  # Functions for regression use class values!!!
    GC_node.stop_training()

    estimated_classes_training = GC_node.label(expanded_data_training)
    estimated_classes_test = GC_node.label(expanded_data_test)

    CR_expansion_training = correct_classif_rateC(correct_classes_training, estimated_classes_training)
    CR_expansion_test = correct_classif_rateC(correct_classes_test, estimated_classes_test)

    return CR_expansion_training, CR_expansion_test


def SFANode_reduce_output_dim(sfa_node, new_output_dim, verbose=False):
    """ This function modifies an already trained SFA node (or GSFA node), 
    reducing the number of preserved SFA features to new_output_dim features.
    The modification takes place in place
    """
    if verbose:
        print("Updating the output dimensionality of SFA node")
    if new_output_dim > sfa_node.output_dim:
        er = "Can only reduce output dimensionality of SFA node, not increase it"
        raise Exception(er)
    if verbose:
        print("Before: sfa_node.d.shape=", sfa_node.d.shape, " sfa_node.sf.shape=", sfa_node.sf.shape)
        print(" sfa_node._bias.shape=", sfa_node._bias.shape)
    sfa_node.d = sfa_node.d[:new_output_dim]
    sfa_node.sf = sfa_node.sf[:, :new_output_dim]
    sfa_node._bias = sfa_node._bias[:new_output_dim]
    sfa_node._output_dim = new_output_dim
    if verbose:
        print("After: sfa_node.d.shape=", sfa_node.d.shape, " sfa_node.sf.shape=", sfa_node.sf.shape)
        print(" sfa_node._bias.shape=", sfa_node._bias.shape)


def f_residual(x_app_i, node, y_i):
    """Computes output errors dimension by dimension for a single sample: y - node.execute(x_app)
    
    The library fails when dim(x_app) > dim(y), thus filling of x_app with zeros is recommended.
    """
    # print "f_residual: x_appi_i=", x_app_i, "node.execute=", node.execute, "y_i=", y_i
    res_long = numpy.zeros_like(y_i)
    y_i = y_i.reshape((1, -1))
    y_i_short = y_i[:, 0:node.output_dim]
    #    x_app_i = x_app_i[0:node.input_dim]
    res = (y_i_short - node.execute(x_app_i.reshape((1, -1)))).flatten()
    # print "res_long=", res_long, "y_i=", y_i, "res", res
    res_long[0:len(res)] = res
    #    res = (y_i - node.execute(x_app_i))
    # print "returning resudial res=", res
    return res_long


# input: exp_x_noisy.shape=[1,dim_exp_x],
# outputs: one dim vectors
# Improved version
def invert_exp_funcs2(exp_x_noisy, dim_x, exp_funcs, distance=sfa_libs.distance_best_squared_Euclidean,
                      use_hint=False, max_steady_factor=5, delta_factor=0.7, min_delta=0.0001, k=0.5, verbose=False):
    """ Function that approximates a preimage of exp_x_noisy notice 
    that distance, max_steady_factor, delta, min_delta are deprecated and useless
    """
    num_samples = exp_x_noisy.shape[0]

    if isinstance(use_hint, numpy.ndarray):
        if verbose:
            print("Using suggested approximation!")
        app_x = use_hint.copy()
    elif use_hint:
        if verbose:
            print("Using lowest dim_x=%d elements of input for first approximation!" % (dim_x))
        app_x = exp_x_noisy[:, 0:dim_x].copy()
    else:
        app_x = numpy.random.normal(size=(num_samples, dim_x))

    for row in range(num_samples):
        # app_x_row = app_x[row].reshape(1, dim_x)
        # exp_x_noisy_row = exp_x_noisy[row].reshape(1, dim_exp_x)
        # app_exp_x_row = app_exp_x[row].reshape(1, dim_exp_x)
        # Definition:       scipy.optimize.leastsq(func, x0, args=(), Dfun=None, full_output=0, col_deriv=0,
        #                                         ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=0.0,
        #                                         factor=100, diag=None, warning=True)
        plsq = scipy.optimize.leastsq(residuals, app_x[row], args=(exp_x_noisy[row], exp_funcs, app_x[row], k),
                                      ftol=1.49012e-06, xtol=1.49012e-06, gtol=0.0, maxfev=50*dim_x, epsfcn=0.0,
                                      factor=1.0)
        app_x[row] = plsq[0]

    app_exp_x = sfa_libs.apply_funcs_to_signal(exp_funcs, app_x)
    return app_x, app_exp_x
