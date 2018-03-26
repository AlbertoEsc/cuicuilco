#####################################################################################################################
# igsfa_node: This module implements the Information-Preserving Graph-Based SFA Node                                #
#                                                                                                                   #
# See the following publication for details:                                                                        #
# Escalante-B., A.-N. and Wiskott, L., "Improved graph-based {SFA}: Information preservation complements the        #
# slowness principle", e-print arXiv:1601.03945, http://arxiv.org/abs/1601.03945, 2017                              #
#                                                                                                                   #
# By Alberto Escalante. Alberto.Escalante@ini.ruhr-uni-bochum.de                                        #
# Ruhr-University-Bochum, Institute for Neural Computation, Group of Prof. Dr. Wiskott                              #
#####################################################################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy
import scipy
import scipy.optimize

import mdp
from mdp.utils import pinv

from .sfa_libs import select_rows_from_matrix, distance_squared_Euclidean
from .more_nodes import GeneralExpansionNode
from .gsfa_node import GSFANode


class iGSFANode(mdp.Node):
    """This node implements "information-preserving graph-based SFA (iGSFA)", which is the main component of
    hierarchical iGSFA (HiGSFA). 
    
    For further information, see: Escalante-B., A.-N. and Wiskott, L., "Improved graph-based {SFA}: Information
    preservation complements the slowness principle", e-print arXiv:1601.03945, http://arxiv.org/abs/1601.03945, 2017
    """
    def __init__(self, input_dim=None, output_dim=None, pre_expansion_node_class=None, pre_expansion_out_dim=None,
                 expansion_funcs=None, expansion_output_dim=None, expansion_starting_point=None,
                 max_length_slow_part=None, slow_feature_scaling_method="sensitivity_based", delta_threshold=1.9999,
                 reconstruct_with_sfa=True, verbose=False, **argv):
        """Initializes the iGSFA node.

        pre_expansion_node_class: a node class. An instance of this class is used to filter the data before the
                                  expansion.
        pre_expansion_out_dim: the output dimensionality of the above-mentioned node.
        expansion_funcs: a list of expansion functions to be applied before GSFA.
        expansion_output_dim: this parameter is used to specify an output dimensionality for some expansion functions.
        expansion_starting_point: this parameter is also used by some specific expansion functions.
        max_length_slow_part: fixes an upper bound to the size of the slow part, which is convenient for
                              computational reasons.
        slow_feature_scaling_method: the method used to scale the slow features. Valid entries are: None,
                         "sensitivity_based" (default), "data_dependent", and "QR_decomposition".
        delta_threshold: this parameter has two different meanings depending on its type. If it is real valued (e.g.,
                         1.99), it determines the parameter \Delta_threshold, which is used to decide how many slow
                         features are preserved, depending on their delta values. If it is integer (e.g., 20), it
                         directly specifies the exact size of the slow part.
        reconstruct_with_sfa: this Boolean parameter indicates whether the slow part is removed from the input before
                              PCA is applied.

        More information about parameters 'expansion_funcs' and 'expansion_starting_point' can be found in the
            documentation of GeneralExpansionNode.

        Note: Training is finished after a single call to the train method, unless multi-train is enabled, which
              is done by using reconstruct_with_sfa=False and slow_feature_scaling_method in [None, "data_dependent"]. This
              is necessary to support weight sharing in iGSFA layers (convolutional iGSFA layers).
        """
        super(iGSFANode, self).__init__(input_dim=input_dim, output_dim=output_dim, **argv)
        self.pre_expansion_node_class = pre_expansion_node_class  # Type of node used to expand the data
        self.pre_expansion_node = None  # Node that expands the input data
        self.pre_expansion_output_dim = pre_expansion_out_dim
        self.expansion_output_dim = expansion_output_dim  # Expanded dimensionality
        self.expansion_starting_point = expansion_starting_point  # Initial parameters for the expansion function

        # creates an expansion node
        if expansion_funcs:
            self.exp_node = GeneralExpansionNode(funcs=expansion_funcs, output_dim=self.expansion_output_dim,
                                                 starting_point=self.expansion_starting_point)
        else:
            self.exp_node = None

        self.sfa_node = None
        self.pca_node = None
        self.lr_node = None
        self.max_length_slow_part = max_length_slow_part  # upper limit to the size of the slow part

        # Parameter that defines the size of the slow part. Its meaning depnds on wheather it is an integer or a float
        self.delta_threshold = delta_threshold
        # Indicates whether (nonlinear) SFA components are used for reconstruction
        self.reconstruct_with_sfa = reconstruct_with_sfa
        # Indicates how to scale the slow part
        self.slow_feature_scaling_method = slow_feature_scaling_method

        # Default verbose value if none is explicity provided to the class methods
        self.verbose = verbose

        # Dimensionality of the data after the expansion function
        self.expanded_dim = None

        # The following variables are for internal use only (available after training on a single batch only)
        self.x_mean = None
        self.sfa_x_mean = None
        self.sfa_x_std = None

    @staticmethod
    def is_trainable():
        return True

    # TODO: should train_mode be renamed training_mode?
    def _train(self, x, block_size=None, train_mode=None, node_weights=None, edge_weights=None, verbose=None, **argv):
        """Trains an iGSFA node on data 'x'

        The parameters:  block_size, train_mode, node_weights, and edge_weights are passed to the training function of
        the corresponding gsfa node inside iGSFA (node.gsfa_node).
        """
        self.input_dim = x.shape[1]
        if verbose is None:
           verbose = self.verbose

        if self.output_dim is None:
            self.output_dim = self.input_dim

        if verbose:
            print("Training iGSFANode...")

        if (not self.reconstruct_with_sfa) and (self.slow_feature_scaling_method in [None, "data_dependent"]):
            self.multiple_train(x, block_size=block_size, train_mode=train_mode, node_weights=node_weights,
                                edge_weights=edge_weights)
            return

        if (not self.reconstruct_with_sfa) and (self.slow_feature_scaling_method not in [None, "data_dependent"]):
            er = "'reconstruct_with_sfa' (" + str(self.reconstruct_with_sfa) + ") must be True when the scaling" + \
                 "method (" + str(self.slow_feature_scaling_method) + ") is neither 'None' not 'data_dependent'"
            raise Exception(er)
        # else continue using the regular method:

        # Remove mean before expansion
        self.x_mean = x.mean(axis=0)
        x_zm = x - self.x_mean

        # Reorder or pre-process the data before it is expanded, but only if there is really an expansion
        if self.pre_expansion_node_class and self.exp_node:
            self.pre_expansion_node = self.pre_expansion_node_class(output_dim=self.pre_expansion_output_dim)
            # reasonable options are pre_expansion_node_class = GSFANode or WhitheningNode
            self.pre_expansion_node.train(x_zm, block_size=block_size,
                                          train_mode=train_mode)  # Some arguments might not be necessary
            self.pre_expansion_node.stop_training()
            x_pre_exp = self.pre_expansion_node.execute(x_zm)
        else:
            x_pre_exp = x_zm

        # Expand data
        if self.exp_node:
            if verbose:
                print("expanding x...")
            exp_x = self.exp_node.execute(x_pre_exp)
        else:
            exp_x = x_pre_exp

        self.expanded_dim = exp_x.shape[1]

        if self.max_length_slow_part is None:
            sfa_output_dim = min(self.expanded_dim, self.output_dim)
        else:
            sfa_output_dim = min(self.max_length_slow_part, self.expanded_dim, self.output_dim)

        # Apply SFA to expanded data
        self.sfa_node = GSFANode(output_dim=sfa_output_dim, verbose=verbose)
        #TODO: train_params is only present if patch_mdp has been imported, is this a bug?
        self.sfa_node.train_params(exp_x, params={"block_size": block_size, "train_mode": train_mode,
                                                  "node_weights": node_weights,
                                                  "edge_weights": edge_weights})
        self.sfa_node.stop_training()
        if verbose:
            print("self.sfa_node.d", self.sfa_node.d)

        # Decide how many slow features are preserved (either use Delta_T=delta_threshold when
        # delta_threshold is a float, or preserve delta_threshold features when delta_threshold is an integer)
        if isinstance(self.delta_threshold, float):
            # here self.max_length_slow_part should be considered
            self.num_sfa_features_preserved = (self.sfa_node.d <= self.delta_threshold).sum()
        elif isinstance(self.delta_threshold, int):
            # here self.max_length_slow_part should be considered
            self.num_sfa_features_preserved = self.delta_threshold
        else:
            ex = "Cannot handle type of self.delta_threshold"
            raise Exception(ex)

        if self.num_sfa_features_preserved > self.output_dim:
            self.num_sfa_features_preserved = self.output_dim

        SFANode_reduce_output_dim(self.sfa_node, self.num_sfa_features_preserved)
        if verbose:
            print("sfa execute...")
        sfa_x = self.sfa_node.execute(exp_x)

        # normalize sfa_x
        self.sfa_x_mean = sfa_x.mean(axis=0)
        self.sfa_x_std = sfa_x.std(axis=0)
        if verbose:
            print("self.sfa_x_mean=", self.sfa_x_mean)
            print("self.sfa_x_std=", self.sfa_x_std)
        if (self.sfa_x_std == 0).any():
            er = "zero-component detected"
            raise Exception(er)
        n_sfa_x = (sfa_x - self.sfa_x_mean) / self.sfa_x_std

        if self.reconstruct_with_sfa:
            x_pca = x_zm

            # approximate input linearly, done inline to preserve node for future use
            if verbose:
                print("training linear regression...")
            self.lr_node = mdp.nodes.LinearRegressionNode()
            # Notice that the input "x"=n_sfa_x and the output to learn is "y" = x_pca
            self.lr_node.train(n_sfa_x, x_pca)
            self.lr_node.stop_training()
            x_pca_app = self.lr_node.execute(n_sfa_x)
            x_app = x_pca_app
        else:
            x_app = numpy.zeros_like(x_zm)

        # Remove linear approximation
        sfa_removed_x = x_zm - x_app

        # TODO:Compute variance removed by linear approximation
        if verbose:
            print("ranking method...")
        # AKA Laurenz method for feature scaling( +rotation)
        if self.reconstruct_with_sfa and self.slow_feature_scaling_method == "QR_decomposition":
            M = self.lr_node.beta[1:, :].T  # bias is used by default, we do not need to consider it
            Q, R = numpy.linalg.qr(M)
            self.Q = Q
            self.R = R
            self.Rpinv = pinv(R)
            s_n_sfa_x = numpy.dot(n_sfa_x, R.T)
        # AKA my method for feature scaling (no rotation)
        elif self.reconstruct_with_sfa and (self.slow_feature_scaling_method == "sensitivity_based"):
            beta = self.lr_node.beta[1:, :]  # bias is used by default, we do not need to consider it
            sens = (beta ** 2).sum(axis=1)
            self.magn_n_sfa_x = sens ** 0.5
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x
            if verbose:
                print("method: sensitivity_based enforced")
        elif self.slow_feature_scaling_method is None:
            self.magn_n_sfa_x = 1.0
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x
            if verbose:
                print("method: constant amplitude for all slow features")
        elif self.slow_feature_scaling_method == "data_dependent":
            if verbose:
                print("skiped data_dependent")
        else:
            er = "unknown slow feature scaling method= " + str(self.slow_feature_scaling_method) + \
                 " for reconstruct_with_sfa= " + str(self.reconstruct_with_sfa)
            raise Exception(er)

        print("training PCA...")
        pca_output_dim = self.output_dim - self.num_sfa_features_preserved
        # This allows training of PCA when pca_out_dim is zero
        self.pca_node = mdp.nodes.PCANode(output_dim=max(1, pca_output_dim))  # reduce=True
        self.pca_node.train(sfa_removed_x)
        self.pca_node.stop_training()
        PCANode_reduce_output_dim(self.pca_node, pca_output_dim, verbose=False)

        # TODO:check that pca_out_dim > 0
        if verbose:
            print("executing PCA...")

        pca_x = self.pca_node.execute(sfa_removed_x)

        if self.slow_feature_scaling_method == "data_dependent":
            if pca_output_dim > 0:
               self.magn_n_sfa_x = 1.0 * numpy.median(self.pca_node.d) ** 0.5    # WARNING, why did I have 5.0 there? it is supposed to be 1.0
            else:
               self.magn_n_sfa_x = 1.0
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x  # Scale according to ranking
            if verbose:
                print("method: data dependent")

        if self.pca_node.output_dim + self.num_sfa_features_preserved < self.output_dim:
            er = "Error, the number of features computed is SMALLER than the output dimensionality of the node: " + \
                 "self.pca_node.output_dim=" + str(self.pca_node.output_dim) + ", self.num_sfa_features_preserved=" + \
                 str(self.num_sfa_features_preserved) + ", self.output_dim=" + str(self.output_dim)
            raise Exception(er)

        # Finally, the output is the concatenation of scaled slow features and remaining pca components
        sfa_pca_x = numpy.concatenate((s_n_sfa_x, pca_x), axis=1)

        sfa_pca_x_truncated = sfa_pca_x[:, 0:self.output_dim]

        # Compute explained variance from amplitudes of output compared to amplitudes of input
        # Only works because amplitudes of SFA are scaled to be equal to explained variance, because PCA is
        # a rotation, and because data has zero mean
        self.evar = (sfa_pca_x_truncated ** 2).sum() / (x_zm ** 2).sum()
        if verbose:
            print("s_n_sfa_x:", s_n_sfa_x, "pca_x:", pca_x)
            print("sfa_pca_x_truncated:", sfa_pca_x_truncated, "x_zm:", x_zm)
            print("Variance(output) / Variance(input) is ", self.evar)
        self.stop_training()

    def multiple_train(self, x, block_size=None, train_mode=None, node_weights=None,
                       edge_weights=None, verbose=None):  # scheduler = None, n_parallel=None
        """This function should not be called directly. Use instead the train method, which will decide whether
        multiple-training is enabled, and call this function if needed. """
        # TODO: is the following line needed? or also self.set_input_dim? or self._input_dim?
        self.input_dim = x.shape[1]
        if verbose is None:
           verbose = self.verbose

        if verbose:
            print("Training iGSFANode (multiple train method)...")

        # Data mean is ignored by the multiple train method
        if self.x_mean is None:
            self.x_mean = numpy.zeros(self.input_dim)
        x_zm = x

        # Reorder or pre-process the data before it is expanded, but only if there is really an expansion.
        # WARNING, why the last condition???
        if self.pre_expansion_node_class and self.exp_node:
            er = "Unexpected parameters"
            raise Exception(er)
        else:
            x_pre_exp = x_zm

        if self.exp_node:
            if verbose:
                print("expanding x...")
            exp_x = self.exp_node.execute(x_pre_exp)  # x_zm
        else:
            exp_x = x_pre_exp

        self.expanded_dim = exp_x.shape[1]

        if self.max_length_slow_part is None:
            sfa_output_dim = min(self.expanded_dim, self.output_dim)
        else:
            sfa_output_dim = min(self.max_length_slow_part, self.expanded_dim, self.output_dim)

        # Apply SFA to expanded data
        if self.sfa_node is None:
            self.sfa_node = GSFANode(output_dim=sfa_output_dim, verbose=verbose)
        self.sfa_x_mean = 0
        self.sfa_x_std = 1.0

        self.sfa_node.train_params(exp_x, params={"block_size": block_size, "train_mode": train_mode,
                                                  "node_weights": node_weights,
                                                  "edge_weights": edge_weights})

        if verbose:
            print("training PCA...")
        pca_output_dim = self.output_dim
        if self.pca_node is None:
            # WARNING: WHY WAS I EXTRACTING ALL PCA COMPONENTS!!?? INEFFICIENT!!!!
            self.pca_node = mdp.nodes.PCANode(output_dim=pca_output_dim)  # reduce=True) #output_dim = pca_out_dim)
        sfa_removed_x = x
        self.pca_node.train(sfa_removed_x)

    def _stop_training(self, verbose=None):
        if verbose is None:
           verbose = self.verbose
        if self.reconstruct_with_sfa or (self.slow_feature_scaling_method not in [None, "data_dependent"]):
            return
        # else, continue with multi-train method

        self.sfa_node.stop_training()
        if verbose:
            print("self.sfa_node.d", self.sfa_node.d)
        self.pca_node.stop_training()

        # Decide how many slow features are preserved (either use Delta_T=delta_threshold when
        # delta_threshold is a float, or preserve delta_threshold features when delta_threshold is an integer)
        if isinstance(self.delta_threshold, float):
            # here self.max_length_slow_part should be considered
            self.num_sfa_features_preserved = (self.sfa_node.d <= self.delta_threshold).sum()
        elif isinstance(self.delta_threshold, int):
            # here self.max_length_slow_part should be considered
            self.num_sfa_features_preserved = self.delta_threshold
        else:
            ex = "Cannot handle type of self.delta_threshold:"+str(type(self.delta_threshold))
            raise Exception(ex)

        if self.num_sfa_features_preserved > self.output_dim:
            self.num_sfa_features_preserved = self.output_dim

        SFANode_reduce_output_dim(self.sfa_node, self.num_sfa_features_preserved)
        if verbose:
            print ("size of slow part:", self.num_sfa_features_preserved)

        final_pca_node_output_dim = self.output_dim - self.num_sfa_features_preserved
        if final_pca_node_output_dim > self.pca_node.output_dim:
            er = "Error, the number of features computed is SMALLER than the output dimensionality of the node: " + \
                 "self.pca_node.output_dim=" + str(self.pca_node.output_dim) + ", self.num_sfa_features_preserved=" + \
                 str(self.num_sfa_features_preserved) + ", self.output_dim=" + str(self.output_dim)
            raise Exception(er)
        PCANode_reduce_output_dim(self.pca_node, final_pca_node_output_dim, verbose=False)

        if verbose:
            print("self.pca_node.d", self.pca_node.d)
            print("ranking method...")
        if self.slow_feature_scaling_method is None:
            self.magn_n_sfa_x = 1.0
            if verbose:
                print("method: constant amplitude for all slow features")
        elif self.slow_feature_scaling_method == "data_dependent":
            # SFA components have an std equal to that of the least significant principal component
            if self.pca_node.d.shape[0] > 0:
               self.magn_n_sfa_x = 1.0 * numpy.median(self.pca_node.d) ** 0.5
               # 100.0 * self.pca_node.d[-1] ** 0.5 + 0.0 # Experiment: use 5.0 instead of 1.0
            else:
               self.magn_n_sfa_x = 1.0
            if verbose:
                print("method: data dependent")
        else:
            er = "Unknown slow feature scaling method" + str(self.slow_feature_scaling_method)
            raise Exception(er)
        self.evar = self.pca_node.explained_variance

    @staticmethod
    def _is_invertible():
        return True

    def _execute(self, x):
        """Extracts iGSFA features from some data. The node must have been already trained. """
        x_zm = x - self.x_mean

        if self.pre_expansion_node:
            x_pre_exp = self.pre_expansion_node.execute(x_zm)
        else:
            x_pre_exp = x_zm

        if self.exp_node:
            exp_x = self.exp_node.execute(x_pre_exp)
        else:
            exp_x = x_pre_exp

        sfa_x = self.sfa_node.execute(exp_x)

        n_sfa_x = (sfa_x - self.sfa_x_mean) / self.sfa_x_std

        if self.reconstruct_with_sfa:
            # approximate input linearly, done inline to preserve node
            x_pca_app = self.lr_node.execute(n_sfa_x)
            x_app = x_pca_app
        else:
            x_app = numpy.zeros_like(x_zm)

        # Remove linear approximation
        sfa_removed_x = x_zm - x_app

        # AKA Laurenz method for feature scaling( +rotation)
        if self.reconstruct_with_sfa and self.slow_feature_scaling_method == "QR_decomposition":
            s_n_sfa_x = numpy.dot(n_sfa_x, self.R.T)
        # AKA my method for feature scaling (no rotation)
        elif self.reconstruct_with_sfa and self.slow_feature_scaling_method == "sensitivity_based":
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x 
        elif self.slow_feature_scaling_method is None:
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x 
        # Scale according to ranking
        elif self.slow_feature_scaling_method == "data_dependent":
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x 
        else:
            er = "unknown feature scaling method" + str(self.slow_feature_scaling_method)
            raise Exception(er)

        # Apply PCA to sfa removed data
        if self.pca_node.output_dim > 0:
            pca_x = self.pca_node.execute(sfa_removed_x)
        else:
            # No reconstructive components present
            pca_x = numpy.zeros((x.shape[0], 0))

        # Finally output is the concatenation of scaled slow features and remaining pca components
        sfa_pca_x = numpy.concatenate((s_n_sfa_x, pca_x), axis=1)

        return sfa_pca_x  # sfa_pca_x_truncated

    def _inverse(self, y, linear_inverse=True):
        """This method approximates an inverse function to the feature extraction.

        if linear_inverse is True, a linear method is used. Otherwise, a gradient-based non-linear method is used.
        """
        if linear_inverse:
            return self.linear_inverse(y)
        else:
            return self.non_linear_inverse(y)

    def non_linear_inverse(self, y, verbose=None):
        """Non-linear inverse approximation method. """
        if verbose is None:
           verbose = self.verbose
        x_lin = self.linear_inverse(y)
        rmse_lin = ((y - self.execute(x_lin)) ** 2).sum(axis=1).mean() ** 0.5
        # scipy.optimize.leastsq(func, x0, args=(), Dfun=None, full_output=0, col_deriv=0, ftol=1.49012e-08,
        # xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=0.0, factor=100, diag=None)
        x_nl = numpy.zeros_like(x_lin)
        y_dim = y.shape[1]
        x_dim = x_lin.shape[1]
        if y_dim < x_dim:
            num_zeros_filling = x_dim - y_dim
        else:
            num_zeros_filling = 0
        if verbose:
            print("x_dim=", x_dim, "y_dim=", y_dim, "num_zeros_filling=", num_zeros_filling)
        y_long = numpy.zeros(y_dim + num_zeros_filling)

        for i, y_i in enumerate(y):
            y_long[0:y_dim] = y_i
            if verbose:
                print("x_0=", x_lin[i])
                print("y_long=", y_long)
            plsq = scipy.optimize.leastsq(func=f_residual, x0=x_lin[i], args=(self, y_long), full_output=False)
            x_nl_i = plsq[0]
            if verbose:
                print("x_nl_i=", x_nl_i, "plsq[1]=", plsq[1])
            if plsq[1] != 2:
                print("Quitting: plsq[1]=", plsq[1])
                # quit()
            x_nl[i] = x_nl_i
            if verbose:
                print("|E_lin(%d)|=" % i, ((y_i - self.execute(x_lin[i].reshape((1, -1)))) ** 2).sum() ** 0.5)
                print("|E_nl(%d)|=" % i, ((y_i - self.execute(x_nl_i.reshape((1, -1)))) ** 2).sum() ** 0.5)
        rmse_nl = ((y - self.execute(x_nl)) ** 2).sum(axis=1).mean() ** 0.5
        if verbose:
            print("rmse_lin(all samples)=", rmse_lin, "rmse_nl(all samples)=", rmse_nl)
        return x_nl

    def linear_inverse(self, y, verbose=None):
        """Linear inverse approximation method. """
        if verbose is None:
           verbose = self.verbose
        num_samples = y.shape[0]
        if y.shape[1] != self.output_dim:
            er = "Serious dimensionality inconsistency:", y.shape[0], self.output_dim
            raise Exception(er)

        sfa_pca_x_full = numpy.zeros(
            (num_samples, self.pca_node.output_dim + self.num_sfa_features_preserved))  # self.input_dim
        sfa_pca_x_full[:, 0:self.output_dim] = y

        s_n_sfa_x = sfa_pca_x_full[:, 0:self.num_sfa_features_preserved]
        pca_x = sfa_pca_x_full[:, self.num_sfa_features_preserved:]

        if pca_x.shape[1] > 0:
            sfa_removed_x = self.pca_node.inverse(pca_x)
        else:
            sfa_removed_x = numpy.zeros((num_samples, self.input_dim))

        # AKA Laurenz method for feature scaling (+rotation)
        if self.reconstruct_with_sfa and self.slow_feature_scaling_method == "QR_decomposition":
            n_sfa_x = numpy.dot(s_n_sfa_x, self.Rpinv.T)
        else:
            n_sfa_x = s_n_sfa_x / self.magn_n_sfa_x

        # sfa_x = n_sfa_x * self.sfa_x_std + self.sfa_x_mean
        if self.reconstruct_with_sfa:
            x_pca_app = self.lr_node.execute(n_sfa_x)
            x_app = x_pca_app
        else:
            x_app = numpy.zeros_like(sfa_removed_x)

        x_zm = sfa_removed_x + x_app

        x = x_zm + self.x_mean

        if verbose:
            print("Data_variance(x_zm)=", data_variance(x_zm))
            print("Data_variance(x_app)=", data_variance(x_app))
            print("Data_variance(sfa_removed_x)=", data_variance(sfa_removed_x))
            print("x_app.mean(axis=0)=", x_app)
            print("x[0]=",x[0])
            print("zm_x[0]=", zm_x[0])
            print("exp_x[0]=", exp_x[0])
            print("s_x_1[0]=", s_x_1[0])
            print("proj_sfa_x[0]=", proj_sfa_x[0])
            print("sfa_removed_x[0]=", sfa_removed_x[0])
            print("pca_x[0]=", pca_x[0])
            print("n_pca_x[0]=", n_pca_x[0])
            print("sfa_x[0]=", sfa_x[0])

        return x


def SFANode_reduce_output_dim(sfa_node, new_output_dim, verbose=False):
    """ This function modifies an already trained SFA node (or GSFA node), 
    reducing the number of preserved SFA features to new_output_dim features.
    The modification is done in place
    """
    if verbose:
        print("Updating the output dimensionality of SFA node")
    if new_output_dim > sfa_node.output_dim:
        er = "Can only reduce output dimensionality of SFA node, not increase it (%d > %d)" % \
             (new_output_dim, sfa_node.output_dim)
        raise Exception(er)
    if verbose:
        print("Before: sfa_node.d.shape=", sfa_node.d.shape, " sfa_node.sf.shape=", sfa_node.sf.shape)
        print("sfa_node._bias.shape=", sfa_node._bias.shape)
    sfa_node.d = sfa_node.d[:new_output_dim]
    sfa_node.sf = sfa_node.sf[:, :new_output_dim]
    sfa_node._bias = sfa_node._bias[:new_output_dim]
    sfa_node._output_dim = new_output_dim
    if verbose:
        print("After: sfa_node.d.shape=", sfa_node.d.shape, " sfa_node.sf.shape=", sfa_node.sf.shape)
        print(" sfa_node._bias.shape=", sfa_node._bias.shape)


def PCANode_reduce_output_dim(pca_node, new_output_dim, verbose=False):
    """ This function modifies an already trained PCA node, 
    reducing the number of preserved SFA features to new_output_dim features.
    The modification is done in place. Also the explained variance field is updated
    """
    if verbose:
        print("Updating the output dimensionality of PCA node")
    if new_output_dim > pca_node.output_dim:
        er = "Can only reduce output dimensionality of PCA node, not increase it"
        raise Exception(er)
    if verbose:
        print("Before: pca_node.d.shape=", pca_node.d.shape, " pca_node.v.shape=", pca_node.v.shape)
        print(" pca_node.avg.shape=", pca_node.avg.shape)

    # if new_output_dim > 0:
    original_total_variance = pca_node.d.sum()
    original_explained_variance = pca_node.explained_variance
    pca_node.d = pca_node.d[0:new_output_dim]
    pca_node.v = pca_node.v[:, 0:new_output_dim]
    # pca_node.avg is not affected by this method!
    pca_node._output_dim = new_output_dim
    pca_node.explained_variance = original_explained_variance * pca_node.d.sum() / original_total_variance
    # else:

    if verbose:
        print("After: pca_node.d.shape=", pca_node.d.shape, " pca_node.v.shape=", pca_node.v.shape)
        print(" pca_node.avg.shape=", pca_node.avg.shape)


# Computes output errors dimension by dimension for a single sample: y - node.execute(x_app)
# The library fails when dim(x_app) > dim(y), thus filling of x_app with zeros is recommended
def f_residual(x_app_i, node, y_i):
    res_long = numpy.zeros_like(y_i)
    y_i = y_i.reshape((1, -1))
    y_i_short = y_i[:, 0:node.output_dim]
    res = (y_i_short - node.execute(x_app_i.reshape((1, -1)))).flatten()
    res_long[0:len(res)] = res
    return res_long


########################################################################################
#   AN EXAMPLE OF HOW iGSFA CAN BE USED                                                #
########################################################################################

def example_iGSFA():
    print("\n\n**************************************************************************")
    print("*Example of training iGSFA on random data")
    num_samples = 1000
    dim = 20
    verbose = False
    x = numpy.random.normal(size=(num_samples, dim))
    x[:,0] += 2.0 * numpy.arange(num_samples) / num_samples
    x[:,1] += 1.0 * numpy.arange(num_samples) / num_samples
    x[:,2] += 0.5 * numpy.arange(num_samples) / num_samples

    x_test = numpy.random.normal(size=(num_samples, dim))
    x_test[:,0] += 2.0 * numpy.arange(num_samples) / num_samples
    x_test[:,1] += 1.0 * numpy.arange(num_samples) / num_samples
    x_test[:,2] += 0.5 * numpy.arange(num_samples) / num_samples

    import cuicuilco.patch_mdp
    from cuicuilco.gsfa_node import comp_delta
    from cuicuilco.sfa_libs import zero_mean_unit_var
    print("Node creation and training")
    n = iGSFANode(output_dim=15, reconstruct_with_sfa=False, slow_feature_scaling_method="data_dependent",
                  verbose=verbose)
    n.train(x, train_mode="regular")
    n.stop_training()

    y = n.execute(x)
    y_test = n.execute(x_test)

    print("y=", y)
    print("y_test=", y_test)
    print("Standard delta values of output features y:", comp_delta(y))
    print("Standard delta values of output features y_test:", comp_delta(y_test))
    y_norm = zero_mean_unit_var(y)
    y_test_norm = zero_mean_unit_var(y_test)
    print("Standard delta values of output features y after constraint enforcement:", comp_delta(y_norm))
    print("Standard delta values of output features y_test after constraint enforcement:", comp_delta(y_test_norm))


if __name__ == "__main__":
    example_iGSFA()
