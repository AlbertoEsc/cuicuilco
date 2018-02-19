#####################################################################################################################
# gsfa_node: This module implements the Graph-Based SFA Node and is part of the Cuicuilco framework                 #
#                                                                                                                   #
# See the following publication for details:                                                                        #
# Escalante-B A.-N., Wiskott L, "How to solve classification and regression problems on high-dimensional data with   #
# a supervised extension of Slow Feature Analysis". Journal of Machine Learning Research 14:3683-3719, 2013         #
#                                                                                                                   #
# An example of using GSFA is provided at the end of the file                                                       #
#                                                                                                                   #
# By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de                                        #
# Ruhr-University-Bochum, Institute for Neural Computation, Group of Prof. Dr. Wiskott                              #
#####################################################################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy
import scipy
import scipy.optimize
import sys

import mdp
from mdp.utils import (mult, pinv, CovarianceMatrix, SymeigException)  # , symeig


# This class is derived from SFANode only to reuse the function _set_range
class GSFANode(mdp.nodes.SFANode):
    """ This node implements "Graph-Based SFA (GSFA)", which is the main component of hierarchical GSFA (HGSFA).

    For further information, see: Escalante-B A.-N., Wiskott L, "How to solve classication and regression
    problems on high-dimensional data with a supervised extension of Slow Feature Analysis". Journal of Machine
    Learning Research 14:3683-3719, 2013
    """
    def __init__(self, input_dim=None, output_dim=None, dtype=None, block_size=None, train_mode=None):
        """Initializes the GSFA node, which is a subclass of the SFA node.

        The parameters block_size and train_mode are not necessary and it is better to provide them during training.
        See the _train method for details.
        """
        super(GSFANode, self).__init__(input_dim, output_dim, dtype)
        self.pinv = None
        self.block_size = block_size
        self.train_mode = train_mode

        self._covdcovmtx = CovDCovMatrix(block_size=block_size)  # why block_size and not input_dim???
        # Parameters accepted during training
        self.list_train_params = ["scheduler", "n_parallel", "train_mode", "block_size"]

    def _train_without_scheduler(self, x, block_size=None, train_mode=None, node_weights=None, edge_weights=None):
        """ This is the main training function of GSFA. It is called internally by _train.
        
        x: training data (each sample is a row)
        the usage of the parameters depends on the training mode (train_mode)
        to train as in standard SFA:
            set train_mode="regular" (the scale of the features should then be corrected)
        to train using the clustered graph:
            set train_mode="clustered". The cluster size is given by block_size (integer). 
               Variable cluster sizes are possible if block_size is a list of integers.
        to train for classification:
            set train_mode=("classification", labels, weight), where labels is an array with the class information and
            weight is a scalar value.
        to train for regression:
            set train_mode=("serial_regression#", labels, weight), where # is the block size used by a serial graph,
            labels is an array with the label information and weight is a scalar value.
        to train using a graph without edges:
            set train_mode="unlabeled".
        to train using the serial graph:
            set train_mode="serial", and use block_size (integer) to specify the group size. 
        to train using the mixed graph:
            set train_mode="mixed", and use block_size (integer) to specify the group size.           
        to train using an arbitrary graph:
            set train_mode="graph", specify the node_weights (numpy 1D array), and the
            edge_weights (numpy 2D array).
        """
        if train_mode is None:
            train_mode = self.train_mode
        if block_size is None:
            print("parameter block_size was not provided, using default value self.block_size")
            block_size = self.block_size

        self.set_input_dim(x.shape[1])

        print("train_mode=", train_mode)

        if isinstance(train_mode, list):
            train_modes = train_mode
        else:
            train_modes = [train_mode]

        for train_mode in train_modes:
            if isinstance(train_mode, tuple):
                method = train_mode[0]
                labels = train_mode[1]
                weight = train_mode[2]
                if method == "classification":
                    print("update classification")
                    ordering = numpy.argsort(labels)
                    x2 = x[ordering, :]
                    unique_labels = numpy.unique(labels)
                    unique_labels.sort()
                    block_sizes = []
                    for label in unique_labels:
                        block_sizes.append((labels == label).sum())
                    self._covdcovmtx.update_clustered(x2, block_sizes=block_sizes, weight=weight)
                elif method.startswith("serial_regression"):
                    block_size = int(method[len("serial_regression"):])
                    print("update serial_regression, block_size=", block_size)
                    ordering = numpy.argsort(labels)
                    x2 = x[ordering, :]
                    self._covdcovmtx.update_serial(x2, block_size=block_size, weight=weight)
                else:
                    er = "method unknown: %s" % (str(method))
                    raise Exception(er)
            else:
                if train_mode == 'unlabeled':
                    print("update_unlabeled")
                    self._covdcovmtx.update_unlabeled(x, weight=0.00015)  # Warning, set this weight appropriately!
                elif train_mode == "regular":
                    print("update_regular")
                    self._covdcovmtx.update_regular(x, weight=1.0)
                elif train_mode == 'clustered':
                    print("update_clustered")
                    self._covdcovmtx.update_clustered(x, block_sizes=block_size, weight=1.0)
                elif train_mode.startswith('compact_classes'):
                    print("update_compact_classes:", train_mode)
                    J = int(train_mode[len('compact_classes'):])
                    self._covdcovmtx.update_compact_classes(x, block_sizes=block_size, Jdes=J, weight=1.0)
                elif train_mode == 'serial':
                    print("update_serial")
                    self._covdcovmtx.update_serial(x, block_size=block_size)
                elif train_mode.startswith('DualSerial'):
                    print("updateDualSerial")
                    num_blocks = len(x) // block_size
                    dual_num_blocks = int(train_mode[len("DualSerial"):])
                    dual_block_size = len(x) // dual_num_blocks
                    chunk_size = block_size // dual_num_blocks
                    print("dual_num_blocks = ", dual_num_blocks)
                    self._covdcovmtx.update_serial(x, block_size=block_size)
                    x2 = numpy.zeros_like(x)
                    for i in range(num_blocks):
                        for j in range(dual_num_blocks):
                            x2[j * dual_block_size + i * chunk_size:j * dual_block_size + (i + 1) * chunk_size] = \
                                x[i * block_size + j * chunk_size:i * block_size + (j + 1) * chunk_size]
                    self._covdcovmtx.update_serial(x2, block_size=dual_block_size, weight=0.0)
                elif train_mode == 'mixed':
                    print("update mixed")
                    bs = block_size
                    self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[0:bs], weight=2.0,
                                                                              block_size=block_size)
                    self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[bs:-bs], weight=1.0,
                                                                              block_size=block_size)
                    self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[-bs:], weight=2.0,
                                                                              block_size=block_size)
                    self._covdcovmtx.update_serial(x, block_size=block_size)
                elif train_mode[0:6] == 'window':
                    window_halfwidth = int(train_mode[6:])
                    print("Window (%d)" % window_halfwidth)
                    self._covdcovmtx.update__sliding_window(x, weight=1.0, window_halfwidth=window_halfwidth)
                elif train_mode[0:7] == 'fwindow':
                    window_halfwidth = int(train_mode[7:])
                    print("Fast Window (%d)" % window_halfwidth)
                    self._covdcovmtx.update_fast_sliding_window(x, weight=1.0, window_halfwidth=window_halfwidth)
                elif train_mode[0:13] == 'mirror_window':
                    window_halfwidth = int(train_mode[13:])
                    print("Mirroring Window (%d)" % window_halfwidth)
                    self._covdcovmtx.update_mirroring_sliding_window(x, weight=1.0, window_halfwidth=window_halfwidth)
                elif train_mode[0:14] == 'smirror_window':
                    window_halfwidth = int(train_mode[14:])
                    print("Slow Mirroring Window (%d)" % window_halfwidth)
                    self._covdcovmtx.update_slow_mirroring_sliding_window(x, weight=1.0, window_halfwidth=window_halfwidth)
                elif train_mode == 'graph':
                    print("update_graph")
                    self._covdcovmtx.update_graph(x, node_weights=node_weights, edge_weights=edge_weights, weight=1.0)
                elif train_mode == 'graph_old':
                    print("update_graph_old")
                    self._covdcovmtx.update_graph_old(x, node_weights=node_weights, edge_weights=edge_weights, weight=1.0)
                elif train_mode == 'smart_unlabeled2':
                    print("smart_unlabeled2")
                    N2 = x.shape[0]

                    N1 = Q1 = self._covdcovmtx.num_samples * 1.0
                    R1 = self._covdcovmtx.num_diffs * 1.0
                    sum_x_labeled_2D = self._covdcovmtx.sum_x.reshape((1, -1)) + 0.0
                    sum_prod_x_labeled = self._covdcovmtx.sum_prod_x + 0.0
                    print("Original sum_x[0]/num_samples=", self._covdcovmtx.sum_x[0] / self._covdcovmtx.num_samples)

                    weight_fraction_unlabeled = 0.2  # 0.1, 0.25
                    additional_weight_unlabeled = -0.025  # 0.02 0.25, 0.65?

                    w1 = Q1 * 1.0 / R1 * (1.0 - weight_fraction_unlabeled)
                    print("weight_fraction_unlabeled=", weight_fraction_unlabeled)
                    print("N1=Q1=", Q1, "R1=", R1, "w1=", w1)
                    print("")

                    self._covdcovmtx.sum_prod_diffs *= w1
                    self._covdcovmtx.num_diffs *= w1
                    print("After diff scaling: num_samples=", self._covdcovmtx.num_samples)
                    print("num_diffs=", self._covdcovmtx.num_diffs, "\n")

                    node_weights2 = Q1 * weight_fraction_unlabeled / N2  # w2*N1
                    w12 = node_weights2 / N1  # One directional weights
                    print("w12 (one dir)", w12)

                    sum_x_unlabeled_2D = x.sum(axis=0).reshape((1, -1))
                    sum_prod_x_unlabeled = mdp.utils.mult(x.T, x)

                    self._covdcovmtx.add_samples(sum_prod_x_unlabeled, sum_x_unlabeled_2D.flatten(), num_samples=N2,
                                                weight=node_weights2)
                    print("After adding unlabeled nodes: num_samples=", self._covdcovmtx.num_samples)
                    print("num_diffs=", self._covdcovmtx.num_diffs)
                    print("sum_x[0]/num_samples=", self._covdcovmtx.sum_x[0] / self._covdcovmtx.num_samples)
                    print("")

                    print("N2=", N2, "node_weights2=", node_weights2)

                    additional_diffs = sum_prod_x_unlabeled * N1 - \
                                       mdp.utils.mult(sum_x_labeled_2D.T, sum_x_unlabeled_2D) - \
                                       mdp.utils.mult(sum_x_unlabeled_2D.T, sum_x_labeled_2D) + sum_prod_x_labeled * N2
                    print("w12=", w12, "additional_diffs=", additional_diffs)
                    self._covdcovmtx.add_diffs(2 * additional_diffs, 2 * N1 * N2,
                                              weight=w12)  # to account for both directions
                    print("After mixed diff addition: num_samples=", self._covdcovmtx.num_samples)
                    print("num_diffs=", self._covdcovmtx.num_diffs)
                    print("sum_x[0]/num_samples=", self._covdcovmtx.sum_x[0] / self._covdcovmtx.num_samples)

                    print("\n Adding complete graph for unlabeled data")
                    self._covdcovmtx.update_clustered_homogeneous_block_sizes(x, weight=additional_weight_unlabeled,
                                                                              block_size=N2)
                    print("After complete x2 addition: num_samples=", self._covdcovmtx.num_samples)
                    print("num_diffs=", self._covdcovmtx.num_diffs)
                    print("sum_x[0]/num_samples=", self._covdcovmtx.sum_x[0] / self._covdcovmtx.num_samples)

                elif train_mode == 'smart_unlabeled3':
                    print("smart_unlabeled3")
                    N2 = x.shape[0]

                    N1 = Q1 = self._covdcovmtx.num_samples * 1.0
                    R1 = self._covdcovmtx.num_diffs * 1.0
                    print("N1=Q1=", Q1, "R1=", R1, "N2=", N2)

                    v = 2.0 ** (-9.5)  # 500.0/4500 #weight of unlabeled samples (making it "500" vs "500")
                    C = 10.0  # 10.0 #Clustered graph assumed, with C classes, and each one having N1/C samples
                    print("v=", v, "C=", C)

                    v_norm = v / C
                    N1_norm = N1 / C

                    sum_x_labeled = self._covdcovmtx.sum_x.reshape((1, -1)) + 0.0
                    sum_prod_x_labeled = self._covdcovmtx.sum_prod_x + 0.0

                    print("Original (Diag(C')/num_diffs.avg)**0.5 =", ((numpy.diagonal(
                        self._covdcovmtx.sum_prod_diffs) / self._covdcovmtx.num_diffs).mean()) ** 0.5)

                    weight_adjustment = (N1_norm - 1) / (N1_norm - 1 + v_norm * N2)
                    print("weight_adjustment =", weight_adjustment, "w11=", 1 / (N1_norm - 1 + v_norm * N2))
                    # w1 = Q1*1.0/R1 * (1.0-weight_fraction_unlabeled)

                    self._covdcovmtx.sum_x *= weight_adjustment
                    self._covdcovmtx.sum_prod_x *= weight_adjustment
                    self._covdcovmtx.num_samples *= weight_adjustment
                    self._covdcovmtx.sum_prod_diffs *= weight_adjustment
                    self._covdcovmtx.num_diffs *= weight_adjustment
                    node_weights_complete_1 = weight_adjustment
                    print("num_diffs (w11) after weight_adjustment=", self._covdcovmtx.num_diffs)
                    w11 = 1 / (N1_norm - 1 + v_norm * N2)
                    print("After adjustment (Diag(C')/num_diffs.avg)**0.5 =", ((numpy.diagonal(
                        self._covdcovmtx.sum_prod_diffs) / self._covdcovmtx.num_diffs).mean()) ** 0.5)
                    print("")

                    # ##Connections within unlabeled data (notice that C times this is equivalent to
                    # v*v/(N1+v*(N2-1)) once)
                    w22 = 0.5 * 2 * v_norm * v_norm / (N1_norm + v_norm * (N2 - 1))
                    sum_x_unlabeled = x.sum(axis=0).reshape((1, -1))
                    sum_prod_x_unlabeled = mdp.utils.mult(x.T, x)
                    node_weights_complete_2 = w22 * (N2 - 1) * C
                    self._covdcovmtx.update_clustered_homogeneous_block_sizes(x, weight=node_weights_complete_2,
                                                                              block_size=N2)
                    print("w22=", w22, "node_weights_complete_2*N2=", node_weights_complete_2 * N2)
                    print("After adding complete 2: num_samples=", self._covdcovmtx.num_samples)
                    print("num_diffs=", self._covdcovmtx.num_diffs)
                    print(" (Diag(C')/num_diffs.avg)**0.5 =", ((numpy.diagonal(
                        self._covdcovmtx.sum_prod_diffs) / self._covdcovmtx.num_diffs).mean()) ** 0.5)
                    print("")

                    # Connections between labeled and unlabeled samples
                    w12 = 2 * 0.5 * v_norm * (1 / (N1_norm - 1 + v_norm * N2) + 1 / (
                        N1_norm + v_norm * (N2 - 1)))  # Accounts for transitions in both directions
                    print("(twice) w12=", w12)
                    sum_prod_diffs_mixed = w12 * (N1 * sum_prod_x_unlabeled -
                                                  (mdp.utils.mult(sum_x_labeled.T, sum_x_unlabeled) +
                                                   mdp.utils.mult(sum_x_unlabeled.T, sum_x_labeled)) +
                                                  N2 * sum_prod_x_labeled)
                    self._covdcovmtx.sum_prod_diffs += sum_prod_diffs_mixed
                    self._covdcovmtx.num_diffs += C * N1_norm * N2 * w12  # w12 already counts twice
                    print(" (Diag(mixed)/num_diffs.avg)**0.5 =", ((numpy.diagonal(sum_prod_diffs_mixed) /
                                                                   (C * N1_norm * N2 * w12)).mean()) ** 0.5, "\n")

                    # Additional adjustment for node weights of unlabeled data
                    missing_weight_unlabeled = v - node_weights_complete_2
                    missing_weight_labeled = 1.0 - node_weights_complete_1
                    print("missing_weight_unlabeled=", missing_weight_unlabeled)
                    print("Before two final add_samples: num_samples=", self._covdcovmtx.num_samples)
                    print("num_diffs=", self._covdcovmtx.num_diffs)
                    self._covdcovmtx.add_samples(sum_prod_x_unlabeled, sum_x_unlabeled, N2, missing_weight_unlabeled)
                    self._covdcovmtx.add_samples(sum_prod_x_labeled, sum_x_labeled, N1, missing_weight_labeled)
                    print("Final transformation: num_samples=", self._covdcovmtx.num_samples)
                    print("num_diffs=", self._covdcovmtx.num_diffs)
                    print("Summary v11=%f+%f, v22=%f+%f" % (weight_adjustment, missing_weight_labeled,
                                                            node_weights_complete_2, missing_weight_unlabeled))
                    print("Summary w11=%f, w22=%f, w12(two ways)=%f" % (w11, w22, w12))
                    print("Summary (N1/C-1)*w11=%f, N2*w12 (one way)=%f" % ((N1 / C - 1) * w11, N2 * w12 / 2))
                    print("Summary (N2-1)*w22*C=%f, N1*w12 (one way)=%f" % ((N2 - 1) * w22 * C, N1 * w12 / 2))
                    print("Summary (Diag(C')/num_diffs.avg)**0.5 =", ((numpy.diagonal(
                        self._covdcovmtx.sum_prod_diffs) / self._covdcovmtx.num_diffs).mean()) ** 0.5)
                elif train_mode == 'ignore_data':
                    print("Training graph: ignoring data")
                else:
                    ex = "Unknown training method"
                    raise Exception(ex)

    def _inverse(self, y):
        """ This function uses a pseudoinverse of the matrix sf to approximate an inverse to the transformation.
        """
        if self.pinv is None:
            self.pinv = pinv(self.sf)
        return mult(y, self.pinv) + self.avg

    # TODO: write correct interface to enable for excecution_read, excecution_save
    # TODO: check integer vs float arguments in parallelization
    # TODO: This function could also be called _train_with_scheduler
    def _train(self, x, block_size=None, train_mode=None, node_weights=None, edge_weights=None, scheduler=None,
               n_parallel=None):
        """ This code is the training funcion when an mdp scheduler is provided (otherwise it resorts to the
        _train_without_scheduler version).

        The use of an scheduler is still in an experimental and buggy state, please do not use the scheduler.
        The use of parallelization through Intel MKL is preferred to using this method.
        """
        self._train_phase_started = True
        if train_mode is None:
            train_mode = self.train_mode
        if block_size is None:
            block_size = self.block_size
        if scheduler is None or n_parallel is None or train_mode is None:
            print("BLOCK_SIZE=", block_size)
            return self._train_without_scheduler(x, block_size=block_size, train_mode=train_mode,
                                                 node_weights=node_weights, edge_weights=edge_weights)
        else:
            num_chunks = min(n_parallel, x.shape[0] // block_size)  # WARNING, cover case division is not exact
            # here chunk_size is given in blocks!!!
            # chunk_size = int(numpy.ceil((x.shape[0]/block_size)*1.0/num_chunks))
            chunk_size = int((x.shape[0] // block_size) // num_chunks)

            # Notice that parallel training doesn't work with clustered mode and inhomogeneous blocks
            # TODO:Fix this
            print("%d chunks, of size %d blocks, last chunk contains %d blocks" % \
                  (num_chunks, chunk_size, (x.shape[0] // block_size) % chunk_size))
            if train_mode == 'clustered':
                # TODO: Implement this
                for i in range(num_chunks):
                    print("Adding scheduler task")
                    sys.stdout.flush()
                    if i < num_chunks - 1:
                        scheduler.add_task(
                            (x[i * block_size * chunk_size:(i + 1) * block_size * chunk_size], block_size, 1.0),
                            ComputeCovDcovMatrixClustered)
                    else:  # i == num_chunks - 1
                        scheduler.add_task((x[i * block_size * chunk_size:], block_size, 1.0),
                                           ComputeCovDcovMatrixClustered)
                    print("Done Adding scheduler task ///////////////////")
                    sys.stdout.flush()
            elif train_mode == 'serial':
                for i in range(num_chunks):
                    if i == 0:
                        print("adding task %d from sample " % i, i * block_size * chunk_size)
                        print("to", (i + 1) * block_size * chunk_size)
                        scheduler.add_task(
                            (x[i * block_size * chunk_size:(i + 1) * block_size * chunk_size], block_size),
                            ComputeCovDcovMatrixSerial)
                    elif i == num_chunks - 1:  # Add one previous block to the chunk
                        print("adding task %d from sample " % i, i * block_size * chunk_size - block_size, "to the end")
                        scheduler.add_task((x[i * block_size * chunk_size - block_size:], block_size),
                                           ComputeCovDcovMatrixSerial)
                    else:
                        print("adding task %d from sample " % i, i * block_size * chunk_size - block_size)
                        print("to", (i + 1) * block_size * chunk_size)
                        scheduler.add_task((x[i * block_size * chunk_size -
                                              block_size:(i + 1) * block_size * chunk_size], block_size),
                                           ComputeCovDcovMatrixSerial)
            elif train_mode == 'mixed':
                bs = block_size
                self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[0:bs], weight=0.5)

                # self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[bs:-bs], weight=1.0)
                x2 = x[bs:-bs]
                # num_chunks2 = int(numpy.ceil((x2.shape[0]/block_size-2)*1.0/chunk_size))
                num_chunks2 = int((x2.shape[0] / block_size - 2) / chunk_size)
                for i in range(num_chunks2):
                    if i < num_chunks2 - 1:
                        scheduler.add_task(
                            (x2[i * block_size * chunk_size:(i + 1) * block_size * chunk_size], block_size, 1.0),
                            ComputeCovDcovMatrixClustered)
                    else:
                        scheduler.add_task((x2[i * block_size * chunk_size:], block_size, 1.0),
                                           ComputeCovDcovMatrixClustered)

                self._covdcovmtx.update_clustered_homogeneous_block_sizes(x[-bs:], weight=0.5)

                # self._covdcovmtx.update_serial(x)
                for i in range(num_chunks):
                    if i == 0:
                        scheduler.add_task(
                            (x[i * block_size * chunk_size:(i + 1) * block_size * chunk_size], block_size),
                            ComputeCovDcovMatrixSerial)
                    elif i == num_chunks - 1:  # Add one previous block to the chunk
                        scheduler.add_task((x[i * block_size * chunk_size - block_size:], block_size),
                                           ComputeCovDcovMatrixSerial)
                    else:  # Add one previous block to the chunk
                        scheduler.add_task(
                            (x[i * block_size * chunk_size - block_size:(i + 1) * block_size * chunk_size], block_size),
                            ComputeCovDcovMatrixSerial)
            elif train_mode == 'unlabeled':
                # TODO: IMPLEMENT THIS
                for i in range(num_chunks):
                    print("Adding scheduler task")
                    sys.stdout.flush()
                    # BUG:Why am I adding here the clustered graph!!!
                    scheduler.add_task(
                        (x[i * block_size * chunk_size:(i + 1) * block_size * chunk_size], block_size, 1.0),
                        ComputeCovDcovMatrixClustered)
                    print("Done Adding scheduler task")
                    sys.stdout.flush()
            # TODO:ADD support for regular/standard mode
            else:
                ex = "Unknown training method:", self.train_mode
                raise Exception(ex)

            print("Getting results")
            sys.stdout.flush()

            results = scheduler.get_results()
            sys.stdout.flush()

            for covdcovmtx in results:
                self._covdcovmtx.add_cov_dcov_matrix(covdcovmtx)

    # TODO: consider removing pca_term and pca_exp
    def _stop_training(self, debug=False, verbose=False, pca_term=0.995, pca_exp=2.0):
        if verbose or True:
            print("stop_training: self.block_size=", self.block_size)
        print("self._covdcovmtx.num_samples = ", self._covdcovmtx.num_samples)
        print("self._covdcovmtx.num_diffs= ", self._covdcovmtx.num_diffs)
        self.cov_mtx, self.avg, self.dcov_mtx = self._covdcovmtx.fix()

        print("Finishing GSFA training: ", self._covdcovmtx.num_samples)
        print(" num_samples, and ", self._covdcovmtx.num_diffs, " num_diffs")
        print("DCov[0:3,0:3] is", self.dcov_mtx[0:3, 0:3])

        if pca_term != 0.0 and False:
            signs = numpy.sign(self.dcov_mtx)
            self.dcov_mtx = ((1.0 - pca_term) * signs * numpy.abs(self.dcov_mtx) ** (1.0 / pca_exp) +
                             pca_term * numpy.identity(self.dcov_mtx.shape[0])) ** pca_exp
            self.dcov_mtx = numpy.identity(self.dcov_mtx.shape[0])
        rng = self._set_range()

        # Solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        # WARNING, moved dcov to the second argument!!!
        # self.d, self.sf = self._symeig(self.dcov_mtx, self.cov_mtx, range=rng, overwrite=(not debug))
        # TODO: Only remove first eigenvalue and ignore negative eigenvalues (now there are features with
        # negative delta value)
        try:
            print("***Range used=", rng)
            # ##            if self.sfa_expo != None and self.pca_expo!=None:
            # ##                self.d, self.sf = _symeig_fake_regularized(self.dcov_mtx, self.cov_mtx, range=rng,
            # overwrite=(not debug), sfa_expo=self.sfa_expo, pca_expo=self.pca_expo,
            # magnitude_sfa_biasing=self.magnitude_sfa_biasing)
            # ##            else:
            self.d, self.sf = self._symeig(self.dcov_mtx, self.cov_mtx, range=rng, overwrite=(not debug))
            d = self.d
            # check that we get only *positive* eigenvalues
            if d.min() < 0:
                raise SymeigException("Got negative eigenvalues: %s." % str(d))
        except SymeigException as exception:
            errstr = str(exception) + "\n Covariance matrices may be singular."
            raise Exception(errstr)

        del self._covdcovmtx
        del self.cov_mtx
        del self.dcov_mtx
        self.cov_mtx = self.dcov_mtx = self._covdcovmtx = None
        self._bias = mult(self.avg, self.sf)
        print("shape of GSFANode.sf is=", self.sf.shape)


##############################################################################################################
#                              HELPER FUNCTIONS                                                              #
##############################################################################################################

def graph_delta_values(y, edge_weights):
    """ Computes delta values from an arbitrary graph as in the objective 
    function of GSFA. The feature vectors are not normalized to weighted 
    unit variance or weighted zero mean.
    """
    R = 0
    deltas = 0
    for (i, j) in edge_weights.keys():
        w_ij = edge_weights[(i, j)]
        deltas += w_ij * (y[j] - y[i]) ** 2
        R += w_ij
    return deltas / R


def comp_delta(x):
    """ Computes delta values as in the objective function of SFA.
    The feature vectors are not normalized to unit variance or zero mean.
    """
    xderiv = x[1:, :] - x[:-1, :]
    return (xderiv ** 2).mean(axis=0)


def Hamming_weight(integer_list):
    """ Computes the Hamming weight of an integer or a list of integers (number of bits equal to one) 
    """
    if isinstance(integer_list, list):
        return [Hamming_weight(k) for k in integer_list]
    elif isinstance(integer_list, int):
        w = 0
        n = integer_list
        while n > 0:
            if n % 2:
                w += 1
            n = n // 2
        return w
    else:
        er = "unsupported input type for Hamming_weight:", integer_list
        raise Exception(er)


# TODO: Remove unneeded global variables sum_diffs and last_block
# TODO: fix function names (lowercase_notation)
class CovDCovMatrix(object):
    """Special purpose class to compute the covariance matrices used by GSFA.
       It supports efficiently training methods for various graphs: e.g., clustered, serial, mixed.
    """
    def __init__(self, block_size=None):
        self.block_size = block_size
        self.sum_x = None
        self.sum_prod_x = None
        self.num_samples = 0
        self.sum_diffs = None  # marked for deletion
        self.sum_prod_diffs = None
        self.num_diffs = 0
        self.last_block = None  # marked for deletion

        # Variables used to store final results
        self.cov_mtx = None
        self.avg = None
        self.dcov_mtx = None
        self.tlen = 0

    def add_samples(self, sum_prod_x, sum_x, num_samples, weight=1.0):
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

    def add_diffs(self, sum_prod_diffs, num_diffs, weight=1.0):
        weighted_sum_prod_diffs = sum_prod_diffs * weight
        weighted_num_diffs = num_diffs * weight

        if self.sum_prod_diffs is None:
            self.sum_prod_diffs = weighted_sum_prod_diffs
        else:
            self.sum_prod_diffs = self.sum_prod_diffs + weighted_sum_prod_diffs

        self.num_diffs = self.num_diffs + weighted_num_diffs

        # TODO:Add option to skip last sample from Cov part.

    # Add unlabeled samples to Cov matrix (DCov remains unmodified)
    def update_unlabeled(self, x, weight=1.0):
        num_samples, dim = x.shape

        sum_x = x.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

    # TODO:Add option to skip last sample from Cov part.
    def update_regular(self, x, weight=1.0):
        """This is equivalent to regular SFA training. """
        num_samples, dim = x.shape

        # Update Cov Matrix
        sum_x = x.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

        # Update DCov Matrix
        diffs = x[1:, :] - x[:-1, :]
        num_diffs = num_samples - 1
        # sum_diffs = diffs.sum(axis=0)
        sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
        self.add_diffs(sum_prod_diffs, num_diffs, weight)

    def update_graph(self, x, node_weights=None, edge_weights=None, weight=1.0):
        """Updates the covariance/second moment matrices using a graph (samples, node weights, edge weights).

         Usually sum(node_weights) = num_samples.
         """
        num_samples, dim = x.shape

        if node_weights is None:
            node_weights = numpy.ones(num_samples)

        if len(node_weights) != num_samples:
            er = "Node weights should be the same length %d as the number of samples %d" % \
                 (len(node_weights), num_samples)
            raise Exception(er)

        if edge_weights is None:
            er = "edge_weights should be a dictionary with entries: d[(i,j)] = w_{i,j} or an NxN array"
            raise Exception(er)

        if isinstance(edge_weights, numpy.ndarray):
            # TODO: make sure edge_weights are symmetric
            # TODO: make sure consistency restriction is fulfilled
            if edge_weights.shape != (num_samples, num_samples):
                er = "Error, dimensions of edge_weights should be (%d,%d) but is (%d,%d)" % \
                     (num_samples, num_samples, edge_weights.shape[0], edge_weights.shape[1])
                raise Exception(er)

        node_weights_column = node_weights.reshape((num_samples, 1))
        # Update Cov Matrix
        weighted_x = x * node_weights_column

        weighted_sum_x = weighted_x.sum(axis=0)
        weighted_sum_prod_x = mdp.utils.mult(x.T, weighted_x)
        weighted_num_samples = node_weights.sum()
        # print "weighted_num_samples=",  weighted_num_samples
        self.add_samples(weighted_sum_prod_x, weighted_sum_x, weighted_num_samples, weight=weight)

        # Update DCov Matrix
        if isinstance(edge_weights, numpy.ndarray):
            weighted_num_diffs = edge_weights.sum()  # R
            prod1 = weighted_sum_prod_x  # TODO: USE THE CORRECT EQUATIONS; THIS ONLY WORKS IF Q==R!!!!!!!
            prod2 = mdp.utils.mult(mdp.utils.mult(x.T, edge_weights), x)
            weighted_sum_prod_diffs = 2 * prod1 - 2 * prod2
            self.add_diffs(weighted_sum_prod_diffs, weighted_num_diffs, weight=weight)
        else:
            num_diffs = len(edge_weights)
            diffs = numpy.zeros((num_diffs, dim))
            weighted_diffs = numpy.zeros((num_diffs, dim))
            weighted_num_diffs = 0
            for ii, (i, j) in enumerate(edge_weights.keys()):
                diff = x[j, :] - x[i, :]
                diffs[ii] = diff
                w_ij = edge_weights[(i, j)]
                weighted_diff = diff * w_ij
                weighted_diffs[ii] = weighted_diff
                weighted_num_diffs += w_ij

            weighted_sum_prod_diffs = mdp.utils.mult(diffs.T, weighted_diffs)
            self.add_diffs(weighted_sum_prod_diffs, weighted_num_diffs, weight=weight)

    def update_graph_old(self, x, node_weights=None, edge_weights=None, weight=1.0):
        """This method performs the same task as update_graph, however it has not been optimized."""
        num_samples, dim = x.shape

        if node_weights is None:
            node_weights = numpy.ones(num_samples)

        if len(node_weights) != num_samples:
            er = "Node weights should be the same length %d as the number of samples %d" % \
                 (len(node_weights), num_samples)
            raise Exception(er)

        if edge_weights is None:
            er = "edge_weights should be a dictionary with entries: d[(i,j)] = w_{i,j} or an NxN array"
            raise Exception(er)

        if isinstance(edge_weights, numpy.ndarray):
            if edge_weights.shape == (num_samples, num_samples):
                e_w = {}
                for i in range(num_samples):
                    for j in range(num_samples):
                        if edge_weights[i, j] != 0:
                            e_w[(i, j)] = edge_weights[i, j]
                edge_weights = e_w
            else:
                er = "Error, dimensions of edge_weights should be (%d,%d) but is (%d,%d)" % \
                     (num_samples, num_samples, edge_weights.shape[0], edge_weights.shape[1])
                raise Exception(er)
        node_weights_column = node_weights.reshape((num_samples, 1))
        # Update Cov Matrix
        weighted_x = x * node_weights_column

        weighted_sum_x = weighted_x.sum(axis=0)
        weighted_sum_prod_x = mdp.utils.mult(x.T, weighted_x)
        weighted_num_samples = node_weights.sum()
        # print "weighted_num_samples=",  weighted_num_samples
        self.add_samples(weighted_sum_prod_x, weighted_sum_x, weighted_num_samples, weight=weight)

        # Update DCov Matrix
        num_diffs = len(edge_weights)
        diffs = numpy.zeros((num_diffs, dim))
        weighted_diffs = numpy.zeros((num_diffs, dim))
        weighted_num_diffs = 0
        for ii, (i, j) in enumerate(edge_weights.keys()):
            diff = x[j, :] - x[i, :]
            diffs[ii] = diff
            w_ij = edge_weights[(i, j)]
            weighted_diff = diff * w_ij
            weighted_diffs[ii] = weighted_diff
            weighted_num_diffs += w_ij

        weighted_sum_prod_diffs = mdp.utils.mult(diffs.T, weighted_diffs)
        self.add_diffs(weighted_sum_prod_diffs, weighted_num_diffs, weight=weight)

    # Note: this method makes sense according to the consistency restriction for "larger" windows.
    def update_mirroring_sliding_window(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth  # window_halfwidth is way too long to write it
        if 2 * width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples!" % (width, num_samples)
            raise Exception(ex)

        # Update Cov Matrix
        # All samples have same weight
        sum_x = x.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

        # Update DCov Matrix
        # First mirror the borders
        x_mirror = numpy.zeros((num_samples + 2 * width, dim))
        x_mirror[width:-width] = x  # center part
        x_mirror[0:width, :] = x[0:width, :][::-1, :]  # first end
        x_mirror[-width:, :] = x[-width:, :][::-1, :]  # second end

        # Center part
        x_full = x
        sum_prod_x_full = mdp.utils.mult(x_full.T, x_full)
        #        print "sum_prod_x_full[0]=", sum_prod_x_full[0]
        #        print "(2*width+1)*sum_prod_x_full=", (2*width+1)*sum_prod_x_full[0:3,0:3]

        Aacc123 = numpy.zeros((dim, dim))
        for i in range(0, 2 * width):  # [0, 2*width-1]
            Aacc123 += (i + 1) * mdp.utils.mult(x_mirror[i:i + 1, :].T, x_mirror[i:i + 1, :])  # (i+1)=1,2,3..., 2*width

        for i in range(num_samples, num_samples + 2 * width):  # [num_samples-width, num_samples-1]
            Aacc123 += (num_samples + 2 * width - i) * mdp.utils.mult(x_mirror[i:i + 1, :].T, x_mirror[i:i + 1, :])
        #            print (num_samples+2*width-i)
        # Warning [-1,0] does not work!!!
        # for i in range(0, 2*width): # [0, 2*width-1]
        #    Aacc123 += (i+1)*mdp.utils.mult(x_mirror[-(i+1):-i,:].T, x_mirror[-(i+1):-i,:]) #(i+1)=1,2,3..., 2*width
        x_middle = x_mirror[2 * width:-2 * width, :]  # intermediate values of x, which are connected 2*width+1 times
        Aacc123 += (2 * width + 1) * mdp.utils.mult(x_middle.T, x_middle)
        #        print "Aacc123[0:3,0:3]=", Aacc123[0:3,0:3]

        b = numpy.zeros((num_samples + 1 + 2 * width, dim))
        b[1:] = x_mirror.cumsum(axis=0)
        B = b[2 * width + 1:] - b[0:-2 * width - 1]
        Bprod = mdp.utils.mult(x_full.T, B)
        #        print "Bprod[0:3,0:3]=", Bprod[0:3,0:3]

        sum_prod_diffs_full = (2 * width + 1) * sum_prod_x_full + Aacc123 - Bprod - Bprod.T
        num_diffs = num_samples * (2 * width)  # removed zero differences
        #         print "N3=", num_diffs
        self.add_diffs(sum_prod_diffs_full, num_diffs, weight)

        # Note: this method makes sense according to the consistency restriction for "larger" windows.

    # This is an unoptimized version of update_mirroring_sliding_window
    def update_slow_mirroring_sliding_window(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth  # window_halfwidth is way too long to write it
        if 2 * width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples!" % (width, num_samples)
            raise Exception(ex)

        # Update Cov Matrix
        # All samples have same weight
        sum_x = x.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

        # Update DCov Matrix
        # window = numpy.ones(2*width+1) # Rectangular window
        x_mirror = numpy.zeros((num_samples + 2 * width, dim))
        x_mirror[width:-width] = x  # center part
        x_mirror[0:width, :] = x[0:width, :][::-1, :]  # first end
        x_mirror[-width:, :] = x[-width:, :][::-1, :]  # second end

        # diffs = numpy.zeros((num_samples, dim))
        # print "width=", width
        for offset in range(-width, width + 1):
            if offset == 0:
                pass
            else:
                diffs = x_mirror[offset + width:offset + width + num_samples, :] - x

                #                sum_diffs = diffs.sum(axis=0)
                sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
                num_diffs = len(diffs)
                # WARNING!
                #                self.add_diffs(sum_prod_diffs, sum_diffs, num_diffs, weight * window[width+offset])
                self.add_diffs(sum_prod_diffs, num_diffs, weight)

                # ################# Truncating Window ####################

    def update_slow_truncating_sliding_window(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth  # window_halfwidth is way too long to write it
        if 2 * width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples!" % (width, num_samples)
            raise Exception(ex)

        # Update Cov Matrix
        # All samples have same weight
        sum_x = x.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

        # Update DCov Matrix
        # window = numpy.ones(2*width+1) # Rectangular window
        x_extended = numpy.zeros((num_samples + 2 * width, dim))
        x_extended[width:-width] = x  # center part is preserved, extreme samples are zero

        # diffs = numpy.zeros((num_samples, dim))
        # print "width=", width
        # Negative offset is not considered because it is equivalent to the positive one, thereore the factor 2
        for offset in range(1, width + 1):
            diffs = x_extended[offset + width:width + num_samples, :] - x[0:-offset, :]
            sum_prod_diffs = 2 * mdp.utils.mult(diffs.T, diffs)
            num_diffs = 2 * (num_samples - offset)
            self.add_diffs(sum_prod_diffs, num_diffs, weight)
        #            diffs = x_extended[offset+width:offset+width+num_samples,:]-x
        #            sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
        #            num_diffs = 2*(num_samples-offset)
        #            self.add_diffs(sum_prod_diffs, num_diffs, weight)


    # ################## Sliding window with node-weight correction #################
    def update_fast_sliding_window(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth
        if 2 * width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples!" % (width, num_samples)
            raise Exception(ex)

        # MOST CORRECT VERSION
        x_sel = x + 0.0
        w_up = numpy.arange(width, 2 * width) / (2.0 * width)
        w_up = w_up.reshape((width, 1))
        w_down = numpy.arange(2 * width - 1, width - 1, -1) / (2.0 * width)
        w_down = w_down.reshape((width, 1))
        x_sel[0:width, :] = x_sel[0:width, :] * w_up
        x_sel[-width:, :] = x_sel[-width:, :] * w_down

        sum_x = x_sel.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x_sel.T, x)  # There was a bug here, x_sel used twice!!!
        self.add_samples(sum_prod_x, sum_x, num_samples - (0.5 * window_halfwidth - 0.5), weight)

        # Update DCov Matrix
        # First we compute the borders.
        # Left border
        for i in range(0, width):  # [0, width -1]
            diffs = x[0:width + i + 1, :] - x[i, :]
            sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
            num_diffs = len(diffs) - 1  # removed zero differences
            # print "N1=", num_diffs
            # print "sum_prod_diffs[0]=", sum_prod_diffs[0]
            self.add_diffs(sum_prod_diffs, num_diffs, weight)
        # Right border
        for i in range(num_samples - width, num_samples):  # [num_samples-width, num_samples-1]
            diffs = x[i - width:num_samples, :] - x[i, :]
            sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
            num_diffs = len(diffs) - 1  # removed zero differences
            # print "N2=", num_diffs
            # print "sum_prod_diffs[0]=", sum_prod_diffs[0]
            self.add_diffs(sum_prod_diffs, num_diffs, weight)

        # Center part
        x_full = x[width:num_samples - width, :]
        sum_prod_x_full = mdp.utils.mult(x_full.T, x_full)
        #        print "sum_prod_x_full[0]=", sum_prod_x_full[0]

        Aacc123 = numpy.zeros((dim, dim))
        for i in range(0, 2 * width):  # [0, 2*width-1]
            Aacc123 += (i + 1) * mdp.utils.mult(x[i:i + 1, :].T, x[i:i + 1, :])  # (i+1)=1,2,3..., 2*width

        for i in range(num_samples - 2 * width, num_samples):  # [num_samples-width, num_samples-1]
            Aacc123 += (num_samples - i) * mdp.utils.mult(x[i:i + 1, :].T,
                                                          x[i:i + 1, :])  # (num_samples-1)=2*width,...,3,2,1

        # intermediate values of x, which are connected 2*width+1 times
        x_middle = x[2 * width:num_samples - 2 * width, :]

        Aacc123 += (2 * width + 1) * mdp.utils.mult(x_middle.T, x_middle)

        b = numpy.zeros((num_samples + 1, dim))
        b[1:] = x.cumsum(axis=0)
        #        for i in range(1,num_samples+1):
        #            b[i] = b[i-1] + x[i-1,:]
        #        A = a[2*width+1:]-a[0:-2*width-1]
        B = b[2 * width + 1:] - b[0:-2 * width - 1]
        #        Aacc = A.sum(axis=0)
        #        print "Aacc[0]=", Aacc[0]
        Bprod = mdp.utils.mult(x_full.T, B)
        #        print "Bprod[0]=", Bprod[0]
        #        print sum_prod_x_full.shape, Aacc.shape, Bprod.shape
        sum_prod_diffs_full = (2 * width + 1) * sum_prod_x_full + Aacc123 - Bprod - Bprod.T
        num_diffs = (num_samples - 2 * width) * (2 * width)  # removed zero differences
        #         print "N3=", num_diffs
        self.add_diffs(sum_prod_diffs_full, num_diffs, weight)

    def update__sliding_window(self, x, weight=1.0, window_halfwidth=2):
        num_samples, dim = x.shape
        width = window_halfwidth
        if 2 * width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples!" % (width, num_samples)
            raise Exception(ex)

        # Update Cov Matrix
        # Warning, truncating samples to avoid edge problems
        # TRUNCATED VERSION
        #        x_sel = x[window_halfwidth:-window_halfwidth,:]
        #        sum_x = x_sel.sum(axis=0)
        #        sum_prod_x = mdp.utils.mult(x_sel.T, x_sel)
        #        self.add_samples(sum_prod_x, sum_x, num_samples-2*window_halfwidth, weight)

        # MOST CORRECT VERSION
        x_sel = x + 0.0
        w_up = numpy.arange(width, 2 * width) / (2.0 * width)
        w_up = w_up.reshape((width, 1))
        w_down = numpy.arange(2 * width - 1, width - 1, -1) / (2.0 * width)
        w_down = w_down.reshape((width, 1))
        x_sel[0:width, :] = x_sel[0:width, :] * w_up
        x_sel[-width:, :] = x_sel[-width:, :] * w_down

        sum_x = x_sel.sum(axis=0)
        # print "F:)",
        sum_prod_x = mdp.utils.mult(x_sel.T, x)  # Bug fixed!!! computing w * X^T * X, with X=(x1,..xN)^T
        self.add_samples(sum_prod_x, sum_x, num_samples - (0.5 * window_halfwidth - 0.5), weight)  # weights verified

        # Update DCov Matrix
        # window = numpy.ones(2*width+1) # Rectangular window, used always here!

        # diffs = numpy.zeros((num_samples - 2 * width, dim))
        # print "width=", width
        # This can be made faster (twice) due to symmetry
        for offset in range(-width, width + 1):
            if offset == 0:
                pass
            else:
                if offset > 0:
                    diffs = x[offset:, :] - x[0:num_samples - offset, :]
                    #                else: #offset < 0, only makes sense if window asymmetric!
                    #                    abs_offset = -1 * offset
                    #                    diffs = x[0:num_samples-abs_offset,:] - x[abs_offset:,:]
                    sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
                    num_diffs = len(diffs)
                    self.add_diffs(sum_prod_diffs, num_diffs, weight)

                # ERRONEOUS VERSION USED FOR PERFORMANCE REPORT: (leaves out several differences due to extreme samples)
                #                diffs = x[width+offset:num_samples-width+offset,:]-x[width:num_samples-width,:]
                #
                #                sum_diffs = diffs.sum(axis=0)
                #                sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
                #                num_diffs = len(diffs)
                # # WARNING!
                # #                self.add_diffs(sum_prod_diffs, sum_diffs, num_diffs, weight * window[width+offset])
                #                self.add_diffs(sum_prod_diffs, sum_diffs, num_diffs, weight)

                #            def add_diffs(self, sum_prod_diffs, sum_diffs, num_diffs, weight=1.0):

    # Add samples belonging to a serial training graph
    # NOTE: include_last_block not needed
    def update_serial(self, x, block_size, weight=1.0, include_last_block=True):
        num_samples, dim = x.shape
        if block_size is None:
            er = "block_size must be specified"
            raise Exception(er)
            block_size = self.block_size
        if isinstance(block_size, numpy.ndarray):
            err = "Inhomogeneous block sizes not yet supported in update_serial"
            raise Exception(err)
        elif isinstance(block_size, list):
            block_size_0 = block_size[0]
            for bs in block_size:
                if bs != block_size_0:
                    er = "for serial graph all groups must have same group size (block_size constant), but " + \
                         str(bs) + "!=" + str(block_size_0)
                    raise Exception(er)
            block_size = block_size_0

        if num_samples % block_size > 0:
            err = "Consistency error: num_samples is not a multiple of block_size"
            raise Exception(err)
        num_blocks = num_samples // block_size

        # warning, plenty of dtype missing!!!!!!!!
        # Optimize computation of x.T ???
        # Warning, remove last element of x (incremental computation)!!!

        # Correlation Matrix. Computing sum of outer products (the easy part)
        xp = x[block_size:num_samples - block_size]
        x_b_ini = x[0:block_size]
        x_b_end = x[num_samples - block_size:]
        sum_x = x_b_ini.sum(axis=0) + 2 * xp.sum(axis=0) + x_b_end.sum(axis=0)

        #            print "Sum_x[0:3] is ", sum_x[0:3]
        sum_prod_x = mdp.utils.mult(x_b_ini.T, x_b_ini) + 2 * mdp.utils.mult(xp.T, xp) + mdp.utils.mult(x_b_end.T,
                                                                                                        x_b_end)
        num_samples = 2 * block_size + 2 * (num_samples - 2 * block_size)

        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

        self.last_block = x[num_samples - block_size:]

        # DCorrelation Matrix. Compute medias signal
        media = numpy.zeros((num_blocks, dim))
        for i in range(num_blocks):
            media[i] = x[i * block_size:(i + 1) * block_size].sum(axis=0) * (1.0 / block_size)

        media_a = media[0:-1]
        media_b = media[1:]
        sum_prod_mixed_meds = (mdp.utils.mult(media_a.T, media_b) + mdp.utils.mult(media_b.T, media_a))
        #        prod_first_media = numpy.outer(media[0], media[0]) * block_size
        #        prod_last_media = numpy.outer(media[num_blocks-1], media[num_blocks-1]) * block_size
        prod_first_block = mdp.utils.mult(x[0:block_size].T, x[0:block_size])
        prod_last_block = mdp.utils.mult(x[num_samples - block_size:].T, x[num_samples - block_size:])

        # next line causes an exception index out of bounds: n_threads=10
        #        sum_diffs = (media[num_blocks-1] - media[0]) * (block_size * block_size)  * (1.0 / block_size)
        #        num_diffs = (block_size * block_size) * (num_blocks - 1)
        #       WARNING? why did I remove one factor block_size?
        num_diffs = block_size * (num_blocks - 1)

        # warning with factors!!!! they should be 2.0, 1.0, -1.0
        #        sum_prod_diffs = block_size * (2.0 * sum_prod_x + 2.0 * prod_last_block - 0.0 *prod_first_block) -
        # (block_size * block_size) * sum_prod_mixed_meds
        sum_prod_diffs = (block_size * sum_prod_x -
                          (block_size * block_size) * sum_prod_mixed_meds) * (1.0 / block_size)
        #        sum_prod_diffs = block_size * (2.0 * sum_prod_x + 2.0 * prod_last_block + 0.0 *prod_first_block) -
        # (block_size * block_size) * sum_prod_mixed_meds
        self.add_diffs(2 * sum_prod_diffs, 2 * num_diffs, weight)  # NEW: Factor 2 to account for both directions

    # Weight should refer to node weights
    def update_clustered(self, x, block_sizes=None, weight=1.0, include_self_loops=True):
        num_samples, dim = x.shape

        if isinstance(block_sizes, int):
            return self.update_clustered_homogeneous_block_sizes(x, weight=weight, block_size=block_sizes,
                                                                 include_self_loops=include_self_loops)

        if block_sizes is None:
            er = "error, block_size not specified!!!!"
            raise Exception(er)
            # block_sizes = self.block_size

        if num_samples != numpy.array(block_sizes).sum():
            err = "Inconsistency error: num_samples (%d) is not equal to sum of block_sizes:" % num_samples, block_sizes
            raise Exception(err)

        counter_sample = 0
        for block_size in block_sizes:
            normalized_weight = weight
            self.update_clustered_homogeneous_block_sizes(x[counter_sample:counter_sample + block_size, :],
                                                          weight=normalized_weight, block_size=block_size,
                                                          include_self_loops=include_self_loops)
            counter_sample += block_size

    def update_clustered_homogeneous_block_sizes(self, x, weight=1.0, block_size=None, include_self_loops=True):
        print("update_clustered_homogeneous_block_sizes ")
        if block_size is None:
            er = "error, block_size not specified!!!!"
            raise Exception(er)
            # block_size = self.block_size

        if isinstance(block_size, numpy.ndarray):
            er = "Error: inhomogeneous block sizes not supported by this function"
            raise Exception(er)

        # Assuming block_size is an integer:
        num_samples, dim = x.shape
        if num_samples % block_size > 0:
            err = "Inconsistency error: num_samples (%d) is not a multiple of block_size (%d)" % \
                  (num_samples, block_size)
            raise Exception(err)
        num_blocks = num_samples // block_size

        # warning, plenty of dtype missing!!!!!!!!
        sum_x = x.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

        self.last_block = None
        # DCorrelation Matrix. Compute medias signal
        media = numpy.zeros((num_blocks, dim))
        for i in range(num_blocks):
            media[i] = x[i * block_size:(i + 1) * block_size].sum(axis=0) * (1.0 / block_size)

        sum_prod_meds = mdp.utils.mult(media.T, media)
        # FIX1: AFTER DT in (0,4) normalization
        num_diffs = num_blocks * block_size  # ## * (block_size-1+1) / (block_size-1)
        print("num_diffs in block:", num_diffs, " num_samples:", num_samples)
        if include_self_loops:
            sum_prod_diffs = 2.0 * block_size * (sum_prod_x - block_size * sum_prod_meds) / block_size
        else:
            sum_prod_diffs = 2.0 * block_size * (sum_prod_x - block_size * sum_prod_meds) / (block_size - 1)

        self.add_diffs(sum_prod_diffs, num_diffs, weight)
        print("(Diag(complete)/num_diffs.avg)**0.5 =", ((numpy.diagonal(sum_prod_diffs) / num_diffs).mean()) ** 0.5)

    def update_compact_classes(self, x, block_sizes=None, Jdes=None, weight=1.0):
        num_samples, dim = x.shape

        print("block_sizes=", block_sizes, type(block_sizes))
        if isinstance(block_sizes, list):
            block_sizes = numpy.array(block_sizes)

        if isinstance(block_sizes, numpy.ndarray):
            if len(block_sizes) > 1:
                if block_sizes.var() > 0:
                    er = "for compact_classes all groups must have the same number of elements (block_sizes)!!!!"
                    raise Exception(er)
                else:
                    block_size = block_sizes[0]
            else:
                block_size = block_sizes[0]
        elif block_sizes is None:
            er = "error, block_size not specified!!!!"
            raise Exception(er)
        else:
            block_size = block_sizes

        if num_samples % block_size != 0:
            err = "Inconsistency error: num_samples (%d) must be a multiple of block_size: " % num_samples, block_sizes
            raise Exception(err)

        num_classes = num_samples // block_size
        J = int(numpy.log2(num_classes))
        if Jdes is None:
            Jdes = J
        extra_label = Jdes - J  # 0, 1, 2

        print("Besides J=%d labels, also adding %d labels" % (J, extra_label))

        if num_classes != 2 ** J:
            err = "Inconsistency error: num_clases %d does not appear to be a power of 2" % num_classes
            raise Exception(err)

        N = num_samples
        labels = numpy.zeros((N, J + extra_label))
        for j in range(J):
            labels[:, j] = (numpy.arange(N) // block_size // (2 ** (J - j - 1)) % 2) * 2 - 1
        eigenvalues = numpy.concatenate(([1.0] * (J - 1), numpy.arange(1.0, 0.0, -1.0 / (extra_label + 1))))
        #        eigenvalues = numpy.concatenate(([1.0]*(J-1), 0.98**numpy.arange(0, extra_label+1)))

        n_taken = [2 ** k for k in range(J)]
        n_free = list(set(range(num_classes)) - set(n_taken))
        n_free_weights = Hamming_weight(n_free)
        order = numpy.argsort(n_free_weights)[::-1]

        for j in range(extra_label):
            digit = n_free[order[j]]
            label = numpy.ones(N)
            for c in range(J):
                if (digit >> c) % 2:
                    label *= labels[:, c]
            if n_free_weights[order[j]] % 2 == 0:
                label *= -1
            labels[:, J + j] = label

        eigenvalues = numpy.array(eigenvalues)

        print("Eigenvalues:", eigenvalues)
        eigenvalues /= eigenvalues.sum()
        print("Eigenvalues normalized:", eigenvalues)

        for j in range(J + extra_label):
            print("labels[%d]=" % j, labels[:, j])

        for j in range(J + extra_label):
            set10 = x[labels[:, j] == -1]
            self.update_clustered_homogeneous_block_sizes(set10, weight=eigenvalues[j],
                                                          block_size=N // 2)  # first cluster
            set10 = x[labels[:, j] == 1]
            self.update_clustered_homogeneous_block_sizes(set10, weight=eigenvalues[j],
                                                          block_size=N // 2)  # second cluster

    def add_cov_dcov_matrix(self, cov_dcov_mat, adding_weight=1.0, own_weight=1.0):
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

    def fix(self, divide_by_num_samples_or_differences=True, verbose=False, center_dcov=False):  # include_tail=False,
        if verbose:
            print("Fixing CovDCovMatrix, with block_size=", self.block_size)

        avg_x = self.sum_x * (1.0 / self.num_samples)

        # THEORY: This computation has a bias
        # exp_prod_x = self.sum_prod_x * (1.0 / self.num_samples)
        # prod_avg_x = numpy.outer(avg_x, avg_x)
        # cov_x = exp_prod_x - prod_avg_x
        prod_avg_x = numpy.outer(avg_x, avg_x)
        if divide_by_num_samples_or_differences:  # as specified by the theory on Training Graphs
            cov_x = (self.sum_prod_x - self.num_samples * prod_avg_x) / (1.0 * self.num_samples)
        else:  # standard unbiased estimation
            cov_x = (self.sum_prod_x - self.num_samples * prod_avg_x) / (self.num_samples - 1.0)

        # Finalize covariance matrix of dx
        if divide_by_num_samples_or_differences or True:
            cov_dx = self.sum_prod_diffs / (1.0 * self.num_diffs)
        else:
            cov_dx = self.sum_prod_diffs / (self.num_diffs - 1.0)

        self.cov_mtx = cov_x
        self.avg = avg_x
        self.tlen = self.num_samples
        self.dcov_mtx = cov_dx

        # Safely uncomment the following lines for debugging
        # print "Finishing training CovDcovMtx: ",  self.num_samples, " num_samples, and ", self.num_diffs, " num_diffs"
        # print "Avg[0:3] is", self.avg[0:4]
        # print "Prod_avg_x[0:3,0:3] is", prod_avg_x[0:3,0:3]
        # print "Cov[0:3,0:3] is", self.cov_mtx[0:3,0:3]
        # print "DCov[0:3,0:3] is", self.dcov_mtx[0:3,0:3]
        # print "AvgDiff[0:4] is", avg_diff[0:4]
        # print "Prod_avg_diff[0:3,0:3] is", prod_avg_diff[0:3,0:3]
        # print "Sum_prod_diffs[0:3,0:3] is", self.sum_prod_diffs[0:3,0:3]
        # print "exp_prod_diffs[0:3,0:3] is", exp_prod_diffs[0:3,0:3]
        return self.cov_mtx, self.avg, self.dcov_mtx

    # ####### Helper functions for parallel processing and CovDcovMatrices #########


# This function is used by patch_mdp
def ComputeCovMatrix(x, verbose=False):
    print("PCov")
    if verbose:
        print("Computation Began!!! **********************************************************")
        sys.stdout.flush()
    covmtx = CovarianceMatrix(bias=True)
    covmtx.update(x)
    if verbose:
        print("Computation Ended!!! **********************************************************")
        sys.stdout.flush()
    return covmtx


def ComputeCovDcovMatrixClustered(params, verbose=False):
    print("PComp")
    if verbose:
        print("Computation Began!!! **********************************************************")
        sys.stdout.flush()
    x, block_size, weight = params
    covdcovmtx = CovDCovMatrix(block_size)
    covdcovmtx.update_clustered_homogeneous_block_sizes(x, block_size=block_size, weight=weight)
    if verbose:
        print("Computation Ended!!! **********************************************************")
        sys.stdout.flush()
    return covdcovmtx


def ComputeCovDcovMatrixSerial(params, verbose=False):
    print("PSeq")
    if verbose:
        print("Computation Began!!! **********************************************************")
        sys.stdout.flush()
    x, block_size = params
    covdcovmtx = CovDCovMatrix(block_size)
    covdcovmtx.update_serial(x, block_size=block_size)
    if verbose:
        print("Computation Ended!!! **********************************************************")
        sys.stdout.flush()
    return covdcovmtx


def ComputeCovDcovMatrixMixed(params, verbose=False):
    print("PMixed")
    if verbose:
        print("Computation Began!!! **********************************************************")
        sys.stdout.flush()
    x, block_size = params
    bs = block_size
    covdcovmtx = CovDCovMatrix(block_size)
    covdcovmtx.update_clustered_homogeneous_block_sizes(x[0:bs], block_size=block_size, weight=0.5)
    covdcovmtx.update_clustered_homogeneous_block_sizes(x[bs:-bs], block_size=block_size, weight=1.0)
    covdcovmtx.update_clustered_homogeneous_block_sizes(x[-bs:], block_size=block_size, weight=0.5)
    covdcovmtx.update_serial(x, block_size=block_size)
    if verbose:
        print("Computation Ended!!! **********************************************************")
        sys.stdout.flush()
    return covdcovmtx


#########################################################################################################
#                      AN EXAMPLE:                                                                      #
#########################################################################################################


def example_clustered_graph():
    print("")
    print("**************************************************************************")
    print("*Example of training GSFA using a clustered graph")
    cluster_size = 20
    num_clusters = 5
    num_samples = cluster_size * num_clusters
    dim = 20
    output_dim = 2
    x = numpy.random.normal(size=(num_samples, dim))
    x += 0.1 * numpy.arange(num_samples).reshape((num_samples, 1))

    print("x=", x)

    GSFA_n = GSFANode(output_dim=output_dim)

    def identity(xx):
        return xx

    def norm2(xx):  # Computes the norm of each sample returning an Nx1 array
        return ((xx ** 2).sum(axis=1) ** 0.5).reshape((-1, 1))

    Exp_n = mdp.nodes.GeneralExpansionNode([identity, norm2])

    exp_x = Exp_n.execute(x)  # Expanded data
    GSFA_n.train(exp_x, train_mode="clustered", block_size=cluster_size)
    GSFA_n.stop_training()

    print("GSFA_n.d=", GSFA_n.d)

    y = GSFA_n.execute(Exp_n.execute(x))
    print("y", y)
    print("Standard delta values of output features y:", comp_delta(y))

    x_test = numpy.random.normal(size=(num_samples, dim))
    x_test += 0.1 * numpy.arange(num_samples).reshape((num_samples, 1))
    y_test = GSFA_n.execute(Exp_n.execute(x_test))
    print("y_test", y_test)
    print("Standard delta values of output features y_test:", comp_delta(y_test))


def example_pathological_outputs(experiment):
    print("")
    print("**************************************************************************")
    print("*Pathological responses. Experiment on graph with weakly connected samples")
    x = numpy.random.normal(size=(20, 19))
    x2 = numpy.random.normal(size=(20, 19))

    l = numpy.random.normal(size=20)
    l -= l.mean()
    l /= l.std()
    l.sort()

    half_width = 3

    v = numpy.ones(20)
    e = {}
    for t in range(19):
        e[(t, t + 1)] = 1.0
    e[(0, 0)] = 0.5
    e[(19, 19)] = 0.5
    train_mode = "graph"

    # experiment = 0 #Select the experiment to perform, from 0 to 11
    print("experiment", experiment)
    if experiment == 0:
        exp_title = "Original linear SFA graph"
    elif experiment == 1:
        v[0] = 10.0
        v[10] = 0.1
        v[19] = 10.0
        exp_title = "Modified node weights 1"
    elif experiment == 2:
        v[0] = 10.0
        v[19] = 0.1
        exp_title = "Modified node weights 2"
    elif experiment == 3:
        e[(0, 1)] = 0.1
        e[(18, 19)] = 10.0
        exp_title = "Modified edge weights 3"
    elif experiment == 4:
        e[(0, 1)] = 0.01
        e[(18, 19)] = 0.01
        e[(15, 17)] = 0.5
        e[(16, 18)] = 0.5
        e[(12, 14)] = 0.5
        e[(3, 5)] = 0.5
        e[(4, 6)] = 0.5
        e[(5, 7)] = 0.5

        # e[(1,2)] = 0.1
        exp_title = "Modified edge weights 4"
    elif experiment == 5:
        e[(10, 11)] = 0.02
        e[(1, 2)] = 0.02
        e[(3, 5)] = 1.0
        e[(7, 9)] = 1.0
        e[(17, 19)] = 1.0
        e[(14, 16)] = 1.0

        exp_title = "Modified edge weights 5"
    elif experiment == 6:
        e[(6, 7)] = 0.1
        e[(5, 6)] = 0.1
        exp_title = "Modified edge weights 6"
    elif experiment == 7:
        e = {}
        for j1 in range(19):
            for j2 in range(j1 + 1, 20):
                e[(j1, j2)] = 1 / (l[j2] - l[j1] + 0.00005)
        exp_title = "Modified edge weights for labels as w12 = 1/(l2-l1+0.00005) 7"
    elif experiment == 8:
        e = {}
        for j1 in range(19):
            for j2 in range(j1 + 1, 20):
                e[(j1, j2)] = numpy.exp(-0.25 * (l[j2] - l[j1]) ** 2)
        exp_title = "Modified edge weights for labels as w12 = exp(-0.25*(l2-l1)**2) 8"
    elif experiment == 9:
        e = {}
        for j1 in range(19):
            for j2 in range(j1 + 1, 20):
                if l[j2] - l[j1] < 0.6:
                    e[(j1, j2)] = 1 / (l[j2] - l[j1] + 0.00005)
        exp_title = "Modified edge weights w12 = 1/(l2-l1+0.00005), for l2-l1<0.6 9"
    elif experiment == 10:
        exp_title = "Mirroring training graph, w=%d" % half_width + "10"
        train_mode = "smirror_window%d" % half_width
        e = {}
    elif experiment == 11:
        exp_title = "Node weight adjustment training graph, w=%d " % half_width + "11"
        train_mode = "window%d" % half_width
        e = {}
    else:
        er = "Unknown experiment " + str(experiment)
        raise Exception(er)

    n = GSFANode(output_dim=5)
    if experiment in (10, 11):
        n.train(x, train_mode=train_mode)
    else:
        n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()

    print("/" * 20, "Brute delta values of GSFA features (training/test):")
    y = n.execute(x)
    y2 = n.execute(x2)
    if e != {}:
        print(graph_delta_values(y, e))
        print(graph_delta_values(y2, e))

    D = numpy.zeros(20)
    for (j1, j2) in e:
        D[j1] += e[(j1, j2)] / 2.0
        D[j2] += e[(j1, j2)] / 2.0

    import matplotlib as mpl
    mpl.use('Qt4Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.title("Overfitted outputs on training data, v (node weights)=" + str(v))
    plt.xlabel(exp_title + "\n With D (half the sum of all edges from/to each vertex)=" + str(D))
    plt.xticks(numpy.arange(0, 20, 1))
    plt.plot(y)
    if experiment in (6, 7, 8):
        if y[0, 0] > 0:
            l *= -1
        plt.plot(l, "*")
    plt.show()


def example_continuous_edge_weights():
    print("")
    print("**************************************************************************")
    print("*Testing continuous edge weigths w_{n,n'} = 1/(|l_n'-l_n|+k)")
    x = numpy.random.normal(size=(20, 19))
    x2 = numpy.random.normal(size=(20, 19))

    l = numpy.random.normal(size=20)
    l -= l.mean()
    l /= l.std()
    l.sort()
    k = 0.0001

    v = numpy.ones(20)
    e = {}
    for n1 in range(20):
        for n2 in range(20):
            if n1 != n2:
                e[(n1, n2)] = 1.0 / (numpy.abs(l[n2] - l[n1]) + k)

    exp_title = "Original linear SFA graph"
    n = GSFANode(output_dim=5)
    n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()

    print("/" * 20, "Brute delta values of GSFA features (training/test):")
    y = n.execute(x)
    y2 = n.execute(x2)
    if e != {}:
        print(graph_delta_values(y, e))
        print(graph_delta_values(y2, e))

    D = numpy.zeros(20)
    for (j1, j2) in e:
        D[j1] += e[(j1, j2)] / 2.0
        D[j2] += e[(j1, j2)] / 2.0

    import matplotlib as mpl
    mpl.use('Qt4Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.title("Overfitted outputs on training data,v=" + str(v))
    plt.xlabel(exp_title + "\n With D=" + str(D))
    plt.xticks(numpy.arange(0, 20, 1))
    plt.plot(y)
    plt.plot(l, "*")
    plt.show()


if __name__ == "__main__":
    for experiment_number in range(0, 12):
        example_pathological_outputs(experiment=experiment_number)
    example_continuous_edge_weights()
    example_clustered_graph()
