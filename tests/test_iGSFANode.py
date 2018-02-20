#####################################################################################################################
# test_iGSFANode: Tests for the Information-Preserving Graph-Based SFA Node (iGSFANode) as defined by                #
#                 the Cuicuilco framework                                                                           #
#                                                                                                                   #
# By Alberto-N Escalante-B. Alberto.Escalante@ini.rub.de                                                            #
# Ruhr-University Bochum, Institute for Neural Computation, Group of Prof. Dr. Wiskott                              #
#####################################################################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy

import pytest
import mdp

import cuicuilco.patch_mdp
from cuicuilco.gsfa_node import comp_delta
from cuicuilco.igsfa_node import iGSFANode

# TODO: *test_automatic_stop_training
# what combinations of reconstruct_with_sfa and offsetting_mode are defined
# TODO: rename offsetting_mode -> slow_feature_scaling_method
#       *test_equivalence_GSFA_PCA_for_DT_m0.5
#       test all offsetting_modes("QR_decomposition", "sensitivity_based_pure", "sensitivity_based_offset",
#                                 "sensitivity_based_normalized", None, "data_dependent")
#       *test_SFANode_reduce_output_dim (extraction and inverse)
#       *test_PCANode_reduce_output_dim (extraction and inverse)

#iGSFANode(input_dim=None, output_dim=None, pre_expansion_node_class=None, pre_expansion_out_dim=None,
#                 expansion_funcs=None, expansion_output_dim=None, expansion_starting_point=None,
#                 max_lenght_slow_part=None, offsetting_mode="sensitivity_based_pure", max_preserved_sfa=1.9999,
#                 reconstruct_with_sfa=True, **argv)

def test_automatic_stop_training():
    """ Test that verifies that iGSFA automatically calls stop training when trained on single batch mode
    """
    x = numpy.random.normal(size=(300, 15))

    n = iGSFANode(output_dim=15, reconstruct_with_sfa=True, offsetting_mode=None)
    n.train(x, train_mode="regular")
    with pytest.raises(mdp.TrainingFinishedException):
        n.train(x, train_mode="regular")

    n = iGSFANode(output_dim=15, reconstruct_with_sfa=True, offsetting_mode="data_dependent")
    n.train(x, train_mode="regular")
    with pytest.raises(mdp.TrainingFinishedException):
        n.train(x, train_mode="regular")

    n = iGSFANode(output_dim=15, reconstruct_with_sfa=True, offsetting_mode="sensitivity_based_pure")
    n.train(x, train_mode="regular")
    with pytest.raises(mdp.TrainingFinishedException):
        n.train(x, train_mode="regular")

    n = iGSFANode(output_dim=15, reconstruct_with_sfa=True, offsetting_mode="QR_decomposition")
    n.train(x, train_mode="regular")
    with pytest.raises(mdp.TrainingFinishedException):
        n.train(x, train_mode="regular")


def test_no_automatic_stop_training():
    """ Test that verifies that iGSFA does not call stop training when when multiple-train is used
    """
    x = numpy.random.normal(size=(300, 15))
    n = iGSFANode(output_dim=5, reconstruct_with_sfa=False, offsetting_mode=None)
    n.train(x, train_mode="regular")
    n.train(x, train_mode="regular")
    n.stop_training()

    n = iGSFANode(output_dim=5, reconstruct_with_sfa=False, offsetting_mode="data_dependent")
    n.train(x, train_mode="regular")
    n.train(x, train_mode="regular")
    n.stop_training()


def test_equivalence_GSFA_iGSFA_for_DT_4_0():
    """ Test of iGSFA and GSFA when delta_threshold is larger than 4.0
    """
    x = numpy.random.normal(size=(300, 15))

    n = iGSFANode(output_dim=5, offsetting_mode=None, max_preserved_sfa=4.10)
    n.train(x, train_mode="regular")
    # n.stop_training() has been automatically called

    y = n.execute(x)
    deltas_igsfa = comp_delta(y)

    n2 = cuicuilco.gsfa_node.GSFANode(output_dim=5)
    n2.train(x, train_mode="regular")
    n2.stop_training()

    y2 = n2.execute(x)
    deltas_gsfa = comp_delta(y2)
    assert ((deltas_igsfa - deltas_gsfa) == pytest.approx(0.0)).all()


def test_equivalence_GSFA_PCA_for_DT_0():
    """ Test of iGSFA and PCA when delta_threshold is smaller than 0.0
    """
    x = numpy.random.normal(size=(300, 15))

    n = iGSFANode(output_dim=5, offsetting_mode=None, max_preserved_sfa=0.0)
    n.train(x, train_mode="regular")
    # n.stop_training() has been automatically called

    y = n.execute(x)
    deltas_igsfa = comp_delta(y)

    n2 = mdp.nodes.PCANode(output_dim=5)
    n2.train(x)
    n2.stop_training()

    y2 = n2.execute(x)
    deltas_pca = comp_delta(y2)
    assert ((deltas_igsfa - deltas_pca) == pytest.approx(0.0)).all()


# def atest_basic_GSFA_edge_dict():
#     """ Basic test of GSFA on random data and graph, edge dictionary mode
#     """
#     x = numpy.random.normal(size=(200, 15))
#     v = numpy.ones(200)
#     e = {}
#     for i in range(1500):
#         n1 = numpy.random.randint(200)
#         n2 = numpy.random.randint(200)
#         e[(n1, n2)] = numpy.random.normal() + 1.0
#     n = GSFANode(output_dim=5)
#     n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
#     n.stop_training()
#
#     y = n.execute(x)
#     delta_values_training_data = graph_delta_values(y, e)
#
#     x2 = numpy.random.normal(size=(200, 15))
#     y2 = n.execute(x2)
#     y2 = y2 - y2.mean(axis=0)  # enforce zero mean
#     y2 /= ((y2**2).mean(axis=0) ** 0.5)  # enforce zero-mean
#     # print("y2 means:", y2.mean(axis=0))
#     # print("y2 std:", (y2**2).mean(axis=0))
#
#     delta_values_test_data = graph_delta_values(y2, e)
#     assert (delta_values_training_data < delta_values_test_data).all()
#     # print("Graph delta values of training data", graph_delta_values(y, e))
#     # print("Graph delta values of test data (should be larger than for training)", graph_delta_values(y2, e))
#
#
# def atest_equivalence_SFA_GSFA_linear_graph():
#     """ Tests the equivalence of Standard SFA and GSFA when trained using an appropriate linear graph (graph mode)
#     """
#     x = numpy.random.normal(size=(200, 15))
#     x2 = numpy.random.normal(size=(200, 15))
#
#     v = numpy.ones(200)
#     # PENDING: Add an additional node without any edge to compensate no bias of SFA
#     # v[200] = -1.0
#     e = {}
#     for t in range(199):
#         e[(t, t + 1)] = 1.0
#     e[(0, 0)] = 0.5
#     e[(199, 199)] = 0.5
#
#     print("Training GSFA:")
#     n = GSFANode(output_dim=5)
#     n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
#     n.stop_training()
#
#     print("Training SFA:")
#     n_sfa = mdp.nodes.SFANode(output_dim=5)
#     n_sfa.train(x)
#     n_sfa.stop_training()
#
#     y = n.execute(x)
#     print("y[0]:", y[0])
#     print("y.mean:", y.mean(axis=0))
#     print("y.var:", (y**2).mean(axis=0))
#     y2 = n.execute(x2)
#
#     y_sfa = n_sfa.execute(x)
#     print("y_sfa[0]:", y_sfa[0])
#     print("y_sfa.mean:", y_sfa.mean(axis=0))
#     print("y_sfa.var:", (y_sfa**2).mean(axis=0))
#     y2_sfa = n_sfa.execute(x2)
#
#     signs_sfa = numpy.sign(y_sfa[0,:])
#     signs_gsfa = numpy.sign(y[0,:])
#     y = y * signs_gsfa * signs_sfa
#     y2 = y2 * signs_gsfa * signs_sfa
#
#     assert ((y_sfa - y) == pytest.approx(0.0)).all()
#     assert ((y2_sfa - y2) == pytest.approx(0.0)).all()
#
#
# # FUTURE: Is it worth it to have so many methods? I guess the mirroring windows are enough, they have constant
# # node weights and the edge weights almost fulfill consistency
# def test_equivalence_window3_fwindow3():
#     """Tests the equivalence of slow and fast mirroring sliding windows for GSFA
#     """
#     x = numpy.random.normal(size=(200, 15))
#     training_modes = ("window3", "fwindow3")
#
#     delta_values = []
#     for training_mode in training_modes:
#         n = GSFANode(output_dim=5)
#         n.train(x, train_mode=training_mode)
#         n.stop_training()
#
#         y = n.execute(x)
#         delta = comp_delta(y)
#         # print("**Brute Delta Values of mode %s are: " % training_mode, delta)
#         delta_values.append(delta)
#
#     # print(delta_values)
#     assert ((delta_values[1] - delta_values[0]) == pytest.approx(0.0)).all()
#
#
# def test_equivalence_smirror_window3_mirror_window3():
#     """Tests the equivalence of slow and fast mirroring sliding windows for GSFA
#     """
#     x = numpy.random.normal(size=(200, 15))
#     training_modes = ("smirror_window3", "mirror_window3")
#
#     delta_values = []
#     for training_mode in training_modes:
#         n = GSFANode(output_dim=5)
#         n.train(x, train_mode=training_mode)
#         n.stop_training()
#
#         y = n.execute(x)
#         delta = comp_delta(y)
#         # print("**Brute Delta Values of mode %s are: " % training_mode, delta)
#         delta_values.append(delta)
#
#     # print(delta_values)
#     assert ((delta_values[1] - delta_values[0]) == pytest.approx(0.0)).all()
#
#
# def test_equivalence_smirror_window32_mirror_window32():
#     """Tests the equivalence of slow and fast mirroring sliding windows for GSFA
#     """
#     x = numpy.random.normal(size=(200, 15))
#     training_modes = ("smirror_window32", "mirror_window32")
#
#     delta_values = []
#     for training_mode in training_modes:
#         n = GSFANode(output_dim=5)
#         n.train(x, train_mode=training_mode)
#         n.stop_training()
#
#         y = n.execute(x)
#         delta = comp_delta(y)
#         # print("**Brute Delta Values of mode %s are: " % training_mode, delta)
#         delta_values.append(delta)
#
#     # print(delta_values)
#     assert ((delta_values[1] - delta_values[0]) == pytest.approx(0.0)).all()
