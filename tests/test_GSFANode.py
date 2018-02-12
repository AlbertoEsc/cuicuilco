#####################################################################################################################
# test_GSFANode: Tests for the Graph-Based SFA Node (GSFANode) as defined by the Cuicuilco framework                #
#                                                                                                                   #
# By Alberto Escalante. Alberto.Escalante@ini.rub.de                                                                #
# Ruhr-University Bochum, Institute for Neural Computation, Group of Prof. Dr. Wiskott                              #
#####################################################################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy

import pytest
import mdp

from cuicuilco.gsfa_node import GSFANode, graph_delta_values, comp_delta


def atest_GSFA_zero_mean_unit_variance_graph():
    """ Test of GSFA for zero-mean unit variance constraints on random data and graph, edge dictionary mode
    """
    x = numpy.random.normal(size=(200, 15))
    v = numpy.ones(200)
    e = {}
    for i in range(1500):
        n1 = numpy.random.randint(200)
        n2 = numpy.random.randint(200)
        e[(n1, n2)] = numpy.random.normal() + 1.0
    n = GSFANode(output_dim=5)
    n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()

    y = n.execute(x)
    assert (y.mean(axis=0) == pytest.approx(0.0)).all()
    assert ((y**2).mean(axis=0) == pytest.approx(1.0)).all()


def atest_basic_GSFA_edge_dict():
    """ Basic test of GSFA on random data and graph, edge dictionary mode
    """
    x = numpy.random.normal(size=(200, 15))
    v = numpy.ones(200)
    e = {}
    for i in range(1500):
        n1 = numpy.random.randint(200)
        n2 = numpy.random.randint(200)
        e[(n1, n2)] = numpy.random.normal() + 1.0
    n = GSFANode(output_dim=5)
    n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()

    y = n.execute(x)
    delta_values_training_data = graph_delta_values(y, e)

    x2 = numpy.random.normal(size=(200, 15))
    y2 = n.execute(x2)
    y2 = y2 - y2.mean(axis=0)  # enforce zero mean
    y2 /= ((y2**2).mean(axis=0) ** 0.5)  # enforce zero-mean
    # print("y2 means:", y2.mean(axis=0))
    # print("y2 std:", (y2**2).mean(axis=0))

    delta_values_test_data = graph_delta_values(y2, e)
    assert (delta_values_training_data < delta_values_test_data).all()
    # print("Graph delta values of training data", graph_delta_values(y, e))
    # print("Graph delta values of test data (should be larger than for training)", graph_delta_values(y2, e))


# SUMMARY:
# Mirroring windows work fine in slow and fast versions
# Truncating window does not work in optimized version, I need to clarify the algebraic optimization
# Plain Sliding window (node-weight adjusting) seems to be broken in optimized version.
# Is it worth it to have so many methods? I guess the mirroring windows are enough, they have constant node weights
# and edge weights almost fulfill consistency


def test_equivalence_SFA_GSFA_linear_graph():
    """ Tests the equivalence of Standard SFA and GSFA when trained using an appropriate linear graph (graph mode)
    """
    x = numpy.random.normal(size=(200, 15))
    x2 = numpy.random.normal(size=(200, 15))

    v = numpy.ones(200)
    e = {}
    for t in range(199):
        e[(t, t + 1)] = 1.0
    e[(0, 0)] = 0.5
    e[(199, 199)] = 0.5
    print("Training GSFA:")
    n = GSFANode(output_dim=5)
    n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()

    print("Training SFA:")
    n_sfa = mdp.nodes.SFANode(output_dim=5)
    n_sfa.train(x)
    n_sfa.stop_training()

    y = n.execute(x)
    print("y[0]:", y[0])
    print("y.mean:", y.mean(axis=0))
    print("y.var:", (y**2).mean(axis=0))
    y2 = n.execute(x2)

    y_sfa = n_sfa.execute(x)
    print("y_sfa[0]:", y_sfa[0])
    print("y_sfa.mean:", y_sfa.mean(axis=0))
    print("y_sfa.var:", (y_sfa**2).mean(axis=0))
    y2_sfa = n_sfa.execute(x2)

    signs_sfa = numpy.sign(y_sfa[0,:])
    signs_gsfa = numpy.sign(y[0,:])
    y = y * signs_gsfa * signs_sfa
    y2 = y2 * signs_gsfa * signs_sfa

    assert ((y_sfa - y) == pytest.approx(0.0)).all()
    assert ((y2_sfa - y2) == pytest.approx(0.0)).all()


def atest_fast_windows():
    print("")
    print("***********************************************************************")
    print("Testing equivalence of slow and fast mirroring sliding windows for GSFA")
    x = numpy.random.normal(size=(200, 15))

    training_modes = ("window3", "fwindow3", "smirror_window3", "mirror_window3")
    # Other training modes: "truncate_window3", "ftruncate_window3")
    #    training_modes = ("truncate_window3", "ftruncate_window3",)
    #    training_modes = ("smirror_window32", "mirror_window32") #Test passed

    delta_values = []
    for training_mode in training_modes:
        n = GSFANode(output_dim=5)
        n.train(x, train_mode=training_mode)
        n.stop_training()

        y = n.execute(x)
        delta = comp_delta(y)
        print("**Brute Delta Values of mode %s are: " % training_mode, delta)
        delta_values.append(delta)

    print(delta_values)
    # quit()


def atest_continuous_edge_weights():
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


    #basic_test_GSFA_edge_dict()
    #test_equivalence_SFA_GSFA_linear_graph()
    #test_fast_windows()
    #for experiment_number in range(0, 12):
    #    test_pathological_outputs(experiment=experiment_number)
    #test_continuous_edge_weights()