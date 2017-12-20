#####################################################################################################################
# exact_label_learning: This module implements methods for the analysis and construction of training graphs for     #
#                       GSFA and further extensions (e.g., iGSFA, HiGSFA). It is part of the Cuicuilco framework.   #
#                                                                                                                   #
# The module contains functions needed to compute ELL graphs as well as to compute optimal free responses of        #
# arbitrary training-graphs.                                                                                        #
#                                                                                                                   #
# See the following publication for details: Escalante-B AN, Wiskott L, "Theoretical analysis of the optimal        #
# free responses of graph-based SFA for the design of training graphs", Journal of Machine Learning Research,       #
# 17(157):1--36, 2016                                                                                               #
#                                                                                                                   #
# By Alberto Escalante. Alberto.Escalante@ini.ruhr-uni-bochum.de                                                    #
# Ruhr-University-Bochum, Institute for Neural Computation, Group of Prof. Dr. Wiskott                              #
#####################################################################################################################

from __future__ import print_function
import numpy

#####################################################################################################################
# ########################### FUNCTIONS FOR CONSTRUCTING PREDEFINED TRAINING GRAPHS #################################
#####################################################################################################################


def add_edge(edge_weights, n1, n2, w):
    """ Adds a (possibly new) weighted edge to an edge_weight dictionary. """
    if (n1, n2) in edge_weights.keys():
        edge_weights[(n1, n2)] += w
    else:
        edge_weights[(n1, n2)] = w


def GenerateLinearGraphConsistent(num_samples):
    """ Computes a linear graph compatible with SFA.

    Self-loops are added to first and last samples for consistency constraints.
    """
    N = num_samples
    node_weights = numpy.ones(N)
    edge_weights = {}
    for n in range(N - 1):
        edge_weights[(n, n + 1)] = 1
        edge_weights[(n + 1, n)] = 1
    edge_weights[(0, 0)] = 1
    edge_weights[(N - 1, N - 1)] = 1
    return node_weights, edge_weights


def GenerateClusteredGraph(num_samples_per_cluster):
    """ Generates a (consistent) clustered graph without self-loops. """
    N = num_samples_per_cluster.sum()
    node_weights = numpy.ones(N)
    current_sample = 0
    edge_weights = {}
    for cluster_size in num_samples_per_cluster:
        w = 1.0 / (cluster_size - 1)
        for i in range(cluster_size):
            for j in range(i + 1, cluster_size):
                edge_weights[(i + current_sample, j + current_sample)] = w
                edge_weights[(j + current_sample, i + current_sample)] = w
        current_sample += cluster_size

    return node_weights, edge_weights


def GenerateCompactClassesGraph(label_block_size, J):
    """ Creates a graph for dense classification features.

    The number of classes must be C = 2^J. The labels computed are binary (0 or 1) and it includes no self-loops.
    """
    num_classes = 2 ** J
    N = label_block_size * num_classes
    labels = numpy.zeros((N, J))
    for j in range(J):
        labels[:, j] = numpy.arange(N) / label_block_size / (2 ** (J - j - 1)) % 2

    node_weights = numpy.ones(N)
    all_nodes = numpy.arange(N)
    edge_weights = {}
    for j in range(J):
        w = 1.0 / (N / 2 - 1)
        c10 = all_nodes[labels[:, j] == 1]
        print("c10=", c10)
        for n1 in c10:
            for n2 in c10:
                if n1 != n2:
                    add_edge(edge_weights, n1, n2, w)
        c10 = all_nodes[labels[:, j] == 0]
        print("c10=", c10)
        for n1 in c10:
            for n2 in c10:
                if n1 != n2:
                    add_edge(edge_weights, n1, n2, w)
    return node_weights, edge_weights


def GenerateSerialGraph(num_samples, block_size):
    """ Generates a (consistent) serial graph. """
    N = num_samples
    num_blocks = N / block_size

    if N % block_size != 0:
        err = "num_samples(%d) must be a multiple of block_size (%d)" % (num_samples, block_size)
        raise Exception(err)

    if num_blocks < 2:
        err = "the number of blocks %d should be at least 2 (%d/%d)" % (num_blocks, num_samples, block_size)
        raise Exception(err)

    node_weights = numpy.ones(N) * 2.0
    node_weights[:block_size] = 1.0
    node_weights[-block_size:] = 1.0

    edge_weights = {}
    w = 1.0
    for block in range(num_blocks - 1):
        for i in range(block_size):
            for j in range(block_size):
                edge_weights[(i + block * block_size, j + (block + 1) * block_size)] = w
                edge_weights[(j + (block + 1) * block_size, i + block * block_size)] = w  # Loops are simply overwritten
    return node_weights, edge_weights


def GenerateMixedGraph(num_samples, block_size):
    """ Generates a (consistent) mixed graph. """
    N = num_samples
    num_blocks = N / block_size

    if N % block_size != 0:
        err = "num_samples(%d) must be a multiple of block_size (%d)" % (num_samples, block_size)
        raise Exception(err)

    if num_blocks < 2:
        err = "the number of blocks %d should be at least 2 (%d/%d)" % (num_blocks, num_samples, block_size)
        raise Exception(err)

    node_weights = numpy.ones(N) * 1.0

    edge_weights = {}
    # Inter-group connections
    for block in range(num_blocks - 1):
        w = 1.0
        for i in range(block_size):
            for j in range(block_size):
                edge_weights[(i + block * block_size, j + (block + 1) * block_size)] = w
                edge_weights[(j + (block + 1) * block_size, i + block * block_size)] = w  # Loops are simply overwritten

    # Inter-group connections
    # First add the edges of the first and last blocks
    for block in [0, num_blocks - 1]:
        w = 2.0
        for i in range(block_size):
            for j in range(block_size):
                edge_weights[(i + block * block_size, j + block * block_size)] = w
                edge_weights[(j + block * block_size, i + block * block_size)] = w  # Loops are simply overwritten
    # Now add the edges of intermediate blocks
    for block in range(1, num_blocks - 1):
        w = 1.0
        for i in range(block_size):
            for j in range(block_size):
                edge_weights[(i + block * block_size, j + block * block_size)] = w
                edge_weights[(j + block * block_size, i + block * block_size)] = w  # Loops are simply overwritten

    return node_weights, edge_weights


def GenerateSlidingWindowConsistentGraph(num_samples, d):
    """ Generates a (consistent) sliding window graph.

    Self-loops added in the end to fix consistency.
    """
    N = num_samples

    # w = 1.0 / (2 * d)

    node_weights = numpy.ones(N)

    edge_weights = {}
    for i in range(1, N + 1):  # here i and j go from 1 to N, but the graph uses indices 0 to N-1
        L = list(range(1, i))
        L.extend(range(i + 1, N + 1))
        for j in L:
            # w = 0.0
            if i + j <= d + 1 or i + j >= 2 * N - 1:
                w = 2.0 / (2 * d)
                edge_weights[(i - 1, j - 1)] = w
                edge_weights[(j - 1, i - 1)] = w
            elif numpy.abs(j - i) <= d and i + j > d + 1 and i + j < 2 * N - 1:
                w = 1.0 / (2 * d)
                edge_weights[(i - 1, j - 1)] = w
                edge_weights[(j - 1, i - 1)] = w

    connectivity = numpy.zeros(N)
    for (i, j) in edge_weights.keys():
        connectivity[i] += edge_weights[(i, j)]

    print("Connectivities are:", connectivity)
    for i in range(N):
        if connectivity[i] < 1.0:
            edge_weights[(i, i)] = 1 - connectivity[i]

    connectivity = numpy.zeros(N)
    for (i, j) in edge_weights.keys():
        connectivity[i] += edge_weights[(i, j)]

    print("Connectivities are:", connectivity)
    for i in range(N):
        if connectivity[i] < 1.0:
            edge_weights[(i, i)] = 1 - connectivity[i]

    return node_weights, edge_weights


def GenerateImprovedGraph(num_samples, block_size, balance_factor=1.0):
    """ Generates a graph useful for semi-supervised learning.

    One attempt at semi-supervised learning.
    """
    N = num_samples
    num_blocks = N / block_size

    if N % block_size != 0:
        err = "num_samples(%d) must be a multiple of block_size (%d)" % (num_samples, block_size)
        raise Exception(err)

    if num_blocks < 2:
        err = "the number of blocks %d should be at least 2 (%d/%d)" % (num_blocks, num_samples, block_size)
        raise Exception(err)

    node_weights = numpy.ones(N) * 1.0

    edge_weights = {}
    # Inter-group connections
    for block in range(num_blocks - 1):
        w = 1.0
        for i in range(block_size):
            for j in range(block_size):
                edge_weights[(i + block * block_size, j + (block + 1) * block_size)] = w
                edge_weights[(j + (block + 1) * block_size, i + block * block_size)] = w

    # Inter-group connections
    # First and last blocks
    for block in [0, num_blocks - 1]:
        c = block_size * 2.0 + (block_size - 1) * balance_factor
        w = (c - block_size) / (block_size - 1)
        for i in range(block_size):
            L = list(range(0, i))
            L.extend(range(i + 1, block_size))
            for j in L:
                edge_weights[(i + block * block_size, j + block * block_size)] = w
                edge_weights[(j + block * block_size, i + block * block_size)] = w  # Loops are omitted
    # Intermediate blocks
    for block in range(1, num_blocks - 1):
        w = 1.0 * balance_factor
        for i in range(block_size):
            L = list(range(0, i))
            L.extend(range(i + 1, block_size))
            for j in L:
                edge_weights[(i + block * block_size, j + block * block_size)] = w
                edge_weights[(j + block * block_size, i + block * block_size)] = w  # Loops are omitted

    return node_weights, edge_weights


def scale_edge_weights(edge_weights, w):
    """ Scales all the edge-weights described by a dictionary. """
    edge_weights2 = {}
    for (n1, n2) in edge_weights.keys():
        edge_weights2[(n1, n2)] = w * edge_weights[(n1, n2)]
    return edge_weights2


def add_unsupervised_samples(node_weights1, edge_weights1, N2, w1=1.0, w2=1.0):
    """ Yet another attempt for semi-supervised learning. """
    N1 = len(node_weights1)
    N = N1 + N2

    Q1 = ComputeQ(node_weights1)
    R1 = ComputeR(edge_weights1)
    w1 = Q1 / R1 * 0.75

    edge_weights2 = scale_edge_weights(edge_weights1, w1)
    w2 = Q1 * 0.25 / (N1 * N2)

    node_weights2 = w2 * N1 * numpy.ones(N)
    node_weights2[0:N1] = node_weights1

    for n2 in range(N1, N):  # element of unlabeled set
        for n1 in range(N1):  # element of labeled set
            add_edge(edge_weights2, n1, n2, w2)
            add_edge(edge_weights2, n2, n1, w2)

    return node_weights2, edge_weights2


#####################################################################################################################
# ################################################### FUNCTIONS FOR GRAPH ANALYSIS ##################################
#####################################################################################################################
def IsGraphConsistent(node_weights, edge_weights):
    """ Verifies whether a graph fulfills the consistency constraints. """
    Q = ComputeQ(node_weights)
    R = ComputeR(edge_weights)
    N = len(node_weights)
    c_out = numpy.zeros(N)
    c_in = numpy.zeros(N)

    for (n, np) in edge_weights.keys():
        w = edge_weights[(n, np)]
        c_out[n] += w
        c_in[np] += w

    if (node_weights / Q - c_out / R > 0.000001).any():
        print("Graph is not consistent! node_weights/Q=", node_weights / Q, " but c_out/R=", c_out / R)
        return 0
    if (node_weights / Q - c_in / R > 0.000001).any():
        print("Graph is not consistent! node_weights/Q=", node_weights / Q, " but c_in/R=", c_out / R)
        return 0
    return 1


def ExtractGammaFromGraph(node_weights, edge_weights):
    """ Creates matrix Gamma from a given edge-weight dictionary. """
    N = len(node_weights)
    Gamma = numpy.zeros((N, N))
    for (i, j) in edge_weights.keys():
        Gamma[i, j] = edge_weights[(i, j)]
    return Gamma


def ComputeSymmetricGamma(Gamma):
    """ Computes a symmetric Gamma matrix given an arbitrary Gamma matrix. """
    return 0.5 * (Gamma + Gamma.T)


def ComputeMFromGamma(node_weights, Gamma):
    """ Computes matrix M from matrix Gamma. """
    D_m12 = numpy.diag(node_weights ** -0.5)
    return numpy.dot(D_m12, numpy.dot(Gamma, D_m12))


def ComputeGammaFromM(node_weights, M):
    """ Computes matrix Gamma from matrix M. """
    # The following code might be correct, but it is too slow!!!
    # D_12 = numpy.diag(node_weights**0.5)
    # return numpy.dot(D_12, numpy.dot(M, D_12))
    v_12_row = (node_weights ** 0.5).reshape((1, -1))
    v_12_col = (node_weights ** 0.5).reshape((-1, 1))
    return v_12_col * M * v_12_row


def ExtractEigenvectors(M):
    """ Computes the eigenvectors of M, ordered by increasing eigenvalues.

    The sign of the eigenvectors is fixed so that their first component is negative.
    """
    l, eigv = numpy.linalg.eigh(M)

    signs = numpy.sign(eigv[0, :])
    signs[signs == 0] = 1.0
    eigv = eigv * signs * -1

    order = numpy.argsort(l)
    l2 = l[order]
    eigv2 = eigv[:, order]
    return l2, eigv2


def ComputeQ(node_weights):
    """ Computes the value of Q (sum of node weights) given the node weights of a graph. """
    return node_weights.sum()


def ComputeR(edge_weights):
    """ Computes the value of R (sum of edge weights) given the edge weights of a graph. """
    sume = 0
    for (i, j) in edge_weights.keys():
        sume += edge_weights[(i, j)]
    return sume


def ComputeFreeResponses(eigv, node_weights):
    """ Maps eigenvectors to the corresponding free responses. """
    Q = ComputeQ(node_weights)
    node_weights_m12 = numpy.diag(node_weights ** -0.5)
    return (Q ** 0.5) * numpy.dot(node_weights_m12, eigv)


def Extract_Deltas_from_Eigenvalues(l, Q, R):
    """ Maps eigenvalues to delta values. """
    return 2 - 2.0 / R * l * Q  # error here!!!????? mixing of two features, both with D=0


# TODO: Check this function, there might be an easier way to compute this
def EnforceEigenvectorConst(l, eigv, node_weights):
    """  Ensures the last eigenvector corresponds to constant free response,
    and decorrelates it from all other eigenvectors.
    """
    Q = ComputeQ(node_weights)
    s = Q ** -0.5 * node_weights ** 0.5  # Assuring s^T s = Q/Q = 1
    #    energy_s = numpy.dot(s, s)

    print("s=", s)
    print("eigvT=", eigv)

    constant_added = False
    for j in range(eigv.shape[1])[::-1]:
        prod = numpy.dot(s, eigv[:, j])
        if numpy.abs(prod) > 0.000001 and not constant_added:
            print("Setting eigenvector %d as constant" % j)
            eigv[:, j] = s
            constant_added = True
    print("Constant added: eigvT=", eigv)

    for j in range(eigv.shape[1])[::-1]:
        energy_j = numpy.dot(eigv[:, j], eigv[:, j])
        if energy_j < 0.0000001:
            energy_j = 1.0  # Prevent case when energy is 0, check if this is a bug
        eigv[:, j] = eigv[:, j] / energy_j ** 0.5
        for k in range(1, j):
            prod = numpy.dot(eigv[:, j], eigv[:, k])
            if numpy.abs(prod) > 0.000001:
                eigv[:, k] = eigv[:, k] - eigv[:, j] * prod
    print("Re-orthogonalization: eigvT=", eigv)

    return l, eigv


# Notation: y[:,j] is the jth feature
def ComputeDeltasFromGraph(y, edge_weights):
    """ Computes the delta values of a feature representation 'y' given the edge weights of the graph. """
    R = ComputeR(edge_weights)
    deltas = numpy.zeros(y.shape[1])
    for (i, j) in edge_weights.keys():
        deltas += edge_weights[i, j] * (y[j] - y[i]) ** 2
    return deltas / R


def RemoveNegativeEdgeWeights(node_weights, Gamma):
    """ This method transforms a graph with edge weights Gamma into an equivalent graph
    with non-negative edge weights.
    """
    N = len(node_weights)
    m = Gamma.min()
    if m < 0:
        print("m=", m)
        print("node_weights=", node_weights)
        # correct_but_slow:
        # node_weights_m1 = numpy.diag(1.0/node_weights)
        # c = -1 * numpy.dot(node_weights_m1, numpy.dot(Gamma, node_weights_m1)).min()
        node_weights_m1_row = (1.0 / node_weights).reshape((1, -1))
        node_weights_m1_col = (1.0 / node_weights).reshape((-1, 1))
        c = -1 * (node_weights_m1_col * Gamma * node_weights_m1_row).min()

        print("c=", c)
        Z = numpy.dot(node_weights.reshape((N, 1)), node_weights.reshape((1, N)))
        print("Z=", Z)
        Q = node_weights.sum()
        R = Gamma.sum()
        print("R=", R)
        b = c * Q ** 2 / R
        Gamma = (Gamma + c * Z) / (1.0 + b)
    else:
        print("Graph has already non-negative weights")
    return Gamma


#####################################################################################################################
# ################################################ FUNCTIONS FOR GRAPH CREATION ######################################
#####################################################################################################################

def MapLabelsToFreeResponses(labels, node_weights):
    """ Transforms a set of arbitrary labels to a feasible set of free responses.

    The weighted mean, weighted variance, and weighted decorrelation constrains are enforced.
    """
    N = len(node_weights)
    J = labels.shape[1]

    Q = ComputeQ(node_weights)
    print("Q=", Q)
    # node_weights_12 = node_weights ** 0.5

    free_responses = numpy.zeros((N, J + 1))  # Response zero is constant

    free_responses[:, 0] = 1.0  # TODO: shouldn't this be 1.0??? was 1.0/Q
    for j in range(J):
        free_responses[:, j + 1] = labels[:, j]

    # Decorrelate the labels and normalize their amplitude
    for j in range(J + 1):
        wstd_free_response = numpy.sqrt(numpy.dot(free_responses[:, j] * node_weights, free_responses[:, j]) / Q)
        free_responses[:, j] = free_responses[:, j] / wstd_free_response
        # norm_free_response = numpy.sqrt(numpy.dot(free_responses[:,j], free_responses[:,j]))
        for k in range(j + 1, J + 1):
            proj = numpy.dot(free_responses[:, k] * node_weights, free_responses[:, j]) / Q
            # free_responses[:,k] = free_responses[:,k] -
            # numpy.dot(free_responses[:,k], free_responses[:,j]) * free_responses[:,j] / norm_free_response**2
            free_responses[:, k] = free_responses[:, k] - proj * free_responses[:, j]

    print("Free responses:", free_responses)
    for j in range(J + 1):
        print("l'[%d]^T D(v) l'[%d]= " % (j, j), numpy.dot(free_responses[:, j] * node_weights, free_responses[:, j]))

    return free_responses


def MapFreeResponsesToEigenvectors(free_responses, node_weights):
    """ Given a set of free responses the corresponding eigenvectors are computed. """
    J = free_responses.shape[1]

    Q = ComputeQ(node_weights)
    eigenvectors = numpy.zeros(free_responses.shape)

    for j in range(J):
        eigenvectors[:, j] = free_responses[:, j] * node_weights ** 0.5 / Q ** 0.5

    print("verifying eigenvectors have unit norm")
    for j in range(J):
        print("Norm ev[%d] is %f" % (j, numpy.dot(eigenvectors[:, j], eigenvectors[:, j])))
    return eigenvectors


def ConstructMFromEigenvectors(eigenvectors, deltas, Q, R):
    """ Matrix M is computed given the corresponding eigenvectors. """
    N, J = eigenvectors.shape
    M = numpy.zeros((N, N))

    eigenvalues = (2.0 - deltas) * R / (2 * Q)
    print("eigenvalues being used:", eigenvalues)
    print("assuming given eigenvectors have unit norm")
    for j in range(J):
        term = eigenvalues[j] * numpy.dot(eigenvectors[:, j:j + 1], eigenvectors[:, j:j + 1].T)
        print("adding term: ", term)
        M += term
    return M


def ConstructGammaFromLabels(labels, node_weights, constant_deltas=False):
    """ A matrix Gamma is computed that encodes the given arbitrary labels. """
    free_responses = MapLabelsToFreeResponses(labels, node_weights)
    eigenvectors = MapFreeResponsesToEigenvectors(free_responses, node_weights)
    J = eigenvectors.shape[1]

    Q = node_weights.sum()
    R = Q

    deltas = numpy.ones(J) * 1.0  # why 1.0 here??? anyways it is overwritten
    if constant_deltas:
        deltas[0:J] = 0.0
    else:
        deltas[0:J] = 2.0 * numpy.arange(J) / J
    print("deltas=", deltas)
    M = ConstructMFromEigenvectors(eigenvectors, deltas, Q, R)
    Gamma = ComputeGammaFromM(node_weights, M)
    print("Gamma computed")
    return Gamma


def MapGammaToEdgeWeights(Gamma):
    """ Computes an edge weights dictionary given the matrix representation Gamma. """
    N = Gamma.shape[0]
    edge_weights = {}
    for i in range(N):
        for j in range(N):
            if Gamma[i, j] != 0:
                edge_weights[(i, j)] = Gamma[i, j]
    return edge_weights
