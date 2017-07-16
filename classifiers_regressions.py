# Contains basic metrics for classification and regression
# By Alberto Escalante: Alberto.Escalante@ini.rub.de First Version 3 August 2009
# Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy
import mdp
import patch_mdp


def mean_average_error(ground_truth, regression, verbose=False):
    """ Computes the mean average error (MAE).
     
    Args:    
        ground_truth (list or 1-dim ndarray): ground truth labels.
        regression (list or 1-dim ndarray): label estimations. Must have the same length as the ground truth.
        verbose (bool): verbosity parameter.
    Returns:
        (float): the MAE.
    """
    if len(ground_truth) != len(regression):
        ex = "ERROR in regression labels in mean_average_error:" + \
             "len(ground_truth)=%d != len(regression)=%d" % (len(ground_truth), len(regression))
        print ex
        raise Exception(ex)

    d1 = numpy.array(ground_truth).flatten()
    d2 = numpy.array(regression).flatten()
    if verbose:
        print "ground_truth=", d1
        print "regression=", d2
    mae = numpy.abs(d2 - d1).mean()
    return mae


def correct_classif_rate(ground_truth, classified, verbose=False):
    """ Computes the classification rate (i.e., fraction of samples correctly classified) 

    Args:    
        ground_truth (list or 1-dim ndarray): ground truth classes. Assumed to be integer.
        classified (list or 1-dim ndarray): class estimations. Assumed to be integer. Must have the 
                                            same length as the ground truth.
        verbose (bool): verbosity parameter.
    Returns:
        (float): the classification rate (i.e., number of successful classifications / total number of samples)
    """
    num = len(ground_truth)
    if len(ground_truth) != len(classified):
        ex = "ERROR in class sizes, in correct_classif_rate:" + \
             "len(ground_truth)=%d != len(classified)=%d" % (len(ground_truth), len(classified))
        raise Exception(ex)

    d1 = numpy.array(ground_truth, dtype="int")
    d2 = numpy.array(classified, dtype="int")
    if verbose:
        print "ground_truth=", d1
        print "classified=", d2
    return (d1 == d2).sum() * 1.0 / num


# The next code is obsolete and can be deleted
define_obsolete_code = False
if define_obsolete_code:
    class GaussianApprox(object):
        def __init__(self):
            self.n_classes = None
            self.labels = None
            self.vectorsM = None

        def train(self, data, labels=None):
            self.n_classes = data.shape[0]
            self.vectorsM = data.copy()
            if labels is None:
                self.labels = numpy.array(range(self.n_classes))
            else:
                if labels.shape[0] != self.n_classes:
                    ex = "Wrong number of labels: %d != %d n_classes" % (labels.shape[0], self.n_classes)
                    raise Exception(ex)
                self.labels = labels
                # print "vectorsM=", self.vectorsM

        def classify(self, data):
            diffs = data[:, :, numpy.newaxis].repeat(self.n_classes, 2).swapaxes(1, 2) - self.vectorsM
            square_distances = (diffs ** 2).sum(2)
            return square_distances.argmin(1)


    class ClosestDistanceClassifierNew(object):
        def __init__(self):
            self.n_classes = None
            self.labels = None
            self.vectorsM = None

        def train(self, data, labels=None):
            self.n_classes = data.shape[0]
            self.vectorsM = data.copy()
            if labels is None:
                self.labels = numpy.array(range(self.n_classes))
            else:
                if isinstance(labels, numpy.ndarray):
                    if labels.shape[0] != self.n_classes:
                        ex = "Wrong number of labels: %d != %d n_classes" % (labels.shape[0], self.n_classes)
                        raise Exception(ex)
                else:
                    if len(labels) != self.n_classes:
                        ex = "Wrong number of labels: %d != %d n_classes [2]" % (labels.shape[0], self.n_classes)
                        raise Exception(ex)
                    print "labels=", labels
                    labels = numpy.array(labels)
                self.labels = labels
                # print "vectorsM=", self.vectorsM

        def classifyNoMem(self, data):
            closest = numpy.zeros(len(data))
            for i, sample in enumerate(data):
                diffs = sample[:, numpy.newaxis].repeat(self.n_classes, axis=1).swapaxes(0, 1) - self.vectorsM
                square_distances = (diffs ** 2).sum(axis=1)
                closest[i] = square_distances.argmin()
            return self.labels[closest.astype('int')]


    class ClosestVectorClassifier(object):
        def __init__(self):
            self.n_classes = None
            self.vectorsM = None
            self.class_names = None

        def train(self, data):
            self.n_classes = data.shape[0]
            self.vectorsM = scipy.matrix(data).T
            if class_names is None:
                self.class_names = numpy.array(range(self.n_classes))
            else:
                if len(class_names) is not self.n_classes:
                    ex = "Wrong number of class names"
                    raise Exception(ex)
                self.class_names = class_names
            print "vectorsM=", self.vectorsM

        def classify(self, data):
            dataM = numpy.matrix(data)
            print "dataM=", dataM

            prod = dataM * self.vectorsM  # alternative: use numpy.dot, with 2D arrays
            print "prod=", prod
            result = prod.argmax(axis=1)
            return result


    def blocker_computer(data, labels, classes=None, block_size=1, spacing=None):
        """Learns vectors used by a classifier (one vector per class)."""
        if spacing is None:
            spacing = block_size

        if classes is None:
            classes_copy = numpy.arange(data.shape[0])
        else:
            classes_copy = classes

        n_means = int((data.shape[0] - block_size + spacing) / spacing)

        if (data.shape[0] - block_size + spacing) % spacing != 0:
            print "Warning, not exact signal sizes in means_computer"

        print "data.shape=", data.shape, "n_means is ", n_means

        input_dim = data.shape[1]
        means = numpy.zeros((n_means, input_dim))
        blocks = numpy.zeros((n_means, block_size, input_dim))
        b_labels = numpy.zeros((n_means, block_size))
        b_classes = numpy.zeros((n_means, block_size))

        if labels is None:
            labels_copy = numpy.arange(data.shape[0])
        else:
            labels_copy = labels

        print "b_labels.shape is ", b_labels.shape, "labels_copy.shape is ", labels_copy.shape
        for i in range(n_means):
            blocks[i, :, :] = data[i * spacing:i * spacing + block_size].copy()
            #        print "labels_copy[****].shape is ", labels_copy[i*spacing:i*spacing+block_size].shape

            b_labels[i] = labels_copy[i * spacing:i * spacing + block_size].copy()
            b_classes[i] = classes_copy[i * spacing:i * spacing + block_size].copy()
            means[i] = blocks[i].mean(axis=0)

        return means, blocks, b_labels, b_classes


    # This needs serious improvements!!!!
    # No need for this mixed classifier, split into two normal classifiers
    class Simple_2_Stage_Classifier(object):
        """This is a composed classifier designed with SFA signals in mind
        (however, it must be improved)
        the input samples (array data) can be divided in blocks of block_size rows, 
        in which each block belongs to the same class
        In addition a label (e.g., a real value) can be assigned to every sample
        """
        def __init__(self):
            self.input_dim = None
            self.block_size = None
            self.set_classes = None
            self.class_index = None
            self.GC_L0 = None
            self.CDC_L0 = None
            self.blocks_L0 = None
            self.classes_L0 = None
            self.labels_L0 = None
            self.means_L0 = None
            self.CDC_nodes_L1 = None
            self.average_Labels_L0 = None
            self.average_Labels_L0_class = None

        def train(self, data, labels=None, classes=None, block_size=None, spacing=None):
            """  A Gaussian classifier and a (possible two step) closest center classified are trained
            """
            # print "training data: ", data
            self.input_dim = data.shape[1]
            self.block_size = block_size

            if classes is None:
                classes = numpy.arange(data.shape[0]) / block_size
                num_classes = int(numpy.ceil(data.shape[0] * 1.0 / block_size))
                print "No classes? Fine???!"
                quit()
            else:
                num_classes = len(set(classes))

            self.set_classes = list(set(classes))
            self.set_classes.sort()
            print "list of classes=", self.set_classes

            # CDC
            self.CDC_L0 = ClosestDistanceClassifier()

            if isinstance(block_size, int):
                self.means_L0, self.blocks_L0, self.labels_L0, self.classes_L0 = blocker_computer(data, labels=labels,
                                                                                                  classes=classes,
                                                                                                  block_size=block_size,
                                                                                                  spacing=spacing)
            elif isinstance(block_size, (list, numpy.ndarray)) or (block_size is None):
                self.means_L0 = numpy.zeros((num_classes, data.shape[1]))
                self.blocks_L0 = []
                self.labels_L0 = []
                self.classes_L0 = []
                # numpy.zeros((num_classes, data.shape[1]))
                #            self.labels_L0 = numpy.zeros(num_classes)
                for class_number, class_value in enumerate(self.set_classes):  # set(classes):
                    self.means_L0[class_number] = data[classes == class_value].mean(axis=0)
                    self.blocks_L0.append(data[classes == class_value, :])
                    self.labels_L0.append(labels[classes == class_value])
                    self.classes_L0.append(classes[classes == class_value])
                    #        print self.means_L0.shape, "(shape) of means=", self.means_L0
                    #        print self.labels_L0.shape, " (shape) of labels=", self.labels_L0
            self.CDC_L0.train(self.means_L0, self.labels_L0)

            if isinstance(self.classes_L0, numpy.ndarray):
                self.CDC_L0.classes = self.classes_L0[:, 0]  # take first class of each block as representative
            else:
                self.CDC_L0.classes = numpy.array(self.set_classes)

            if block_size is not None:
                self.GC_L0 = mdp.nodes.GaussianClassifierNode()

                if isinstance(block_size, int):
                    if data.shape[0] % block_size != 0:
                        print "Warning, training data is not a multiple of block_size"
                        quit()

                self.GC_L0.train(data, classes)
                self.GC_L0.stop_training()
                #           This assumes that classes go from 0 to C-1, for exactly C classes
                self.average_Labels_L0_class = {}
                for class_value in self.set_classes:
                    self.average_Labels_L0_class[class_value] = labels[classes == class_value].mean()
                    # print "average label class[", class_value, "]=", self.average_Labels_L0_class[class_value]

                self.class_index = {}
                self.average_Labels_L0 = numpy.zeros(num_classes)

                for index, class_value in enumerate(self.set_classes):
                    self.average_Labels_L0[index] = self.average_Labels_L0_class[class_value]
                    self.class_index[class_value] = index
                    #                print "average label[", index, "]=", self.average_Labels_L0[index]

            if block_size is not None:
                self.CDC_nodes_L1 = range(self.means_L0.shape[0])
                for i in range(self.means_L0.shape[0]):
                    self.CDC_nodes_L1[i] = ClosestDistanceClassifier()
                    self.CDC_nodes_L1[i].train(self.blocks_L0[i], self.labels_L0[i])

        def classifyCDC(self, data):
            """  Classify according to the closest center classifier
            """
            #        classes = numpy.zeros(data.shape[0])
            labels = numpy.zeros(data.shape[0])
            classes = numpy.zeros(data.shape[0])
            for i in range(data.shape[0]):
                xx = data[i].reshape((1, self.input_dim))
                cx = self.CDC_L0.classify(xx)
                classes[i] = self.CDC_L0.classes[cx]

                if self.block_size is not None:
                    c_L1 = self.CDC_nodes_L1[cx].classify(xx)
                    labels[i] = self.CDC_nodes_L1[cx].labels[c_L1]
                else:
                    labels[i] = self.CDC_L0.labels[cx]
                    # print xx, " classified to ", cx, " subclass ", c_L1, " with label: ",
                    # self.CDC_nodes_L1[cx].labels[c_L1]
            return classes, labels

        def classifyGaussian(self, data):
            """  Classify according to the Gaussian classifier
            """
            print "Warning, this function works for classes, "
            print "but it looks strange for labels because it also uses closest distance classifier, check this"
            #        classes = numpy.zeros(data.shape[0])
            labels = numpy.zeros(data.shape[0]) - 10
            classes = self.GC_L0.classify(data)
            return classes, labels

        def GaussianRegression(self, data):
            """  Use the class probabilities to generate a better label
            If the computation of the class probabilities were perfect,
            and if the Gaussians were just a delta, then the output value
            minimizes the squared error 
            """
            probabilities = self.GC_L0.class_probabilities(data)

            value = numpy.dot(probabilities, self.average_Labels_L0)
            return value


    class ClosestDistanceClassifier(object):
        def __init__(self):
            self.n_classes = None
            self.vectorsM = None
            self.labels = None

        def train(self, data, labels=None):
            self.n_classes = data.shape[0]
            self.vectorsM = data.copy()
            if labels is None:
                self.labels = numpy.array(range(self.n_classes))
            else:
                if isinstance(labels, numpy.ndarray):
                    if labels.shape[0] != self.n_classes:
                        ex = "Wrong number of labels: %d != %d n_classes" % (labels.shape[0], self.n_classes)
                        raise Exception(ex)
                else:
                    if len(labels) != self.n_classes:
                        ex = "Wrong number of labels: %d != %d n_classes [2]" % (labels.shape[0], self.n_classes)
                        raise Exception(ex)
                self.labels = labels
                # print "vectorsM=", self.vectorsM

        def classify(self, data):
            diffs = data[:, :, numpy.newaxis].repeat(self.n_classes, axis=2).swapaxes(1, 2) - self.vectorsM
            # diffs = data[:,numpy.newaxis,:].repeat(self.n_classes, axis=1) - self.vectorsM
            square_distances = (diffs ** 2).sum(axis=2)
            return square_distances.argmin(axis=1)
