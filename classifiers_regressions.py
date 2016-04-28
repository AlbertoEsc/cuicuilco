#Basic Classes useful for classifiying SFA output signals
#By Alberto Escalante, strongly based on/reusing code by Niko Wilbert
# Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 3 August 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy
import mdp
import patch_mdp

def mean_average_error(ground_truth, regression, verbose=False):
    """Computes a float value indicating the mean average error
        both input lists/arrays must have the same length and float values
    """
    num = len(ground_truth)
    if len(ground_truth) != len(regression):
        ex = "ERROR in regression labels in mean_average_error: len(ground_truth)=%d != len(regression)=%d"%(len(ground_truth), len(regression))
        print ex
        raise Exception(ex)

    d1 = numpy.array(ground_truth).flatten()
    d2 = numpy.array(regression).flatten()
    if verbose:
        print "ground_truth=", d1
        print "regression=", d2
    mae = numpy.abs(d2-d1).mean()
    return mae

def remove_nans_from_array(array):
    ex = "Function obsolete, use the function array.nan_to_num"
    raise Exception(ex)
    array[numpy.isnan(array)] = 0

#This code is not being used... send to prehistoric file!!!
class GaussianApprox(object):
    def train(self, data, labels=None):
        self.n_classes = data.shape[0]
        self.vectorsM = data.copy()
        if labels is None:
            self.labels = numpy.array(range(self.n_classes))
        else:
            if labels.shape[0] != self.n_classes:
                ex = "Wrong number of labels: %d != %d n_classes"%(labels.shape[0], self.n_classes)
                raise Exception(ex)
            self.labels = labels

#       print "vectorsM=", self.vectorsM
    def classify(self, data):
        diffs = data[:,:,numpy.newaxis].repeat(self.n_classes, 2).swapaxes(1,2) - self.vectorsM
        square_distances = (diffs**2).sum(2)
        return square_distances.argmin(1)

class ClosestDistanceClassifier(object): 
    def train(self, data, labels=None):
        self.n_classes = data.shape[0]
        self.vectorsM = data.copy()
        if labels == None:
            self.labels = numpy.array(range(self.n_classes))
        else:
            if isinstance(labels, numpy.ndarray):
                if labels.shape[0] != self.n_classes:
                    ex = "Wrong number of labels: %d != %d n_classes"%(labels.shape[0], self.n_classes)
                    raise Exception(ex)
            else:
                if len(labels) != self.n_classes:
                    ex = "Wrong number of labels: %d != %d n_classes [2]"%(labels.shape[0], self.n_classes)
                    raise Exception(ex)
            self.labels = labels
#       print "vectorsM=", self.vectorsM
    def classify(self, data):
        diffs = data[:,:,numpy.newaxis].repeat(self.n_classes, axis=2).swapaxes(1,2) - self.vectorsM
#        diffs = data[:,numpy.newaxis,:].repeat(self.n_classes, axis=1) - self.vectorsM
        square_distances = (diffs**2).sum(axis=2)
        return square_distances.argmin(axis=1)



class ClosestDistanceClassifierNew(object):
    def train(self, data, labels=None):
        self.n_classes = data.shape[0]
        self.vectorsM = data.copy()
        if labels == None:
            self.labels = numpy.array(range(self.n_classes))
        else:
            if isinstance(labels, numpy.ndarray):
                if labels.shape[0] != self.n_classes:
                    ex = "Wrong number of labels: %d != %d n_classes"%(labels.shape[0], self.n_classes)
                    raise Exception(ex)
            else:
                if len(labels) != self.n_classes:
                    ex = "Wrong number of labels: %d != %d n_classes [2]"%(labels.shape[0], self.n_classes)
                    raise Exception(ex)
                print "labels=", labels
                labels = numpy.array(labels)
            self.labels = labels
#       print "vectorsM=", self.vectorsM
    def classifyNoMem(self, data):
        closest = numpy.zeros(len(data))
        for i, sample in enumerate(data):
            diffs = sample[:,numpy.newaxis].repeat(self.n_classes, axis=1).swapaxes(0,1) - self.vectorsM
            square_distances = (diffs**2).sum(axis=1)
            closest[i] = square_distances.argmin()
        return self.labels[closest.astype('int')]

    
class ClosestVectorClassifier(object):
    def train(self, data, labels=None):
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

        prod = dataM * self.vectorsM #alternative: use numpy.dot, with 2D arrays
        print "prod=", prod
        result = prod.argmax(axis=1)
        return result
    
def blocker_computer(data, labels, classes = None, block_size=1, spacing=None, verbose=False):
    """Learns the vectors, one for each class."""
    if spacing is None:
        spacing = block_size
        
    if classes == None:
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
        blocks[i,:,:] = data[i*spacing:i*spacing+block_size].copy()
#        print "labels_copy[****].shape is ", labels_copy[i*spacing:i*spacing+block_size].shape
        
        b_labels[i] = labels_copy[i*spacing:i*spacing+block_size].copy()
        b_classes[i] = classes_copy[i*spacing:i*spacing+block_size].copy()
        means[i] = blocks[i].mean(axis=0)
    
    return means, blocks, b_labels, b_classes



#This needs serious improvements!!!!
#No need for this mixed classifier, split into two normal classifiers
class Simple_2_Stage_Classifier(object):
    """This is a composed classifier designed with SFA signals in mind
    (however, it must be improved)
    the input samples (array data) can be divided in blocks of block_size rows, 
    in which each block belongs to the same class
    In addition a label (e.g., a real value) can be assigned to every sample
    """
    def train(self, data, labels=None, classes=None, block_size=None, spacing=None, verbose=True):
        """  A Gaussian classifier and a (possible two step) closest center classified are trained
        """
#        print "training data: ", data
        self.input_dim = data.shape[1]
        self.block_size = block_size

        if classes == None:
            classes = numpy.arange(data.shape[0])/block_size
            num_classes = int(numpy.ceil(data.shape[0]*1.0/block_size))
            print "No classes? Fine???!"
            quit()
        else:
            num_classes = len(set(classes))

        self.set_classes = list(set(classes))
        self.set_classes.sort()
        print "list of classes=", self.set_classes

#CDC
        self.CDC_L0 = ClosestDistanceClassifier()
        
        if isinstance(block_size, int):
            self.means_L0, self.blocks_L0, self.labels_L0, self.classes_L0 = blocker_computer(data, labels=labels, classes = classes, block_size=block_size, spacing=spacing)
        elif isinstance(block_size, (list, numpy.ndarray)) or (block_size == None):        
            self.means_L0 = numpy.zeros((num_classes, data.shape[1]))
            self.blocks_L0 = []
            self.labels_L0 = []
            self.classes_L0 = []
            # numpy.zeros((num_classes, data.shape[1]))
#            self.labels_L0 = numpy.zeros(num_classes)
            for class_number, class_value in enumerate(self.set_classes): # set(classes):
                self.means_L0[class_number] = data[classes == class_value].mean(axis=0)
                self.blocks_L0.append( data[classes == class_value, :])
                self.labels_L0.append( labels[classes == class_value] )
                self.classes_L0.append( classes[classes == class_value])
#        print self.means_L0.shape, "(shape) of means=", self.means_L0
#        print self.labels_L0.shape, " (shape) of labels=", self.labels_L0
##        else: #block_size is None???
##            self.means_L0 = numpy.zeros((num_classes, data.shape[1]))
##            self.blocks_L0 = []
##            self.labels_L0 = []
##            self.classes_L0 = []
##            # numpy.zeros((num_classes, data.shape[1]))
###            self.labels_L0 = numpy.zeros(num_classes)
##            for class_number, class_value in enumerate(self.set_classes): # set(classes):
##                self.means_L0[class_number] = data[classes == class_value].mean(axis=0)
##                self.blocks_L0.append( data[classes == class_value, :])
##                self.labels_L0.append( labels[classes == class_value] )
##                self.classes_L0.append( classes[classes == class_value])
            
            
        self.CDC_L0.train(self.means_L0, self.labels_L0)

        if isinstance(self.classes_L0, numpy.ndarray):
            self.CDC_L0.classes = self.classes_L0[:,0] #take first class of each block as representative
        else:
#            self.CDC_L0.classes = numpy.zeros(num_classes)
#            for i in range(num_classes):
#                self.CDC_L0.classes[i] = self.classes_L0[:,0]
            self.CDC_L0.classes = numpy.array(self.set_classes)
             
        if block_size != None:
            self.GC_L0 = mdp.nodes.GaussianClassifierNode()

            if isinstance(block_size, int):
                if data.shape[0] % block_size != 0:
                    print "Warning, training data is not a multiple of block_size"
                    quit()
                    
            self.GC_L0.train(data, classes)
            self.GC_L0.stop_training()     
#            self.average_Labels_L0 = numpy.zeros(num_classes)
#            for i in range(num_classes):
#                self.average_Labels_L0[i] = labels[i*block_size:(i+1)*block_size].mean()
#           This assumes that classes go from 0 to C-1, for exactly C classes
            self.average_Labels_L0_class = {}
            for class_value in self.set_classes:
                self.average_Labels_L0_class[class_value] = labels[classes == class_value].mean()
#                print "average label class[", class_value, "]=", self.average_Labels_L0_class[class_value]
            
            self.class_index = {}                
            self.average_Labels_L0 = numpy.zeros(num_classes)

            for index, class_value in enumerate(self.set_classes):
                self.average_Labels_L0[index] = self.average_Labels_L0_class[class_value]               
                self.class_index[class_value] = index  
#                print "average label[", index, "]=", self.average_Labels_L0[index]
            
            
        if block_size != None:
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
            xx = data[i].reshape((1,self.input_dim))
            cx = self.CDC_L0.classify(xx)
            classes[i] = self.CDC_L0.classes[cx]

            if self.block_size != None:
                c_L1= self.CDC_nodes_L1[cx].classify(xx)
                labels[i] = self.CDC_nodes_L1[cx].labels[c_L1]
            else:
                labels[i] = self.CDC_L0.labels[cx]
#            print xx, " classified to ", cx, " subclass ", c_L1, " with label: ", self.CDC_nodes_L1[cx].labels[c_L1]
        return classes, labels

    def classifyGaussian(self, data):
        """  Classify according to the Gaussian classifier
        """
        print "Warning, this function works for classes, "
        print "but it looks strange for labels because it also uses closest distance classifier, check this"
#        classes = numpy.zeros(data.shape[0])
        labels = numpy.zeros(data.shape[0]) - 10
        classes = self.GC_L0.classify(data)      
        
#        for i, class_value in enumerate(classes):
#            labels[i] = self.average_Labels_L0_class[class_value]

#        print "labels=", labels
#        print "classes=", classes

        #Just in case... Nope, classes are integers!!!
        #classes = numpy.nan_to_num(classes)

#WARNING!!!! WTF is doing the gaussian classifier, it is screwing with the labels now!
#Warning, code might be removed because it looks strange and it is useless now
##        for i in range(data.shape[0]):
##            xx = data[i].reshape((1,self.input_dim))
##            cx = classes[i]
###            print "i=", i,"cx=", cx
##            if self.block_size != None:
##                index = list(self.set_classes).index(cx)
##
##                if index >= len(self.CDC_nodes_L1):
##                    index = len(self.CDC_nodes_L1) - 1
##                c_L1= self.CDC_nodes_L1[index].classify(xx)
##                labels[i] = self.CDC_nodes_L1[index].labels[c_L1]
##            else:
##                labels[i] = self.CDC_L0.labels[cx]

        return classes, labels
    
    def GaussianRegression(self, data):
        """  Use the class probabilities to generate a better label
        If the computation of the class probabilities were perfect,
        and if the Gaussians were just a delta, then the output value
        minimizes the squared error 
        """

#        classes = numpy.zeros(data.shape[0])
#        value = numpy.zeros(data.shape[0])
        probabilities = self.GC_L0.class_probabilities(data)

        #now class probabilities should always return something that makes sense
#WARNING, EXPERIMENTAL CODE!!!        
#        sum_probs = probabilities_nan.sum(axis=1)
#        print "sum_probs[0:40] counting nans is:", sum_probs[0:40]
#    
#        probabilities = numpy.nan_to_num(probabilities_nan)

#        if (probabilities - probabilities_nan).any():
#            print "Warning, nan_to_num changed something"
#
#        sum_probs = probabilities.sum(axis=1)
#        print "sum_probs[0:40] without nans is:", sum_probs[0:40]
#        print "Total sum is:", probabilities.sum()
#        
#        print "probabilities.shape=", probabilities.shape
#        print "average_Labels_L0.shape=", self.average_Labels_L0.shape
#        print "average_Labels_L0=", self.average_Labels_L0

#        for i in range(probabilities_nan.shape[0]):
#            if (probabilities_nan[i] == probabilities[i]).all():
#                pass
#            else:
#                print "DIF1: ", probabilities_nan[i]
#                print "DIF2: ", probabilities[i]

#        for i in range(data.shape[0]):
#            value[i] = numpy.dot(probabilities[i], self.average_Labels_L0)
#            
#        print "probabilities[0] =", probabilities[0]
#        print "average_Labels_L0 =", self.average_Labels_L0
#        quit()
        
        value = value2 = numpy.dot(probabilities, self.average_Labels_L0)
#        print "value.shape=", value.shape
#        print "value2.shape=", value2.shape
#        
#        print "value2 - value=", value2-value
#        print "value1 == value2???", (value2 == value).all()
#
#        for i in range(value.shape[0]):
#            if (value[i] == value2[i]).all():
#                pass
#            else:
#                print "i=", i
#                print "value : ", value[i]
#                print "value2: ", value[i]
        return value

def correct_classif_rate(ground_truth, classified, verbose=False):
    """Computes a float value indicating the classification rate
    
        output = number of success classifications / total number of samples
        both input arrays must have the same length and integer values
    """
    num = len(ground_truth)
    if len(ground_truth) != len(classified):
        ex = "ERROR in class sizes, in correct_classif_rate: len(ground_truth)=%d != len(classified)=%d"%(len(ground_truth), len(classified))
        print ex
        raise Exception(ex)

    d1 = numpy.array(ground_truth, dtype="int")
    d2 = numpy.array(classified, dtype="int")
    if verbose:
        print "ground_truth=", d1
        print "classified=", d2
    return (d1 == d2).sum() * 1.0 / num

