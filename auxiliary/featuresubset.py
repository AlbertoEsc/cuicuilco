# -*- coding: latin-1 -*-
#Program that reads an libsvm data file, and saves it keeping at most d features
#First version: 8.11.2012 by Alberto N. Escalante-B.
#Theory of Neural Systems (Prof. Dr. Laurenz Wiskott)
#Institur für Neuroinformatik, Ruhr-University of Bochum
import string, numpy, sys

def read_libsvm_compact_datafile(filename):
    """ returns 1) an array with the N float labels or if possible integer classes, 
    and 2) a list with N dictionaries, each dictionary has entries "index:feature". """
    file = open(filename, "rb")
    
    labels_classes = []
    all_features = [] 
    
    for i, line in enumerate(file.readlines()):
        entries = string.split(line)
        label_class = float(entries[0])
        current_features = {}
        for feature in entries [1:]:
            index, val = string.split(feature, ":")
            index = int(index)
            val = float(val)
            current_features[index] = val
        labels_classes.append(label_class)
        all_features.append(current_features)
    file.close()
    labels_classes_int = numpy.array(labels_classes, dtype="int")
    labels_classes_float = numpy.array(labels_classes, dtype="float")
    if numpy.alltrue(labels_classes_int == labels_classes_float):
        labels_classes_array = labels_classes_int
    else:
        labels_classes_array = labels_classes_float
    
    keys = all_features[0].keys()
    keys.sort()
    num_feats = keys[-1]
    #print "keys=", keys

    num_samples = len(labels_classes_array)
    features_array = numpy.zeros((num_samples, num_feats))
    for sample, current_features in enumerate(all_features):
        #print "s=",sample,"i=",
        current_features_keys = current_features.keys() #Zero entries might not be included in file!
        for index in range(num_feats):
            if index+1 in current_features_keys:
                features_array[sample, index] =  current_features[index+1]
            #print "%d->%f"%(index,features_array[sample, index]),
    return labels_classes_array, features_array
           
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
    
if __name__ == "__main__":
    print "Program that reads an libsvm data file, and saves it keeping at most 'd' features"
    
    if len(sys.argv) != 4:
        print "Incorrect number of parameters. Usage: %s libsvm_data_file_in libsvm_data_file_out d"%sys.argv[0]
        print "libsvm_data_file_in -- input data file (labels + features) in libsvm format. Must be compact: no features have a default zero value"
        print "libsvm_data_file_out -- output file (labels + 'd'-features) . The features are truncated to the first 'd'"
        print "d -- number of features to be preserved"
#        print "centering_mode -- either 'mid_eyes_mouth' or 'mid_eyes_inferred-mouth' or 'noFace' " 
#        print "rotation_mode -- either 'noRotation' or 'EyeLineRotation' "
        quit()
        
    #CREATE NORMALIZATION FILE
    input_filename = sys.argv[1] #CAS_PEAL_coordinates.txt
    output_filename = sys.argv[2] # image%05d.jpg
    d = int(sys.argv[3])
    
    print "reading input file", input_filename
    labels, features = read_libsvm_compact_datafile(input_filename)
    features = features[:,:d]
    print "writing output file", output_filename
    export_to_libsvm(labels, features, output_filename)
    print "%s finished."%sys.argv[0]
    
