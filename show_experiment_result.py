#! /usr/bin/env python

# Program that opens a given ExperimentResult object and displays the most important
# information. 
# An ExperimentResult object stores the execution performance of a given
# hierarchical network and training/test data in terms of
# classification rates, and mean square errors
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 17 Sept 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import sys, os
sys.path.append("/home/escalafl/workspace/hiphi/src/hiphi/utils")
import misc

num_relevant_sfa_signals = 5
for arg in sys.argv[1:]:
    results = misc.unpickle_from_disk(arg)
    print "Experiment Name:", results.name
    print "Network Name: ", results.network_name
    for i, lay in enumerate(results.layers_name):
        print "Layer %d has Name: "%i, results.layers_name[i]
            
    print "Training Data: ", results.iTrain.name,
    print "Consisting of %d Images"%results.iTrain.num_images
    print "Seen Id Test Data: ", results.iSeenid.name,
    print "Consisting of %d Images"%results.iSeenid.num_images
    print "New Id Test Data: ", results.iNewid.name,
    print "Consisting of %d Images"%results.iNewid.num_images
        
#        self.sTrain = None
#        self.sSeenid = None
#        self.iNewid = None
#        self.sNewid = None

    print "reg_num_signals=", results.reg_num_signals
    print "typical_delta_train=", results.typical_delta_train[0:num_relevant_sfa_signals]
    print "typical_eta_train=", results.typical_eta_train[0:num_relevant_sfa_signals]
    print "brute_delta_train=", results.brute_delta_train[0:num_relevant_sfa_signals]
    print "brute_eta_train=", results.brute_eta_train[0:num_relevant_sfa_signals]
    
    print "typical_delta_seenid=", results.typical_delta_seenid[0:num_relevant_sfa_signals]
    print "typical_eta_seenid=", results.typical_eta_seenid[0:num_relevant_sfa_signals]
    print "brute_delta_seenid=", results.brute_delta_seenid[0:num_relevant_sfa_signals]
    print "brute_eta_seenid=", results.brute_eta_seenid[0:num_relevant_sfa_signals]

    print "typical_delta_newid=", results.typical_delta_newid[0:num_relevant_sfa_signals]
    print "typical_eta_newid=", results.typical_eta_newid[0:num_relevant_sfa_signals]
    print "brute_delta_newid=", results.brute_delta_newid[0:num_relevant_sfa_signals]
    print "brute_eta_newid=", results.brute_eta_newid[0:num_relevant_sfa_signals]
            
    print "Train: %0.3f classification rate, MSE %0.3f, MSEGauss %0.3f" % (results.class_rate_train, results.mse_train, results.msegauss_train)
    print "Seen Id: %0.3f classification rate, MSE %0.3f, MSEGauss %0.3f" % (results.class_rate_seenid, results.mse_seenid, results.msegauss_seenid)
    print "New Id: %0.3f classification rate, MSE %0.3f, MSEGauss %0.3f" % (results.class_rate_newid, results.mse_newid, results.msegauss_newid)


        