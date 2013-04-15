#! /usr/bin/env python
#
#General purpose raw image processing/analysis
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 26 Feb 2010
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import Image
import mdp
import more_nodes
import patch_mdp
import object_cache as cache
import os, sys
import glob
import random
import sfa_libs
from sfa_libs import (scale_to, distance_squared_Euclidean, str3, wider_1Darray)
 
import SystemParameters
from imageLoader import *
import classifiers_regressions as classifiers
import network_builder
import time
from matplotlib.ticker import MultipleLocator
import copy
import string

import object_cache


#from SystemParameters import *

#list holding the benchmark information with entries: ("description", time as float in seconds)
benchmark=[]

#The next lines are only relevant if cache or network saving is enabled.
on_lok21 = os.path.lexists("/local2/tmp/escalafl/")
#on_lok21 = True
if on_lok21:
    network_baseir = "/local2/tmp/escalafl/Alberto/SavedNetworks"
else:
    network_baseir = "/local/tmp/escalafl/Alberto/SavedNetworks"

#t0 = time.time()
#print "LOADING INPUT INFORMATION"        

##from GenerateSystemParameters import Linear4LNetwork as Network
##This defines the sequences used for training, and testing
##See also: ParamsGender, ParamsAngle, ParamsIdentity,  ParamsTransX, ParamsAge,  ParamsRTransX
from GenerateSystemParameters import ParamsGender as Parameters


#TODO: Consider reduces size images
enable_reduced_image_sizes = True
if enable_reduced_image_sizes:
    reduction_factor = 4 # (the inverse of a zoom factor)
    Parameters.name = Parameters.name + ". Half-size images"
    for iSeq in (Parameters.iTrain, Parameters.iSeenid, Parameters.iNewid): 
        # iSeq.trans = iSeq.trans / 2
        pass

    for sSeq in (Parameters.sTrain, Parameters.sSeenid, Parameters.sNewid): 
        sSeq.subimage_width = sSeq.subimage_width / reduction_factor
        sSeq.subimage_height = sSeq.subimage_height / reduction_factor 
        sSeq.pixelsampling_x = sSeq.pixelsampling_x * reduction_factor
        sSeq.pixelsampling_y = sSeq.pixelsampling_y * reduction_factor
    
iTrain = Parameters.iTrain
sTrain = Parameters.sTrain
iSeenid = Parameters.iSeenid
sSeenid = Parameters.sSeenid
iNewid = Parameters.iNewid
sNewid = Parameters.sNewid

image_files_training = iTrain.input_files
num_images_training = num_images = iTrain.num_images

block_size = Parameters.block_size
#WARNING!!!!!!!
train_mode = Parameters.train_mode
# = "mixed", "sequence", "complete"

block_size_L0=block_size
block_size_L1=block_size
block_size_L2=block_size
block_size_L3=block_size
block_size_exec=block_size #(Used only for random walk)

seq = sTrain

#oTrain = NetworkOutput()

#WARNING!!!!!!!
#Move this after network loadings
small_image_size_128_128 = False
if small_image_size_128_128 and (seq.subimage_height != 128 or seq.subimage_width != 128):
    print "Warning, modifying input image sizes to 128x128"
    seq.subimage_height = seq.subimage_width = 128
    sSeenid.subimage_height = sSeenid.subimage_width = 128
    sNewid.subimage_height = sNewid.subimage_width = 128   


hack_image_sizes = [135, 90, 64, 32]
#Warning!!! hack_image_size = False
hack_image_size = 32
enable_hack_image_size = True
if enable_hack_image_size:
    seq.subimage_height = seq.subimage_width = hack_image_size
    sSeenid.subimage_height = sSeenid.subimage_width = hack_image_size
    sNewid.subimage_height = sNewid.subimage_width = hack_image_size

#Filter used for loading images with transparent background
#filter = generate_color_filter2((seq.subimage_height, seq.subimage_width))
alpha = 4.0 # mask 1 / f^(alpha/2) => power 1/f^alpha
filter = filter_colored_noise2D_imp((seq.subimage_height, seq.subimage_width), alpha)
#back_type = None
#filter = None


results = SystemParameters.ExperimentResult()
results.name = Parameters.name
results.network_name = "Raw Processing"
results.layers_name = []
results.iTrain = iTrain
results.sTrain = sTrain
results.iSeenid = iSeenid
results.sSeenid = sSeenid
results.iNewid = iNewid
results.sNewid = sNewid

subimages_train = load_image_data(seq.input_files, seq.image_width, seq.image_height, seq.subimage_width, \
                            seq.subimage_height, seq.pixelsampling_x, seq.pixelsampling_y, \
                            seq.subimage_first_row, seq.subimage_first_column, seq.add_noise_L0, \
                            seq.convert_format, seq.translations_x, seq.translations_y, seq.trans_sampled, background_type=seq.background_type, color_background_filter=filter, verbose=False)

t1=t0=0.0
print seq.num_images, "Train Images loaded in %0.3f s"% ((t1-t0))
#benchmark.append(("Load Info and Train Images", t1-t0))  

sl_seq_training = subimages_train
print "Setting correct labels/classes data for training"
correct_classes_training = iTrain.correct_classes
correct_labels_training = iTrain.correct_labels



enable_raw_analysis = True
if enable_raw_analysis:
    if enable_raw_analysis:
        print "Training Raw Signal Analysis (Classifier/Regression)..."

        cfr_sl = subimages_train
        cfr_num_samples = cfr_sl.shape[0]
        cfr_correct_labels = correct_labels_training
        cfr_spacing = cfr_block_size = iTrain.block_size

        results.reg_num_signals = reg_num_signals = cfr_sl.shape[1]

        #cfr_sl = sl_seq_seenid
        #cfr_correct_labels = iSeenid.correct_labels
        #cfr_spacing = cfr_block_size = iSeenid.block_size

        enable_ccc_Gauss_cfr = False
        enable_svm_cfr = False
        enable_lr_cfr = True

        if enable_ccc_Gauss_cfr == True:
            S2SC = classifiers.Simple_2_Stage_Classifier()
            S2SC.train(data=cf_sl[:,0:reg_num_signals], labels=cfr_correct_labels, block_size=cfr_block_size,spacing=cfr_block_size)        
            print "Training CCC Classifier... Done"
                
        num_blocks = cfr_num_samples/cfr_block_size
        if enable_svm_cfr == True:
            import svm as libsvm
            svm_node = mdp.contrib.LibSVMNode(probability=True)
            svm_node.train(cfr_sl[:,0:reg_num_signals], cf_correct_labels)
            svm_node.stop_training(svm_type=libsvm.C_SVC, kernel_type=libsvm.RBF, C=1.0, gamma=1.0/(num_blocks))
            print "Training SVM Classifier... Done"
        
        if enable_lr_cfr == True:
            lr_node = mdp.nodes.LinearRegressionNode(with_bias=True, use_pinv=False)
            lr_node.train(cfr_sl[:,0:reg_num_signals], cfr_correct_labels.reshape((cfr_sl.shape[0], 1)))
            lr_node.stop_training()
            print "Training LR ... Done"
      
        print "Classification/Regression over training signal..."
        if enable_ccc_Gauss_cfr == True:
            classes_ccc_training, labels_ccc_training = S2SC.classifyCDC(cfr_sl[:,0:reg_num_signals])
            classes_Gauss_training, labels_Gauss_training = S2SC.classifyGaussian(cfr_sl[:,0:reg_num_signals])
            regression_Gauss_training = S2SC.GaussianRegression(cfr_sl[:,0:reg_num_signals])
            probs_training = S2SC.GC_L0.class_probabilities(cfr_sl[:,0:reg_num_signals])
        else:
            classes_ccc_training = labels_ccc_training = classes_Gauss_training =  labels_Gauss_training = regression_Gauss_training  =   numpy.zeros(cfr_num_samples) 
            probs_training = numpy.zeros((cfr_num_samples, 2))
        
        if enable_svm_cfr == True:
            classes_svm_training= svm_node.classify(sl_seq_training[:,0:reg_num_signals])
            regression_svm_training= svm_node.label_of_class(classes_svm_training)
        else:
            classes_svm_training=regression_svm_training = numpy.zeros(cfr_num_samples)
        
        if enable_lr_cfr == True:
            regression_lr_training = lr_node.execute(sl_seq_training[:,0:reg_num_signals]).flatten()
        else:
            regression_lr_training = numpy.zeros(cfr_num_samples)

image_files_seenid = iSeenid.input_files
num_images_seenid = iSeenid.num_images
block_size_seeneid = iSeenid.block_size
seq = sSeenid

print "Loading Seenid test sequence..."
subimages_seenid = load_image_data(seq.input_files, seq.image_width, seq.image_height, seq.subimage_width, \
                            seq.subimage_height, seq.pixelsampling_x, seq.pixelsampling_y, \
                            seq.subimage_first_row, seq.subimage_first_column, seq.add_noise_L0, \
                            seq.convert_format, seq.translations_x, seq.translations_y, seq.trans_sampled, background_type=seq.background_type, color_background_filter=filter, verbose=False)

sl_seq_seenid = subimages_seenid
print "Setting correct labels/classes data for seenid"
correct_classes_seenid = iSeenid.correct_classes
correct_labels_seenid = iSeenid.correct_labels

if enable_ccc_Gauss_cfr == True:
    classes_ccc_seenid, labels_ccc_seenid = S2SC.classifyCDC(sl_seq_seenid[:,0:reg_num_signals])
    classes_Gauss_seenid, labels_Gauss_seenid = S2SC.classifyGaussian(sl_seq_seenid[:,0:reg_num_signals])
    print "Classification of Seen id test images: ", labels_ccc_seenid
    regression_Gauss_seenid = S2SC.GaussianRegression(sl_seq_seenid[:,0:reg_num_signals])
    probs_seenid = S2SC.GC_L0.class_probabilities(sl_seq_seenid[:,0:reg_num_signals])
else:
    classes_ccc_seenid = labels_ccc_seenid =  classes_Gauss_seenid = labels_Gauss_seenid = regression_Gauss_seenid = numpy.zeros(num_images_seenid) 
    probs_seenid = numpy.zeros((num_images_seenid, 2))

if enable_svm_cfr == True:
    classes_svm_seenid = svm_node.classify(sl_seq_seenid[:,0:reg_num_signals])
    regression_svm_seenid = svm_node.label_of_class(classes_svm_seenid)
else:
    classes_svm_seenid=regression_svm_seenid = numpy.zeros(num_images_seenid)
        
if enable_lr_cfr == True:
    regression_lr_seenid = lr_node.execute(sl_seq_seenid[:,0:reg_num_signals]).flatten()
else:
    regression_lr_seenid = numpy.zeros(num_images_seenid)

print "labels_ccc_seenid.shape=", labels_ccc_seenid.shape

#correct_labels_seenid = wider_1Darray(numpy.arange(iSeenid.MIN_GENDER, iSeenid.MAX_GENDER, iSeenid.GENDER_STEP), iSeenid.block_size)
print "correct_labels_seenid.shape=", correct_labels_seenid.shape
#correct_classes_seenid = numpy.arange(len(labels_ccc_seenid)) * len(labels_ccc_training) / len(labels_ccc_seenid) / block_size


print "Loading test images, new ids..."

image_files_newid = iNewid.input_files
num_images_newid = iNewid.num_images
block_size_newid = iNewid.block_size
seq = sNewid

subimages_newid = load_image_data(seq.input_files, seq.image_width, seq.image_height, seq.subimage_width, \
                            seq.subimage_height, seq.pixelsampling_x, seq.pixelsampling_y, \
                            seq.subimage_first_row, seq.subimage_first_column, seq.add_noise_L0, \
                            seq.convert_format, seq.translations_x, seq.translations_y, seq.trans_sampled, background_type=seq.background_type, color_background_filter=filter, verbose=False)

t_load_images1 = t_load_images0 = 0
print num_images_newid, " Images loaded in %0.3f s"% ((t_load_images1 - t_load_images0))

sl_seq_newid = subimages_newid

##WARNING!!!
#print "WARNING!!! SCALING NEW ID SLOW SIGNALS!!!"
#corr_factor = numpy.array([ 1.06273968,  1.0320762 ,  1.06581665,  1.01598426,  1.08355725,
#        1.10316477,  1.08731609,  1.05887109,  1.08185727,  1.09867758,
#        1.10567757,  1.08268021])
#print corr_factor
#
#corr_factor = numpy.array([ 1.06273968,  1.0320762 ,  1.06581665,  1.01598426,  1.08355725,
#        1.10316477,  1.08731609,  1.05887109,  1.08185727,  1.09867758]) * 0.98
#print corr_factor
#
#corr_factor =  numpy.sqrt(sl_seq_training.var(axis=0) / sl_seq_newid.var(axis=0))[0:reg_num_signals].mean()
#print corr_factor
#
#corr_factor =  numpy.sqrt(sl_seq_training.var(axis=0) / sl_seq_newid.var(axis=0))[0:reg_num_signals] * 0.98
#print corr_factor
#
#corr_factor =  0.977 * numpy.sqrt(sl_seq_training.var(axis=0)[0:reg_num_signals].mean() / sl_seq_newid.var(axis=0)[0:reg_num_signals].mean())
#print corr_factor
#
#corr_factor=1
#print corr_factor
#sl_seq_newid[:,0:reg_num_signals] = sl_seq_newid[:,0:reg_num_signals] * corr_factor

t_class0 = time.time()
correct_classes_newid = iNewid.correct_classes
correct_labels_newid = iNewid.correct_labels

if enable_ccc_Gauss_cfr == True:
    classes_ccc_newid, labels_ccc_newid = S2SC.classifyCDC(sl_seq_newid[:,0:reg_num_signals])
    classes_Gauss_newid, labels_Gauss_newid = S2SC.classifyGaussian(sl_seq_newid[:,0:reg_num_signals])
    print "Classification of New Id test images: ", labels_ccc_newid
    regression_Gauss_newid = S2SC.GaussianRegression(sl_seq_newid[:,0:reg_num_signals])
    probs_newid = S2SC.GC_L0.class_probabilities(sl_seq_newid[:,0:reg_num_signals])
else:
    classes_ccc_newid = labels_ccc_newid = classes_Gauss_newid = labels_Gauss_newid = regression_Gauss_newid = numpy.zeros(num_images_newid) 
    probs_newid = numpy.zeros((num_images_newid, 2))

if enable_svm_cfr == True:
    classes_svm_newid = svm_node.classify(sl_seq_newid[:,0:reg_num_signals])
    regression_svm_newid = svm_node.label_of_class(classes_svm_newid)
else:
    classes_svm_newid=regression_svm_newid = numpy.zeros(num_images_newid)

if enable_lr_cfr == True:
    regression_lr_newid = lr_node.execute(sl_seq_newid[:,0:reg_num_signals]).flatten()
else:
    regression_lr_newid = numpy.zeros(num_images_newid)

t_class1 = time.time()
print "Classification/Regression over New Id in %0.3f s"% ((t_class1 - t_class0))

print "Computing typical delta, eta values for New Id SFA Signal"
telta_eta0 = time.time()
results.typical_delta_newid, results.typical_eta_newid = sfa_libs.comp_typical_delta_eta(sl_seq_newid, block_size, num_reps=200)
results.brute_delta_newid = sfa_libs.comp_delta(sl_seq_newid)
results.brute_eta_newid= sfa_libs.comp_eta(sl_seq_newid)
telta_eta1 = time.time()
print "delta_newid=", results.typical_delta_newid
print "eta_newid=", results.typical_eta_newid
#print "bruteelta_newid=", results.bruteelta_newid
#print "brute_eta_newid=", results.brute_eta_newid
print "computed delta/eta in %0.3f ms"% ((telta_eta1-telta_eta0)*1000.0)
print "Computations Finished!"

print "** Displaying Benchmark data: **"
for task_name, task_time in benchmark:
    print "     ", task_name, " done in %0.3f s"%task_time


print "Classification/Regression Performance: "
results.class_ccc_rate_train = classifiers.correct_classif_rate(correct_classes_training, classes_ccc_training)
results.class_Gauss_rate_train = classifiers.correct_classif_rate(correct_classes_training, classes_Gauss_training)
results.class_svm_rate_train = classifiers.correct_classif_rate(correct_classes_training, classes_svm_training)
results.mse_ccc_train = distance_squared_Euclidean(correct_labels_training, labels_ccc_training)/len(labels_ccc_training)
results.mse_gauss_train = distance_squared_Euclidean(correct_labels_training, regression_Gauss_training)/len(labels_ccc_training)
results.mse_svm_train = distance_squared_Euclidean(correct_labels_training, regression_svm_training)/len(labels_ccc_training)
results.mse_lr_train = distance_squared_Euclidean(correct_labels_training, regression_lr_training)/len(labels_ccc_training)

results.class_ccc_rate_seenid = classifiers.correct_classif_rate(correct_classes_seenid, classes_ccc_seenid)
results.class_Gauss_rate_seenid = classifiers.correct_classif_rate(correct_classes_seenid, classes_Gauss_seenid)
results.class_svm_rate_seenid = classifiers.correct_classif_rate(correct_classes_seenid, classes_svm_seenid)
results.mse_ccc_seenid = distance_squared_Euclidean(correct_labels_seenid, labels_ccc_seenid)/len(labels_ccc_seenid)
results.mse_gauss_seenid = distance_squared_Euclidean(correct_labels_seenid, regression_Gauss_seenid)/len(labels_ccc_seenid)
results.mse_svm_seenid = distance_squared_Euclidean(correct_labels_seenid, regression_svm_seenid)/len(labels_ccc_seenid)
results.mse_lr_seenid = distance_squared_Euclidean(correct_labels_seenid, regression_lr_seenid)/len(labels_ccc_seenid)

results.class_ccc_rate_newid = classifiers.correct_classif_rate(correct_classes_newid, classes_ccc_newid)
results.class_Gauss_rate_newid = classifiers.correct_classif_rate(correct_classes_newid, classes_Gauss_newid)
results.class_svm_rate_newid = classifiers.correct_classif_rate(correct_classes_newid, classes_svm_newid)
results.mse_ccc_newid = distance_squared_Euclidean(correct_labels_newid, labels_ccc_newid)/len(labels_ccc_newid)
results.mse_gauss_newid = distance_squared_Euclidean(correct_labels_newid, regression_Gauss_newid)/len(labels_ccc_newid)
results.mse_svm_newid = distance_squared_Euclidean(correct_labels_newid, regression_svm_newid)/len(labels_ccc_newid)
results.mse_lr_newid = distance_squared_Euclidean(correct_labels_newid, regression_lr_newid)/len(labels_ccc_newid)

save_results=False
if save_results:
    object_cache.pickle_toisk(results, "results_" + str(int(time.time()))+ ".pckl")

print "sl_seq_training.mean(axis=0)=", sl_seq_training.mean(axis=0)
print "sl_seq_seenid.mean(axis=0)=", sl_seq_seenid.mean(axis=0)
print "sl_seq_newid.mean(axis=0)=", sl_seq_newid.mean(axis=0)
print "sl_seq_training.var(axis=0)=", sl_seq_training.var(axis=0)
print "sl_seq_seenid.var(axis=0)=", sl_seq_seenid.var(axis=0)
print "sl_seq_newid.var(axis=0)=", sl_seq_newid.var(axis=0)
    
print "Train: %0.3f CR_CCC, CR_Gauss %0.3f, CR_SVM %0.3f, MSE_CCC %0.3f, MSE_Gauss %0.3f, MSE_SVM %0.3f, MSE_LR %0.3f  "%(results.class_ccc_rate_train, results.class_Gauss_rate_train, results.class_svm_rate_train, results.mse_ccc_train, results.mse_gauss_train, results.mse_svm_train, results.mse_lr_train)
print "Seen Id: %0.3f CR_CCC, CR_Gauss %0.3f, CR_SVM %0.3f, MSE_CCC %0.3f, MSE_Gauss %0.3f, MSE_SVM %0.3f, MSE_LR %0.3f"%(results.class_ccc_rate_seenid, results.class_Gauss_rate_seenid, results.class_svm_rate_seenid, results.mse_ccc_seenid, results.mse_gauss_seenid, results.mse_svm_seenid, results.mse_lr_seenid)
print "New Id: %0.3f CR_CCC, CR_Gauss %0.3f, CR_SVM %0.3f, MSE_CCC %0.3f, MSE_Gauss %0.3f, MSE_SVM %0.3f, MSE_LR %0.3f "%(results.class_ccc_rate_newid, results.class_Gauss_rate_newid, results.class_svm_rate_newid, results.mse_ccc_newid, results.mse_gauss_newid, results.mse_svm_newid, results.mse_lr_newid)

#quit()


print "Creating GUI..."

print "****** Displaying Typical Images used for Training and Testing **********"
tmp_fig = plt.figure()
plt.suptitle(Parameters.name + ". Image Datasets")
 
num_images_per_set = 4 

subimages_training = subimages_train
num_images_training

sequences = [subimages_training, subimages_seenid, subimages_newid]
messages = ["Train Images", "Seen Id Images", "New Id Images"]
nums_images = [num_images_training, num_images_seenid, num_images_newid]
sizes = [(sTrain.subimage_height, sTrain.subimage_width), (sSeenid.subimage_height, sSeenid.subimage_width), \
         (sNewid.subimage_height, sNewid.subimage_width)]

for seqn in range(3):
    for im in range(num_images_per_set):
        tmp_sb = plt.subplot(3, num_images_per_set, num_images_per_set*seqn + im+1)
        y = im * (nums_images[seqn]-1) / (num_images_per_set - 1)
        subimage_im = sequences[seqn][y].reshape(sizes[seqn])
        tmp_sb.imshow(subimage_im.clip(0,255), norm=None, vmin=0, vmax=255, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
        if im == 0:
            plt.ylabel(messages[seqn])
        else:
            tmp_sb.axis('off')
    
print "************ Displaying Classification / Regression Results **************"
#Create Figure
f2 = plt.figure()
plt.suptitle("Classification Results (Class Numbers)  using %d Slow Signals"%reg_num_signals)
#plt.title("Training Set")
p11 = f2.add_subplot(311, frame_on=False)

p11.plot(numpy.arange(len(correct_classes_training)), correct_classes_training, 'r.', markersize=1, markerfacecolor='red')
p11.plot(numpy.arange(len(classes_ccc_training)), classes_ccc_training, 'b.', markersize=1, markerfacecolor='blue')
p11.plot(numpy.arange(len(classes_svm_training)), classes_svm_training, 'g.', markersize=1, markerfacecolor='green')
plt.xlabel("Image Number, Training Set. Classification Rate CCC=%.3f, CR_Gauss=%.3f, CR_SVM=%.3f" % (results.class_ccc_rate_train, results.class_Gauss_rate_train, results.class_svm_rate_train))
plt.ylabel("Class Number")
p11.grid(True)
#draw horizontal and vertical lines
#majorLocator_x   = MultipleLocator(block_size)
#majorLocator_y   = MultipleLocator(1)
#p11.xaxis.set_major_locator(majorLocator_x)
##p11.yaxis.set_major_locator(majorLocator_y)
#plt.xticks(numpy.arange(0, len(labels_ccc_training), block_size)) 
#plt.yticks(numpy.arange(0, len(labels_ccc_training), block_size)) 
#print "Block_size is: ", block_size

p12 = f2.add_subplot(312, frame_on=False)
p12.plot(numpy.arange(len(correct_classes_seenid)), correct_classes_seenid, 'r.', markersize=2, markerfacecolor='red')
p12.plot(numpy.arange(len(classes_ccc_seenid)), classes_ccc_seenid, 'm.', markersize=2, markerfacecolor='magenta')
p12.plot(numpy.arange(len(classes_ccc_seenid)), classes_svm_seenid, 'g.', markersize=2, markerfacecolor='green')
plt.xlabel("Image Number, Seen Id Set. Classification Rate CCC=%f, CR_Gauss=%.3f, CR_SVM=%.3f" % ( results.class_ccc_rate_seenid, results.class_Gauss_rate_seenid, results.class_svm_rate_seenid))
plt.ylabel("Class Number")
p12.grid(True)
#p12.plot(numpy.arange(len(labels_ccc_seenid)), correct_classes_seenid, 'mo', markersize=3, markerfacecolor='magenta')
#majorLocator_y   = MultipleLocator(block_size)
##majorLocator_x   = MultipleLocator(block_size_seenid)
#majorLocator_x   = MultipleLocator(block_size_seenid)
#p12.xaxis.set_major_locator(majorLocator_x)
#p12.yaxis.set_major_locator(majorLocator_y)
#majorLocator_y   = MultipleLocator(block_size)

p13 = f2.add_subplot(313, frame_on=False)
p13.plot(numpy.arange(len(correct_classes_newid)), correct_classes_newid, 'r.', markersize=2, markerfacecolor='red')
p13.plot(numpy.arange(len(classes_ccc_newid)), classes_ccc_newid, 'm.', markersize=2, markerfacecolor='magenta')
p13.plot(numpy.arange(len(classes_svm_newid)), classes_svm_newid, 'g.', markersize=2, markerfacecolor='green')

plt.xlabel("Image Number, New Id Set. Classification Rate CCC=%f, CR_Gauss=%.3f, CR_SVM=%.3f" % ( results.class_ccc_rate_newid, results.class_Gauss_rate_newid, results.class_svm_rate_newid))
plt.ylabel("Class Number")
p13.grid(True)
#majorLocator_y = MultipleLocator(block_size)
##majorLocator_x   = MultipleLocator(block_size_seenid)
#majorLocator_x   = MultipleLocator(block_size_newid)
#p13.xaxis.set_major_locator(majorLocator_x)
##p13.yaxis.set_major_locator(majorLocator_y)

f3 = plt.figure()
regression_text_labels = ["Closest Center Class.", "SVM", "Linear Regression", "Gaussian Class/Regr.", "Ground Truth"]
plt.suptitle("Regression Results (Labels) using %d Slow Signals"%reg_num_signals)
#plt.title("Training Set")
p11 = f3.add_subplot(311, frame_on=False)
#correct_classes_training = numpy.arange(len(labels_ccc_training)) / block_size

p11.plot(numpy.arange(len(labels_ccc_training)), labels_ccc_training, 'b.', markersize=3, markerfacecolor='blue')
p11.plot(numpy.arange(len(regression_svm_training)), regression_svm_training, 'g.', markersize=3, markerfacecolor='green')
p11.plot(numpy.arange(len(correct_labels_training)), regression_lr_training, 'c.', markersize=3, markerfacecolor='cyan')
p11.plot(numpy.arange(len(regression_Gauss_training)), regression_Gauss_training, 'm.', markersize=3, markerfacecolor='magenta')
p11.plot(numpy.arange(len(correct_labels_training)), correct_labels_training, 'r.', markersize=3, markerfacecolor='red')

##draw horizontal and vertical lines
#majorLocator   = MultipleLocator(block_size)
#p11.xaxis.set_major_locator(majorLocator)
##p11.yaxis.set_major_locator(majorLocator)
#plt.xticks(numpy.arange(0, len(labels_ccc_training), block_size)) 
#plt.yticks(numpy.arange(0, len(labels_ccc_training), block_size)) 
plt.xlabel("Image Number, Training Set. CCC Classification Rate=%f, MSE_CCC=%f, MSE_Gauss=%f, MSE_SVM=%f, MSE_LR=%f" % ( results.class_ccc_rate_train, results.mse_ccc_train, results.mse_gauss_train, results.mse_svm_train, results.mse_lr_train))
plt.ylabel("Label")
plt.legend( (regression_text_labels[0],  regression_text_labels[1], regression_text_labels[2], regression_text_labels[3], regression_text_labels[4]), loc=2 )
p11.grid(True)


p12 = f3.add_subplot(312, frame_on=False)
#correct_classes_seenid = numpy.arange(len(labels_ccc_seenid)) * len(labels_ccc_training) / len(labels_ccc_seenid) / block_size
p12.plot(numpy.arange(len(labels_ccc_seenid)), labels_ccc_seenid, 'b.', markersize=4, markerfacecolor='blue')
p12.plot(numpy.arange(len(regression_svm_seenid)), regression_svm_seenid, 'g.', markersize=4, markerfacecolor='green')
p12.plot(numpy.arange(len(regression_lr_seenid)), regression_lr_seenid, 'c.', markersize=4, markerfacecolor='cyan')
p12.plot(numpy.arange(len(regression_Gauss_seenid)), regression_Gauss_seenid, 'm.', markersize=4, markerfacecolor='magenta')
p12.plot(numpy.arange(len(correct_labels_seenid)), correct_labels_seenid, 'r.', markersize=4, markerfacecolor='red')
##majorLocator_y   = MultipleLocator(block_size)
##majorLocator_x   = MultipleLocator(block_size_seenid)
#majorLocator_x   = MultipleLocator( len(labels_ccc_seenid) * block_size / len(labels_ccc_training))
#p12.xaxis.set_major_locator(majorLocator_x)
##p12.yaxis.set_major_locator(majorLocator_y)
plt.xlabel("Image Number, Seen Id Set. Classification Rate=%f, MSE_CCC=%f, MSE_Gauss=%f, MSE_SVM=%f, MSE_LR=%f" % (results.class_ccc_rate_seenid, results.mse_ccc_seenid, results.mse_gauss_seenid, results.mse_svm_seenid, results.mse_lr_seenid))
plt.ylabel("Label")
plt.legend( (regression_text_labels[0],  regression_text_labels[1], regression_text_labels[2], regression_text_labels[3], regression_text_labels[4]), loc=2 )
p12.grid(True)


p13 = f3.add_subplot(313, frame_on=False)
p13.plot(numpy.arange(len(labels_ccc_newid)), labels_ccc_newid, 'b.', markersize=4, markerfacecolor='blue')
p13.plot(numpy.arange(len(regression_svm_newid)), regression_svm_newid, 'g.', markersize=4, markerfacecolor='green')
p13.plot(numpy.arange(len(regression_lr_newid)), regression_lr_newid, 'c.', markersize=4, markerfacecolor='cyan')
p13.plot(numpy.arange(len(regression_Gauss_newid)), regression_Gauss_newid, 'm.', markersize=4, markerfacecolor='magenta')
p13.plot(numpy.arange(len(correct_labels_newid)), correct_labels_newid, 'r.', markersize=4, markerfacecolor='red')
##majorLocator_y   = MultipleLocator(block_size)
##majorLocator_x   = MultipleLocator(block_size_seenid)
#majorLocator_x   = MultipleLocator( len(labels_ccc_newid) * block_size / len(labels_ccc_training))
#p13.xaxis.set_major_locator(majorLocator_x)
##p12.yaxis.set_major_locator(majorLocator_y)
plt.xlabel("Image Number, New Id Set. Classification Rate=%f, MSE_CCC=%f, MSE_Gauss=%f, MSE_SVM=%f, MSE_LR=%f" % ( results.class_ccc_rate_newid, results.mse_ccc_newid, results.mse_gauss_newid, results.mse_svm_newid, results.mse_lr_newid))
plt.ylabel("Label")
plt.legend( (regression_text_labels[0],  regression_text_labels[1], regression_text_labels[2], regression_text_labels[3], regression_text_labels[4]), loc=2 )
p13.grid(True)


print "************** Displaying Probability Profiles ***********"
f4 = plt.figure()
plt.suptitle("Probability Profiles of Gaussian Classifier Using %d Signals for Classification"%reg_num_signals)
  
#display Probability Profile of Training Set
ax11 = plt.subplot(1,3,1)
plt.title("(Network) Training Set")
#cax = p11.add_axes([1, 10, 1, 10])
pic = ax11.imshow(probs_training, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.hot)
plt.xlabel("Class Number")
plt.ylabel("Image Number, Training Set")
f4.colorbar(pic)

#display Probability Profile of Seen Id
ax11 = plt.subplot(1,3,2)
plt.title("Seen Id Test Set")
pic = ax11.imshow(probs_seenid, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.hot)
plt.xlabel("Class Number")
plt.ylabel("Image Number, Seen Id Set")
f4.colorbar(pic)

#display Probability Profile of New Id
ax11 = plt.subplot(1,3,3)
plt.title("New Id Test Set")
pic = ax11.imshow(probs_newid, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.hot)
plt.xlabel("Class Number")
plt.ylabel("Image Number, New Id Set")
f4.colorbar(pic)

print "GUI Created, showing!!!!"
plt.show()
print "GUI Finished!"

