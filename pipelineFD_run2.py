#! /usr/bin/env python

#Special purpose hierarchical network pipeline for data processing
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 7 Juni 2010
#Ruhr-University-Bochum, Institute of Neural Computation, Group of Prof. Dr. Wiskott

import numpy
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import Image
import mdp
import more_nodes
import patch_mdp

import svm as libsvm
import object_cache as cache
import os, sys
import glob
import random
import sfa_libs
from sfa_libs import (scale_to, distance_squared_Euclidean, str3, wider_1Darray, ndarray_to_string, cutoff)
 
import SystemParameters
from imageLoader import *
import classifiers_regressions as classifiers
import network_builder
import time
from matplotlib.ticker import MultipleLocator
import copy
import string




#sys.path.append("/home/escalafl/workspace/hiphi/src/hiphi/utils")
#list holding the benchmark information with entries: ("description", time as float in seconds)
benchmark=[]

#The next lines are only relevant if cache or network saving is enabled.
on_lok21 = os.path.lexists("/local2/tmp/escalafl/")
on_lok09 = os.path.lexists("/local/escalafl/on_lok09")
on_lok10 = os.path.lexists("/local/escalafl/on_lok10")
on_zappa01 = os.path.lexists("/local/escalafl/on_zappa01")
if on_lok21:
    network_base_dir = "/local2/tmp/escalafl/Alberto/SavedNetworks"
    n_parallel = 3
    print "Working on lok21"
elif on_lok09:
    network_base_dir = "/local/escalafl/Alberto/SavedNetworks"
    n_parallel = 3
    print "Working on lok09"
elif on_lok10:
    network_base_dir = "/local/escalafl/Alberto/SavedNetworks"
    n_parallel = 3
    print "Working on lok10"
else:
    network_base_dir = "/local/tmp/escalafl/Alberto/SavedNetworks"
    n_parallel = 9
    print "Working on unknown host"

pipeline_base_dir = "/local/escalafl/Alberto/Pipelines"
images_base_dir = "/local/escalafl/Alberto/ImagesFaceDetection"
networks_base_dir = "/local/tmp/escalafl/Alberto/SavedNetworks"
classifiers_base_dir = "/local/tmp/escalafl/Alberto/SavedClassifiers"

#scheduler = mdp.parallel.ThreadScheduler(n_threads=n_parallel)
scheduler = None

cache_obj = cache.Cache("", "")


numpy.random.seed(123456)

print "LOADING PIPELINE DESCRIPTION FILE"        

pipeline_filenames = cache.find_filenames_beginning_with(pipeline_base_dir, "Pipeline", recursion=False, extension=".txt")
print "%d pipelines found:"%len(pipeline_filenames), pipeline_filenames

if len(pipeline_filenames) <= 0:
    print "ERROR: No pipelines found in directory", pipeline_base_dir
    quit()

enable_select_pipeline = False
for i, pipeline_filename in enumerate(pipeline_filenames):
    pipeline_base_filename = string.split(pipeline_filename, sep=".")[0] #Remove extension          
#    (NetworkType, Imsize) = pipeline_info = cache_read.load_obj_from_cache(base_dir="/", base_filename=pipeline_base_filename, verbose=True)
    print "Pipeline %d: %s"%(i, pipeline_base_filename)
                    
if enable_select_pipeline ==True:
    selected_pipeline = int( raw_input("Please select a pipeline [0--%d]:"%(len(pipeline_filenames)-1) ))
else:
    selected_pipeline = 0
print "Pipeline %d was selected"%selected_pipeline           

pipeline_filename = pipeline_filenames[selected_pipeline]
#pipeline_base_filename = string.split(pipeline_filename, sep=".")[0]

t0 = time.time()

pipeline_file = open(pipeline_filename, "rb")

num_networks = int(pipeline_file.readline())
print "Pipeline contains %d network/classifier pairs"%num_networks
#num_networks = 1

tmp_string = pipeline_file.readline()
tmp_strings = string.split(tmp_string, " ")
net_Dx = int(tmp_strings[0])
net_Dy = int(tmp_strings[1])
net_mins = float(tmp_strings[2])
net_maxs = float(tmp_strings[3])

subimage_width = int(tmp_strings[4])
subimage_height = int(tmp_strings[5])
#WARNING! HACK!!!
regression_width = int(tmp_strings[6])
regression_height = int(tmp_strings[7])

#regression_width = regression_height = 128 #Regression data assumes subimage has this size

network_filenames = []
classifier_filenames = [] 
for i in range(num_networks):
    network_filename = pipeline_file.readline().rstrip()[0:-5]
    network_filenames.append(network_filename)    
    classifier_filename = pipeline_file.readline().rstrip()[0:-5]
    classifier_filenames.append(classifier_filename)
    
print "networks:", network_filenames
print "classifiers:", classifier_filenames

networks = []
for network_filename in network_filenames:
    #load_obj_from_cache(self, hash_value=None, base_dir = None, base_filename=None, verbose=True)
    #[flow, layers, benchmark, Network]
    all_data = cache_obj.load_obj_from_cache(None, base_dir=networks_base_dir, base_filename=network_filename, verbose=True) 
    networks.append(all_data[0]) #Keep only the flows

classifiers = []
for classifier_filename in classifier_filenames:
    #load_obj_from_cache(self, hash_value=None, base_dir = None, base_filename=None, verbose=True)
    classifier = cache_obj.load_obj_from_cache(None, base_dir=classifiers_base_dir, base_filename=classifier_filename, verbose=True) 
    classifiers.append(classifier)

t1 = time.time()
benchmark.append(("Network and Classifier loading", t1-t0))
                 
image_filename_prefix = "image"
image_filenames = cache.find_filenames_beginning_with(images_base_dir, image_filename_prefix, recursion=False, extension=".jpg")
image_filenames.sort()
print "Images:", image_filenames
num_images = len(image_filenames)
if num_images <= 0:
    raise Exception("No images Found")
num_images = 1 #TODO: more images

#sampling_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
sampling_values = [0.475, 0.95, 1.9, 3.8]

sampling_value = 1.6 #1.9, 1.6 for group small, 0.8 for smaller images sport, 1.0 for medium

images = load_images(image_filenames, format="L", sampling=sampling_value)

t2 = time.time()
benchmark.append(("Image loading", t2-t1))
    
im_height = images[0].size[1]
im_width = images[0].size[0]
    
    
patch_width = subimage_width * sampling_value 
patch_height = subimage_height * sampling_value 

#add random inicialization between 0 and net_Dx * 2.0 * patch_width/regression_width, the same for Y
posX_values = numpy.arange(0, im_width-(patch_width-1), net_Dx * 2.0 * patch_width/regression_width)
posY_values = numpy.arange(0, im_height-(patch_height-1), net_Dy * 2.0 * patch_height/regression_height)

max_Dx_diff = net_Dx * patch_width/regression_width
max_Dy_diff = net_Dy * patch_height/regression_height

print "posX_values=", posX_values
print "posY_values=", posY_values

#actually all resolutions can be processed also at once!
orig_num_subimages = len(posX_values) * len(posY_values)
orig_subimage_coordinates = numpy.zeros((orig_num_subimages, 4))

#subimage_width, subimage_height
for j, posY in enumerate(posY_values):
    for i, posX in enumerate(posX_values):
        orig_subimage_coordinates[j*len(posX_values)+i] = numpy.array([posX, posY, posX+patch_width-1, posY+patch_height-1])

base_magnitude = patch_width **2 + patch_height**2

#print "subimage_coordinates", subimage_coordinates
#base_estimation = orig_subimage_coordinates + 0.0

orig_image_indices = numpy.zeros(num_images * orig_num_subimages, dtype="int")
for im, image in enumerate(images):
#    for xx in range(orig_num_subimages):
    orig_image_indices[im * orig_num_subimages:(im+1)*orig_num_subimages] = im 

#Check that this is not memory inefficient
#subimage_coordinates = subimage_coordinates times num_images
#image = images[0] #TODO => loop

orig_colors = numpy.random.uniform(0.0, 1.0, size=(orig_num_subimages,3))

curr_num_subimages = orig_num_subimages + 0
curr_subimage_coordinates = orig_subimage_coordinates + 0
curr_invalid_subimages = numpy.zeros(curr_num_subimages, dtype='bool')
curr_image_indices = orig_image_indices + 0
curr_orig_index = numpy.arange(curr_num_subimages)

f0 = plt.figure()
plt.suptitle("Iterative Face Detection")
sel_image = 0
p11 = plt.subplot(3,4,1)
p12 = plt.subplot(3,4,2)
p13 = plt.subplot(3,4,3)
p14 = plt.subplot(3,4,4)
p21 = plt.subplot(3,4,5)
p22 = plt.subplot(3,4,6)
p23 = plt.subplot(3,4,7)
p24 = plt.subplot(3,4,8)
p31 = plt.subplot(3,4,9)
p32 = plt.subplot(3,4,10)
p33 = plt.subplot(3,4,11)
p34 = plt.subplot(3,4,12)
p41 = p34

im_disp = numpy.asarray(images[sel_image])
p11.imshow(im_disp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
#color=(r_color[sig], g_color[sig], b_color[sig])

for ii, (x0, y0, x1, y1) in enumerate(orig_subimage_coordinates):
#    p11.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=colors[ii] )
    p11.plot([x0, x1], [y0, y1], color=orig_colors[ii] )

###from matplotlib.lines import Line2D
##line = mpl.lines.Line2D([10,20,30,40,500], [10,40, 35, 20, 500],
##              linewidth=4, color='green', transform=f0.transFigure)
##f0.lines.append(line)


subplots = [p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34, p41]

t3 = time.time()
benchmark.append(("Window Creation", t3-t2))

t_latest = t3
for i in range(num_networks):
    #Get arrays
    subimages = extract_subimages(images, curr_image_indices, curr_subimage_coordinates, (subimage_width, subimage_height) )
    subimages_arr = images_asarray(subimages)+0.0
    t_afterloading = time.time()

    sl = networks[i].execute(subimages_arr, benchmark=benchmark)
    print "Network %d processed all subimages"%i
    reg_num_signals = classifiers[i].input_dim
    t_class = time.time()
    reg_out = classifiers[i].GaussianRegression(sl[:,0:reg_num_signals])

    if i%4 == 0: #POS_X       
        width = curr_subimage_coordinates[:, 2] - curr_subimage_coordinates[:, 0]
        reg_out = reg_out * width / regression_width
#        print "Regression Output scaled:", reg_out
#        print "Correcting coordinates (X)"
        curr_subimage_coordinates[:, 0] = curr_subimage_coordinates[:, 0] - reg_out  #X0
        curr_subimage_coordinates[:, 2] = curr_subimage_coordinates[:, 2] - reg_out  #X1
    elif i%4 == 1:     #POS_Y
        height = curr_subimage_coordinates[:, 3] - curr_subimage_coordinates[:, 1]
        reg_out = reg_out * height / regression_height
#        print "Regression Output scaled:", reg_out
#        print "Correcting coordinates (Y)"
        curr_subimage_coordinates[:, 1] = curr_subimage_coordinates[:, 1] - reg_out  #Y0
        curr_subimage_coordinates[:, 3] = curr_subimage_coordinates[:, 3] - reg_out  #Y1
    elif i%4 == 2: #SCALE
        old_width = curr_subimage_coordinates[:, 2] - curr_subimage_coordinates[:, 0]
        old_height = curr_subimage_coordinates[:, 3] - curr_subimage_coordinates[:, 1]
        x_center = (curr_subimage_coordinates[:, 2] + curr_subimage_coordinates[:, 0])/2.0
        y_center = (curr_subimage_coordinates[:, 3] + curr_subimage_coordinates[:, 1])/2.0

        desired_sampling = 0.825 #1 or better: (0.55 + 1.1)/2

        width = old_width / reg_out * desired_sampling
        height = old_height / reg_out * desired_sampling
#        print "Regression Output scaled:", reg_out
#        print "Correcting scale (X)"
        curr_subimage_coordinates[:, 0] = x_center - width / 2.0
        curr_subimage_coordinates[:, 2] = x_center + width / 2.0
        curr_subimage_coordinates[:, 1] = y_center - height / 2.0
        curr_subimage_coordinates[:, 3] = y_center + height / 2.0
    else:
        pass #regression does not need to modify subimage coordinates

    if i%4 in [0,1,2]: 
        magic = 1.2 #1.2 for group small
        #out of image
        out_of_borders_images = (curr_subimage_coordinates[:,0]<0) | (curr_subimage_coordinates[:,1]<0) | \
        (curr_subimage_coordinates[:,2]>=im_width) | (curr_subimage_coordinates[:,3]>=im_height)      
        #too large or small
        
        subimage_magnitudes = ((curr_subimage_coordinates[:,0:2] - curr_subimage_coordinates[:,2:4])**2).sum(axis=1)
        #sqrt(2)/2*orig_diagonal = 1/sqrt(2)*orig_diagonal < subimage_diagonal < sqrt(2)*orig_diagonal ???
        too_large_small_images = (subimage_magnitudes > 4*base_magnitude) | (subimage_magnitudes < 0.25 * base_magnitude)      
        #too far away horizontally

        subimage_deltas_x = (curr_subimage_coordinates[:,2] + curr_subimage_coordinates[:,0])/2 - (orig_subimage_coordinates[curr_orig_index][:,2] + orig_subimage_coordinates[curr_orig_index][:,0])/2
        subimage_deltas_y = (curr_subimage_coordinates[:,3] + curr_subimage_coordinates[:,1])/2 - (orig_subimage_coordinates[curr_orig_index][:,3] + orig_subimage_coordinates[curr_orig_index][:,1])/2

        print "subimage_deltas is: ", subimage_deltas_x
        x_far_images = numpy.abs(subimage_deltas_x) > (max_Dx_diff * magic)
        print "wrong x_center is:", (curr_subimage_coordinates[:,2][x_far_images] + curr_subimage_coordinates[:,0][x_far_images])/2
        print "wrong x_center was:", (orig_subimage_coordinates[:,2][curr_orig_index[x_far_images]] + orig_subimage_coordinates[:,0][curr_orig_index[x_far_images]])/2
        
        y_far_images = numpy.abs(subimage_deltas_y) > (max_Dy_diff * magic)
        print "wrong y_center is:", (curr_subimage_coordinates[:,3][y_far_images] + curr_subimage_coordinates[:,1][y_far_images])/2
        print "wrong y_center was:", (orig_subimage_coordinates[:,3][curr_orig_index[y_far_images]] + orig_subimage_coordinates[:,1][curr_orig_index[y_far_images]])/2
       
        new_wrong_images = out_of_borders_images | too_large_small_images | x_far_images | y_far_images
         
        print "new_wrong_images %d = out_of_borders_images %d + too_large_small_images %d + x_far_images %d + y_far_images %d" % \
        (new_wrong_images.sum(), out_of_borders_images.sum(), too_large_small_images.sum(), x_far_images.sum(), y_far_images.sum())

#Also discard too similar subimages: 
    else: #Classifier
        cut_off_face = 0.25
        new_wrong_images = reg_out >= cut_off_face

#Update subimage patch information, on temporal variables
    new_num_subimages = curr_num_subimages - new_wrong_images.sum()
    new_subimage_coordinates = curr_subimage_coordinates[new_wrong_images==0]+0.0
    new_invalid_subimages = numpy.zeros(new_num_subimages, dtype='bool')
    new_image_indices = curr_image_indices[new_wrong_images==0] + 0
    new_orig_index = curr_orig_index[new_wrong_images==0] + 0


    print "%d / %d valid images"%(new_num_subimages, curr_num_subimages)
#Overwrite current values
    curr_num_subimages = new_num_subimages
    curr_subimage_coordinates = new_subimage_coordinates
    curr_invalid_subimages = new_invalid_subimages
    curr_image_indices = new_image_indices 
    curr_orig_index = new_orig_index


    subplot = subplots[i]
    subplot.imshow(im_disp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
    for j, (x0, y0, x1, y1) in enumerate(curr_subimage_coordinates):
#        if invalid_subimages[j] == False and False:
#            subplot.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=orig_colors[curr_orig_index[j]] )
#        if invalid_subimages[j] == False:
        subplot.plot([x0, x1], [y0, y1], color=orig_colors[curr_orig_index[j]] )
            
#    print "subimage_coordinates", subimage_coordinates
    
    t_new = time.time()
    benchmark.append(("Pipeline Image Loading Step %d"%i, t_afterloading-t_latest))
    benchmark.append(("Pipeline Network Step %d"%i, t_class-t_afterloading))
    benchmark.append(("Pipeline Classifier Step %d"%i, t_new-t_class))

    t_latest = t_new
##print "Done loading subimages_train: %d Samples"%len(subimages_train)
##flow, layers, benchmark, Network = cache.load_obj_from_cache(network_hash, network_base_dir, "Network", verbose=True)   
##print "Done loading network: " + Network.name

for (msg,t) in benchmark:
    print msg, " ", t

print "Total detection time: ", t_latest - t3, "patches processed:", orig_num_subimages
plt.show()
#from GenerateSystemParameters import Linear4LNetwork as Network
#This defines the sequences used for training, and testing
#See also: ParamsGender, ParamsAngle, ParamsIdentity,  ParamsTransX, ParamsAge,  ParamsRTransX, ParamsRTransY, ParamsRScale


##from GenerateSystemParameters import ParamsRTransY as Parameters
###from GenerateSystemParameters import ParamsRScale as Parameters
##
##min_cutoff = -30.0
##max_cutoff = 30.0
##
##enable_reduced_image_sizes = True
##if enable_reduced_image_sizes:
##    reduction_factor = 2.0 # (the inverse of a zoom factor)
##    Parameters.name = Parameters.name + ". Resized images"
##    for iSeq in (Parameters.iTrain, Parameters.iSeenid, Parameters.iNewid): 
##        # iSeq.trans = iSeq.trans / 2
##        pass
##
##    for sSeq in (Parameters.sTrain, Parameters.sSeenid, Parameters.sNewid): 
##        sSeq.subimage_width = sSeq.subimage_width / reduction_factor
##        sSeq.subimage_height = sSeq.subimage_height / reduction_factor 
##        sSeq.pixelsampling_x = sSeq.pixelsampling_x * reduction_factor
##        sSeq.pixelsampling_y = sSeq.pixelsampling_y * reduction_factor
##        if sSeq.trans_sampled == True:
##            sSeq.translations_x = sSeq.translations_x / reduction_factor
##            sSeq.translations_y = sSeq.translations_y / reduction_factor
##    
##iTrain = Parameters.iTrain
##sTrain = Parameters.sTrain
##iSeenid = Parameters.iSeenid
##sSeenid = Parameters.sSeenid
##iNewid = Parameters.iNewid
##sNewid = Parameters.sNewid
##
##image_files_training = iTrain.input_files
##num_images_training = num_images = iTrain.num_images
##
##block_size = Parameters.block_size
###WARNING!!!!!!!
##train_mode = Parameters.train_mode
### = "mixed", "sequence", "complete"
##
##block_size_L0=block_size
##block_size_L1=block_size
##block_size_L2=block_size
##block_size_L3=block_size
##block_size_exec=block_size #(Used only for random walk)
##
##seq = sTrain
##
###oTrain = NetworkOutput()
##
###WARNING!!!!!!!
###Move this after network loadings
##hack_image_sizes = [135, 90, 64, 32]
###Warning!!! hack_image_size = False
##hack_image_size = 64
##enable_hack_image_size = True
##if enable_hack_image_size:
##    seq.subimage_height = seq.subimage_width = hack_image_size
##    sSeenid.subimage_height = sSeenid.subimage_width = hack_image_size
##    sNewid.subimage_height = sNewid.subimage_width = hack_image_size
##    
###Filter used for loading images with transparent background
###filter = generate_color_filter2((seq.subimage_height, seq.subimage_width))
##alpha = 4.0 # mask 1 / f^(alpha/2) => power 1/f^alpha
##filter = filter_colored_noise2D_imp((seq.subimage_height, seq.subimage_width), alpha)
###back_type = None
###filter = None

        
#TODO: CREATE USER INTERFACE: users should be able to select a trained network, or train a new one
#                             and to set the training variables: type of training
#                             Also, users should set the analysis parameters: num. of signals 

###Work in process, for now keep cache disabled
##cache_read_enabled = True
##if cache_read_enabled:
##    if on_lok21:
##        cache_read = cache.Cache("/local2/tmp/escalafl/Alberto/SavedNetworks", "")
##    else:
##        cache_read = cache.Cache("/local/tmp/escalafl/Alberto/SavedNetworks", "")
##else:
##    cache_read = None
##
##cache_write_enabled = True
##if cache_write_enabled:
##    if on_lok21:
##        cache_write = cache.Cache("/local2/tmp/escalafl/Alberto/SavedNetworks", "")
##    else:
##        cache_write = cache.Cache("/local/tmp/escalafl/Alberto/SavedNetworks", "")
##else:
##    cache_write = None
##
##network_saving_enabled = True
##if network_saving_enabled:
##    if on_lok21:
##        network_write = cache.Cache("/local2/tmp/escalafl/Alberto/SavedNetworks", "")
##    elif on_lok09:
##        network_write = cache.Cache("/local/escalafl/Alberto/SavedNetworks", "")
##    else:
##        network_write = cache.Cache("/local/tmp/escalafl/Alberto/SavedNetworks", "")
##else:
##    network_write = None
##
##classifier_read_enabled = False
##if classifier_read_enabled:
##    if on_lok21:
##        classifier_read = cache.Cache("/local2/tmp/escalafl/Alberto/SavedClassifiers", "")
##    elif on_lok09:
##        classifier_read = cache.Cache("/local/escalafl/Alberto/SavedClassifiers", "")
##    else:
##        classifier_read = cache.Cache("/local/tmp/escalafl/Alberto/SavedClassifiers", "")
##else:
##    classifier_read = None
##    
##classifier_saving_enabled = True
##if classifier_saving_enabled:
##    if on_lok21:
##        classifier_write = cache.Cache("/local2/tmp/escalafl/Alberto/SavedClassifiers", "")
##    elif on_lok09:
##        classifier_write = cache.Cache("/local/escalafl/Alberto/SavedClassifiers", "")
##    else:
##        classifier_write = cache.Cache("/local/tmp/escalafl/Alberto/SavedClassifiers", "")
##else:
##    classifier_write = None
    

###network_filenames = cache.find_filenames_beginning_with(network_base_dir, "Network", recursion=False, extension=".pckl")
###print "%d networks found:"%len(network_filenames), network_filenames
###
###load_latest = False
###
###if len(network_filenames) > 0 and load_latest==True:
###    print "******************************************"
###    print "Loading Trained Network from Disk         "
###    print "******************************************"
####    flow, layers, benchmark, Network, subimages_train, sl_seq_training = cache.unpickle_from_disk(network_filenames[-1])
###    network_filename = network_filenames[-1]
###    
###    network_base_filename = string.split(string.split(network_filename, sep=".")[0], sep="/")[-1]
###    network_hash = string.split(network_base_filename, sep="_")[1]
###
###    print "network_base_dir", network_base_dir
###    print "network_filename:", network_filename
###    print "network_basefilename:", network_base_filename
###    print "network_hash:", network_hash
###
####    flow, layers, benchmark, Network = cache.unpickle_from_disk(network_filename)   
###    subimages_train = cache.load_array_from_cache(hash_value = "1259249913", base_dir = network_base_dir, base_filename="subimages_train_Network", verbose=True)
###    print "Done loading subimages_train: %d Samples"%len(subimages_train)
###
###    flow, layers, benchmark, Network = cache.load_obj_from_cache(network_hash, network_base_dir, "Network", verbose=True)   
###    print "Done loading network: " + Network.name
###
###    
###    
####    subimages_train_iter = cache.UnpickleLoader2(path=network_base_dir, basefilename="subimages_train_" + network_basefilename, verbose=True)
####   
####    subimages_train = cache.from_iter_to_array(subimages_train_iter, continuous=False, block_size=1, verbose=False)
####    del subimages_train_iter
###    
###    sl_seq_training = cache.load_array_from_cache(hash_value = "1259249913", base_dir = network_base_dir, base_filename="sl_seq_training_Network", verbose=True)
###        
####    sl_seq_training_iter = cache.UnpickleLoader2(path=network_base_dir, basefilename="sl_seq_training_" + network_basefilename)
####    sl_seq_training = cache.from_iter_to_array(sl_seq_training_iter, continuous=False, block_size=1, verbose=False)
####    del sl_seq_training_iter
###
####      
####    subimages_train = load_image_data(seq.input_files, seq.image_width, seq.image_height, seq.subimage_width, \
####                                seq.subimage_height, seq.pixelsampling_x, seq.pixelsampling_y, \
####                                seq.subimage_first_row, seq.subimage_first_column, seq.add_noise_L0, \
####                                seq.convert_format, seq.translations_x, seq.translations_y, seq.trans_sampled, background_type=seq.background_type, color_background_filter=filter, verbose=False)
####    sl_seq_training = flow.execute(subimages_train)
###
###else:
###    print "Generating Network..."
###    # Networks available: voidNetwork1L, Test_Network, linearNetwork4L, NL_Network5L, linearNetwork5L, linearNetworkT6L, TestNetworkT6L, 
###    # linearNetworkU11L, TestNetworkU11L, nonlinearNetworkU11L, TestNetworkPCASFAU11L, linearPCANetworkU11L,  u08expoNetworkU11L 
###    # linearWhiteningNetwork11L
###    from GenerateSystemParameters import TestNetworkU11L as Network
####    from GenerateSystemParameters import TestNetworkU11L as Network
###
###    #Usually true for voidNetwork1L, but might be also activated for other networks
###    use_full_sl_output = False
###
###    #WARNING
###    #Warning
###    #WARNING
###    #Todo: Correct network creation, to avoid redundant layers (without enough fan in)
###    #Todo: Correct update of last PCA Node into Whitening
####    Network.L7 = None
####    Network.L8 = None
###    Network.L9 = None
###    Network.L10 = None
####  RTransX  
####    Network.L8.sfa_node_class = mdp.nodes.WhiteningNode
####    Network.L8.sfa_out_dim = 50
###
####    Network.L2=None
####    Network.L3=None
####    Network.L4=None
####    Network.L5=None
###    #TODO: try loading subimage_data from cache...
###    #TODO: Add RGB support
###    #TODO: Verify that at least iSeq is the same
###    load_subimages_train_signal_from_cache = True
###    enable_select_train_signal = True
###    
###
###    subimages_train_signal_in_cache = False
###    if cache_read and load_subimages_train_signal_from_cache:
###        print "Looking for subimages_train in cache..."
###        
###        info_beginning_filename = "subimages_info"  
###        subimages_info_filenames = cache.find_filenames_beginning_with(network_base_dir, info_beginning_filename, recursion=False, extension=".pckl")
###        print "The following possible training sequences were found:"
###        if len(subimages_info_filenames) > 0:
###            for i, info_filename in enumerate(subimages_info_filenames):
###                info_base_filename = string.split(info_filename, sep=".")[0] #Remove extension          
###                (iTrainInfo, sTrainInfo) = subimages_info = cache_read.load_obj_from_cache(base_dir="/", base_filename=info_base_filename, verbose=True)
###                print "%d: %s, with %d images of width=%d, height=%d"%(i, iTrainInfo.name, iTrainInfo.num_images, sTrainInfo.subimage_width, sTrainInfo.subimage_height)
###                    
###            if enable_select_train_signal==True:
###                selected_train_sequence = int(raw_input("Please select a training sequence (-1=Reload new data):"))
###            else:
###                selected_train_sequence = 0
###            print "Training sequence %d was selected"%selected_train_sequence
###
###            if selected_train_sequence >= 0:
###                info_filename = subimages_info_filenames[selected_train_sequence]
###                info_base_filename = string.split(info_filename, sep=".")[0] #Remove extension          
###                (iTrain, sTrain) = cache_read.load_obj_from_cache(base_dir="/", base_filename=info_base_filename, verbose=True)
###                signal_base_filename = string.replace(info_base_filename, "subimages_info", "subimages_train")
###                
###                if cache_read.is_splitted_file_in_filesystem(base_dir="/", base_filename=signal_base_filename):
###                    print "Subimages train signal found in cache..."
###                    subimages_train = cache_read.load_array_from_cache(base_dir="/", base_filename=signal_base_filename, verbose=True)
###                    subimages_train_signal_in_cache = True
###                    print "Subimages train signal loaded from cache with shape: ",
###                    print subimages_train.shape
###                    if cache_write:
###                        subimages_train_hash = cache.hash_object(subimages_train).hexdigest()
###                else:
###                    print "Subimages training signal UNEXPECTEDLY NOT FOUND in cache:", signal_base_filename
###                    quit()
###            
####        subimages_ndim = sTrain.subimage_height * sTrain.subimage_width
####        signal_beginning_filename = "subimages_train_%d"%subimages_ndim
####        subimages_train_filenames = cache.find_filenames_beginning_with(network_base_dir, signal_beginning_filename, recursion=False, extension=".pckl")
####        
####        print "The following possible subimage_train_signals were found:", subimages_train_filenames
####        if len(subimages_train_filenames) > 0:
####            #remove extension .pckl
####            signal_filename = subimages_train_filenames[-1]
####            print "Signal_filename selected:", signal_filename 
####            signal_base_filename = string.split(signal_filename, sep=".")[0]
####            #remove last 6 characters: "_S0000"
####            signal_base_filename = signal_base_filename[0:-6]
###    
###    if subimages_train_signal_in_cache == False:        
###        subimages_train = load_image_data(seq.input_files, seq.image_width, seq.image_height, seq.subimage_width, \
###                                    seq.subimage_height, seq.pixelsampling_x, seq.pixelsampling_y, \
###                                    seq.subimage_first_row, seq.subimage_first_column, seq.add_noise_L0, \
###                                    seq.convert_format, seq.translations_x, seq.translations_y, seq.trans_sampled, background_type=seq.background_type, color_background_filter=filter, verbose=False)
###
###
###    if cache_write and subimages_train_signal_in_cache == False:
###        print "Caching Train Signal..."
###        subimages_ndim = subimages_train.shape[1]
###        subimages_time = str(int(time.time()))
###        iTrain_hash = cache.hash_object(iTrain).hexdigest()
###        sTrain_hash = cache.hash_object(sTrain).hexdigest()   
###        subimages_base_filename = "subimages_train_%s_%s_%s_%s"%((subimages_ndim, subimages_time, iTrain_hash, sTrain_hash))
###        subimages_train_hash = cache_write.update_cache(subimages_train, base_filename=subimages_base_filename, overwrite=True, verbose=True)
###        subimages_info = (iTrain, sTrain)
###        subimages_info_filename = "subimages_info_%s_%s_%s_%s"%((subimages_ndim, subimages_time, iTrain_hash, sTrain_hash))
###        subimages_info_hash = cache_write.update_cache(subimages_info, base_filename=subimages_info_filename, overwrite=True, verbose=True)
###        
###    t1 = time.time()
###    print seq.num_images, "Train Images loaded in %0.3f s"% ((t1-t0))
###    #benchmark.append(("Load Info and Train Images", t1-t0))  
###
###    save_images_training = False
###    save_images_training_base_dir = "/local/tmp/escalafl/Alberto/saved_images"
###    if save_images_training:
###        print "saving images..."
###        decimate =  45
###        for i, x in enumerate(subimages_train):
###            if i%decimate == 0:
###                im_raw = numpy.reshape(x, (seq.subimage_width, seq.subimage_height))
###                im = scipy.misc.toimage(im_raw, mode=seq.convert_format)
###                fullname = os.path.join(save_images_training_base_dir, "image%05d.png"%(i/decimate))
###                im.save(fullname)
###        print "done, finishing"
###        quit()
###
###    print "******************************************"
###    print "Creating hierarchy through network_builder"
###    print "******************************************"
###    flow, layers, benchmark = network_builder.CreateNetwork(Network, sTrain.subimage_width, sTrain.subimage_height, block_size, train_mode, benchmark)
###    
###    print "*****************************"
###    print "Training hierarchy ..."
###    print "*****************************"
###    
###
####    enable_parallel = False
####    if enable_parallel == True:
#####        flow = mdp.Flow(flow[0:])
####        print "original flow is", flow
####        scheduler = mdp.parallel.ProcessScheduler(n_processes=4)
####        print "***********1"
####        flow = make_flow_parallel(flow)
####        print "parallel flow is", flow
####        print "***********2"
###
###        
###    subimages_p = subimages = subimages_train
###    #subimages_p = subimages
###    #DEFINE TRAINING TYPE. SET ONE OF THE FOLLOWING VARIABLES TO TRUE
###    #Either use special (most debugged and efficient) or storage_iterator (saves memory)
###    special_training = True
###    iterator_training = False
###    storage_iterator_training = False    
###    
###    if special_training is True:
###        ttrain0 = time.time()
###        sl_seq = sl_seq_training = flow.special_train_cache_scheduler(subimages_p, verbose=True, benchmark=benchmark, cache_read=cache_read, cache_write=cache_write, scheduler=scheduler, n_parallel=n_parallel)
###        
###        ttrain1 = time.time()
###        print "Network trained (specialized way) in time %0.3f s"% ((ttrain1-ttrain0))
###        benchmark.append(("Network training  (specialized way)", ttrain1-ttrain0))
###    elif iterator_training is True:
###        ttrain0 = time.time()
###    #WARNING, introduce smart way of computing chunk_sizes
###        input_iter = cache.chunk_iterator(subimages_p, 4, block_size, continuous=False)
###        
###    #    sl_seq = sl_seq_training = flow.iterator_train(input_iter)
###        flow.iterator_train(input_iter, block_size, continuous=True)
###        sl_seq = sl_seq_training = flow.execute(subimages_p)
###        
###        ttrain1 = time.time()
###        print "Network trained (iterator way) in time %0.3f s"% ((ttrain1-ttrain0))
###        benchmark.append(("Network training (iterator way)", ttrain1-ttrain0))
###    elif storage_iterator_training is True:
###        ttrain0 = time.time()
###    #Warning: introduce smart way of computing chunk_sizes
###    #    input_iter = chunk_iterator(subimages_p, 15 * 15 / block_size, block_size, continuous=False)
###        input_iter = cache.chunk_iterator(subimages_p, 4, block_size, continuous=False)
###        
###    #    sl_seq = sl_seq_training = flow.iterator_train(input_iter)
###    #WARNING, continuous should not always be true
###        flow.storage_iterator_train(input_iter, "/local/tmp/escalafl/simulations/gender", "trainseq", block_size, continuous=True)
###    
###        output_iterator = cache.UnpickleLoader2(path="/local/tmp/escalafl/simulations/gender", \
###                                          basefilename="trainseq"+"_N%03d"%(len(flow)-1))
###            
###        sl_seq = sl_seq_training = cache.from_iter_to_array(output_iterator, continuous=False, block_size=block_size, verbose=0)
###        del output_iterator
###        
###        ttrain1 = time.time()
###        print "Network trained (storage iterator way) in time %0.3f s"% ((ttrain1-ttrain0))
###        benchmark.append(("Network training (storage iterator way)", ttrain1-ttrain0))
###    else:
###        ttrain0 = time.time()
###        flow.train(subimages_p)
###        y = flow.execute(subimages_p[0:1]) #stop training
###        sl_seq = sl_seq_training = flow.execute(subimages_p)
###        ttrain1 = time.time()
###        print "Network trained (MDP way) in time %0.3f s"% ((ttrain1-ttrain0))
###        benchmark.append(("Network training (MDP way)", ttrain1-ttrain0))
###    
###    nodes_in_flow = len(flow)
###    last_sfa_node = flow[nodes_in_flow-1]
###    if isinstance(last_sfa_node, mdp.hinet.CloneLayer) or \
###    isinstance(last_sfa_node, mdp.hinet.Layer):
###        last_sfa_node = last_sfa_node.nodes[0]
###
###    if isinstance(last_sfa_node, mdp.nodes.SFANode):
###        sl_seq = sl_seq_training = more_nodes.sfa_pretty_coefficients(last_sfa_node, sl_seq_training)
###    else:
###        print "SFA coefficients not made pretty, last node was not SFA!!!"
###
####    try:
####        cache.pickle_to_disk([flow, layers, benchmark, Network, subimages_train, sl_seq_training], os.path.join(network_base_dir, "Network_" + str(int(time.time()))+ ".pckl"))
###    network_hash = str(int(time.time()))
####    network_filename = "Network_" + network_hash + ".pckl"
###
###    if network_write:
###        print "Saving flow, layers, benchmark, Network ..."
###        #update cache is not adding the hash to the filename,so we add it manually
###        network_write.update_cache(flow, None, network_base_dir, "JustFlow"+sTrain.name+"_"+network_hash, overwrite=True, use_hash=network_hash, verbose=True)
###        network_write.update_cache(layers, None, network_base_dir, "JustLayers"+sTrain.name+"_"+network_hash, overwrite=True, use_hash=network_hash, verbose=True)
###        network_write.update_cache(benchmark, None, network_base_dir, "JustBenchmark"+sTrain.name+"_"+network_hash, overwrite=True, use_hash=network_hash, verbose=True)
###        network_write.update_cache(Network, None, network_base_dir, "JustNetwork"+sTrain.name+"_"+network_hash, overwrite=True, use_hash=network_hash, verbose=True)
###
###        network_write.update_cache([flow, layers, benchmark, Network], None, network_base_dir, "Network"+Network.name+"_ParName"+Parameters.name+"_"+network_hash, overwrite=True, use_hash=network_hash, verbose=True)
###        #obj, obj_data=None, base_dir = None, base_filename=None, overwrite=True, use_hash=None, verbose=True
###
###    if cache_write:
###        print "Caching sl_seq_training  Signal... (however, it's never read!)"
###        signal_ndim = sl_seq_training.shape[1]
###        signal_time = str(int(time.time()))
###        flow_hash = cache.hash_object(flow).hexdigest()   
###        signal_base_filename = "sfa_signal_%s_%s_%s_%s"%((signal_ndim, signal_time, subimages_train_hash, flow_hash))
###        cache_write.update_cache(sl_seq_training, base_filename=signal_base_filename, overwrite=True, verbose=True)
###    
###
####    cache.pickle_to_disk([flow, layers, benchmark, Network], os.path.join(network_base_dir, network_filename ))
####    print "2"
####    subimages_train_iter = chunk_iterator(subimages_train, chunk_size=5000, block_size=1, continuous=False, verbose=True)
####    print "3"
####    cache.save_iterable2(subimages_train_iter, path=network_base_dir, basefilename="subimages_train_" + network_filename)   
####    print "4"
####    del subimages_train_iter
####    print "5"
####    sl_seq_training_iter = chunk_iterator(sl_seq_training, chunk_size=200000, block_size=1, continuous=False, verbose=True)
####    print "6"
####    cache.save_iterable2(sl_seq_training_iter, path=network_base_dir, basefilename="sl_seq_training_" + network_filename)
####    print "7"
####    del sl_seq_training_iter
####    print "8"
###    print "Saving Finished"
####    except:
####        print "Saving Failed, reason:", ex
###        
###
####Improve this!
####Fixing some unassigned variables
###subimages_p = subimages = subimages_train
###sl_seq = sl_seq_training
###
###print "Done creating / training / loading network"   
###y = flow.execute(subimages_p[0:1])
###more_nodes.describe_flow(flow)
###hierarchy_out_dim = y.shape[1]
###
###results = SystemParameters.ExperimentResult()
###results.name = Parameters.name
###results.network_name = Network.name
###results.layers_name = []
###for lay in layers:
###    results.layers_name.append(lay.name)
###results.iTrain = iTrain
###results.sTrain = sTrain
###results.iSeenid = iSeenid
###results.sSeenid = sSeenid
###results.iNewid = iNewid
###results.sNewid = sNewid
###
###
###print "Computing typical delta, eta values for Train SFA Signal"
###t_delta_eta0 = time.time()
###results.typical_delta_train, results.typical_eta_train = sfa_libs.comp_typical_delta_eta(sl_seq_training, block_size, num_reps=200)
###results.brute_delta_train = sfa_libs.comp_delta(sl_seq_training)
###results.brute_eta_train = sfa_libs.comp_eta(sl_seq_training)
###t_delta_eta1 = time.time()
###print "typical_delta_train=", results.typical_delta_train
####print "typical_eta_train=", results.typical_eta_train
####print "brute_delta_train=", results.brute_delta_train
####print "brute_eta_train=", results.brute_eta_train
###
###print "computed delta/eta in %0.3f ms"% ((t_delta_eta1-t_delta_eta0)*1000.0)
###benchmark.append(("Computation of delta, eta values for Train SFA Signal", t_delta_eta1-t_delta_eta0))
###
###
###print "Setting correct classes and labels for the Classifier/Regression, Train SFA Signal"
###correct_classes_training = iTrain.correct_classes
###correct_labels_training = iTrain.correct_labels
###    
####print "Testing for bug in first node..."
####print "subimages_train[0:1, 0:10]="
####print subimages_train[0:1, 0:10]
####print "flow[0].execute(subimages_train[0:1])[:,0:10]="
####print flow[0].execute(subimages_train[0:1])[:, 0:10]
###
###
###
###print "Loading test images, seen ids..."
###t_load_images0 = time.time()
####im_seq_base_dir = "/local/tmp/escalafl/Alberto/testing_seenid"
####im_seq_base_dir = "/local/tmp/escalafl/Alberto/training"
###
####FOR LEARNING IDENTITIES, INVARIANT TO VERTICAL ANGLE AND TRANSLATIONS
####im_seq_base_dir = "/local/tmp/escalafl/Alberto/Renderings20x500"
####ids=range(0,4)
####expressions=[0]
####morphs=[0]
####poses=range(0,500)
####lightings=[0]
#####slow_signal=0
####step=4
####offset=1
####image_files_seenid = create_image_filenames(im_seq_base_dir, slow_signal, ids, expressions, morphs, poses, lightings, step, offset)
###
###print "LOADING KNOWNID TEST INFORMATION"      
###image_files_seenid = iSeenid.input_files
###num_images_seenid = iSeenid.num_images
###block_size_seenid = iSeenid.block_size
###seq = sSeenid
###
###subimages_seenid = load_image_data(seq.input_files, seq.image_width, seq.image_height, seq.subimage_width, \
###                            seq.subimage_height, seq.pixelsampling_x, seq.pixelsampling_y, \
###                            seq.subimage_first_row, seq.subimage_first_column, seq.add_noise_L0, \
###                            seq.convert_format, seq.translations_x, seq.translations_y, seq.trans_sampled, background_type=seq.background_type, color_background_filter=filter, verbose=False)
###
###t_load_images1 = time.time()
###
####average_subimage_seenid = subimages_seenid.sum(axis=0)*1.0 / num_images_seenid
####average_subimage_seenid_I = scipy.cache.toimage(average_subimage_seenid.reshape(sSeenid.subimage_height, sSeenid.subimage_width, 3), mode="RGB")
####average_subimage_seenid_I.save("average_image_seenidRGB2.jpg", mode="RGB")
###
###print num_images_seenid, " Images loaded in %0.3f s"% ((t_load_images1-t_load_images0))
###
###t_exec0 = time.time()
###print "Execution over known id testing set..."
###print "Input Signal: Known Id test images"
###sl_seq_seenid = flow.execute(subimages_seenid)
###
###sl_seq_seenid = cutoff(sl_seq_seenid, min_cutoff, max_cutoff)
###
###t_exec1 = time.time()
###print "Execution over Known Id in %0.3f s"% ((t_exec1 - t_exec0))
###
###
###print "Computing typical delta, eta values for Seen Id SFA Signal"
###t_delta_eta0 = time.time()
###results.typical_delta_seenid, results.typical_eta_seenid = sfa_libs.comp_typical_delta_eta(sl_seq_seenid, block_size, num_reps=200)
###results.brute_delta_seenid = sfa_libs.comp_delta(sl_seq_seenid)
###results.brute_eta_seenid= sfa_libs.comp_eta(sl_seq_seenid)
###t_delta_eta1 = time.time()
###print "delta_seenid=", results.typical_delta_seenid
###print "eta_seenid=", results.typical_eta_seenid
####print "brute_delta_seenid=", results.brute_delta_seenid
####print "brute_eta_seenid=", results.brute_eta_seenid
###print "computed delta/eta in %0.3f ms"% ((t_delta_eta1-t_delta_eta0)*1000.0)
###
###
###print "Setting correct labels/classes data for seenid"
###correct_classes_seenid = iSeenid.correct_classes
###correct_labels_seenid = iSeenid.correct_labels
###
###
###t8 = time.time()
###t_classifier_train0 = time.time()
###print "*** Training Classifier/Regression"
###
###if use_full_sl_output == True:
###    results.reg_num_signals = reg_num_signals = sl_seq_training.shape[1]
###else:
###    results.reg_num_signals = reg_num_signals = 15
###
###
###
####WARNIGN!!! USING NEWID INSTEAD OF TRAINING
####cf_sl = sl_seq_training
####cf_num_samples = cf_sl.shape[0]
####cf_correct_labels = correct_labels_training
####cf_spacing = cf_block_size = iTrain.block_size
###
###cf_sl = sl_seq_seenid
###cf_num_samples = cf_sl.shape[0]
###cf_correct_labels = iSeenid.correct_labels
###cf_spacing = cf_block_size = iSeenid.block_size
###
###
####correct_labels_training = numpy.arange(len(labels_ccc_training))
####labels_classif = wider_1Darray(numpy.arange(iTrain.MIN_GENDER, iTrain.MAX_GENDER, iTrain.GENDER_STEP), iTrain.block_size)
####correct_labels_training = labels_classif
####print "labels_classif.shape = ", labels_classif.shape, " blocksize= ", block_size
####correct_classes_training = numpy.arange(len(labels_classif)) / block_size
###
###if reg_num_signals <= 128:
###    enable_ccc_Gauss_cfr = True
###else:
###    enable_ccc_Gauss_cfr = False
###
###if reg_num_signals <= 64 and False:
###    enable_svm_cfr = True
###else:
###    enable_svm_cfr = False
###    
###if reg_num_signals <= 8192:
###    enable_lr_cfr = True
###else:
###    enable_lr_cfr = False
###
####WARNING!!!!
####enable_svm_cfr = False
####enable_lr_cfr = False
###
###if enable_ccc_Gauss_cfr == True:
###    print "Training Classifier/Regression GC..."
###    S2SC = classifiers.Simple_2_Stage_Classifier()
###    S2SC.train(data=cf_sl[:,0:reg_num_signals], labels=cf_correct_labels, block_size=cf_block_size,spacing=cf_block_size)
###    t_classifier_train1 = time.time()
###    benchmark.append(("Training Classifier/Regression GC", t_classifier_train1-t_classifier_train0))
###t_classifier_train1 = time.time()
###
###def my_sigmoid(x):
###    return numpy.tanh(5*x)
###
###import svm as libsvm
###num_blocks = cf_sl.shape[0]/cf_block_size
###if enable_svm_cfr == True:
###    print "Training SVM..."
###    svm_node = mdp.contrib.LibSVMNode(probability=True)
###    svm_node.train(cf_sl[:,0:reg_num_signals], cf_correct_labels)
###    svm_node.stop_training(svm_type=libsvm.C_SVC, kernel_type=libsvm.RBF, C=1.0, gamma=1.0/(num_blocks), nu=0.6, eps=0.0001, expo=1.6)
####    svm_node.train_probability(cf_sl[:,0:reg_num_signals], cf_block_size, activation_func = my_sigmoid)
###
####    svm_node.eban_probability(cf_sl[:,0:reg_num_signals])
####    quit()
###
###if enable_lr_cfr == True:
###    print "Training LR..."
###    lr_node = mdp.nodes.LinearRegressionNode(with_bias=True, use_pinv=False)
###    lr_node.train(cf_sl[:,0:reg_num_signals], cf_correct_labels.reshape((cf_sl.shape[0], 1)))
###    lr_node.stop_training()
###
###
###if classifier_write:
###    print "Saving Gaussian Classifier"
###    cf_sl_hash = cache.hash_array(cf_sl).hexdigest() 
###    #update cache is not adding the hash to the filename,so we add it manually
###    classifier_filename = "GaussianClassifier_NetName"+Network.name+"_ParName"+Parameters.name+"_NetH" + network_hash + "_CFSlowH"+ cf_sl_hash +"_NumSig%03d"%reg_num_signals
###    classifier_write.update_cache(S2SC, None, None, classifier_filename, overwrite=True, verbose=True)
####****************************************************************
#######TODO: make classifier cash work!
#######TODO: review eban_svm model & implementation! beat normal svm!
###
###print "Executing/Executed over training set..."
###print "Input Signal: Training Data"
###subimages_training = subimages
###num_images_training = num_images
###
###print "Classification/Regression over training set..."
###t_class0 = time.time()
###if enable_ccc_Gauss_cfr == True:
###    print "GC classify..."
###    classes_ccc_training, labels_ccc_training = S2SC.classifyCDC(sl_seq_training[:,0:reg_num_signals])
###    classes_Gauss_training, labels_Gauss_training = S2SC.classifyGaussian(sl_seq_training[:,0:reg_num_signals])
###    regression_Gauss_training = S2SC.GaussianRegression(sl_seq_training[:,0:reg_num_signals])
###    probs_training = S2SC.GC_L0.class_probabilities(sl_seq_training[:,0:reg_num_signals])
###else:
###    classes_ccc_training = labels_ccc_training = classes_Gauss_training =  labels_Gauss_training = regression_Gauss_training  =   numpy.zeros(num_images_training) 
###    probs_training = numpy.zeros((num_images_training, 2))
###
###skip_svm_training = False
###
###if enable_svm_cfr == True and skip_svm_training==False:
###    print "SVM classify..."
###    classes_svm_training= svm_node.classify(sl_seq_training[:,0:reg_num_signals])
###    regression_svm_training= svm_node.label_of_class(classes_svm_training)
###    regression2_svm_training= svm_node.regression(sl_seq_training[:,0:reg_num_signals])
###    regression3_svm_training= regression2_svm_training
####    regression3_svm_training= svm_node.eban_regression(sl_seq_training[:,0:reg_num_signals])
###    #Warning!!!
####    raw_input("please enter something to continue")
###        
####    eban_probs = svm_node.eban_probability2(sl_seq_training[0:2,0:reg_num_signals])
####    print "len(eban_probs[0])= ", len(eban_probs[0])
####    print eban_probs
####
####    
####    eban_probs = svm_node.eban_probability(sl_seq_training[num_images_training/2:num_images_training/2+2,0:reg_num_signals])
####    print "len(eban_probs[0])= ", len(eban_probs[0])
####    print eban_probs
####
####    eban_probs = svm_node.eban_probability(sl_seq_training[-2:,0:reg_num_signals])
####    print "len(eban_probs[0])= ", len(eban_probs[0])
####    print eban_probs
####
####    quit()
###else:
###    classes_svm_training=regression_svm_training = regression2_svm_training = regression3_svm_training=numpy.zeros(num_images_training)
###
###if enable_lr_cfr == True:
###    print "LR execute..."
###    regression_lr_training = lr_node.execute(sl_seq_training[:,0:reg_num_signals]).flatten()
###else:
###    regression_lr_training = numpy.zeros(num_images_training)
###    
###print "Classification of training data: ", labels_ccc_training
###t_classifier_train2 = time.time()
###
###print "Classifier trained in time %0.3f s"% ((t_classifier_train1 - t_classifier_train0))
###print "Training Images Classified in time %0.3f s"% ((t_classifier_train2 - t_classifier_train1))
###benchmark.append(("Classification of Training Images", t_classifier_train2-t_classifier_train1))
###
###t_class1 = time.time()
###print "Classification/Regression over Training Set in %0.3f s"% ((t_class1 - t_class0))
###
###
###t_class0 = time.time()
###if enable_ccc_Gauss_cfr == True:
###    classes_ccc_seenid, labels_ccc_seenid = S2SC.classifyCDC(sl_seq_seenid[:,0:reg_num_signals])
###    classes_Gauss_seenid, labels_Gauss_seenid = S2SC.classifyGaussian(sl_seq_seenid[:,0:reg_num_signals])
###    print "Classification of Seen id test images: ", labels_ccc_seenid
###    regression_Gauss_seenid = S2SC.GaussianRegression(sl_seq_seenid[:,0:reg_num_signals])
###    probs_seenid = S2SC.GC_L0.class_probabilities(sl_seq_seenid[:,0:reg_num_signals])
###else:
###    classes_ccc_seenid = labels_ccc_seenid =  classes_Gauss_seenid = labels_Gauss_seenid = regression_Gauss_seenid = numpy.zeros(num_images_seenid) 
###    probs_seenid = numpy.zeros((num_images_seenid, 2))
###
###if enable_svm_cfr == True:
###    classes_svm_seenid = svm_node.classify(sl_seq_seenid[:,0:reg_num_signals])
###    regression_svm_seenid = svm_node.label_of_class(classes_svm_seenid)
###    regression2_svm_seenid = svm_node.regression(sl_seq_seenid[:,0:reg_num_signals])
###    regression3_svm_seenid = regression2_svm_seenid
####    regression3_svm_seenid = svm_node.eban_regression(sl_seq_seenid[:,0:reg_num_signals], hint=probs_seenid)
####    raw_input("please enter something to continue & save")
###    
###    #network_write.update_cache(probs_seenid, None, network_base_dir, "GCProbs", overwrite=True, use_hash=network_hash, verbose=True)
###
###    
####    quit()
###else:
###    classes_svm_seenid=regression_svm_seenid = regression2_svm_seenid = regression3_svm_seenid = numpy.zeros(num_images_seenid)
###        
###if enable_lr_cfr == True:
###    regression_lr_seenid = lr_node.execute(sl_seq_seenid[:,0:reg_num_signals]).flatten()
###else:
###    regression_lr_seenid = numpy.zeros(num_images_seenid)
###
###print "labels_ccc_seenid.shape=", labels_ccc_seenid.shape
###
####correct_labels_seenid = wider_1Darray(numpy.arange(iSeenid.MIN_GENDER, iSeenid.MAX_GENDER, iSeenid.GENDER_STEP), iSeenid.block_size)
###print "correct_labels_seenid.shape=", correct_labels_seenid.shape
####correct_classes_seenid = numpy.arange(len(labels_ccc_seenid)) * len(labels_ccc_training) / len(labels_ccc_seenid) / block_size
###
###t_class1 = time.time()
###print "Classification/Regression over Seen Id in %0.3f s"% ((t_class1 - t_class0))
###
###t10 = time.time()
###t_load_images0 = time.time()
###print "Loading test images, new ids..."
###
###image_files_newid = iNewid.input_files
###num_images_newid = iNewid.num_images
###block_size_newid = iNewid.block_size
###seq = sNewid
###
###subimages_newid = load_image_data(seq.input_files, seq.image_width, seq.image_height, seq.subimage_width, \
###                            seq.subimage_height, seq.pixelsampling_x, seq.pixelsampling_y, \
###                            seq.subimage_first_row, seq.subimage_first_column, seq.add_noise_L0, \
###                            seq.convert_format, seq.translations_x, seq.translations_y, seq.trans_sampled, background_type=seq.background_type, color_background_filter=filter, verbose=False)
###
###t_load_images1 = time.time()
###t11 = time.time()
###print num_images_newid, " Images loaded in %0.3f s"% ((t_load_images1 - t_load_images0))
###
###t_exec0 = time.time()
###print "Execution over New Id testing set..."
###print "Input Signal: New Id test images"
###sl_seq_newid = flow.execute(subimages_newid)
###sl_seq_newid = cutoff(sl_seq_newid, min_cutoff, max_cutoff)
###
####WARNING!!!
####print "WARNING!!! SCALING NEW ID SLOW SIGNALS!!!"
####corr_factor = numpy.array([ 1.06273968,  1.0320762 ,  1.06581665,  1.01598426,  1.08355725,
####        1.10316477,  1.08731609,  1.05887109,  1.08185727,  1.09867758,
####        1.10567757,  1.08268021])
####print corr_factor
####
####corr_factor = numpy.array([ 1.06273968,  1.0320762 ,  1.06581665,  1.01598426,  1.08355725,
####        1.10316477,  1.08731609,  1.05887109,  1.08185727,  1.09867758]) * 0.98
####print corr_factor
####
####corr_factor =  numpy.sqrt(sl_seq_training.var(axis=0) / sl_seq_newid.var(axis=0))[0:reg_num_signals].mean()
####print corr_factor
####
####corr_factor =  numpy.sqrt(sl_seq_training.var(axis=0) / sl_seq_newid.var(axis=0))[0:reg_num_signals] * 0.98
####print corr_factor
####
####corr_factor =  0.977 * numpy.sqrt(sl_seq_training.var(axis=0)[0:reg_num_signals].mean() / sl_seq_newid.var(axis=0)[0:reg_num_signals].mean())
####print corr_factor
###
###corr_factor=1.0
###print corr_factor
###sl_seq_newid[:,0:reg_num_signals] = sl_seq_newid[:,0:reg_num_signals] * corr_factor
###
###t_exec1 = time.time()
###print "Execution over New Id in %0.3f s"% ((t_exec1 - t_exec0))
###
###t_class0 = time.time()
###
###correct_classes_newid = iNewid.correct_classes
###correct_labels_newid = iNewid.correct_labels
###
###if enable_ccc_Gauss_cfr == True:
###    classes_ccc_newid, labels_ccc_newid = S2SC.classifyCDC(sl_seq_newid[:,0:reg_num_signals])
###    classes_Gauss_newid, labels_Gauss_newid = S2SC.classifyGaussian(sl_seq_newid[:,0:reg_num_signals])
###    print "Classification of New Id test images: ", labels_ccc_newid
###    regression_Gauss_newid = S2SC.GaussianRegression(sl_seq_newid[:,0:reg_num_signals])
###    probs_newid = S2SC.GC_L0.class_probabilities(sl_seq_newid[:,0:reg_num_signals])
###else:
###    classes_ccc_newid = labels_ccc_newid = classes_Gauss_newid = labels_Gauss_newid = regression_Gauss_newid = numpy.zeros(num_images_newid) 
###    probs_newid = numpy.zeros((num_images_newid, 2))
###
###if enable_svm_cfr == True:
###    classes_svm_newid = svm_node.classify(sl_seq_newid[:,0:reg_num_signals])
###    regression_svm_newid = svm_node.label_of_class(classes_svm_newid)
###    regression2_svm_newid = svm_node.regression(sl_seq_newid[:,0:reg_num_signals])
###    regression3_svm_newid = regression2_svm_newid
####    regression3_svm_newid = svm_node.eban_regression(sl_seq_newid[:,0:reg_num_signals], hint=probs_newid)
####WARNING
####    regression3_svm_newid = svm_node.eban_regression3(sl_seq_newid[:,0:reg_num_signals], activation_func_app = my_sigmoid, hint=probs_newid)
####    raw_input("please enter something to continue")
###    #Hack, reusing probs_newid for displaying probs_newid_eban_svm
####    probs_newid = svm_node.eban_probability2(sl_seq_seenid[:,0:reg_num_signals], hint=probs_seenid)
###    probs_training[0, 10] = 1.0
###    probs_newid[0, 10] = 1.0
###    probs_seenid[0, 10] = 1.0
####    m_err1 = svm_node.model_error(probs, l)
###else:
###    classes_svm_newid=regression_svm_newid = regression2_svm_newid = regression3_svm_newid = numpy.zeros(num_images_newid)
###
###if enable_lr_cfr == True:
###    regression_lr_newid = lr_node.execute(sl_seq_newid[:,0:reg_num_signals]).flatten()
###else:
###    regression_lr_newid = numpy.zeros(num_images_newid)
###
###t_class1 = time.time()
###print "Classification/Regression over New Id in %0.3f s"% ((t_class1 - t_class0))
###
###
####print "Saving train/test_data for external analysis"
####ndarray_to_string(sl_seq_training, "/local/tmp/escalafl/training_samples.txt")
####ndarray_to_string(correct_labels_training, "/local/tmp/escalafl/training_labels.txt")
####ndarray_to_string(sl_seq_seenid, "/local/tmp/escalafl/seenid_samples.txt")
####ndarray_to_string(correct_labels_seenid, "/local/tmp/escalafl/seenid_labels.txt")
####ndarray_to_string(sl_seq_newid, "/local/tmp/escalafl/newid_samples.txt")
####ndarray_to_string(correct_labels_newid, "/local/tmp/escalafl/newid_labels.txt")
###
###print "Computing typical delta, eta values for Training SFA Signal"
###t_delta_eta0 = time.time()
###results.typical_delta_train, results.typical_eta_newid = sfa_libs.comp_typical_delta_eta(sl_seq_training, block_size, num_reps=200)
###results.brute_delta_train = sfa_libs.comp_delta(sl_seq_training)
###results.brute_eta_train= sfa_libs.comp_eta(sl_seq_training)
###t_delta_eta1 = time.time()
###print "delta_train=", results.typical_delta_newid
###print "eta_train=", results.typical_eta_newid
###
###print "Computing typical delta, eta values for New Id SFA Signal"
###t_delta_eta0 = time.time()
###results.typical_delta_newid, results.typical_eta_newid = sfa_libs.comp_typical_delta_eta(sl_seq_newid, block_size, num_reps=200)
###results.brute_delta_newid = sfa_libs.comp_delta(sl_seq_newid)
###results.brute_eta_newid= sfa_libs.comp_eta(sl_seq_newid)
###t_delta_eta1 = time.time()
###print "delta_newid=", results.typical_delta_newid
###print "eta_newid=", results.typical_eta_newid
####print "brute_delta_newid=", results.brute_delta_newid
####print "brute_eta_newid=", results.brute_eta_newid
###print "computed delta/eta in %0.3f ms"% ((t_delta_eta1-t_delta_eta0)*1000.0)
###
####print "Generating (new random) input sequence..."
####use_average = False
####use_first_id = False
####use_random_walk = False
####use_training_data = True
####use_new_identity = False
####new_id = 5
####if block_size_exec is not None and block_size_exec > 1:
####    print "block_size_exec > 1"
####    num_images2 = num_images / block_size_exec
####    subimages2 = numpy.zeros((num_images2, subimage_width * subimage_height))
####    if use_first_id is True:
####        print "Input Signal: First ID / First Pose of each ID"
####        for i in range(num_images2):
####            subimages2[i] = subimages[block_size_exec * i]       
####    elif use_average is True:
####        print "Input Signal: Average of IDs"
####        for i in range(num_images2):
####            subimages2[i] = subimages[block_size_exec * i: block_size_exec * (i+1)].sum(axis=0) / block_size_exec
####    elif use_random_walk is True:
####        print "Input Signal: Random Walk"
####        for i in range(num_images2):
####            id = numpy.random.randint(block_size_exec)
#####            subimages[block_size * i + id]
#####            subimages2[i] = subimages[0]
####            subimages2[i] = subimages[block_size_exec * i + id]
####    elif use_training_data is True:
####        print "Input Signal: Training Data"
####        subimages2 = subimages
####        num_images2 = num_images
####    elif use_new_identity is True:
####        print "Input Signal: New ID random%03d*.tif"%(new_id)
####        test_image_files1 = glob.glob(im_seq_base_dir + "/random%03d*.tif"%(new_id))
####        test_image_files1.sort()
####
####        test_image_files = []
####        for i in range(len(test_image_files1)):
####            test_image_files.append(test_image_files1[i])
####
####        num_images2 = num_test_images = len(test_image_files)
####        
####        subimages2 = numpy.zeros((num_test_images, subimage_width * subimage_height))
####        act_im_num = 0
####        for image_file in test_image_files:
####            im = Image.open(image_file)
####            im = im.convert("L")
####            im_arr = numpy.asarray(im)
####            im_small = im_arr[subimage_first_row:(subimage_first_row+subimage_height*subimage_pixelsampling):subimage_pixelsampling,
####                              subimage_first_column:(subimage_first_column+subimage_width*subimage_pixelsampling):subimage_pixelsampling].astype(float)
####            subimages2[act_im_num] = im_small.flatten()
####            act_im_num = act_im_num+1
####            del im_small
####            del im_arr
####            del im
####    else:
####        print "******************* No input sequence specified !!!!!!!!!!!!!!!!!"
#####
####    subimages = subimages2
####    num_images = num_images2
####
#####flow.
#####print "Training finished, ed in %0.3f ms"% ((t2-t1)*1000.0)
####sl_seq = flow.execute(subimages)
####inverted_im = flow.inverse(sl_seq)
###
###
###print "virtual sequence length complete = ", num_images_training  * (block_size - 1)/2
###print "virtual sequence length sequence = ", (num_images_training - block_size)  * block_size 
###print "virtual sequence length mixed = ", num_images_training  * (block_size - 1)/2 + (num_images_training - block_size)  * block_size
###
###print "Estimating explained variance for Train SFA Signal"
###number_samples_explained_variance = 1000
###fast_inverse_available = False
####WARNING!!!!
###if fast_inverse_available:
###    print "Estimated explained variance (train) is; ", more_nodes.estimated_explained_variance(subimages_train, flow, sl_seq_training, number_samples_explained_variance)
###else:
###    print "Fast inverse not available, not estimating explained variance"
###
###print "Computations Finished!"
###
###
###print "** Displaying Benchmark data: **"
###for task_name, task_time in benchmark:
###    print "     ", task_name, " done in %0.3f s"%task_time
###
###
###print "Classification/Regression Performance: "
###results.class_ccc_rate_train = classifiers.correct_classif_rate(correct_classes_training, classes_ccc_training)
###results.class_Gauss_rate_train = classifiers.correct_classif_rate(correct_classes_training, classes_Gauss_training)
###results.class_svm_rate_train = classifiers.correct_classif_rate(correct_classes_training, classes_svm_training)
###results.mse_ccc_train = distance_squared_Euclidean(correct_labels_training, labels_ccc_training)/len(labels_ccc_training)
###results.mse_gauss_train = distance_squared_Euclidean(correct_labels_training, regression_Gauss_training)/len(labels_ccc_training)
###results.mse_svm_train = distance_squared_Euclidean(correct_labels_training, regression_svm_training)/len(labels_ccc_training)
###results.mse2_svm_train = distance_squared_Euclidean(correct_labels_training, regression2_svm_training)/len(labels_ccc_training)
###results.mse3_svm_train = distance_squared_Euclidean(correct_labels_training, regression3_svm_training)/len(labels_ccc_training)
###results.mse_lr_train = distance_squared_Euclidean(correct_labels_training, regression_lr_training)/len(labels_ccc_training)
###
###results.class_ccc_rate_seenid = classifiers.correct_classif_rate(correct_classes_seenid, classes_ccc_seenid)
###results.class_Gauss_rate_seenid = classifiers.correct_classif_rate(correct_classes_seenid, classes_Gauss_seenid)
###results.class_svm_rate_seenid = classifiers.correct_classif_rate(correct_classes_seenid, classes_svm_seenid)
###results.mse_ccc_seenid = distance_squared_Euclidean(correct_labels_seenid, labels_ccc_seenid)/len(labels_ccc_seenid)
###results.mse_gauss_seenid = distance_squared_Euclidean(correct_labels_seenid, regression_Gauss_seenid)/len(labels_ccc_seenid)
###results.mse_svm_seenid = distance_squared_Euclidean(correct_labels_seenid, regression_svm_seenid)/len(labels_ccc_seenid)
###results.mse2_svm_seenid = distance_squared_Euclidean(correct_labels_seenid, regression2_svm_seenid)/len(labels_ccc_seenid)
###results.mse3_svm_seenid = distance_squared_Euclidean(correct_labels_seenid, regression3_svm_seenid)/len(labels_ccc_seenid)
###results.mse_lr_seenid = distance_squared_Euclidean(correct_labels_seenid, regression_lr_seenid)/len(labels_ccc_seenid)
###
###results.class_ccc_rate_newid = classifiers.correct_classif_rate(correct_classes_newid, classes_ccc_newid)
###results.class_Gauss_rate_newid = classifiers.correct_classif_rate(correct_classes_newid, classes_Gauss_newid)
###results.class_svm_rate_newid = classifiers.correct_classif_rate(correct_classes_newid, classes_svm_newid)
###results.mse_ccc_newid = distance_squared_Euclidean(correct_labels_newid, labels_ccc_newid)/len(labels_ccc_newid)
###results.mse_gauss_newid = distance_squared_Euclidean(correct_labels_newid, regression_Gauss_newid)/len(labels_ccc_newid)
###results.mse_svm_newid = distance_squared_Euclidean(correct_labels_newid, regression_svm_newid)/len(labels_ccc_newid)
###results.mse2_svm_newid = distance_squared_Euclidean(correct_labels_newid, regression2_svm_newid)/len(labels_ccc_newid)
###results.mse3_svm_newid = distance_squared_Euclidean(correct_labels_newid, regression3_svm_newid)/len(labels_ccc_newid)
###results.mse_lr_newid = distance_squared_Euclidean(correct_labels_newid, regression_lr_newid)/len(labels_ccc_newid)
###
###save_results=False
###if save_results:
###    cache.pickle_to_disk(results, "results_" + str(int(time.time()))+ ".pckl")
###
###print "sl_seq_training.mean(axis=0)=", sl_seq_training.mean(axis=0)
###print "sl_seq_seenid.mean(axis=0)=", sl_seq_seenid.mean(axis=0)
###print "sl_seq_newid.mean(axis=0)=", sl_seq_newid.mean(axis=0)
###print "sl_seq_training.var(axis=0)=", sl_seq_training.var(axis=0)
###print "sl_seq_seenid.var(axis=0)=", sl_seq_seenid.var(axis=0)
###print "sl_seq_newid.var(axis=0)=", sl_seq_newid.var(axis=0)
###    
###print "Train: %0.3f CR_CCC, CR_Gauss %0.3f, CR_SVM %0.3f, MSE_CCC %0.3f, MSE_Gauss %0.3f, MSE3_SVM %0.3f, MSE2_SVM %0.3f, MSE_SVM %0.3f, MSE_LR %0.3f  "%(results.class_ccc_rate_train, results.class_Gauss_rate_train, results.class_svm_rate_train, results.mse_ccc_train, results.mse_gauss_train, results.mse3_svm_train, results.mse2_svm_train, results.mse_svm_train, results.mse_lr_train)
###print "Seen Id: %0.3f CR_CCC, CR_Gauss %0.3f, CR_SVM %0.3f, MSE_CCC %0.3f, MSE_Gauss %0.3f, MSE3_SVM %0.3f, MSE2_SVM %0.3f, MSE_SVM %0.3f, MSE_LR %0.3f"%(results.class_ccc_rate_seenid, results.class_Gauss_rate_seenid, results.class_svm_rate_seenid, results.mse_ccc_seenid, results.mse_gauss_seenid, results.mse3_svm_seenid, results.mse2_svm_seenid, results.mse_svm_seenid, results.mse_lr_seenid)
###print "New Id: %0.3f CR_CCC, CR_Gauss %0.3f, CR_SVM %0.3f, MSE_CCC %0.3f, MSE_Gauss %0.3f, MSE3_SVM %0.3f, MSE2_SVM %0.3f, MSE_SVM %0.3f, MSE_LR %0.3f "%(results.class_ccc_rate_newid, results.class_Gauss_rate_newid, results.class_svm_rate_newid, results.mse_ccc_newid, results.mse_gauss_newid, results.mse3_svm_newid, results.mse2_svm_newid, results.mse_svm_newid, results.mse_lr_newid)
###
####quit()
###scale_disp = 1
###
###print "Computing average SFA..."
###num_blocks_train = iTrain.num_images/iTrain.block_size
###sl_seq_training_mean = numpy.zeros((iTrain.num_images/iTrain.block_size, hierarchy_out_dim))
###for block in range(num_blocks_train):
###    sl_seq_training_mean[block] = sl_seq_training[block*block_size:(block+1)*block_size,:].mean(axis=0)
###print "%d Blocks used for averages"%num_blocks_train
###
###print "Creating GUI..."
###
###print "****** Displaying Typical Images used for Training and Testing **********"
###tmp_fig = plt.figure()
###plt.suptitle(Parameters.name + ". Image Datasets")
### 
###num_images_per_set = 4 
###
###subimages_training = subimages
###num_images_training
###
###sequences = [subimages_training, subimages_seenid, subimages_newid]
###messages = ["Train Images", "Seen Id Images", "New Id Images"]
###nums_images = [num_images_training, num_images_seenid, num_images_newid]
###sizes = [(sTrain.subimage_height, sTrain.subimage_width), (sSeenid.subimage_height, sSeenid.subimage_width), \
###         (sNewid.subimage_height, sNewid.subimage_width)]
###
###for seqn in range(3):
###    for im in range(num_images_per_set):
###        tmp_sb = plt.subplot(3, num_images_per_set, num_images_per_set*seqn + im+1)
###        y = im * (nums_images[seqn]-1) / (num_images_per_set - 1)
###        subimage_im = sequences[seqn][y].reshape(sizes[seqn])
###        tmp_sb.imshow(subimage_im.clip(0,255), norm=None, vmin=0, vmax=255, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###        if im == 0:
###            plt.ylabel(messages[seqn])
###        else:
###            tmp_sb.axis('off')
###            
###print "************ Displaying SFA Output Signals **************"
####Create Figure
###f0 = plt.figure()
###plt.suptitle(Parameters.name + ". Slow Signals")
###  
####display SFA of Training Set
###p11 = plt.subplot(1,3,1)
###plt.title("Output Signals (Training Set)")
###sl_seqdisp = sl_seq_training[:, range(0,hierarchy_out_dim)]
###sl_seqdisp = scale_to(sl_seqdisp, sl_seq_training.mean(axis=0), sl_seq_training.max(axis=0)-sl_seq_training.min(axis=0), 127.5, 255.0, scale_disp, 'tanh')
###p11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###plt.xlabel("min[0]=%.3f, max[0]=%.3f, scale=%.3f\n e[]=" % (sl_seq_training.min(axis=0)[0], sl_seq_training.max(axis=0)[0], scale_disp)+str3(sfa_libs.comp_eta(sl_seq_training)[0:5]))
###plt.ylabel("Train Images")
###
####display SFA of Known Id testing Set
###p12 = plt.subplot(1,3,2)
###plt.title("Output Signals (Seen Id Test Set)")
###sl_seqdisp = sl_seq_seenid[:, range(0,hierarchy_out_dim)]
###sl_seqdisp = scale_to(sl_seqdisp, sl_seq_training.mean(axis=0), sl_seq_training.max(axis=0)-sl_seq_training.min(axis=0), 127.5, 255.0, scale_disp, 'tanh')
###p12.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###plt.xlabel("min[0]=%.3f, max[0]=%.3f, scale=%.3f\n e[]=" % (sl_seq_seenid.min(axis=0)[0], sl_seq_seenid.max(axis=0)[0], scale_disp)+str3(sfa_libs.comp_eta(sl_seq_seenid)[0:5]))
###plt.ylabel("Seen Id Images")
###
####display SFA of Known Id testing Set
###p13 = plt.subplot(1,3,3)
###plt.title("Output Signals (New Id Test Set)")
###sl_seqdisp = sl_seq_newid[:, range(0,hierarchy_out_dim)]
###sl_seqdisp = scale_to(sl_seqdisp, sl_seq_training.mean(axis=0), sl_seq_training.max(axis=0)-sl_seq_training.min(axis=0), 127.5, 255.0, scale_disp, 'tanh')
###p13.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###plt.xlabel("min[0]=%.3f, max[0]=%.3f, scale=%.3f\n e[]=" % (sl_seq_newid.min(axis=0)[0], sl_seq_newid.max(axis=0)[0], scale_disp)+str3(sfa_libs.comp_eta(sl_seq_newid)[0:5]))
###plt.ylabel("New Id Images")
###
###
###relevant_out_dim = 3
###if hierarchy_out_dim < relevant_out_dim:
###    relevant_out_dim = hierarchy_out_dim
###
###print "************ Plotting Relevant SFA Output Signals **************"
###ax_5 = plt.figure()
###ax_5.subplots_adjust(hspace=0.5)
###plt.suptitle(Parameters.name + ". Most Relevant Slow Signals")
###
###sp11 = plt.subplot(4,1,1)
###plt.title("SFA Outputs. (Training Set)")
### 
###relevant_sfa_indices = numpy.arange(relevant_out_dim)
###reversed_sfa_indices = range(relevant_out_dim)
###reversed_sfa_indices.reverse()
###
###r_color = (1-relevant_sfa_indices * 1.0 / relevant_out_dim) * 0.8 + 0.2
###g_color = (relevant_sfa_indices * 1.0 / relevant_out_dim) * 0.8 + 0.2
####r_color = (0.5*numpy.cos(relevant_sfa_indices * numpy.pi / relevant_out_dim) + 0.5).clip(0.0,1.0)
####g_color = (0.5*numpy.cos(relevant_sfa_indices * numpy.pi / relevant_out_dim + numpy.pi)+0.5).clip(0.0,1.0)
###b_color = relevant_sfa_indices * 0.0
###max_amplitude_sfa = 2.0
###
###sfa_text_labels = ["Slowest Signal", "2nd Slowest Signal", "3rd Slowest Signal"]
###
###for sig in reversed_sfa_indices:
###    plt.plot(numpy.arange(num_images_training), sl_seq_training[:, sig], ".", color=(r_color[sig], g_color[sig], b_color[sig]), label=sfa_text_labels[sig])
####    plt.plot(numpy.arange(num_images_training), sl_seq_training[:, sig], ".")
###plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
###plt.xlabel("Input Image, Training Set (red=slowest signal, light green=fastest signal)")
###plt.ylabel("Slow Signals")
###
###sp12 = plt.subplot(4,1,2)
###if Parameters.train_mode == 'sequence' or Parameters.train_mode == 'mixed' :
###    plt.title("Example of Ideal SFA Outputs for Training Set (Sequence Training)")
###    num_blocks = num_images_training/block_size
###    sl_optimal = numpy.zeros((num_images, relevant_out_dim))
###    factor = -1.0 * numpy.sqrt(2.0)
###    t_opt=numpy.linspace(0, numpy.pi, num_blocks)
###    for sig in range(relevant_out_dim):
###        sl_optimal[:,sig] = wider_1Darray(factor * numpy.cos((sig+1) * t_opt), block_size)
###    
###    for sig in reversed_sfa_indices:
###        colour = sig * 0.6 / relevant_out_dim 
###        plt.plot(numpy.arange(num_images_training), sl_optimal[:, sig], ".", color=(r_color[sig], g_color[sig], b_color[sig]), label=sfa_text_labels[sig])
###    plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
###    plt.xlabel("Input Image, Training Set (red=slowest signal, light green=fastest signal)")
###    plt.ylabel("Slow Signals")
###else:
###    plt.title("Example of Ideal SFA Outputs for Training Set... Not Available")
###
###sp13 = plt.subplot(4,1,3)
###plt.title("SFA Outputs. (Seen Id Test Set)")
###for sig in reversed_sfa_indices:
###    colour = sig * 1.0 /relevant_out_dim 
###    plt.plot(numpy.arange(num_images_seenid), sl_seq_seenid[:, sig], ".", color=(r_color[sig], g_color[sig], b_color[sig]), label=sfa_text_labels[sig])
###plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
###plt.xlabel("Input Image, Seen Id Test (red=slowest signal, light green=fastest signal)")
###plt.ylabel("Slow Signals")
###plt.legend( (sfa_text_labels[2], sfa_text_labels[1], sfa_text_labels[0]), loc=4)
###
###sp14 = plt.subplot(4,1,4)
###plt.title("SFA Outputs. (New Id Test Set)")
###for sig in reversed_sfa_indices:
###    colour = sig * 1.0 /relevant_out_dim 
###    plt.plot(numpy.arange(num_images_newid), sl_seq_newid[:, sig], ".", color=(r_color[sig], g_color[sig], b_color[sig]), label=sfa_text_labels[sig])
###plt.ylim(-max_amplitude_sfa, max_amplitude_sfa)
###plt.xlabel("Input Image, New Id Test (red=slowest signal, light green=fastest signal)")
###plt.ylabel("Slow Signals")
###
###show_linear_inv    = False
###show_linear_masks  = True
###show_linear_masks_ext  = True
###show_linear_morphs = True
###show_localized_masks  = True
###show_localized_morphs = True
###show_progressive_morph = False
###
###show_translation_x = True
###
###print "************ Displaying Training Set SFA and Inverses **************"
####Create Figure
###f1 = plt.figure()
###ax_5.subplots_adjust(hspace=0.5)
###plt.suptitle("Pseudo-Invertible 4L SFA Hierarchy")
###  
####display SFA
###a11 = plt.subplot(2,3,1)
###plt.title("Output Unit (Top Node)")
###sl_seqdisp = sl_seq[:, range(0,hierarchy_out_dim)]
###sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
###a11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp)+str3(sfa_libs.comp_eta(sl_seq)[0:5]))
###
####display first image
####Alternative: im1.show(command="xv")
###f1a12 = plt.subplot(2,3,2)
###plt.title("A particular image in the sequence")
####im_smalldisp = im_small.copy()
####f1a12.imshow(im_smalldisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###f1a13 = plt.subplot(2,3,3)
###plt.title("Reconstructed Image")
###    
###f1a21 = plt.subplot(2,3,4)
###plt.title("Reconstruction Error")
###    
###f1a22 = plt.subplot(2,3,5)
###plt.title("DIfferential Reconstruction y_pinv_(t+1) - y_pinv_(t)")
###    
###f1a23 = plt.subplot(2,3,6)
###plt.title("Pseudoinverse of 0 / PINV(y) - PINV(0)")
###if show_linear_inv == True:
###    sfa_zero = numpy.zeros((1, hierarchy_out_dim))
###    pinv_zero = flow.inverse(sfa_zero)
###    pinv_zero = pinv_zero.reshape((sTrain.subimage_height, sTrain.subimage_width))
###    error_scale_disp=1.5
###    pinv_zero_disp = scale_to(pinv_zero, pinv_zero.mean(), pinv_zero.max()-pinv_zero.min(), 127.5, 255.0, error_scale_disp, 'tanh')
###    f1a23.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, zero" % (pinv_zero_disp.min(), pinv_zero_disp.max(), pinv_zero_disp.std(), error_scale_disp))
###    f1a23.imshow(pinv_zero_disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###else:
###    pinv_zero = None
###
####Retrieve Image in Sequence
###def on_press(event):
###    global plt, f1, f1a12, f1a13, f1a21, f1a22, fla23, subimages, L2, sTrain, sl_seq, pinv_zero, flow, error_scale_disp
###    print 'you pressed', event.button, event.xdata, event.ydata
###    y = int(event.ydata)
###    if y < 0:
###        y = 0
###    if y >= num_images:
###        y = num_images -1
###    print "y=" + str(y)
###
####Display Original Image
###    subimage_im = subimages[y].reshape((sTrain.subimage_height, sTrain.subimage_width)) + 0.0
###    
###    if show_translation_x:
###        if sTrain.trans_sampled:
###            subimage_im[:, sTrain.subimage_width/2.0-sTrain.translations_x[y]] = 255
###            subimage_im[:, sTrain.subimage_width/2.0-regression_Gauss_training[y]/reduction_factor] = 0
###        else:
###            subimage_im[:, sTrain.subimage_width/2.0-sTrain.translations_x[y]*sTrain.pixelsampling_x] = 255
###            subimage_im[:, sTrain.subimage_width/2.0-regression_Gauss_training[y]/reduction_factor*sTrain.pixelsampling_x] = 0
###    
###    f1a12.imshow(subimage_im.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###
###    if show_linear_inv == False:
###        f1.canvas.draw()
###        return
###
####Display Reconstructed Image
###    data_out = sl_seq[y].reshape((1, hierarchy_out_dim))
###    inverted_im = flow.inverse(data_out)
###    inverted_im = inverted_im.reshape((sTrain.subimage_height, sTrain.subimage_width))
###    f1a13.imshow(inverted_im.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)        
####Display Reconstruction Error
###    error_scale_disp=1.5
###    error_im = subimages[y].reshape((sTrain.subimage_height, sTrain.subimage_width)) - inverted_im 
###    error_im_disp = scale_to(error_im, error_im.mean(), error_im.max()-error_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
###    f1a21.imshow(error_im_disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
###    plt.axis = f1a21
###    f1a21.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y=%d" % (error_im.min(), error_im.max(), error_im.std(), error_scale_disp, y))
####Display Differencial change in reconstruction
###    error_scale_disp=1.5
###    if y >= sTrain.num_images - 1:
###        y_next = 0
###    else:
###        y_next = y+1
###    print "y_next=" + str(y_next)
###    data_out2 = sl_seq[y_next].reshape((1, hierarchy_out_dim))
###    inverted_im2 = flow.inverse(data_out2).reshape((sTrain.subimage_height, sTrain.subimage_width))
###    diff_im = inverted_im2 - inverted_im 
###    diff_im_disp = scale_to(diff_im, diff_im.mean(), diff_im.max()-diff_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
###    f1a22.imshow(diff_im_disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
###    plt.axis = f1a22
###    f1a22.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y=%d" % (diff_im.min(), diff_im.max(), diff_im.std(), error_scale_disp, y))
####Display Difference from PINV(y) and PINV(0)
###    error_scale_disp=1.0
###    dif_pinv = inverted_im - pinv_zero 
###    dif_pinv_disp = scale_to(dif_pinv, dif_pinv.mean(), dif_pinv.max()-dif_pinv.min(), 127.5, 255.0, error_scale_disp, 'tanh')
###    f1a23.imshow(dif_pinv.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
###    plt.axis = f1a23
###    f1a23.set_xlabel("PINV(y) - PINV(0): min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y=%d" % (dif_pinv.min(), dif_pinv.max(), dif_pinv.std(), error_scale_disp, y))
###    
###    f1.canvas.draw()
###    
###f1.canvas.mpl_connect('button_press_event', on_press)
###
###
###print "************ Displaying Classification / Regression Results **************"
####Create Figure
###f2 = plt.figure()
###plt.suptitle("Classification Results (Class Numbers)  using %d Slow Signals"%reg_num_signals)
####plt.title("Training Set")
###p11 = f2.add_subplot(311, frame_on=False)
###
###p11.plot(numpy.arange(len(correct_classes_training)), correct_classes_training, 'r.', markersize=1, markerfacecolor='red')
###p11.plot(numpy.arange(len(classes_ccc_training)), classes_ccc_training, 'b.', markersize=1, markerfacecolor='blue')
###p11.plot(numpy.arange(len(classes_svm_training)), classes_svm_training, 'g.', markersize=1, markerfacecolor='green')
###plt.xlabel("Image Number, Training Set. Classification Rate CCC=%.3f, CR_Gauss=%.3f, CR_SVM=%.3f" % (results.class_ccc_rate_train, results.class_Gauss_rate_train, results.class_svm_rate_train))
###plt.ylabel("Class Number")
###p11.grid(True)
####draw horizontal and vertical lines
####majorLocator_x   = MultipleLocator(block_size)
####majorLocator_y   = MultipleLocator(1)
####p11.xaxis.set_major_locator(majorLocator_x)
#####p11.yaxis.set_major_locator(majorLocator_y)
####plt.xticks(numpy.arange(0, len(labels_ccc_training), block_size)) 
####plt.yticks(numpy.arange(0, len(labels_ccc_training), block_size)) 
####print "Block_size is: ", block_size
###
###p12 = f2.add_subplot(312, frame_on=False)
###p12.plot(numpy.arange(len(correct_classes_seenid)), correct_classes_seenid, 'r.', markersize=2, markerfacecolor='red')
###p12.plot(numpy.arange(len(classes_ccc_seenid)), classes_ccc_seenid, 'm.', markersize=2, markerfacecolor='magenta')
###p12.plot(numpy.arange(len(classes_ccc_seenid)), classes_svm_seenid, 'g.', markersize=2, markerfacecolor='green')
###plt.xlabel("Image Number, Seen Id Set. Classification Rate CCC=%f, CR_Gauss=%.3f, CR_SVM=%.3f" % ( results.class_ccc_rate_seenid, results.class_Gauss_rate_seenid, results.class_svm_rate_seenid))
###plt.ylabel("Class Number")
###p12.grid(True)
####p12.plot(numpy.arange(len(labels_ccc_seenid)), correct_classes_seenid, 'mo', markersize=3, markerfacecolor='magenta')
####majorLocator_y   = MultipleLocator(block_size)
#####majorLocator_x   = MultipleLocator(block_size_seenid)
####majorLocator_x   = MultipleLocator(block_size_seenid)
####p12.xaxis.set_major_locator(majorLocator_x)
####p12.yaxis.set_major_locator(majorLocator_y)
####majorLocator_y   = MultipleLocator(block_size)
###
###p13 = f2.add_subplot(313, frame_on=False)
###p13.plot(numpy.arange(len(correct_classes_newid)), correct_classes_newid, 'r.', markersize=2, markerfacecolor='red')
###p13.plot(numpy.arange(len(classes_ccc_newid)), classes_ccc_newid, 'm.', markersize=2, markerfacecolor='magenta')
###p13.plot(numpy.arange(len(classes_svm_newid)), classes_svm_newid, 'g.', markersize=2, markerfacecolor='green')
###
###plt.xlabel("Image Number, New Id Set. Classification Rate CCC=%f, CR_Gauss=%.3f, CR_SVM=%.3f" % ( results.class_ccc_rate_newid, results.class_Gauss_rate_newid, results.class_svm_rate_newid))
###plt.ylabel("Class Number")
###p13.grid(True)
####majorLocator_y = MultipleLocator(block_size)
#####majorLocator_x   = MultipleLocator(block_size_seenid)
####majorLocator_x   = MultipleLocator(block_size_newid)
####p13.xaxis.set_major_locator(majorLocator_x)
#####p13.yaxis.set_major_locator(majorLocator_y)
###
###f3 = plt.figure()
###regression_text_labels = ["Closest Center Class.", "SVM", "Linear Regression", "Gaussian Class/Regr.", "Ground Truth"]
###plt.suptitle("Regression Results (Labels) using %d Slow Signals"%reg_num_signals)
####plt.title("Training Set")
###p11 = f3.add_subplot(311, frame_on=False)
####correct_classes_training = numpy.arange(len(labels_ccc_training)) / block_size
###
###p11.plot(numpy.arange(len(labels_ccc_training)), labels_ccc_training, 'b.', markersize=3, markerfacecolor='blue')
###p11.plot(numpy.arange(len(regression_svm_training)), regression_svm_training, 'g.', markersize=3, markerfacecolor='green')
###p11.plot(numpy.arange(len(correct_labels_training)), regression_lr_training, 'c.', markersize=3, markerfacecolor='cyan')
###p11.plot(numpy.arange(len(regression_Gauss_training)), regression_Gauss_training, 'm.', markersize=3, markerfacecolor='magenta')
###p11.plot(numpy.arange(len(correct_labels_training)), correct_labels_training, 'r.', markersize=3, markerfacecolor='red')
###
#####draw horizontal and vertical lines
####majorLocator   = MultipleLocator(block_size)
####p11.xaxis.set_major_locator(majorLocator)
#####p11.yaxis.set_major_locator(majorLocator)
####plt.xticks(numpy.arange(0, len(labels_ccc_training), block_size)) 
####plt.yticks(numpy.arange(0, len(labels_ccc_training), block_size)) 
###plt.xlabel("Image Number, Training Set. Classification Rate=%f, MSE_CCC=%f, MSE_Gauss=%f, MSE_SVM=%f, MSE_LR=%f" % ( results.class_ccc_rate_train, results.mse_ccc_train, results.mse_gauss_train, results.mse_svm_train, results.mse_lr_train))
###plt.ylabel("Label")
###plt.legend( (regression_text_labels[0],  regression_text_labels[1], regression_text_labels[2], regression_text_labels[3], regression_text_labels[4]), loc=2 )
###p11.grid(True)
###
###
###p12 = f3.add_subplot(312, frame_on=False)
####correct_classes_seenid = numpy.arange(len(labels_ccc_seenid)) * len(labels_ccc_training) / len(labels_ccc_seenid) / block_size
###p12.plot(numpy.arange(len(labels_ccc_seenid)), labels_ccc_seenid, 'b.', markersize=4, markerfacecolor='blue')
###p12.plot(numpy.arange(len(regression_svm_seenid)), regression_svm_seenid, 'g.', markersize=4, markerfacecolor='green')
###p12.plot(numpy.arange(len(regression_lr_seenid)), regression_lr_seenid, 'c.', markersize=4, markerfacecolor='cyan')
###p12.plot(numpy.arange(len(regression_Gauss_seenid)), regression_Gauss_seenid, 'm.', markersize=4, markerfacecolor='magenta')
###p12.plot(numpy.arange(len(correct_labels_seenid)), correct_labels_seenid, 'r.', markersize=4, markerfacecolor='red')
#####majorLocator_y   = MultipleLocator(block_size)
#####majorLocator_x   = MultipleLocator(block_size_seenid)
####majorLocator_x   = MultipleLocator( len(labels_ccc_seenid) * block_size / len(labels_ccc_training))
####p12.xaxis.set_major_locator(majorLocator_x)
#####p12.yaxis.set_major_locator(majorLocator_y)
###plt.xlabel("Image Number, Seen Id Set. Classification Rate=%f, MSE_CCC=%f, MSE_Gauss=%f, MSE_SVM=%f, MSE_LR=%f" % (results.class_ccc_rate_seenid, results.mse_ccc_seenid, results.mse_gauss_seenid, results.mse_svm_seenid, results.mse_lr_seenid ))
###plt.ylabel("Label")
###plt.legend( (regression_text_labels[0],  regression_text_labels[1], regression_text_labels[2], regression_text_labels[3], regression_text_labels[4]), loc=2 )
###p12.grid(True)
###
###
###p13 = f3.add_subplot(313, frame_on=False)
###p13.plot(numpy.arange(len(labels_ccc_newid)), labels_ccc_newid, 'b.', markersize=4, markerfacecolor='blue')
###p13.plot(numpy.arange(len(regression_svm_newid)), regression_svm_newid, 'g.', markersize=4, markerfacecolor='green')
###p13.plot(numpy.arange(len(regression_lr_newid)), regression_lr_newid, 'c.', markersize=4, markerfacecolor='cyan')
###p13.plot(numpy.arange(len(regression_Gauss_newid)), regression_Gauss_newid, 'm.', markersize=4, markerfacecolor='magenta')
###p13.plot(numpy.arange(len(correct_labels_newid)), correct_labels_newid, 'r.', markersize=4, markerfacecolor='red')
#####majorLocator_y   = MultipleLocator(block_size)
#####majorLocator_x   = MultipleLocator(block_size_seenid)
####majorLocator_x   = MultipleLocator( len(labels_ccc_newid) * block_size / len(labels_ccc_training))
####p13.xaxis.set_major_locator(majorLocator_x)
#####p12.yaxis.set_major_locator(majorLocator_y)
###plt.xlabel("Image Number, New Id Set. Classification Rate=%f, MSE_CCC=%f, MSE_Gauss=%f, MSE_SVM=%f, MSE_LR=%f" % ( results.class_ccc_rate_newid, results.mse_ccc_newid, results.mse_gauss_newid, results.mse_svm_newid, results.mse_lr_newid))
###plt.ylabel("Label")
###plt.legend( (regression_text_labels[0],  regression_text_labels[1], regression_text_labels[2], regression_text_labels[3], regression_text_labels[4]), loc=2 )
###p13.grid(True)
###
###
###print "************** Displaying Probability Profiles ***********"
###f4 = plt.figure()
###plt.suptitle("Probability Profiles of Gaussian Classifier Using %d Signals for Classification"%reg_num_signals)
###  
####display Probability Profile of Training Set
###ax11 = plt.subplot(1,3,1)
###plt.title("(Network) Training Set")
####cax = p11.add_axes([1, 10, 1, 10])
###pic = ax11.imshow(probs_training, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.hot)
###plt.xlabel("Class Number")
###plt.ylabel("Image Number, Training Set")
###f4.colorbar(pic)
###
####display Probability Profile of Seen Id
###ax11 = plt.subplot(1,3,2)
###plt.title("Seen Id Test Set")
###pic = ax11.imshow(probs_seenid, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.hot)
###plt.xlabel("Class Number")
###plt.ylabel("Image Number, Seen Id Set")
###f4.colorbar(pic)
###
####display Probability Profile of New Id
###ax11 = plt.subplot(1,3,3)
###plt.title("New Id Test Set")
###pic = ax11.imshow(probs_newid, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.hot)
###plt.xlabel("Class Number")
###plt.ylabel("Image Number, New Id Set")
###f4.colorbar(pic)
###
###print "************ Displaying Linear (or Non-Linear) Morphs and Masks Learned by SFA **********"
####Create Figure
###ax6 = plt.figure()
###ax6.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
###
####ax_6.subplots_adjust(hspace=0.5)
###plt.suptitle("Linear (or Non-Linear) Morphs using SFA")
###
####display SFA
###ax6_11 = plt.subplot(4,5,1)
###plt.title("Train-Signals in Slow Domain")
###sl_seqdisp = sl_seq[:, range(0,hierarchy_out_dim)]
###sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
###ax6_11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###plt.ylabel("Image number")
####plt.xlabel("Slow Signal S[im][sl]")
###
###ax6_12 = plt.subplot(4,5,2)
###plt.title("Selected Original Image")
###ax6_12.axis('off')
###
###ax6_13 = plt.subplot(4,5,3)
###plt.title("Approx. Image x'")
###ax6_13.axis('off')
###
###ax6_14 = plt.subplot(4,5,4)
###plt.title("Re-Approx Image x''")
###ax6_14.axis('off')
###
###if show_linear_inv == True:
###    ax6_15 = plt.subplot(4,5,5)
###    plt.title("Avg. Image H-1(0)=z'")
###    error_scale_disp=1.0
###    z_p = pinv_zero
###    pinv_zero_disp = scale_to(pinv_zero, pinv_zero.mean(), pinv_zero.max()-pinv_zero.min(), 127.5, 255.0, error_scale_disp, 'tanh')
###    ax6_15.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, zero" % (pinv_zero_disp.min(), pinv_zero_disp.max(), pinv_zero_disp.std(), error_scale_disp))
###    ax6_15.imshow(pinv_zero_disp.clip(0,255), aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###    ax6_15.axis('off')
###
###ax6_21 = plt.subplot(4,5,6)
###plt.title("H-1 (y*), y*[sl]=-2")
###plt.yticks([])
###plt.xticks([])
###plt.ylabel("Modified Projection")
###
###ax6_22 = plt.subplot(4,5,7)
###plt.title("H-1 (y*), y*[sl]=-1")
###ax6_22.axis('off')
###
###ax6_23 = plt.subplot(4,5,8)
###plt.title("H-1 (y*), y*[sl]=0")
###ax6_23.axis('off')
###
###ax6_24 = plt.subplot(4,5,9)
###plt.title("H-1 (y*), y*[sl]=1")
###ax6_24.axis('off')
###
###ax6_25 = plt.subplot(4,5,10)
####plt.title("x' - rec S[im][sl]=2")
###plt.title("H-1 (y*), y*[sl]=2")
###ax6_25.axis('off')
###
###
###ax6_31 = plt.subplot(4,5,11)
###plt.title("x-x'+H-1(S*-S), S*[sl]=-2")
###plt.yticks([])
###plt.xticks([])
###plt.ylabel("Morph")
###
###ax6_32 = plt.subplot(4,5,12)
###plt.title("x-x'+H-1(S*-S), S*[sl]=-1")
###ax6_32.axis('off')
###
###ax6_33 = plt.subplot(4,5,13)
###plt.title("x-x'+H-1(S*-S), S*[sl]=0")
###ax6_33.axis('off')
###
###ax6_34 = plt.subplot(4,5,14)
###plt.title("x-x'+H-1(S*-S), S*[sl]=1")
###ax6_34.axis('off')
###
###ax6_35 = plt.subplot(4,5,15)
###plt.title("x-x'+H-1(S*-S), S*[sl]=2")
###ax6_35.axis('off')
###
###ax6_41 = plt.subplot(4,5,16)
###plt.title("x-x'+H-1(SFA_train[0]-S)")
###plt.yticks([])
###plt.xticks([])
###plt.ylabel("Morph from SFA_Train")
###
###ax6_42 = plt.subplot(4,5,17)
###plt.title("x-x'+H-1(SFA_train[1/4]-S)")
###ax6_42.axis('off')
###
###ax6_43 = plt.subplot(4,5,18)
###plt.title("x-x'+H-1(SFA_train[2/4]-S)")
###ax6_43.axis('off')
###
###ax6_44 = plt.subplot(4,5,19)
###plt.title("x-x'+H-1(SFA_train[3/4]-S)")
###ax6_44.axis('off')
###
###ax6_45 = plt.subplot(4,5,20)
###plt.title("x-x'+H-1(SFA_train[4/4]-S)")
###ax6_45.axis('off')
###
###print "************ Displaying Linear (or Non-Linear) Masks Learned by SFA **********"
####Create Figure
###ax7 = plt.figure()
###ax7.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
###
####ax_6.subplots_adjust(hspace=0.5)
###plt.suptitle("Linear (or Non-Linear) Masks Learned by SFA [0 - 4]")
###
###mask_normalize = False
###lim_delta_sfa = 0.01
###num_masks = 4
###slow_values = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
###axes = range(num_masks)
###for ma in range(num_masks):
###    axes[ma] = range(len(slow_values))
###
###for ma in range(num_masks):
###    for sl, slow_value in enumerate(slow_values):
###        tmp_ax = plt.subplot(4,6,ma*len(slow_values)+sl+1)
###        plt.axes(tmp_ax)
###        plt.title("H-1( S[%d]=%d ) - z'"%(ma, slow_value))
###
###        if sl == 0:
###            plt.yticks([])
###            plt.xticks([])
###            plt.ylabel("Mask[%d]"%ma)
###        else:
###            tmp_ax.axis('off')
###        axes[ma][sl] = tmp_ax
###
####Create Figure
###ax8 = plt.figure()
###ax8.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
###
####ax_6.subplots_adjust(hspace=0.5)
###plt.suptitle("Linear (or Non-Linear) Masks Learned by SFA [4 to 15]")
###
###masks2 = range(4,16) 
###num_masks2 = len(masks2)
###slow_values2 = [-1.0, 1.0]
###axes2 = range(num_masks2)
###for ma in range(num_masks2):
###    axes2[ma] = range(len(slow_values2))
###
###for ma, mask in enumerate(masks2):
###    for sl, slow_value in enumerate(slow_values2):
###        tmp_ax = plt.subplot(4,6,ma*len(slow_values2)+sl+1)
###        plt.axes(tmp_ax)
###        plt.title("H-1( S[%d]=%d ) - z'"%(mask, slow_value))
###        if sl == 0:
###            plt.yticks([])
###            plt.xticks([])
###            plt.ylabel("Mask[%d]"%mask)
###        else:
###            tmp_ax.axis('off')
###        axes2[ma][sl] = tmp_ax
###
###
###print "************ Displaying Localized Morphs and Masks **********"
####Create Figure
###ax9 = plt.figure()
###ax9.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
###
####ax_6.subplots_adjust(hspace=0.5)
###plt.suptitle("Localized Linear (or Non-Linear) Morphs")
###
###ax9_11 = plt.subplot(4,5,1)
###plt.title("Train-Signals in Slow Domain")
###sl_seqdisp = sl_seq[:, range(0,hierarchy_out_dim)]
###sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
###ax9_11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###plt.ylabel("Image number")
###
###ax9_12 = plt.subplot(4,5,2)
###plt.title("Selected Original Image")
###ax9_11.axis('off')
###
###ax9_13 = plt.subplot(4,5,3)
###plt.title("Loc. Approx. Image x'")
###ax9_12.axis('off')
###
###ax9_14 = plt.subplot(4,5,4)
###plt.title("Loc. Re-Approx Image x''")
###ax9_13.axis('off')
###
###ax9_21 = plt.subplot(4,5,6)
###plt.title("-8*Mask(cl -> cl_prev)")
###plt.yticks([])
###plt.xticks([])
###plt.ylabel("Modified Loc. Inv")
###
###ax9_22 = plt.subplot(4,5,7)
###plt.title("-4*Mask(cl -> cl_prev)")
###ax9_22.axis('off')
###
###ax9_23 = plt.subplot(4,5,8)
###plt.title("2*Mask(cl -> cl_prev)")
###ax9_23.axis('off')
###
###ax9_24 = plt.subplot(4,5,9)
###plt.title("4*Mask(cl -> cl_prev)")
###ax9_24.axis('off')
###
###ax9_25 = plt.subplot(4,5,10)
###plt.title("8*Mask(cl -> cl_prev)")
###ax9_25.axis('off')
###
###print "************ Displaying Localized Morphs Learned by SFA **********"
####Create Figure
###ax10 = plt.figure()
###ax10.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
###
###num_morphs = 20
###morph_step = 0.5
###all_axes_morph_inc = []
###all_classes_morph_inc = numpy.arange(num_morphs)*morph_step
###for i in range(len(all_classes_morph_inc)):
###    tmp_ax = plt.subplot(4,5,i+1)
###    plt.title("Morph(cl* -> cl*)")
###    tmp_ax.axis('off')
###    all_axes_morph_inc.append(tmp_ax)
###
###
###ax11 = plt.figure()
###ax11.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
###
###all_axes_morph_dec = []
###all_classes_morph_dec = numpy.arange(0, -1 * num_morphs, -1) * morph_step
###for i in range(len(all_classes_morph_dec)):
###    tmp_ax = plt.subplot(4, 5,20-i)
###    plt.title("Morph (cl*-> cl*)")
###    tmp_ax.axis('off')
###    all_axes_morph_dec.append(tmp_ax)
###
####morphed sequence in SFA domain
###ax12 = plt.figure()
###ax12_1 = plt.subplot(2, 2, 1)
###plt.title("SFA of Morphed Images")
###ax12_2 = plt.subplot(2, 2, 2)
###plt.title("Average SFA for each Class")
###sl_seq_meandisp = scale_to(sl_seq_training_mean, sl_seq_training_mean.mean(), sl_seq_training_mean.max()-sl_seq_training_mean.min(), 127.5, 255.0, scale_disp, 'tanh')
###ax12_2.imshow(sl_seq_meandisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###ax12_3 = plt.subplot(2, 2, 3)
###plt.title("SFA of Selected Image")
###
###print "************ Displaying Localized) Morphs Learned by SFA **********"
####Create Figure
###ax13 = plt.figure()
###ax13.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
###
###all_axes_mask_inc = []
###for i in range(len(all_classes_morph_inc)):
###    tmp_ax = plt.subplot(4,5,i+1)
###    plt.title("Mask(cl* -> cl*)")
###    tmp_ax.axis('off')
###    all_axes_mask_inc.append(tmp_ax)
###
###ax14 = plt.figure()
###ax14.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
###
###all_axes_mask_dec = []
###for i in range(len(all_classes_morph_dec)):
###    tmp_ax = plt.subplot(4, 5,20-i)
###    plt.title("Mask (cl*-> cl*)")
###    tmp_ax.axis('off')
###    all_axes_mask_dec.append(tmp_ax)
###    
###
#####ax_6.subplots_adjust(hspace=0.5)
####plt.suptitle("Localized Linear (or Non-Linear) Masks Learned by SFA [0 - 4]")
####lim_delta_sfa = 0.001
####num_masks3 = 4
####slow_values3 = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
####axes3 = range(num_masks3)
####for ma in range(num_masks3):
####    axes3[ma] = range(len(slow_values3))
####
####for ma in range(num_masks3):
####    for sl, slow_value in enumerate(slow_values3):
####        tmp_ax = plt.subplot(4,6,ma*len(slow_values3)+sl+1)
####        plt.axes(tmp_ax)
####        plt.title("H-1( S[%d]=%d ) - z'"%(ma, slow_value))
####
####        if sl == 0:
####            plt.yticks([])
####            plt.xticks([])
####            plt.ylabel("Mask[%d]"%ma)
####        else:
####            tmp_ax.axis('off')
####        axes3[ma][sl] = tmp_ax
####
#####Create Figure
####ax11 = plt.figure()
####ax11.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
####
#####ax_6.subplots_adjust(hspace=0.5)
####plt.suptitle("Linear (or Non-Linear) Masks Learned by SFA [4 to 15]")
####
####masks4 = range(4,16) 
####num_masks4 = len(masks4)
####slow_values4 = [-1.0, 1.0]
####axes4 = range(num_masks4)
####for ma in range(num_masks4):
####    axes4[ma] = range(len(slow_values4))
####
####for ma, mask in enumerate(masks4):
####    for sl, slow_value in enumerate(slow_values4):
####        tmp_ax = plt.subplot(4,6,ma*len(slow_values4)+sl+1)
####        plt.axes(tmp_ax)
####        plt.title("H-1( S[%d]=%d ) - z'"%(mask, slow_value))
####        if sl == 0:
####            plt.yticks([])
####            plt.xticks([])
####            plt.ylabel("Mask[%d]"%mask)
####        else:
####            tmp_ax.axis('off')
####        axes4[ma][sl] = tmp_ax
###
###
####UNCOMMENT THIS!!!! and add code to show result of localized inversion
####print "************ Displaying Localized Morphs and Masks **********"
#####Create Figure
####ax9 = plt.figure()
####ax9.subplots_adjust(hspace=0.3, wspace=0.03, top=0.93, right=0.96, bottom=0.05, left=0.05)
####
#####ax_6.subplots_adjust(hspace=0.5)
####plt.suptitle("Localized (Linear or Non-Linear) Morphs using SFA")
####
#####display SFA
####ax9_11 = plt.subplot(4,5,1)
####plt.title("Train-Signals in Slow Domain")
####sl_seqdisp = sl_seq[:, range(0,hierarchy_out_dim)]
####sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
####ax9_11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
####plt.ylabel("Image number")
####
####mask_normalize = False
####lim_delta_sfa = 0.001
####num_masks = 4
####slow_values = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
####axes = range(num_masks)
####for ma in range(num_masks):
####    axes[ma] = range(len(slow_values))
####
####for ma in range(num_masks):
####    for sl, slow_value in enumerate(slow_values):
####        tmp_ax = plt.subplot(4,6,ma*len(slow_values)+sl+1)
####        plt.axes(tmp_ax)
####        plt.title("H-1( S[%d]=%d ) - z'"%(ma, slow_value))
####
####        if sl == 0:
####            plt.yticks([])
####            plt.xticks([])
####            plt.ylabel("Mask[%d]"%ma)
####        else:
####            tmp_ax.axis('off')
####        axes[ma][sl] = tmp_ax
###
###
###
####Retrieve Image in Sequence
###def mask_on_press(event):
###    global plt, ax6, ax6_11, ax6_12, ax6_13, ax6_14, ax6_21, ax6_22, ax6_23, ax6_24, ax6_25, ax6_31, ax6_32, ax6_33, ax6_34, ax6_35, ax6_41, ax6_42, ax6_43, ax6_44, ax6_45
###    global ax7, axes, num_masks, slow_values, mask_normalize, lim_delta_sfa
###    global ax8, axes2, masks2, slow_values2
###    global ax9, ax9_11, ax9_12, ax9_13, ax9_14, ax9_21, ax9_22, ax9_23, ax9_24, ax9_25
###
###    global subimages, sTrain, sl_seq, pinv_zero, flow, error_scale_disp
###    
###    print 'you pressed', event.button, event.xdata, event.ydata
###
###    if event.xdata == None or event.ydata==None:
###        mask_normalize = not mask_normalize
###        print "mask_normalize is: ", mask_normalize
###        return
###    
###    y = int(event.ydata)
###    if y < 0:
###        y = 0
###    if y >= num_images:
###        y = num_images - 1
###    x = int(event.xdata)
###    if x < 0:
###        x = 0
###    if x >= hierarchy_out_dim:
###        x = hierarchy_out_dim -1
###    print "Image Selected=" + str(y) + " , Slow Component Selected=" + str(x)
###
###    print "Displaying Original and Reconstructions"
####Display Original Image
###    subimage_im = subimages[y].reshape((sTrain.subimage_height, sTrain.subimage_width))
###    ax6_12.imshow(subimage_im.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###
###    if show_linear_inv == True:
###        #Display Reconstructed Image
###        data_out = sl_seq[y].reshape((1, hierarchy_out_dim))
###        inverted_im = flow.inverse(data_out)
###        inverted_im = inverted_im.reshape((sTrain.subimage_height, sTrain.subimage_width))
###        x_p = inverted_im
###        inverted_im_ori = inverted_im.copy()
###        ax6_13.imshow(inverted_im.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)        
###    
###        #Display Re-Reconstructed Image
###        re_data_out = flow.execute(inverted_im_ori.reshape((1, sTrain.subimage_height * sTrain.subimage_width)))
###        re_inverted_im = flow.inverse(re_data_out)
###        re_inverted_im = re_inverted_im.reshape((sTrain.subimage_height, sTrain.subimage_width))
###        ax6_14.imshow(re_inverted_im.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###    
###    if show_linear_morphs == True:   
###        print "Displaying Morphs, Original Version, no localized inverses"
###        #Display: Altered Reconstructions
###        #each tuple has the form: (val of slow_signal, remove, axes for display)
###        #where remove is None, "avg" or "ori"
###        error_scale_disp=1.0
###        disp_data = [(-2, "inv", ax6_21), (-1, "inv", ax6_22), (0, "inv", ax6_23), (1, "inv", ax6_24), (2, "inv", ax6_25), \
###                     (-2, "mor", ax6_31), (-1, "mor", ax6_32), (0, "mor", ax6_33), (1, "mor", ax6_34), (2, "mor", ax6_35), 
###                     (-2, "mo2", ax6_41), (-1, "mo2", ax6_42), (0, "mo2", ax6_43), (1, "mo2", ax6_44), (2, "mo2", ax6_45)]
###      
###        work_sfa = data_out.copy()
###           
###        for slow_value, display_type, fig_axes in disp_data:
###            work_sfa[0][x] = slow_value
###            inverted_im = flow.inverse(work_sfa)
###            inverted_im = inverted_im.reshape((sTrain.subimage_height, sTrain.subimage_width))
###            if display_type == "inv":
###                fig_axes.imshow(inverted_im.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)        
###            elif display_type == "mor":
###                # delta_sfa = sfa*-sfa
###                delta_sfa = numpy.zeros((1, hierarchy_out_dim))
###                delta_sfa[0][x] = slow_value
###                delta_im = flow.inverse(delta_sfa)
###                delta_im = delta_im.reshape((sTrain.subimage_height, sTrain.subimage_width))                      
###                morphed_im = subimage_im - x_p + delta_im
###    #            morphed_im = subimage_im - x_p + inverted_im - z_p
###    #            morphed_im = morphed.reshape((sTrain.subimage_height, sTrain.subimage_width))           
###    #            inverted_im = inverted_im - pinv_zero
###    #            inverted_im_disp = scale_to(inverted_im, inverted_im.mean(), inverted_im.max()-inverted_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
###    #            morphed_im_disp = scale_to(morphed_im, morphed_im.mean(), morphed_im.max()-morphed_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
###                morphed_im_disp = morphed_im
###                fig_axes.imshow(morphed_im_disp.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)        
###            elif display_type == "mo2":
###                # delta_sfa = sfa*-sfa
###                sfa_asterix = sl_seq[(slow_value + 2) * (num_images-1) / 4].reshape((1, hierarchy_out_dim))
###                delta_sfa = sfa_asterix - data_out
###                delta_im = flow.inverse(delta_sfa)
###                delta_im = delta_im.reshape((sTrain.subimage_height, sTrain.subimage_width))                      
###                morphed_im = subimage_im - x_p + delta_im
###                morphed_im_disp = morphed_im
###                fig_axes.imshow(morphed_im_disp.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)                 
###    ax6.canvas.draw()
###
###    if show_linear_masks == True:  
###        print "Displaying Masks [0-3]"
###        for ma in range(num_masks):
###            for sl, slow_value in enumerate(slow_values):
###                tmp_ax = axes[ma][sl]
###
###                print "Computing mask %d, slow_value %d"%(ma, slow_value)
###                work_sfa = data_out.copy()
###                work_sfa[0][ma] = work_sfa[0][ma] + slow_value * lim_delta_sfa
###                mask_im = flow.inverse(work_sfa)
###                mask_im = (mask_im.reshape((sTrain.subimage_height, sTrain.subimage_width)) - x_p) / lim_delta_sfa
###                if mask_normalize == True:
###                    mask_im_disp = scale_to(mask_im, 0.0, mask_im.max()-mask_im.min(), 127.5, 127.5, error_scale_disp, 'tanh')
###                else:
###                    mask_im_disp = mask_im + 127.5
###                axes[ma][sl].imshow(mask_im_disp.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)          
###        ax7.canvas.draw()
###        
###    if show_linear_masks_ext == True:  
###        print "Displaying Masks [4-15]"
###        for ma, mask in enumerate(masks2):
###            for sl, slow_value in enumerate(slow_values2):
###                tmp_ax = axes2[ma][sl]
###                work_sfa = data_out.copy()
###                work_sfa[0][mask] += slow_value * lim_delta_sfa
###                mask_im = flow.inverse(work_sfa)
###                mask_im = (mask_im.reshape((sTrain.subimage_height, sTrain.subimage_width)) - x_p) / lim_delta_sfa
###                if mask_normalize == True:
###                    mask_im_disp = scale_to(mask_im, 0.0, mask_im.max()-mask_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
###                else:
###                    mask_im_disp = mask_im + 127.5
###                axes2[ma][sl].imshow(mask_im_disp.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)          
###        ax8.canvas.draw()
###
###ax6.canvas.mpl_connect('button_press_event', mask_on_press)
###
####QUESTION: WHEN SHOULD I USE THE KEYWORK global?????
###def localized_on_press(event):
###    global plt, ax9, ax9_11, ax9_12, ax9_13, ax9_14, ax9_21, ax9_22, ax9_23, ax9_24, ax9_25
###    global ax9_31, ax9_32, ax9_33, ax9_34, ax9_35
###    global ax9_41, ax9_42, ax9_43, ax9_44, ax9_45
###    global subimages, sTrain, sl_seq, flow, error_scale_disp, hierarchy_out_dim
###    global mask_normalize, lim_delta_sfa, correct_classes_training, S2SC, block_size
###    global ax10, ax11, all_axes_morph
###    global ax13, ax14
###    
###    print 'you pressed', event.button, event.xdata, event.ydata
###
###    if event.xdata == None or event.ydata==None:
###        mask_normalize = not mask_normalize
###        print "mask_normalize was successfully changed to: ", mask_normalize
###        return
###    
###    y = int(event.ydata)
###    if y < 0:
###        y = 0
###    if y >= num_images:
###        y = num_images - 1
###    x = int(event.xdata)
###    if x < 0:
###        x = 0
###    if x >= hierarchy_out_dim:
###        x = hierarchy_out_dim -1
###    print "Image Selected=" + str(y) + " , Slow Component Selected=" + str(x)
###
###    print "Displaying Original and Reconstructions"
####Display Original Image
###    subimage_im = subimages[y].reshape((sTrain.subimage_height, sTrain.subimage_width))
###    ax9_12.imshow(subimage_im.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###
###    if show_localized_morphs == True: 
###        #Display Localized Reconstructed Image
###        data_in = subimages[y].reshape((1, sTrain.subimage_height * sTrain.subimage_width))
###        data_out = sl_seq[y].reshape((1, hierarchy_out_dim))
###        loc_inverted_im = flow.localized_inverse(data_in, data_out)
###        loc_inverted_im = loc_inverted_im.reshape((sTrain.subimage_height, sTrain.subimage_width))
###        loc_inverted_im_ori = loc_inverted_im.copy()
###        ax9_13.imshow(loc_inverted_im.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)        
###    
###        #Display Re-Reconstructed Image
###        loc_re_data_in = loc_inverted_im_ori.reshape((1, sTrain.subimage_height * sTrain.subimage_width))
###        loc_re_data_out = flow.execute(loc_re_data_in)
###        loc_re_inverted_im = flow.localized_inverse(loc_re_data_in, loc_re_data_out)
###        loc_re_inverted_im = loc_re_inverted_im.reshape((sTrain.subimage_height, sTrain.subimage_width))
###        ax9_14.imshow(loc_re_inverted_im.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
###        
###        print "Displaying Masks Using Localized Inverses"
###        error_scale_disp=1.0
###        disp_data = [(-8, "lmsk", ax9_21), (-4.0, "lmsk", ax9_22), (2.0, "lmsk", ax9_23), (4.0, "lmsk", ax9_24), (8.0, "lmsk", ax9_25)]
###    
###        data_in = subimages[y].reshape((1, sTrain.subimage_height * sTrain.subimage_width))
###        data_out = sl_seq[y].reshape((1, hierarchy_out_dim))  
###        work_sfa = data_out.copy()
###           
###        for scale_factor, display_type, fig_axes in disp_data:
###            #WARNING, this should not be computed this way!!!!
###            current_class = y/block_size
###            print "Current classs is:", current_class
###            if scale_factor < 0:
###                next_class = current_class-1
###                if next_class < 0:
###                    next_class = 0
###            else:
###                next_class = current_class+1
###                if next_class >= hierarchy_out_dim:
###                    next_class = hierarchy_out_dim-1
###                
###            current_avg_sfa = sl_seq[current_class * block_size:(current_class+1)*block_size,:].mean(axis=0)
###            next_avg_sfa = sl_seq[next_class*block_size:(next_class+1)*block_size,:].mean(axis=0)
###                    
###            print "Current class is ", current_class
###            #print "Current class_avg is ", current_avg_sfa
###            #print "Next class_avg is ", next_avg_sfa
###            
###            data_out_next = next_avg_sfa
###            print "Computing from class %d to class %d, slow_value %d"%(current_class, next_class, scale_factor)
###            work_sfa = data_out * (1-lim_delta_sfa) + data_out_next * lim_delta_sfa
###            t_loc_inv0 = time.time()
###            mask_im = flow.localized_inverse(data_in, work_sfa, verbose=False)
###            t_loc_inv1 = time.time()
###            print "Localized inverse computed in %0.3f s"% ((t_loc_inv1-t_loc_inv0)) 
###            mask_im = (mask_im - data_in).reshape((sTrain.subimage_height, sTrain.subimage_width))  / lim_delta_sfa
###            if mask_normalize == True:
###                mask_im_disp = abs(scale_factor) * scale_to(mask_im, 0.0, mask_im.max()-mask_im.min(), 127.5, 127.5, error_scale_disp, 'tanh')
###            else:
###                mask_im_disp = abs(scale_factor) * mask_im + 127.5
###            fig_axes.imshow(mask_im_disp.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)           
###            fig_axes.set_title('%0.2f x Mask: cl %d => %d'%(abs(scale_factor), current_class, next_class))
###        ax9.canvas.draw()
###        
###        error_scale_disp=1.0
###        print "Displaying Morphs Using Localized Inverses Incrementing Class"
###
###        num_morphs_inc = len(all_classes_morph_inc)
###        num_morphs_dec = len(all_classes_morph_dec)
###        #make a function specialized in morphs, use this variable,
###        morph_outputs =  range(num_morphs_inc + num_morphs_dec - 1)
###        morph_sfa_outputs = range(num_morphs_inc + num_morphs_dec - 1)
###        original_class = y/block_size
###
###
###        #WARNING!!! THERE IS A BUG, IN WHICH THE LAST INC MORPHS ARE INCORRECTLY COMBINED USING ZERO??? 
###        for ii, action in enumerate(["inc", "dec"]):
###            current_class = original_class
###            if ii==0:
###                all_axes_morph = all_axes_morph_inc
###                all_axes_mask = all_axes_mask_inc
###                num_morphs = len(all_classes_morph_inc)
###                desired_next_classes = all_classes_morph_inc + current_class
###                max_class = num_images/block_size
###                for i in range(len(desired_next_classes)):
###                    if desired_next_classes[i] >= max_class:
###                        desired_next_classes[i] = -1
###            else:                
###                all_axes_morph = all_axes_morph_dec
###                all_axes_mask = all_axes_mask_dec
###                num_morphs = len(all_classes_morph_dec)
###                desired_next_classes = all_classes_morph_dec + current_class
###                for i in range(len(desired_next_classes)):
###                    if desired_next_classes[i] < 0:
###                        desired_next_classes[i] = -1
###
###            desired_next_sfa=[]
###            for next_class in desired_next_classes:
###                if next_class >= 0 and next_class < max_class:
###                    c1 = numpy.floor(next_class)
###                    c2 = c1  + 1               
###                    if c2 >= max_class:
###                        c2 = max_class-1 
###                    desired_next_sfa.append(sl_seq_training_mean[c1] * (1+c1-next_class) + sl_seq_training_mean[c2]*(next_class-c1))
###                else: #just in case something goes wrong
###                    desired_next_sfa.append(sl_seq_training_mean[0])
###                #sl_seq[next_class*block_size:(next_class+1)*block_size,:].mean(axis=0))
###    
###            data_in = subimages[y].reshape((1, sTrain.subimage_height * sTrain.subimage_width))
###            data_out = sl_seq[y].reshape((1, hierarchy_out_dim))
###            for i, next_class in enumerate(desired_next_classes):
###                if next_class == -1:
###                    if ii==0:
###                        morph_sfa_outputs[i+num_morphs_dec-1] = numpy.zeros(len(data_out[0])) 
###                    else:
###                        morph_sfa_outputs[num_morphs_dec-i-1] = numpy.zeros(len(data_out[0]))
###                    break          
###                data_out_next = desired_next_sfa[i]           
###                print "Morphing to desired class %.2f..."%next_class
###                
###                work_sfa = data_out * (1-lim_delta_sfa) + data_out_next * lim_delta_sfa
###    
###                t_loc_inv0 = time.time()
###                morphed_data = flow.localized_inverse(data_in, work_sfa, verbose=False)
###                t_loc_inv1 = time.time()
###                print "Localized inverse computed in %0.3f s"% ((t_loc_inv1-t_loc_inv0)) 
###                
###                morphed_data = data_in + (morphed_data - data_in)/lim_delta_sfa
###                morphed_im_disp = morphed_data.reshape((sTrain.subimage_height, sTrain.subimage_width)) 
###    
###                if all_axes_morph[i] != None:
###                    all_axes_morph[i].imshow(morphed_im_disp.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)           
###                    all_axes_morph[i].set_title("Morph(cl %.1f -> %.1f)"%(current_class,next_class))
###                else:
###                    print "No plotting Morph. (Reason: axes = None)"
###
###                if all_axes_mask[i] != None:
###                    loc_mask_data = morphed_data[0] - data_in[0]
###                    loc_mask_disp = loc_mask_data.reshape((sTrain.subimage_height, sTrain.subimage_width)) + 127.0
###                    loc_mask_disp = scale_to(loc_mask_disp, loc_mask_disp.mean(), loc_mask_disp.max() - loc_mask_disp.min(), 127.5, 255.0, scale_disp, 'tanh')
###                    all_axes_mask[i].imshow(loc_mask_disp.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)           
###                    all_axes_mask[i].set_title("Mask(cl %.1f -> %.1f)"%(current_class,next_class))
###                else:
###                    print "No plotting Mask. (Reason: axes = None)"
###                
###                current_class = next_class
###                data_in = morphed_data
###                data_out = flow.execute(data_in)
###                if ii==0: #20-29
###                    morph_sfa_outputs[i+num_morphs_dec-1] = data_out[0]
###                else: #0-19
###                    morph_sfa_outputs[num_morphs_dec-i-1] = data_out[0]
###                   
###        ax10.canvas.draw()
###        ax11.canvas.draw()
###        ax13.canvas.draw()
###        ax14.canvas.draw()
###
####        for i, sfa_out in enumerate(morph_sfa_outputs):
####            print "elem %d: "%i, "has shape", sfa_out.shape, "and is= ", sfa_out
###
####        morph_sfa_outputs[num_morphs_dec] = sl_seq[y]
###        sl_morph = numpy.array(morph_sfa_outputs)
###        sl_morphdisp = scale_to(sl_morph, sl_morph.mean(), sl_morph.max()-sl_morph.min(), 127.5, 255.0, scale_disp, 'tanh')
####        extent = (L, R, B, U)
###        extent = (0, hierarchy_out_dim-1, all_classes_morph_inc[-1]+original_class, all_classes_morph_dec[-1]+original_class-0.25)
###        ax12_1.imshow(sl_morphdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray, extent=extent)
####        majorLocator_y   = MultipleLocator(0.5)
####        ax12_1.yaxis.set_major_locator(majorLocator_y)
###        plt.ylabel("Morphs")
###        
### 
###        sl_selected = sl_seq[y][:].reshape((1, hierarchy_out_dim))
###        sl_selected = scale_to(sl_selected, sl_selected.mean(), sl_selected.max()-sl_selected.min(), 127.5, 255.0, scale_disp, 'tanh')
####        extent = (0, hierarchy_out_dim-1, all_classes_morph_inc[-1]+original_class+0.5, all_classes_morph_dec[-1]+original_class-0.5)
###        ax12_3.imshow(sl_selected.clip(0,255), aspect=8.0, interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
####        majorLocator_y   = MultipleLocator(0.5)
####        ax12_1.yaxis.set_major_locator(majorLocator_y)
####        plt.ylabel("Morphs")
###
###              
####        morphed_classes = numpy.concatenate((all_classes_morph_dec[::-1], [0], all_classes_morph_inc))
####        print "morphed_classes=", morphed_classes
####        morphed_classes = morphed_classes + original_class
####        majorLocator_y   = MultipleLocator(1)
####        #ax12_1.yticks(morphed_classes) 
####        ax12_1.yaxis.set_major_locator(majorLocator_y)
###        ax12.canvas.draw()
###        
###ax9.canvas.mpl_connect('button_press_event', localized_on_press)
###
###
####            work_sfa[0][x] = slow_value
####            t_loc_inv0 = time.time()
####            inverted_im = flow.localized_inverse(data_in, work_sfa, verbose=True)
####            t_loc_inv1 = time.time()
####            print "Localized inverse computed in %0.3f ms"% ((t_loc_inv1-t_loc_inv0)*1000.0) 
####
####            inverted_im = inverted_im.reshape((sTrain.subimage_height, sTrain.subimage_width))
####            if display_type == "inv":
####                fig_axes.imshow(inverted_im.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)        
###        
####        print "Displaying Masks Using Localized Inverses"
####        #Display: Altered Reconstructions
####        error_scale_disp=1.0
####        disp_data = [(-1, "inv", ax9_21), (-0.5, "inv", ax9_22), (0, "inv", ax9_23), (0.5, "inv", ax9_24), (1, "inv", ax9_25)]
####    
####        data_in = subimages[y].reshape((1, sTrain.subimage_height * sTrain.subimage_width))
####        data_out = sl_seq[y].reshape((1, hierarchy_out_dim))  
####        work_sfa = data_out.copy()
####           
####        for slow_value, display_type, fig_axes in disp_data:
####            work_sfa[0][x] = slow_value
####            t_loc_inv0 = time.time()
####            inverted_im = flow.localized_inverse(data_in, work_sfa, verbose=True)
####            t_loc_inv1 = time.time()
####            print "Localized inverse computed in %0.3f ms"% ((t_loc_inv1-t_loc_inv0)*1000.0) 
####
####            inverted_im = inverted_im.reshape((sTrain.subimage_height, sTrain.subimage_width))
####            if display_type == "inv":
####                fig_axes.imshow(inverted_im.clip(0,255), vmin=0, vmax=255, aspect='equal', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)        
###
###
###print "GUI Created, showing!!!!"
###plt.show()
###print "GUI Finished!"



#    def __init__(self):
#        self.name = "Training Data"
#        self.num_samples = 0
#        self.sl = []
#        self.correct_classes = []
#        self.correct_labels = []
#        self.classes = []
#        self.labels = []
#        self.block_size = []
#        self.eta_values = []
#        self.delta_values = []
#        self.class_ccc_rate = 0
#        self.gauss_class_rate = 0
#        self.reg_mse = 0
#        self.gauss_reg_mse = 0


#im_seq_base_dir = pFTrain.data_base_dir
#ids = pFTrain.ids
#ages = pFTrain.ages
#MIN_GENDER = pFTrain.MIN_GENDER
#MAX_GENDER = pFTrain.MAX_GENDER
#GENDER_STEP = pFTrain.GENDER_STEP
#genders = map(code_gender, numpy.arange(MIN_GENDER, MAX_GENDER,GENDER_STEP))
#racetweens = pFTrain.racetweens
#expressions = pFTrain.expressions 
#morphs = pFTrain.morphs 
#poses = pFTrain.poses
#lightings = pFTrain.lightings
#slow_signal = pFTrain.slow_signal
#step = pFTrain.step
#offset = pFTrain.offset
#
#image_files_training = create_image_filenames2(im_seq_base_dir, slow_signal, ids, ages, genders, racetweens, \
#                            expressions, morphs, poses, lightings, step, offset)

#params = [ids, expressions, morphs, poses, lightings]
#params2 = [ids, ages, genders, racetweens, expressions, morphs, poses, lightings]

#block_size= num_images / len(params[slow_signal])
#block_size = num_images / len(params2[slow_signal])

##translation = 4
#translation = pDataTraining.translation 
#
##WARNING
##scale_disp = 3
#scale_disp = 1
##image_width  = 640
##image_height = 480
#image_width  = pDataTraining.image_width   
#image_height = pDataTraining.image_height 
#subimage_width  = pDataTraining.subimage_width   
#
#subimage_height = pDataTraining.subimage_height 
##pixelsampling_x = 2
##pixelsampling_y = 2
#pixelsampling_x = pDataTraining.pixelsampling_x 
#pixelsampling_y = pDataTraining.pixelsampling_y 
#subimage_pixelsampling=pDataTraining.subimage_pixelsampling
#subimage_first_row= pDataTraining.subimage_first_row
#subimage_first_column=pDataTraining.subimage_first_column
#add_noise_L0 = pDataTraining.add_noise_L0
#convert_format=pDataTraining.convert_format
#
##translations_x=None
##translations_y=None
#
#translations_x=pDataTraining.translations_x
#translations_y=pDataTraining.translations_y
#trans_sampled=pDataTraining.trans_sampled



#IMPROVEMENT, block_size_L0, L1, L2 should be specified just before training
#print "Using training data for slowly varying vertical angle, and (very!!!) quickly varying user identity"
#print "suggested: include_tails=False, use normal"
#image_files = []
#skip = 1
#for i in range(len(image_files1)):
#    image_files.append(image_files1[i])
#    image_files.append(image_files2[i])
#    image_files.append(image_files3[i])
#    image_files.append(image_files4[i])
#    image_files.append(image_files5[i])
#    image_files.append(image_files6[i])
#    image_files.append(image_files7[i])
#
#    
#num_images = len(image_files)
#
##block_size=num_images/5
#block_size=7
#block_size_L0=block_size
#block_size_L1=block_size
#block_size_L2=block_size
#block_size_L3=block_size
#block_size_exec=block_size #(Used only for random walk)

#Specify images to be loaded
#IMPROVEMENT: implement with a dictionary

#im_seq_base_dir = "/local/tmp/escalafl/Alberto/Renderings20x500"
#ids=range(0,8)
#expressions=[0]
#morphs=[0]
#poses=range(0,500)
#lightings=[0]
#slow_signal=0
#step=2
#offset=0
#
#image_files_training = create_image_filenames(im_seq_base_dir, slow_signal, ids, expressions, morphs, poses, lightings, step, offset)


#    print "Shape of lat_mat_L0 is:", La.lat_mat
#Create L%d sfa node
#sfa_out_dim_La = 12
#pca_out_dim_La = 90
#pca_out_dim_La = sfa_out_dim_L0 x_field_channels_La * x_field_channels_La * 0.75 


##pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
#pca_node_L3 = mdp.nodes.SFANode(output_dim=pca_out_dim_L3, block_size=block_size_L3) 
#
#exp_node_L3 = GeneralExpansionNode(exp_funcs_L3, use_hint=True, max_steady_factor=0.35, \
#                 delta_factor=0.6, min_delta=0.0001)
##exp_out_dim_L3 = exp_node_L3.output_dim
##red_node_L3 = mdp.nodes.WhiteningNode(input_dim=exp_out_dim_L3, output_dim=red_out_dim_L3)   
#red_node_L3 = mdp.nodes.WhiteningNode(output_dim=red_out_dim_L3)   
#
##sfa_node_L3 = mdp.nodes.SFANode(input_dim=preserve_mask_L2.size, output_dim=sfa_out_dim_L3)
##sfa_node_L3 = mdp.nodes.SFANode()
#sfa_node_L3 = mdp.nodes.SFANode(output_dim=sfa_out_dim_L3, block_size=block_size_L3)
#
#t8 = time.time()


#x_field_channels_L0=5
#y_field_channels_L0=5
#x_field_spacing_L0=5
#y_field_spacing_L0=5
#in_channel_dim_L0=1

#    La.v1 = (La.x_field_spacing, 0)
#    La.v2 = (La.x_field_spacing, La.y_field_spacing)
#
#    La.preserve_mask = numpy.ones((La.y_field_channels, La.x_field_channels)) > 0.5
## 6 x 12
#print "About to create (lattice based) perceptive field of widht=%d, height=%d"%(La.x_field_channels, La.y_field_channels) 
#print "With a spacing of horiz=%d, vert=%d, and %d channels"%(La.x_field_spacing, La.y_field_spacing, La.in_channel_dim)
#
#(mat_connections_L0, lat_mat_L0) = compute_lattice_matrix_connections(v1_L0, v2_L0, preserve_mask_L0, subimage_width, subimage_height, in_channel_dim_L0)
#print "matrix connections L0:"
#print mat_connections_L0
#
#t1 = time.time()
#
#switchboard_L0 = PInvSwitchboard(subimage_width * subimage_height, mat_connections_L0)
#switchboard_L0.connections
#
#t2 = time.time()
#print "PInvSwitchboard L0 created in %0.3f ms"% ((t2-t1)*1000.0)
#
##Create single PCA Node
##pca_out_dim_L0 = 20
#num_nodes_RED_L0 = num_nodes_EXP_L0 = num_nodes_SFA_L0 = num_nodes_PCA_L0 = lat_mat_L0.size / 2
#
##pca_out_dim_L0 = 7
##exp_funcs_L0 = [identity, pair_prod_ex, pair_prod_adj1_ex, pair_prod_adj2_ex, pair_sqrt_abs_dif_adj2_ex]
##red_out_dim_L0 = 25
##sfa_out_dim_L0 = 20
##MEGAWARNING!!!
##pca_out_dim_L0 = 10
#
#pca_out_dim_L0 = pSFALayerL0.pca_out_dim
#exp_funcs_L0 = pSFALayerL0.exp_funcs
#red_out_dim_L0 = pSFALayerL0.red_out_dim
#sfa_out_dim_L0 = pSFALayerL0.sfa_out_dim
#
##pca_out_dim_L0 = 16
##
##exp_funcs_L0 = [identity,]
##red_out_dim_L0 = 0.99
##sfa_out_dim_L0 = 15
#
##WARNING!!!!!!!!!!!
##pca_node_L0 = mdp.nodes.WhiteningNode(input_dim=preserve_mask_L0.size, output_dim=pca_out_dim_L0)
##pca_node_L0 = mdp.nodes.SFANode(input_dim=preserve_mask_L0.size, output_dim=pca_out_dim_L0, block_size=block_size_L0)
#pca_node_L0 = pSFALayerL0.pca_node(output_dim=pca_out_dim_L0, **pSFALayerL0.pca_args)
##pca_node_L0 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L0)
##Create array of pca_nodes (just one node, but cloned)
#pca_layer_L0 = mdp.hinet.CloneLayer(pca_node_L0, n_nodes=num_nodes_PCA_L0)
#
##exp_funcs_L0 = [identity, pair_prod_ex, pair_sqrt_abs_dif_ex, pair_sqrt_abs_sum_ex]
##exp_node_L0 = GeneralExpansionNode(exp_funcs_L0, input_dim = pca_out_dim_L0, use_hint=True, max_steady_factor=0.25, \
#exp_node_L0 = GeneralExpansionNode(exp_funcs_L0, use_hint=True, max_steady_factor=0.25, \
#                 delta_factor=0.6, min_delta=0.001)
##exp_out_dim_L0 = exp_node_L0.output_dim
#exp_layer_L0 = mdp.hinet.CloneLayer(exp_node_L0, n_nodes=num_nodes_EXP_L0)
#
##Create Node for dimensionality reduction
##red_out_dim_L0 = 20.....PCANode, WhiteningNode
##red_node_L0 = mdp.nodes.PCANode(output_dim=red_out_dim_L0)
#red_node_L0 = pSFALayerL0.red_node(output_dim=red_out_dim_L0)
##Create array of red_nodes (just one node, but cloned)
#red_layer_L0 = mdp.hinet.CloneLayer(red_node_L0, n_nodes=num_nodes_RED_L0)
#
##Create single SFA Node
##Warning Signal too short!!!!!sfa_out_dim_L0 = 20
##sfa_out_dim_L0 = 10
#sfa_node_L0 = pSFALayerL0.sfa_node(output_dim=sfa_out_dim_L0, block_size=block_size_L0)
##Create array of sfa_nodes (just one node, but cloned)
#num_nodes_SFA_L0 = lat_mat_L0.size / 2
#sfa_layer_L0 = mdp.hinet.CloneLayer(sfa_node_L0, n_nodes=num_nodes_SFA_L0)
#
#t3 = time.time()

#from GenerateSystemParameters import pSFALayerL1 as L1
#
##Create Switchboard L1
##Create L1 sfa node
##x_field_channels_L1 = pSFALayerL1.x_field_channels
##y_field_channels_L1 = pSFALayerL1.y_field_channels
##x_field_spacing_L1 = pSFALayerL1.x_field_spacing
##y_field_spacing_L1 = pSFALayerL1.y_field_spacing
##in_channel_dim_L1 = pSFALayerL1.in_channel_dim
##
##pca_out_dim_L1 = pSFALayerL1.pca_out_dim 
##exp_funcs_L1 = pSFALayerL1.exp_funcs 
##red_out_dim_L1 = pSFALayerL1.red_out_dim 
##sfa_out_dim_L1 = pSFALayerL1.sfa_out_dim
##cloneLayerL1 = pSFALayerL1.cloneLayer
#
#L1.v1 = [L1.x_field_spacing, 0]
#L1.v2 = [L1.x_field_spacing, L1.y_field_spacing]
#
#L1.preserve_mask = numpy.ones((L1.y_field_channels, L1.x_field_channels, L1.in_channel_dim)) > 0.5
#
#print "About to create (lattice based) intermediate layer widht=%d, height=%d"%(L1.x_field_channels, L1.y_field_channels) 
#print "With a spacing of horiz=%d, vert=%d, and %d channels"%(L1.x_field_spacing, L1.y_field_spacing, L1.in_channel_dim)
#
#print "Shape of lat_mat_L0 is:", lat_mat_L0
#L1.y_in_channels, L1.x_in_channels, tmp = lat_mat_L0.shape
#
##remember, here tmp is always two!!!
#
##switchboard_L1 = mdp.hinet.Rectangular2dSwitchboard(12, 6, x_field_channels_L1,y_field_channels_L1,x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)
#
##preserve_mask_L1_3D = wider(preserve_mask_L1, scale_x=in_channel_dim)
#(L1.mat_connections, L1.lat_mat) = compute_lattice_matrix_connections_with_input_dim(L1.v1, L1.v2, L1.preserve_mask, L1.x_in_channels, L1.y_in_channels, L1.in_channel_dim)
#print "matrix connections L1:"
#print L1.mat_connections
#L1.switchboard = PInvSwitchboard(L1.x_in_channels * L1.y_in_channels * L1.in_channel_dim, L1.mat_connections)
#
#L1.switchboard.connections
#
#t4 = time.time()
#
#L1.num_nodes = L1.lat_mat.size / 2
#
#
##Create L1 sfa node
##sfa_out_dim_L1 = 12
##pca_out_dim_L1 = 90
##pca_out_dim_L1 = sfa_out_dim_L0 x_field_channels_L1 * x_field_channels_L1 * 0.75 
#
##MEGAWARNING, "is" is a wrong condition!!!
#if L1.cloneLayer == True:
#    print "Layer L1 with ", L1.num_nodes, " cloned PCA nodes will be created"
#    print "Warning!!! layer L1 using cloned PCA instead of several independent copies!!!"
#    L1.pca_node = L1.pca_node_class(input_dim=L1.preserve_mask.size, output_dim=L1.pca_out_dim, **L1.pca_args)
#    #Create array of sfa_nodes (just one node, but cloned)
#    L1.pca_layer = mdp.hinet.CloneLayer(L1.pca_node, n_nodes=L1.num_nodes)
#else:
#    print "Layer L1 with ", L1.num_nodes, " independent PCA nodes will be created"
#    L1.PCA_nodes = range(L1.num_nodes_PCA)
#    for i in range(L1.num_nodes_PCA):
#        L1.PCA_nodes[i] = L1.pca_node_class(input_dim=L1.preserve_mask.size, output_dim=L1.pca_out_dim, **L1.pca_args)
#    L1.pca_layer_L1 = mdp.hinet.Layer(L1.PCA_nodes, homogeneous = True)
#
#L1.exp_node = GeneralExpansionNode(L1.exp_funcs, use_hint=True, max_steady_factor=0.05, \
#                 delta_factor=0.6, min_delta=0.0001)
##exp_out_dim_L1 = exp_node_L1.output_dim
#L1.exp_layer = mdp.hinet.CloneLayer(L1.exp_node, n_nodes=L1.num_nodes)
#
#if L1.cloneLayer == True: 
#    print "Warning!!! layer L1 using cloned RED instead of several independent copies!!!"
#    L1.red_node = L1.red_node_class(output_dim=L1.red_out_dim)   
#    L1.red_layer = mdp.hinet.CloneLayer(L1.red_node, n_nodes=L1.num_nodes)
#else:    
#    print "Layer L1 with ", L1.num_nodes, " independent RED nodes will be created"
#    L1.RED_nodes = range(L1.num_nodes)
#    for i in range(L1.num_nodes):
#        L1.RED_nodes[i] = L1.red_node_class(output_dim=L1.red_out_dim)
#    L1.red_layer = mdp.hinet.Layer(L1.RED_nodes, homogeneous = True)
#
#if L1.cloneLayer == True: 
#    print "Warning!!! layer L1 using cloned SFA instead of several independent copies!!!"
#    #sfa_node_L1 = mdp.nodes.SFANode(input_dim=switchboard_L1.out_channel_dim, output_dim=sfa_out_dim_L1)
#    L1.sfa_node = L1.sfa_node_class(output_dim=L1.sfa_out_dim, block_size=block_size)    
#    #!!!no ma, ya aniadele el atributo output_channels al PINVSwitchboard    
#    L1.sfa_layer = mdp.hinet.CloneLayer(L1.sfa_node, n_nodes=L1.num_nodes)
#else:    
#    print "Layer L1 with ", L1.num_nodes, " independent SFA nodes will be created"
#    L1.SFA_nodes = range(L1.num_nodes)
#    for i in range(L1.num_nodes):
#        L1.SFA_nodes[i] = L1.sfa_node_class(output_dim=L1.sfa_out_dim, block_size=block_size)
#    L1.sfa_layer = mdp.hinet.Layer(L1.SFA_nodes, homogeneous = True)

#t5 = time.time()
#
#print "LAYER L2"
##Create Switchboard L2
#x_field_channels_L2=3
#y_field_channels_L2=3
#x_field_spacing_L2=3
#y_field_spacing_L2=3
#in_channel_dim_L2=L1.sfa_out_dim
#
#v1_L2 = [x_field_spacing_L2, 0]
#v2_L2 = [x_field_spacing_L2, y_field_spacing_L2]
#
#preserve_mask_L2 = numpy.ones((y_field_channels_L2, x_field_channels_L2, in_channel_dim_L2)) > 0.5
#
#print "About to create (lattice based) third layer (L2) widht=%d, height=%d"%(x_field_channels_L2,y_field_channels_L2) 
#print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L2,y_field_spacing_L2,in_channel_dim_L2)
#
#print "Shape of lat_mat_L1 is:", L1.lat_mat
#y_in_channels_L2, x_in_channels_L2, tmp = L1.lat_mat.shape
#
##preserve_mask_L2_3D = wider(preserve_mask_L2, scale_x=in_channel_dim)
#(mat_connections_L2, lat_mat_L2) = compute_lattice_matrix_connections_with_input_dim(v1_L2, v2_L2, preserve_mask_L2, x_in_channels_L2, y_in_channels_L2, in_channel_dim_L2)
#print "matrix connections L2:"
#print mat_connections_L2
#switchboard_L2 = PInvSwitchboard(x_in_channels_L2 * y_in_channels_L2 * in_channel_dim_L2, mat_connections_L2)
#
#switchboard_L2.connections
#
#t6 = time.time()
#print "PinvSwitchboard L2 created in %0.3f ms"% ((t6-t5)*1000.0)
#num_nodes_PCA_L2 = num_nodes_EXP_L2 = num_nodes_RED_L2 = num_nodes_SFA_L2 = lat_mat_L2.size / 2
#
##Default: cloneLayerL2 = False
#cloneLayerL2 = False
#
##Create L2 sfa node
##sfa_out_dim_L2 = 12
##pca_out_dim_L2 = 120
#pca_out_dim_L2 = 100 
#exp_funcs_L2 = [identity,]
#red_out_dim_L2 = 100
#sfa_out_dim_L2 = 40
#
#if cloneLayerL2 == True:
#    print "Layer L2 with ", num_nodes_PCA_L2, " cloned PCA nodes will be created"
#    print "Warning!!! layer L2 using cloned PCA instead of several independent copies!!!"  
#    
#    pca_node_L2 = mdp.nodes.PCANode(input_dim=preserve_mask_L2.size, output_dim=pca_out_dim_L2)
#    #Create array of sfa_nodes (just one node, but cloned)
#    pca_layer_L2 = mdp.hinet.CloneLayer(pca_node_L2, n_nodes=num_nodes_PCA_L2)
#else:
#    print "Layer L2 with ", num_nodes_PCA_L2, " independent PCA nodes will be created"
#    PCA_nodes_L2 = range(num_nodes_PCA_L2)
#    for i in range(num_nodes_PCA_L2):
#        PCA_nodes_L2[i] = mdp.nodes.PCANode(input_dim=preserve_mask_L2.size, output_dim=pca_out_dim_L2)
#    pca_layer_L2 = mdp.hinet.Layer(PCA_nodes_L2, homogeneous = True)
#
#exp_node_L2 = GeneralExpansionNode(exp_funcs_L2, use_hint=True, max_steady_factor=0.05, \
#                 delta_factor=0.6, min_delta=0.0001)
#exp_out_dim_L2 = exp_node_L2.output_dim
#exp_layer_L2 = mdp.hinet.CloneLayer(exp_node_L2, n_nodes=num_nodes_EXP_L2)
#
#if cloneLayerL2 == True: 
#    print "Warning!!! layer L2 using cloned RED instead of several independent copies!!!"
#    red_node_L2 = mdp.nodes.WhiteningNode(output_dim=red_out_dim_L2)   
#    red_layer_L2 = mdp.hinet.CloneLayer(red_node_L2, n_nodes=num_nodes_RED_L2)
#else:    
#    print "Layer L2 with ", num_nodes_RED_L2, " independent RED nodes will be created"
#    RED_nodes_L2 = range(num_nodes_RED_L2)
#    for i in range(num_nodes_RED_L2):
#        RED_nodes_L2[i] = mdp.nodes.WhiteningNode(output_dim=red_out_dim_L2)
#    red_layer_L2 = mdp.hinet.Layer(RED_nodes_L2, homogeneous = True)
#
#if cloneLayerL2 == True:
#    print "Layer L2 with ", num_nodes_SFA_L2, " cloned SFA nodes will be created"
#    print "Warning!!! layer L2 using cloned SFA instead of several independent copies!!!"      
#    #sfa_node_L2 = mdp.nodes.SFANode(input_dim=switchboard_L2.out_channel_dim, output_dim=sfa_out_dim_L2)
#    sfa_node_L2 = mdp.nodes.SFANode(input_dim=red_out_dim_L2, output_dim=sfa_out_dim_L2, block_size=block_size_L2)
#    #!!!no ma, ya aniadele el atributo output_channels al PINVSwitchboard
#    sfa_layer_L2 = mdp.hinet.CloneLayer(sfa_node_L2, n_nodes=num_nodes_SFA_L2)
#else:
#    print "Layer L2 with ", num_nodes_SFA_L2, " independent PCA/SFA nodes will be created"
#
#    SFA_nodes_L2 = range(num_nodes_SFA_L2)
#    for i in range(num_nodes_SFA_L2):
#        SFA_nodes_L2[i] = mdp.nodes.SFANode(input_dim=red_out_dim_L2, output_dim=sfa_out_dim_L2, block_size=block_size_L2)
#    sfa_layer_L2 = mdp.hinet.Layer(SFA_nodes_L2, homogeneous = True)
#
#t7 = time.time()
#
##Create L3 sfa node
##sfa_out_dim_L3 = 150
##sfa_out_dim_L3 = 78
##WARNING!!! CHANGED PCA TO SFA
##pca_out_dim_L3 = 210
##pca_out_dim_L3 = 0.999
#pca_out_dim_L3 = 300
##exp_funcs_L3 = [identity, pair_prod_ex, pair_prod_adj1_ex, pair_prod_adj2_ex, pair_prod_adj3_ex]
#exp_funcs_L3 = [identity]
#red_out_dim_L3 = 0.999999
#sfa_out_dim_L3 = 40
#
#print "Creating final EXP/DimRed/SFA node L3"
#
##pca_node_L3 = mdp.nodes.WhiteningNode(output_dim=pca_out_dim_L3) 
#pca_node_L3 = mdp.nodes.SFANode(output_dim=pca_out_dim_L3, block_size=block_size_L3) 
#
#exp_node_L3 = GeneralExpansionNode(exp_funcs_L3, use_hint=True, max_steady_factor=0.35, \
#                 delta_factor=0.6, min_delta=0.0001)
##exp_out_dim_L3 = exp_node_L3.output_dim
##red_node_L3 = mdp.nodes.WhiteningNode(input_dim=exp_out_dim_L3, output_dim=red_out_dim_L3)   
#red_node_L3 = mdp.nodes.WhiteningNode(output_dim=red_out_dim_L3)   
#
##sfa_node_L3 = mdp.nodes.SFANode(input_dim=preserve_mask_L2.size, output_dim=sfa_out_dim_L3)
##sfa_node_L3 = mdp.nodes.SFANode()
#sfa_node_L3 = mdp.nodes.SFANode(output_dim=sfa_out_dim_L3, block_size=block_size_L3)
#
#t8 = time.time()

#Join Switchboard and SFA layer in a single flow
#flow = mdp.Flow([switchboard_L0, sfa_layer_L0, switchboard_L1, sfa_layer_L1, switchboard_L2, sfa_layer_L2, sfa_node_L3], verbose=True)
#flow = mdp.Flow([switchboard_L0, pca_layer_L0, exp_layer_L0, red_layer_L0, sfa_layer_L0, switchboard_L1, pca_layer_L1, exp_layer_L1, red_layer_L1, sfa_layer_L1, switchboard_L2, pca_layer_L2, exp_layer_L2, red_layer_L2, sfa_layer_L2, pca_node_L3, exp_node_L3, red_node_L3, sfa_node_L3], verbose=True)


#pFSeenid = ParamsInput()
#pFSeenid.name = "Gender60x200"
#pFSeenid.data_base_dir ="/local/tmp/escalafl/Alberto/RenderingsGender60x200"
#pFSeenid.ids = range(160,200)
#pFSeenid.ages = [999]
#pFSeenid.MIN_GENDER = -3
#pFSeenid.MAX_GENDER = 3
#pFSeenid.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
##pFSeenid.GENDER_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#pFSeenid.genders = map(code_gender, numpy.arange(pFSeenid.MIN_GENDER, pFSeenid.MAX_GENDER, pFSeenid.GENDER_STEP))
#pFSeenid.racetweens = [999]
#pFSeenid.expressions = [0]
#pFSeenid.morphs = [0]
#pFSeenid.poses = [0]
#pFSeenid.lightings = [0]
#pFSeenid.slow_signal = 2
#pFSeenid.step = 1
#pFSeenid.offset = 0                             

#im_seq_base_dir = pFSeenid.data_base_dir
#ids = pFSeenid.ids
#ages = pFSeenid.ages
#MIN_GENDER_SEENID = pFSeenid.MIN_GENDER
#MAX_GENDER_SEENID = pFSeenid.MAX_GENDER
#GENDER_STEP_SEENID = pFSeenid.GENDER_STEP
#genders = map(code_gender, numpy.arange(MIN_GENDER_SEENID, MAX_GENDER_SEENID,GENDER_STEP_SEENID))
#racetweens = pFSeenid.racetweens
#expressions = pFSeenid.expressions 
#morphs = pFSeenid.morphs 
#poses = pFSeenid.poses
#lightings = pFSeenid.lightings
#slow_signal = pFSeenid.slow_signal
#step = pFSeenid.step
#offset = pFSeenid.offset


#FOR LEARNING GENDER, INVARIANT TO IDENTITY
#im_seq_base_dir = "/local/tmp/escalafl/Alberto/RenderingsGender60x200"
#ids=range(160,200)
#ages=[999]
#GENDER_STEP_SEENID = 0.1
#genders = map(code_gender, numpy.arange(-3,3,GENDER_STEP_SEENID)) #4.005, 0.20025
#racetweens = [999]
#expressions=[0]
#morphs=[0]
#poses=[0]
#lightings=[0]
#slow_signal=2
#step=1
#offset=0

#params = [ids, expressions, morphs, poses, lightings]
#params2 = [ids, ages, genders, racetweens, expressions, morphs, poses, lightings]
#block_size_seenid= num_images_seenid / len(params2[slow_signal])

#pDataSeenid = ParamsDataLoading()
#pDataSeenid.input_files = image_files_seenid
#pDataSeenid.num_images = len(image_files_seenid)
#pDataSeenid.image_width = 256
#pDataSeenid.image_height = 192
#pDataSeenid.subimage_width = 135
#pDataSeenid.subimage_height = 135 
#pDataSeenid.pixelsampling_x = 1
#pDataSeenid.pixelsampling_y =  1
#pDataSeenid.subimage_pixelsampling = 2
#pDataSeenid.subimage_first_row =  pDataSeenid.image_height/2-pDataSeenid.subimage_height*pDataSeenid.pixelsampling_y/2
#pDataSeenid.subimage_first_column = pDataSeenid.image_width/2-pDataSeenid.subimage_width*pDataSeenid.pixelsampling_x/2+ 5*pDataSeenid.pixelsampling_x
#pDataSeenid.add_noise_L0 = True
#pDataSeenid.convert_format = "L"
#pDataSeenid.translation = 0
#pDataSeenid.translations_x = numpy.random.random_integers(-pDataSeenid.translation, pDataSeenid.translation, pDataSeenid.num_images)
#pDataSeenid.translations_y = numpy.random.random_integers(-pDataSeenid.translation, pDataSeenid.translation, pDataSeenid.num_images)
#pDataSeenid.trans_sampled = True

#image_width  = pDataSeenid.image_width   
#image_height = pDataSeenid.image_height 
#subimage_width  = pDataSeenid.subimage_width   
#subimage_height = pDataSeenid.subimage_height 
#pixelsampling_x = pDataSeenid.pixelsampling_x 
#pixelsampling_y = pDataSeenid.pixelsampling_y 
#subimage_pixelsampling=pDataSeenid.subimage_pixelsampling
#subimage_first_row= pDataSeenid.subimage_first_row
#subimage_first_column=pDataSeenid.subimage_first_column
#add_noise_L0 = pDataSeenid.add_noise_L0
#convert_format=pDataSeenid.convert_format
#translations_x=pDataSeenid.translations_x
#translations_y=pDataSeenid.translations_y
#trans_sampled=pDataSeenid.trans_sampled

##image_width  = 640
##image_height = 480
#image_width  = 256
#image_height = 192
#subimage_width  = 135
#subimage_height = 135 
#pixelsampling_x = 1
#pixelsampling_y = 1
#subimage_pixelsampling=1
##pixelsampling_x = 2
##pixelsampling_y = 2
##subimage_pixelsampling=2
#subimage_first_row= image_height/2-subimage_height*pixelsampling_y/2
#subimage_first_column=image_width/2-subimage_width*pixelsampling_x/2+ 5*pixelsampling_x
#add_noise_L0 = False
#convert_format="L"
#translations_x=numpy.random.random_integers(-translation, translation, num_images_seenid) 
#translations_y=numpy.random.random_integers(-translation, translation, num_images_seenid)
#trans_sampled=True


#image_files_seenid = create_image_filenames2(im_seq_base_dir, slow_signal, ids, ages, genders, racetweens, \
#                            expressions, morphs, poses, lightings, step, offset)
#
#num_images_seenid = len(image_files_seenid)
#
#subimages_seenid = load_image_data(image_files_seenid, image_width, image_height, subimage_width, subimage_height, \
#                    pixelsampling_x, pixelsampling_y, subimage_first_row, subimage_first_column, \
#                    add_noise_L0, convert_format, translations_x, translations_y, trans_sampled)


#pFNewid = ParamsInput()
#pFNewid.name = "Gender60x200"
#pFNewid.data_base_dir ="/local/tmp/escalafl/Alberto/Renderings20x500"
#pFNewid.ids = range(0,2)
#pFNewid.ages = [999]
#pFNewid.MIN_GENDER = -3
#pFNewid.MAX_GENDER = 3
#pFNewid.GENDER_STEP = 0.10000 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
##pFNewid.GENDER_STEP = 0.80075 #01. default. 0.4 fails, use 0.4005, 0.80075, 0.9005
#pFNewid.genders = map(code_gender, numpy.arange(pFNewid.MIN_GENDER, pFNewid.MAX_GENDER, pFNewid.GENDER_STEP))
#pFNewid.racetweens = [999]
#pFNewid.expressions = [0]
#pFNewid.morphs = [0]
#pFNewid.poses = range(0,500)
#pFNewid.lightings = [0]
#pFNewid.slow_signal = 0
#pFNewid.step = 4
#pFNewid.offset = 0                             
#
#im_seq_base_dir = pFNewid.data_base_dir
#ids = pFNewid.ids
#ages = pFNewid.ages
#MIN_GENDER_NEWID = pFNewid.MIN_GENDER
#MAX_GENDER_NEWID = pFNewid.MAX_GENDER
#GENDER_STEP_NEWID = pFNewid.GENDER_STEP
#genders = map(code_gender, numpy.arange(MIN_GENDER_NEWID , MAX_GENDER_NEWID ,GENDER_STEP_NEWID ))
#racetweens = pFNewid.racetweens
#expressions = pFNewid.expressions 
#morphs = pFNewid.morphs 
#poses = pFNewid.poses
#lightings = pFNewid.lightings
#slow_signal = pFNewid.slow_signal
#step = pFNewid.step
#offset = pFNewid.offset
#
###im_seq_base_dir = "/local/tmp/escalafl/Alberto/testing_newid"
##im_seq_base_dir = "/local/tmp/escalafl/Alberto/Renderings20x500"
##ids=range(0,2)
##expressions=[0]
##morphs=[0]
##poses=range(0,500)
##lightings=[0]
##slow_signal=0
##step=4
##offset=0
#
#image_files_newid = create_image_filenames(im_seq_base_dir, slow_signal, ids, expressions, morphs, poses, lightings, step, offset)
#num_images_newid = len(image_files_newid)
#params = [ids, expressions, morphs, poses, lightings]
#block_size_newid= num_images_newid / len(params[slow_signal])
#
#
#
#
##image_width  = 640
##image_height = 480
##subimage_width  = 135
##subimage_height = 135 
##pixelsampling_x = 2
##pixelsampling_y = 2
##subimage_pixelsampling=2
##subimage_first_row= image_height/2-subimage_height*pixelsampling_y/2
##subimage_first_column=image_width/2-subimage_width*pixelsampling_x/2+ 5*pixelsampling_x
##add_noise_L0 = False
##convert_format="L"
##translations_x=None
##translations_y=None
##trans_sampled=True
#
#
#pDataNewid = ParamsDataLoading()
#pDataNewid.input_files = image_files_seenid
#pDataNewid.num_images = len(image_files_seenid)
#pDataNewid.image_width = 640
#pDataNewid.image_height = 480
#pDataNewid.subimage_width = 135
#pDataNewid.subimage_height = 135 
#pDataNewid.pixelsampling_x = 2
#pDataNewid.pixelsampling_y =  2
#pDataNewid.subimage_pixelsampling = 2
#pDataNewid.subimage_first_row =  pDataNewid.image_height/2-pDataNewid.subimage_height*pDataNewid.pixelsampling_y/2
#pDataNewid.subimage_first_column = pDataNewid.image_width/2-pDataNewid.subimage_width*pDataNewid.pixelsampling_x/2+ 5*pDataNewid.pixelsampling_x
#pDataNewid.add_noise_L0 = False
#pDataNewid.convert_format = "L"
#pDataNewid.translation = 0
#pDataNewid.translations_x = numpy.random.random_integers(-pDataNewid.translation, pDataNewid.translation, pDataNewid.num_images)
#pDataNewid.translations_y = numpy.random.random_integers(-pDataNewid.translation, pDataNewid.translation, pDataNewid.num_images)
#pDataNewid.trans_sampled = True
#
#image_width  = pDataNewid.image_width   
#image_height = pDataNewid.image_height 
#subimage_width  = pDataNewid.subimage_width   
#subimage_height = pDataNewid.subimage_height 
#pixelsampling_x = pDataNewid.pixelsampling_x 
#pixelsampling_y = pDataNewid.pixelsampling_y 
#subimage_pixelsampling=pDataNewid.subimage_pixelsampling
#subimage_first_row= pDataNewid.subimage_first_row
#subimage_first_column=pDataNewid.subimage_first_column
#add_noise_L0 = pDataNewid.add_noise_L0
#convert_format=pDataNewid.convert_format
#translations_x=pDataNewid.translations_x
#translations_y=pDataNewid.translations_y
#trans_sampled=pDataNewid.trans_sampled
#
#subimages_newid = load_image_data(image_files_newid, image_width, image_height, subimage_width, subimage_height, \
#                    pixelsampling_x, pixelsampling_y, subimage_first_row, subimage_first_column, \
#                    add_noise_L0, convert_format, translations_x, translations_y, trans_sampled)

##Testing hash function
##print "Testing for bug in hashing of RandomPermutationNode..."
##xx = numpy.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
##node1 = more_nodes.RandomPermutationNode()
##node1.train(xx)
##node2 = more_nodes.RandomPermutationNode()
##node2.train(xx)
##
##hash1 = cache.hash_object(node1, m=None, recursion=True, verbose=False)
##hash2 = cache.hash_object(node2, m=None, recursion=True, verbose=False)
##
##print "hash1=", hash1.hexdigest()
##print "hash2=", hash2.hexdigest()
##print "hash values should usually differ since there are two trainings"
##print "*********************************************************************"
##
##print "Testing for bug in hashing of RandomPermutationNode..."
##xx = numpy.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
##node1 = more_nodes.GeneralExpansionNode([identity])
##node2 = more_nodes.GeneralExpansionNode([pair_prod_adj2_ex])
##
##hash1 = cache.hash_object(node1, m=None, recursion=True, verbose=False)
##hash2 = cache.hash_object(node2, m=None, recursion=True, verbose=False)
##
##print "hash1=", hash1.hexdigest()
##print "hash2=", hash2.hexdigest()
##print "hash values should be equal if expansion functions are the same"
##print "*********************************************************************"
##
##quit()