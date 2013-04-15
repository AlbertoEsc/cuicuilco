#! /usr/bin/env python

#Some tests on the GaussianClassifier, and on the LinearRegression Nodes
#Goal: Test the reliability of these nodes fpr postprocessing after the hiererchical sfa network
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 27 Jul 2009
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import numpy
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import Image
import mdp
import os
import glob
import random
from  eban_SFA_libs import *
from imageLoader import *
import time


gaussNode = mdp.nodes.GaussianClassifierNode()

num_classes = 8
num_samples = 5000
num_vars = 16

med = numpy.zeros((num_classes, num_vars))
std = numpy.zeros((num_classes, num_vars))

t0 = time.time()
for cl in range(num_classes):
    med[cl] =  numpy.random.normal(loc=0.0, scale = 1.0, size=num_vars)
    std[cl] =  numpy.fabs(numpy.random.normal(loc=1.0, scale = 0.25, size=num_vars))
    x = numpy.zeros((num_vars,num_samples))
    for i in range(num_vars):
        x[i] = numpy.random.normal(loc=med[cl][i], scale = std[cl][i], size=(num_samples))
    x = x.T
    #print "x: has media %f and std %f", med, std
    gaussNode.train(x, cl)
gaussNode.stop_training()
t1 = time.time()
print 'Training %d classes took %0.3f ms' % (num_classes, (t1-t0)*1000.0)

num_samples_test = 10
total_correct = 0
for cl in range(num_classes):
    x = numpy.zeros((num_vars,num_samples_test))
    for i in range(num_vars):
        x[i] = numpy.random.normal(loc=med[cl][i], scale = std[cl][i], size=(num_samples_test))
    x = x.T

    cl_est = gaussNode.classify(x)
    print "cl_est=", cl_est
    eq = ((numpy.array(cl_est) - cl) ==0).sum()
    print eq
    total_correct = total_correct + eq
    prob_est = gaussNode.class_probabilities(x)
t2 = time.time()
print "prob_est=", prob_est

print "overal %f correct classification"%(total_correct * 1.0 / (num_samples_test * num_classes))
quit()

#only one identity, and just changing the angle
t0 = time.time()
im_seq_base_dir = "/local/tmp/escalafl/Alberto/training"
ids=range(0,1)
expressions=[0]
morphs=[0]
poses=range(0,500)
lightings=[0]
slow_signal=0
step=4
offset=0

image_files_training = create_image_filenames(im_seq_base_dir, slow_signal, ids, expressions, morphs, poses, lightings, step, offset)
num_images_training = num_images = len(image_files_training)

params = [ids, expressions, morphs, poses, lightings]
block_size= num_images / len(params[slow_signal])
  
block_size_L0=block_size
block_size_L1=block_size
block_size_L2=block_size
block_size_L3=block_size
block_size_exec=block_size #(Used only for random walk)

scale_disp = 3
image_width  = 640
image_height = 480
subimage_width  = 135
subimage_height = 135 
pixelsampling_x = 2
pixelsampling_y = 2
subimage_pixelsampling=2
subimage_first_row= image_height/2-subimage_height*pixelsampling_y/2
subimage_first_column=image_width/2-subimage_width*pixelsampling_x/2+ 5*pixelsampling_x
add_noise_L0 = True
convert_format="L"
#translations_x=None
#translations_y=None
translations_x=numpy.random.random_integers(-5, 5, num_images) 
translations_y=numpy.random.random_integers(-5, 5, num_images)
trans_sampled=True

subimages = load_image_data(image_files_training, image_width, image_height, subimage_width, subimage_height, \
                    pixelsampling_x, pixelsampling_y, subimage_first_row, subimage_first_column, \
                    add_noise_L0, convert_format, translations_x, translations_y, trans_sampled)
  
t1 = time.time()
print num_images, " Images loaded in %0.3f ms"% ((t1-t0)*1000.0)


print "******************************************"
print "Creating Expanded 4L SFA hierarchy"
print "******************************************"

t0 = time.time()
print "Creating layer L0"
x_field_channels_L0=5
y_field_channels_L0=5
x_field_spacing_L0=5
y_field_spacing_L0=5
in_channel_dim_L0=1

v1_L0 = (x_field_spacing_L0, 0)
v2_L0 = (x_field_spacing_L0, y_field_spacing_L0)

preserve_mask_L0 = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
# 6 x 12
print "About to create (lattice based) perceptive field of widht=%d, height=%d"%(x_field_channels_L0,y_field_channels_L0) 
print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L0, y_field_spacing_L0, in_channel_dim_L0)

(mat_connections_L0, lat_mat_L0) = compute_lattice_matrix_connections(v1_L0, v2_L0, preserve_mask_L0, subimage_width, subimage_height, in_channel_dim_L0)
print "matrix connections L0:"
print mat_connections_L0

t1 = time.time()

switchboard_L0 = PInvSwitchboard(subimage_width * subimage_height, mat_connections_L0)
switchboard_L0.connections

t2 = time.time()
print "PInvSwitchboard L0 created in %0.3f ms"% ((t2-t1)*1000.0)

#Create single PCA Node
#pca_out_dim_L0 = 20
pca_out_dim_L0 = 5
pca_node_L0 = mdp.nodes.PCANode(input_dim=preserve_mask_L0.size, output_dim=pca_out_dim_L0)

#Create array of pca_nodes (just one node, but cloned)
num_nodes_RED_L0 = num_nodes_EXP_L0 = num_nodes_SFA_L0 = num_nodes_PCA_L0 = lat_mat_L0.size / 2
pca_layer_L0 = mdp.hinet.CloneLayer(pca_node_L0, n_nodes=num_nodes_PCA_L0)

#exp_funcs_L0 = [identity, pair_prod_ex, pair_sqrt_abs_dif_ex, pair_sqrt_abs_sum_ex]
exp_funcs_L0 = [identity, pair_prod_ex]
exp_node_L0 = GeneralExpansionNode(exp_funcs_L0, input_dim = pca_out_dim_L0, use_hint=True, max_steady_factor=0.15, \
                 delta_factor=0.6, min_delta=0.0001)
exp_out_dim_L0 = exp_node_L0.output_dim
exp_layer_L0 = mdp.hinet.CloneLayer(exp_node_L0, n_nodes=num_nodes_EXP_L0)

#Create Node for dimensionality reduction
#red_out_dim_L0 = 20
red_out_dim_L0 = 20
red_node_L0 = mdp.nodes.PCANode(input_dim=exp_out_dim_L0, output_dim=red_out_dim_L0)

#Create array of red_nodes (just one node, but cloned)
red_layer_L0 = mdp.hinet.CloneLayer(red_node_L0, n_nodes=num_nodes_RED_L0)

#Create single SFA Node
#Warning Signal too short!!!!!sfa_out_dim_L0 = 20
#sfa_out_dim_L0 = 10
sfa_out_dim_L0 = 15
sfa_node_L0 = mdp.nodes.SFANode(input_dim=exp_out_dim_L0, output_dim=sfa_out_dim_L0, block_size=block_size_L0)

#Create array of sfa_nodes (just one node, but cloned)
num_nodes_SFA_L0 = lat_mat_L0.size / 2
sfa_layer_L0 = mdp.hinet.CloneLayer(sfa_node_L0, n_nodes=num_nodes_SFA_L0)

t3 = time.time()

#Create Switchboard L1
x_field_channels_L1=3
y_field_channels_L1=3
x_field_spacing_L1=3
y_field_spacing_L1=3
in_channel_dim_L1=sfa_out_dim_L0

v1_L1 = [x_field_spacing_L1, 0]
v2_L1 = [x_field_spacing_L1, y_field_spacing_L1]

preserve_mask_L1 = numpy.ones((y_field_channels_L1, x_field_channels_L1, in_channel_dim_L1)) > 0.5

print "About to create (lattice based) intermediate layer widht=%d, height=%d"%(x_field_channels_L1,y_field_channels_L1) 
print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)

print "Shape of lat_mat_L0 is:", lat_mat_L0
y_in_channels_L1, x_in_channels_L1, tmp = lat_mat_L0.shape
#remember, here tmp is always two!!!

#switchboard_L1 = mdp.hinet.Rectangular2dSwitchboard(12, 6, x_field_channels_L1,y_field_channels_L1,x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)

#preserve_mask_L1_3D = wider(preserve_mask_L1, scale_x=in_channel_dim)
(mat_connections_L1, lat_mat_L1) = compute_lattice_matrix_connections_with_input_dim(v1_L1, v2_L1, preserve_mask_L1, x_in_channels_L1, y_in_channels_L1, in_channel_dim_L1)
print "matrix connections L1:"
print mat_connections_L1
switchboard_L1 = PInvSwitchboard(x_in_channels_L1 * y_in_channels_L1 * in_channel_dim_L1, mat_connections_L1)

switchboard_L1.connections

t4 = time.time()

num_nodes_EXP_L1 = num_nodes_PCA_L1 =num_nodes_SFA_L1 = lat_mat_L1.size / 2

#Default: cloneLayerL1 = False
cloneLayerL1 = False

#Create L1 sfa node
#sfa_out_dim_L1 = 12
#pca_out_dim_L1 = 90
pca_out_dim_L1 = 90
exp_funcs_L1 = [identity,]
red_out_dim_L1 = 30

sfa_out_dim_L1 = 20

#WARNING, wrong condition!!!
if cloneLayerL1 is True:
    print "Layer L1 with ", num_nodes_PCA_L1, " cloned PCA nodes will be created"
    print "Warning!!! layer L1 using cloned PCA instead of several independent copies!!!"
    pca_node_L1 = mdp.nodes.PCANode(input_dim=preserve_mask_L1.size, output_dim=pca_out_dim_L1)
    #Create array of sfa_nodes (just one node, but cloned)
    pca_layer_L1 = mdp.hinet.CloneLayer(pca_node_L1, n_nodes=num_nodes_PCA_L1)
else:
    print "Layer L1 with ", num_nodes_PCA_L1, " independent PCA nodes will be created"
    PCA_nodes_L1 = range(num_nodes_PCA_L1)
    for i in range(num_nodes_PCA_L1):
        PCA_nodes_L1[i] = mdp.nodes.PCANode(input_dim=preserve_mask_L1.size, output_dim=pca_out_dim_L1)
    pca_layer_L1 = mdp.hinet.Layer(PCA_nodes_L1)

exp_node_L1 = GeneralExpansionNode(exp_funcs_L1, input_dim = pca_out_dim_L1, use_hint=True, max_steady_factor=0.05, \
                 delta_factor=0.6, min_delta=0.0001)
exp_out_dim_L1 = exp_node_L1.output_dim
exp_layer_L1 = mdp.hinet.CloneLayer(exp_node_L1, n_nodes=num_nodes_EXP_L1)

if cloneLayerL1 is True: 
    red_out_dim_L0 = 20
    red_node_L0 = mdp.nodes.PCANode(input_dim=exp_out_dim_L0, output_dim=red_out_dim_L0)

    print "Warning!!! layer L1 using cloned SFA instead of several independent copies!!!"
    #sfa_node_L1 = mdp.nodes.SFANode(input_dim=switchboard_L1.out_channel_dim, output_dim=sfa_out_dim_L1)
    sfa_node_L1 = mdp.nodes.SFANode(input_dim=exp_out_dim_L1, output_dim=sfa_out_dim_L1, block_size=block_size_L1)    
    #!!!no ma, ya aniadele el atributo output_channels al PINVSwitchboard    
    sfa_layer_L1 = mdp.hinet.CloneLayer(sfa_node_L1, n_nodes=num_nodes_SFA_L1)
else:    
    print "Layer L1 with ", num_nodes_SFA_L1, " independent PCA nodes will be created"
    SFA_nodes_L1 = range(num_nodes_SFA_L1)
    for i in range(num_nodes_SFA_L1):
        SFA_nodes_L1[i] = mdp.nodes.SFANode(input_dim=exp_out_dim_L1, output_dim=sfa_out_dim_L1, block_size=block_size_L1)
    sfa_layer_L1 = mdp.hinet.Layer(SFA_nodes_L1)

t5 = time.time()

print "LAYER L2"
#Create Switchboard L2
x_field_channels_L2=3
y_field_channels_L2=3
x_field_spacing_L2=3
y_field_spacing_L2=3
in_channel_dim_L2=sfa_out_dim_L1

v1_L2 = [x_field_spacing_L2, 0]
v2_L2 = [x_field_spacing_L2, y_field_spacing_L2]

preserve_mask_L2 = numpy.ones((y_field_channels_L2, x_field_channels_L2, in_channel_dim_L2)) > 0.5

print "About to create (lattice based) third layer (L2) widht=%d, height=%d"%(x_field_channels_L2,y_field_channels_L2) 
print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L2,y_field_spacing_L2,in_channel_dim_L2)

print "Shape of lat_mat_L1 is:", lat_mat_L1
y_in_channels_L2, x_in_channels_L2, tmp = lat_mat_L1.shape

#preserve_mask_L2_3D = wider(preserve_mask_L2, scale_x=in_channel_dim)
(mat_connections_L2, lat_mat_L2) = compute_lattice_matrix_connections_with_input_dim(v1_L2, v2_L2, preserve_mask_L2, x_in_channels_L2, y_in_channels_L2, in_channel_dim_L2)
print "matrix connections L2:"
print mat_connections_L2
switchboard_L2 = PInvSwitchboard(x_in_channels_L2 * y_in_channels_L2 * in_channel_dim_L2, mat_connections_L2)

switchboard_L2.connections

t6 = time.time()
print "PinvSwitchboard L2 created in %0.3f ms"% ((t6-t5)*1000.0)
num_nodes_EXP_L2 = num_nodes_SFA_L2 = num_nodes_PCA_L2 = lat_mat_L2.size / 2

#Default: cloneLayerL2 = False
cloneLayerL2 = False

#Create L2 sfa node
#sfa_out_dim_L2 = 12
#pca_out_dim_L2 = 120
pca_out_dim_L2 = 120
exp_funcs_L2 = [identity,]
sfa_out_dim_L2 = 20

if cloneLayerL2 is True:
    print "Layer L2 with ", num_nodes_PCA_L2, " cloned PCA nodes will be created"
    print "Warning!!! layer L2 using cloned PCA instead of several independent copies!!!"  
    
    pca_node_L2 = mdp.nodes.PCANode(input_dim=preserve_mask_L2.size, output_dim=pca_out_dim_L2)
    #Create array of sfa_nodes (just one node, but cloned)
    pca_layer_L2 = mdp.hinet.CloneLayer(pca_node_L2, n_nodes=num_nodes_PCA_L2)
else:
    print "Layer L2 with ", num_nodes_PCA_L2, " independent PCA nodes will be created"
    PCA_nodes_L2 = range(num_nodes_PCA_L2)
    for i in range(num_nodes_PCA_L2):
        PCA_nodes_L2[i] = mdp.nodes.PCANode(input_dim=preserve_mask_L2.size, output_dim=pca_out_dim_L2)
    pca_layer_L2 = mdp.hinet.Layer(PCA_nodes_L2)

exp_node_L2 = GeneralExpansionNode(exp_funcs_L2, input_dim = pca_out_dim_L2, use_hint=True, max_steady_factor=0.05, \
                 delta_factor=0.6, min_delta=0.0001)
exp_out_dim_L2 = exp_node_L2.output_dim
exp_layer_L2 = mdp.hinet.CloneLayer(exp_node_L2, n_nodes=num_nodes_EXP_L2)


if cloneLayerL2 is True:
    print "Layer L2 with ", num_nodes_SFA_L2, " cloned SFA nodes will be created"
    print "Warning!!! layer L2 using cloned SFA instead of several independent copies!!!"      
    #sfa_node_L2 = mdp.nodes.SFANode(input_dim=switchboard_L2.out_channel_dim, output_dim=sfa_out_dim_L2)
    sfa_node_L2 = mdp.nodes.SFANode(input_dim=exp_out_dim_L2, output_dim=sfa_out_dim_L2, block_size=block_size_L2)
    #!!!no ma, ya aniadele el atributo output_channels al PINVSwitchboard
    sfa_layer_L2 = mdp.hinet.CloneLayer(sfa_node_L2, n_nodes=num_nodes_SFA_L2)
else:
    print "Layer L2 with ", num_nodes_SFA_L2, " independent PCA/SFA nodes will be created"

    SFA_nodes_L2 = range(num_nodes_SFA_L2)
    for i in range(num_nodes_SFA_L2):
        SFA_nodes_L2[i] = mdp.nodes.SFANode(input_dim=exp_out_dim_L2, output_dim=sfa_out_dim_L2, block_size=block_size_L2)
    sfa_layer_L2 = mdp.hinet.Layer(SFA_nodes_L2)

t7 = time.time()

#Create L3 sfa node
#sfa_out_dim_L3 = 150
#sfa_out_dim_L3 = 78
sfa_out_dim_L3 = 20

print "Creating final SFA node L3"
#sfa_node_L3 = mdp.nodes.SFANode(input_dim=preserve_mask_L2.size, output_dim=sfa_out_dim_L3)
#sfa_node_L3 = mdp.nodes.SFANode()
sfa_node_L3 = mdp.nodes.SFANode(output_dim=sfa_out_dim_L3, block_size=block_size_L3)

t8 = time.time()

#Join Switchboard and SFA layer in a single flow
#flow = mdp.Flow([switchboard_L0, sfa_layer_L0, switchboard_L1, sfa_layer_L1, switchboard_L2, sfa_layer_L2, sfa_node_L3], verbose=True)
flow = mdp.Flow([switchboard_L0, pca_layer_L0, exp_layer_L0, sfa_layer_L0, switchboard_L1, pca_layer_L1, exp_layer_L1, sfa_layer_L1, switchboard_L2, pca_layer_L2, exp_layer_L2, sfa_layer_L2, sfa_node_L3], verbose=True)
#flow = mdp.Flow([switchboard_L0, sfa_layer_L0, switchboard_L1, sfa_layer_L1])
t9 = time.time()
print "sfa_node_L3 output_dim = ", sfa_node_L3.output_dim

print "Finished hierarchy construction, with total time %0.3f ms"% ((t9-t0)*1000.0) 
print "PinvSwitchboard L1 created in %0.3f ms"% ((t4-t3)*1000.0)
print "SFA Layer L1 created in %0.3f ms"% ((t5-t4)*1000.0)
print "SFA Layer L2 created in %0.3f ms"% ((t7-t6)*1000.0)
print "SFA Node L3 created in %0.3f ms"% ((t8-t7)*1000.0)

print "*****************************"
print "Training hierarchy ..."
print "*****************************"

subimages_p = subimages
optimized_training = False
special_training = True
if optimized_training is True:
    print "Training network... layer by layer"
    #WARNING!
    #subimages_p = subimages.copy()
    ttrain0 = time.time()
    
    data00 = switchboard_L0(subimages_p)
    
    pca_layer_L0.train(data00)
    pca_layer_L0.stop_training()
    data01 = pca_layer_L0.execute(data00)
    
    sfa_layer_L0.train(data01)
    sfa_layer_L0.stop_training()
    data02 = sfa_layer_L0.execute(data01)
    
    data03 = switchboard_L1(data02)
    
    pca_layer_L1.train(data03)
    pca_layer_L1.stop_training()
    data04 = pca_layer_L1.execute(data03)
    
    sfa_layer_L1.train(data04)
    sfa_layer_L1.stop_training()
    data05 = sfa_layer_L1.execute(data04)
    
    data06 = switchboard_L2(data05)
    
    pca_layer_L2.train(data06)
    pca_layer_L2.stop_training()
    data07 = pca_layer_L2.execute(data06)
    
    sfa_layer_L2.train(data07)
    sfa_layer_L2.stop_training()
    data08 = sfa_layer_L2.execute(data07)
    
    sfa_node_L3.train(data08)
    sfa_node_L3.stop_training()
    sl_seq = sl_seq_training = data09 = sfa_node_L3.execute(data08)

    ttrain1 = time.time()
    print "Network trained (artesanal way) in time %0.3f ms"% ((ttrain1-ttrain0)*1000.0)
elif special_training is True:
    ttrain0 = time.time()
    sl_seq = sl_seq_training = flow.special_train(subimages_p)
    ttrain1 = time.time()
    print "Network trained (specialized way) in time %0.3f ms"% ((ttrain1-ttrain0)*1000.0)
else:
    ttrain0 = time.time()
    flow.train(subimages_p)
    y = flow.execute(subimages_p[0:1]) #stop training
    sl_seq = sl_seq_training = flow.execute(subimages_p)
    ttrain1 = time.time()
    print "Network trained (MDP way) in time %0.3f ms"% ((ttrain1-ttrain0)*1000.0)

y = flow.execute(subimages_p[0:1])
hierarchy_out_dim = y.shape[1]
    
t8 = time.time()

print "Executing/Executed over training set..."
print "Input Signal: Training Data"
subimages_training = subimages
num_images_training = num_images

print "Loading test images, known ids..."
#im_seq_base_dir = "/local/tmp/escalafl/Alberto/testing_seenid"
im_seq_base_dir = "/local/tmp/escalafl/Alberto/training"
ids=range(0,2)
expressions=[0]
morphs=[0]
poses=range(0,500)
lightings=[0]
#slow_signal=0
step=4
offset=1

image_files_seenid = create_image_filenames(im_seq_base_dir, slow_signal, ids, expressions, morphs, poses, lightings, step, offset)
num_images_seenid = len(image_files_seenid)

params = [ids, expressions, morphs, poses, lightings]
block_size= num_images_seenid / len(params[slow_signal])

block_size_L0=block_size
block_size_L1=block_size
block_size_L2=block_size
block_size_L3=block_size
block_size_exec=block_size #(Used only for random walk)

scale_disp = 3
image_width  = 640
image_height = 480
subimage_width  = 135
subimage_height = 135 
pixelsampling_x = 2
pixelsampling_y = 2
subimage_pixelsampling=2
subimage_first_row= image_height/2-subimage_height*pixelsampling_y/2
subimage_first_column=image_width/2-subimage_width*pixelsampling_x/2+ 5*pixelsampling_x
add_noise_L0 = False
convert_format="L"
translations_x=numpy.random.random_integers(-5, 5, num_images_seenid) 
translations_y=numpy.random.random_integers(-5, 5, num_images_seenid)
trans_sampled=True


subimages_seenid = load_image_data(image_files_seenid, image_width, image_height, subimage_width, subimage_height, \
                    pixelsampling_x, pixelsampling_y, subimage_first_row, subimage_first_column, \
                    add_noise_L0, convert_format, translations_x, translations_y, trans_sampled)

t9 = time.time()
print num_images_seenid, " Images loaded in %0.3f ms"% ((t9-t8)*1000.0)

print "Execution over known id testing set..."
print "Input Signal: Known Id test images"
sl_seq_seenid = flow.execute(subimages_seenid)

t10 = time.time()
print "Loading test images, unknown ids..."
im_seq_base_dir = "/local/tmp/escalafl/Alberto/testing_newid"
ids=range(16,18)
expressions=[0]
morphs=[0]
poses=range(0,500)
lightings=[0]
slow_signal=0
step=4
offset=0

image_files_newid = create_image_filenames(im_seq_base_dir, slow_signal, ids, expressions, morphs, poses, lightings, step, offset)
num_images_newid = len(image_files_newid)
params = [ids, expressions, morphs, poses, lightings]
block_size= num_images_newid / len(params[slow_signal])
block_size_L0=block_size
block_size_L1=block_size
block_size_L2=block_size
block_size_L3=block_size
block_size_exec=block_size #(Used only for random walk)

image_width  = 640
image_height = 480
subimage_width  = 135
subimage_height = 135 
pixelsampling_x = 2
pixelsampling_y = 2
subimage_pixelsampling=2
subimage_first_row= image_height/2-subimage_height*pixelsampling_y/2
subimage_first_column=image_width/2-subimage_width*pixelsampling_x/2+ 5*pixelsampling_x
add_noise_L0 = False
convert_format="L"
translations_x=None
translations_y=None
trans_sampled=True

subimages_newid = load_image_data(image_files_newid, image_width, image_height, subimage_width, subimage_height, \
                    pixelsampling_x, pixelsampling_y, subimage_first_row, subimage_first_column, \
                    add_noise_L0, convert_format, translations_x, translations_y, trans_sampled)

t11 = time.time()
print num_images_newid, " Images loaded in %0.3f ms"% ((t11-t10)*1000.0)
print "Execution over unknown id testing set..."
print "Input Signal: Unknown Id test images"
sl_seq_newid = flow.execute(subimages_newid)


#print "Generating (new random) input sequence..."
#use_average = False
#use_first_id = False
#use_random_walk = False
#use_training_data = True
#use_new_identity = False
#new_id = 5
#if block_size_exec is not None and block_size_exec > 1:
#    print "block_size_exec > 1"
#    num_images2 = num_images / block_size_exec
#    subimages2 = numpy.zeros((num_images2, subimage_width * subimage_height))
#    if use_first_id is True:
#        print "Input Signal: First ID / First Pose of each ID"
#        for i in range(num_images2):
#            subimages2[i] = subimages[block_size_exec * i]       
#    elif use_average is True:
#        print "Input Signal: Average of IDs"
#        for i in range(num_images2):
#            subimages2[i] = subimages[block_size_exec * i: block_size_exec * (i+1)].sum(axis=0) / block_size_exec
#    elif use_random_walk is True:
#        print "Input Signal: Random Walk"
#        for i in range(num_images2):
#            id = numpy.random.randint(block_size_exec)
##            subimages[block_size * i + id]
##            subimages2[i] = subimages[0]
#            subimages2[i] = subimages[block_size_exec * i + id]
#    elif use_training_data is True:
#        print "Input Signal: Training Data"
#        subimages2 = subimages
#        num_images2 = num_images
#    elif use_new_identity is True:
#        print "Input Signal: New ID random%03d*.tif"%(new_id)
#        test_image_files1 = glob.glob(im_seq_base_dir + "/random%03d*.tif"%(new_id))
#        test_image_files1.sort()
#
#        test_image_files = []
#        for i in range(len(test_image_files1)):
#            test_image_files.append(test_image_files1[i])
#
#        num_images2 = num_test_images = len(test_image_files)
#        
#        subimages2 = numpy.zeros((num_test_images, subimage_width * subimage_height))
#        act_im_num = 0
#        for image_file in test_image_files:
#            im = Image.open(image_file)
#            im = im.convert("L")
#            im_arr = numpy.asarray(im)
#            im_small = im_arr[subimage_first_row:(subimage_first_row+subimage_height*subimage_pixelsampling):subimage_pixelsampling,
#                              subimage_first_column:(subimage_first_column+subimage_width*subimage_pixelsampling):subimage_pixelsampling].astype(float)
#            subimages2[act_im_num] = im_small.flatten()
#            act_im_num = act_im_num+1
#            del im_small
#            del im_arr
#            del im
#    else:
#        print "******************* No input sequence specified !!!!!!!!!!!!!!!!!"
##
#    subimages = subimages2
#    num_images = num_images2
#
##flow.
##print "Training finished, ed in %0.3f ms"% ((t2-t1)*1000.0)
#sl_seq = flow.execute(subimages)

t9 = time.time()
print "Execution over complete training/testing sets took %0.3f ms"% ((t9-t8)*1000.0)

#inverted_im = flow.inverse(sl_seq)

print "Computations Finished!"

print "Hierarchy description:"
print "Layer L0:" 
print "widht=%d, height=%d channels, and %d channel_dim"%(x_field_channels_L0,y_field_channels_L0, in_channel_dim_L0) 
print "Lattice spacing: horiz=%d, vert=%d, and %d sfa_out_dim"%(x_field_spacing_L0,y_field_spacing_L0, sfa_out_dim_L0)
print "Shape of lat_mat_L0 is:", lat_mat_L0.shape
#print "lat_mat_L0 is:", lat_mat_L0

print "Layer L1:" 
print "widht=%d, height=%d channels, and %d channel_dim"%(x_field_channels_L1,y_field_channels_L1, in_channel_dim_L1) 
print "Lattice spacing: horiz=%d, vert=%d, and %d sfa_out_dim"%(x_field_spacing_L1,y_field_spacing_L1, sfa_out_dim_L1)
print "Shape of lat_mat_L1 is:", lat_mat_L1.shape
#print "lat_mat_L1 is:", lat_mat_L1
print "matrix connections L1:"
print mat_connections_L1

print "Layer L2:" 
print "widht=%d, height=%d channels, and %d channel_dim"%(x_field_channels_L2,y_field_channels_L2, in_channel_dim_L2) 
print "Lattice spacing: horiz=%d, vert=%d, and %d sfa_out_dim"%(x_field_spacing_L2,y_field_spacing_L2, sfa_out_dim_L2)
print "Shape of lat_mat_L2 is:", lat_mat_L2.shape
#print "lat_mat_L2 is:", lat_mat_L2
print "matrix connections L2:"
print mat_connections_L2


print "Layer L3:" 
print "output_dim = ", sfa_node_L3.output_dim

print "Timing Information in ms: t1-t0=%0.3f, t2-t1=%0.3f,t3-t2=%0.3f, t4-t3=%0.3f, t5-t4=%0.3f, \n t6-t5=%0.3f, t7-t6=%0.3f, t8-t7=%0.3f, t9-t8=%0.3f"%((t1-t0)/1000,(t2-t1)/1000, (t3-t2)/1000, (t4-t3)/1000, (t5-t4)/1000, (t6-t5)/1000, (t7-t6)/1000, (t8-t7)/1000, (t9-t8)/1000) 


print "Creating GUI..."
print "SFA Outputs..."
#Create Figure
f0 = plt.figure()
plt.suptitle("Learning Person Identity")
  
#display SFA of Training Set
p11 = plt.subplot(1,3,1)
plt.title("Output Unit L3. (Training Set)")
sl_seqdisp = sl_seq_training[:, range(0,hierarchy_out_dim)]
sl_seqdisp = scale_to(sl_seqdisp, sl_seq_training.mean(), sl_seq_training.max()-sl_seq_training.min(), 127.5, 255.0, scale_disp, 'tanh')
p11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq_training.min(), sl_seq_training.max(), scale_disp)+str3(comp_eta(sl_seq_training)[0:5]))

#display SFA of Known Id testing Set
p12 = plt.subplot(1,3,2)
plt.title("Output Unit L3. (Known Id Test Set)")
sl_seqdisp = sl_seq_seenid[:, range(0,hierarchy_out_dim)]
sl_seqdisp = scale_to(sl_seqdisp, sl_seq_training.mean(), sl_seq_training.max()-sl_seq_training.min(), 127.5, 255.0, scale_disp, 'tanh')
p12.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq_seenid.min(), sl_seq_seenid.max(), scale_disp)+str3(comp_eta(sl_seq_seenid)[0:5]))

#display SFA of Known Id testing Set
p13 = plt.subplot(1,3,3)
plt.title("Output Unit L3. (Unknown Id Test Set)")
sl_seqdisp = sl_seq_newid[:, range(0,hierarchy_out_dim)]
sl_seqdisp = scale_to(sl_seqdisp, sl_seq_training.mean(), sl_seq_training.max()-sl_seq_training.min(), 127.5, 255.0, scale_disp, 'tanh')
p13.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq_newid.min(), sl_seq_newid.max(), scale_disp)+str3(comp_eta(sl_seq_newid)[0:5]))


print "Creating GUI..."
#Create Figure
f1 = plt.figure()
plt.suptitle("Pseudo-Invertible 4L SFA Hierarchy")
  
#display SFA
a11 = plt.subplot(2,3,1)
plt.title("Output Unit L3. (Usually Top Node)")
sl_seqdisp = sl_seq[:, range(0,hierarchy_out_dim)]
sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
a11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp)+str3(comp_eta(sl_seq)[0:5]))

#display first image
#Alternative: im1.show(command="xv")
f1a12 = plt.subplot(2,3,2)
plt.title("A particular image in the sequence")
#im_smalldisp = im_small.copy()
#f1a12.imshow(im_smalldisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)


f1a13 = plt.subplot(2,3,3)
plt.title("Reconstructed Image")

f1a21 = plt.subplot(2,3,4)
plt.title("Reconstruction Error")

f1a22 = plt.subplot(2,3,5)
plt.title("DIfferential Reconstruction y_pinv_(t+1) - y_pinv_(t)")

f1a23 = plt.subplot(2,3,6)
plt.title("Pseudoinverse of 0 / PINV(y) - PINV(0)")
sfa_zero = numpy.zeros((1, sfa_out_dim_L3))
pinv_zero = flow.inverse(sfa_zero)
pinv_zero = pinv_zero.reshape((subimage_height, subimage_width))
error_scale_disp=1.5
pinv_zero_disp = scale_to(pinv_zero, pinv_zero.mean(), pinv_zero.max()-pinv_zero.min(), 127.5, 255.0, error_scale_disp, 'tanh')
f1a23.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, zero" % (pinv_zero_disp.min(), pinv_zero_disp.max(), pinv_zero_disp.std(), error_scale_disp))
f1a23.imshow(pinv_zero_disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)        


#Retrieve Image in Sequence
def on_press(event):
    global plt, f1, f1a12, f1a13, f1a21, f1a22, fla23, subimages, subimage_width, subimage_height, num_images, sl_seq, pinv_zero, flow, sfa_out_dim_L2, error_scale_disp
    print 'you pressed', event.button, event.xdata, event.ydata
    y = int(event.ydata)
    if y < 0:
        y = 0
    if y >= num_images:
        y = num_images -1
    print "y=" + str(y)

#Display Original Image
    subimage_im = subimages[y].reshape((subimage_height, subimage_width))
    f1a12.imshow(subimage_im.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
 
#Display Reconstructed Image
    data_out = sl_seq[y].reshape((1, sfa_out_dim_L3))
    inverted_im = flow.inverse(data_out)
    inverted_im = inverted_im.reshape((subimage_height, subimage_width))
    f1a13.imshow(inverted_im.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)        
#Display Reconstruction Error
    error_scale_disp=1.5
    error_im = subimages[y].reshape((subimage_height, subimage_width)) - inverted_im 
    error_im_disp = scale_to(error_im, error_im.mean(), error_im.max()-error_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
    f1a21.imshow(error_im_disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
    plt.axis = f1a21
    f1a21.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y=%d" % (error_im.min(), error_im.max(), error_im.std(), error_scale_disp, y))
#Display Differencial change in reconstruction
    error_scale_disp=1.5
    if y >= num_images - 1:
        y_next = 0
    else:
        y_next = y+1
    print "y_next=" + str(y_next)
    data_out2 = sl_seq[y_next].reshape((1, sfa_out_dim_L3))
    inverted_im2 = flow.inverse(data_out2).reshape((subimage_height, subimage_width))
    diff_im = inverted_im2 - inverted_im 
    diff_im_disp = scale_to(diff_im, diff_im.mean(), diff_im.max()-diff_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
    f1a22.imshow(diff_im_disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
    plt.axis = f1a22
    f1a22.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y=%d" % (diff_im.min(), diff_im.max(), diff_im.std(), error_scale_disp, y))
#Display Difference from PINV(y) and PINV(0)
    error_scale_disp=1.5
    dif_pinv = inverted_im - pinv_zero 
    dif_pinv_disp = scale_to(dif_pinv, dif_pinv.mean(), dif_pinv.max()-dif_pinv.min(), 127.5, 255.0, error_scale_disp, 'tanh')
    f1a23.imshow(dif_pinv.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
    plt.axis = f1a23
    f1a23.set_xlabel("PINV(y) - PINV(0): min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y=%d" % (dif_pinv.min(), dif_pinv.max(), dif_pinv.std(), error_scale_disp, y))
    
    f1.canvas.draw()
    
f1.canvas.mpl_connect('button_press_event', on_press)

plt.show()
print "GUI Finished!"

