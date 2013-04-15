# More complex hierarchichal architecture with arbitrary expansion (work in progress)
#Using modified SFA node to group elements of same slowness in blocks
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 12 Jun 2009
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
import time


print "******************************************"
print "Creating Expanded 4L SFA hierarchy"
print "******************************************"

scale_disp = 3
#MULTI-IDENTITY SEQUENCE OF IMAGES
#image_width  = 256
#image_height = 192
image_width  = 640
image_height = 480

#subimage_width  = image_width/5
#subimage_height = image_height/2 
#subimage_width  = image_width/8
#subimage_height = image_height/4 
subimage_width  = 135
#subimage_height = 122 
subimage_height = 135 

#subimage_width  = 49
#subimage_height = 91
 
#pixelsampling: 1=each pixel, 2= one yes and one not, 3=one every thee pixels
subimage_pixelsampling= 2

subimage_first_row= image_height/2-subimage_height*subimage_pixelsampling/2
subimage_first_column=image_width/2-subimage_width*subimage_pixelsampling/2+ 5*subimage_pixelsampling

print "Images: width=%d, height=%d, subimage_width=%d,subimage_height=%d"%(image_width,image_height, subimage_width,subimage_height)

#Open sequence of images
#im_seq_base_dir = "/home/escalafl/datasets/face_gen"
print "loading training images..."
t0 = time.time()
im_seq_base_dir = "/local/tmp/escalafl/Alberto/training"
image_files0 = glob.glob(im_seq_base_dir + "/random000*tif")
image_files0.sort()
image_files1 = glob.glob(im_seq_base_dir + "/random001*tif")
image_files1.sort()
image_files2 = glob.glob(im_seq_base_dir + "/random002*tif")
image_files2.sort()
image_files3 = glob.glob(im_seq_base_dir + "/random003*tif")
image_files3.sort()
image_files4 = glob.glob(im_seq_base_dir + "/random004*tif")
image_files4.sort()
image_files5 = glob.glob(im_seq_base_dir + "/random005*tif")
image_files5.sort()
image_files6 = glob.glob(im_seq_base_dir + "/random006*tif")
image_files6.sort()
image_files7 = glob.glob(im_seq_base_dir + "/random007*tif")
image_files7.sort()
image_files8 = glob.glob(im_seq_base_dir + "/random008*tif")
image_files8.sort()
image_files9 = glob.glob(im_seq_base_dir + "/random009*tif")
image_files9.sort()
image_files10 = glob.glob(im_seq_base_dir + "/random010*tif")
image_files10.sort()
image_files11 = glob.glob(im_seq_base_dir + "/random011*tif")
image_files11.sort()
image_files12 = glob.glob(im_seq_base_dir + "/random012*tif")
image_files12.sort()
image_files13 = glob.glob(im_seq_base_dir + "/random013*tif")
image_files13.sort()
image_files14 = glob.glob(im_seq_base_dir + "/random014*tif")
image_files14.sort()
image_files15 = glob.glob(im_seq_base_dir + "/random015*tif")
image_files15.sort()

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


print "Using training data for slowly varying ID, and (very!!!) quickly varying angle"
image_files = []
image_files.extend(image_files0)
image_files.extend(image_files1)
image_files.extend(image_files2)
image_files.extend(image_files3)
#image_files.extend(image_files4)
#image_files.extend(image_files5)
#image_files.extend(image_files6)
#image_files.extend(image_files7)
#image_files.extend(image_files8)
#image_files.extend(image_files9)
#image_files.extend(image_files10)
#image_files.extend(image_files11)
#image_files.extend(image_files12)
#image_files.extend(image_files13)
#image_files.extend(image_files14)
#image_files.extend(image_files15)

num_images = len(image_files)
block_size= len(image_files0)
block_size_L0=block_size
block_size_L1=block_size
block_size_L2=block_size
block_size_L3=block_size
block_size_exec=block_size #(Used only for random walk)


add_noise_L0 = True
subimages = numpy.zeros((num_images, subimage_width * subimage_height))
act_im_num = 0
im_ori = []
for image_file in image_files:
    im = Image.open(image_file)
    im = im.convert("L")
    im_arr = numpy.asarray(im)
    im_small = im_arr[subimage_first_row:(subimage_first_row+subimage_height*subimage_pixelsampling):subimage_pixelsampling,
                              subimage_first_column:(subimage_first_column+subimage_width*subimage_pixelsampling):subimage_pixelsampling].astype(float)
    if add_noise_L0 is True:
        noise = numpy.random.normal(loc=0.0, scale=0.05, size=(subimage_height, subimage_width))
        im_small = im_small*1.0 + noise
#    im_ori.append(im_small)
    subimages[act_im_num] = im_small.flatten()
    act_im_num = act_im_num+1
    del im_small
    del im_arr
    del im
    
t1 = time.time()
print num_images, " Images loaded in %0.3f ms"% ((t1-t0)*1000.0)
    
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

switchboard_L0 = PInvSwitchboard(subimage_width * subimage_height, mat_connections_L0)
switchboard_L0.connections

t2 = time.time()
print "PInvSwitchboard L0 created in %0.3f ms"% ((t2-t1)*1000.0)

#Create single SFA Node
#Warning Signal too short!!!!!sfa_out_dim_L0 = 20
#sfa_out_dim_L0 = 10
sfa_out_dim_L0 = 15
sfa_node_L0 = mdp.nodes.SFANode(input_dim=preserve_mask_L0.size, output_dim=sfa_out_dim_L0, block_size=block_size_L0)

#Create array of sfa_nodes (just one node, but cloned)
num_nodes_SFA_L0 = lat_mat_L0.size / 2
sfa_layer_L0 = mdp.hinet.CloneLayer(sfa_node_L0, n_nodes=num_nodes_SFA_L0)

t3 = time.time()
print "SFACloneLayer L0 created in %0.3f ms"% ((t3-t2)*1000.0)

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
print "PinvSwitchboard L1 created in %0.3f ms"% ((t4-t3)*1000.0)

num_nodes_SFA_L1 = lat_mat_L1.size / 2

#Default: cloneLayerL1 = False
cloneLayerL1 = False

#Create L1 sfa node
#sfa_out_dim_L1 = 12
sfa_out_dim_L1 = 20

if cloneLayerL1 is False:
    print "Layer L1 with ", num_nodes_SFA_L1, " cloned SFA nodes will be created"
      
    #sfa_node_L1 = mdp.nodes.SFANode(input_dim=switchboard_L1.out_channel_dim, output_dim=sfa_out_dim_L1)
    sfa_node_L1 = mdp.nodes.SFANode(input_dim=preserve_mask_L1.size, output_dim=sfa_out_dim_L1, block_size=block_size_L1)
    
    print "Warning!!! layer L1 using cloned SFA instead of several independent copies!!!"
    #!!!no ma, ya aniadele el atributo output_channels al PINVSwitchboard
    
    sfa_layer_L1 = mdp.hinet.CloneLayer(sfa_node_L1, n_nodes=num_nodes_SFA_L1)
else:
    print "Layer L1 with ", num_nodes_SFA_L1, " independent SFA nodes will be created"
    
    SFA_nodes_L1 = range(num_nodes_SFA_L1)
    for i in range(num_nodes_SFA_L1):
        SFA_nodes_L1[i] = mdp.nodes.SFANode(input_dim=preserve_mask_L1.size, output_dim=sfa_out_dim_L1, block_size=block_size_L1)
    sfa_layer_L1 = mdp.hinet.Layer(SFA_nodes_L1)
  
t5 = time.time()
print "SFA Layer L1 created in %0.3f ms"% ((t5-t4)*1000.0)

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

num_nodes_SFA_L2 = lat_mat_L2.size / 2

#Default: cloneLayerL2 = False
cloneLayerL2 = False

#Create L2 sfa node
#sfa_out_dim_L2 = 12
sfa_out_dim_L2 = 20

if cloneLayerL2 is True:
    print "Layer L2 with ", num_nodes_SFA_L2, " cloned SFA nodes will be created"
      
    #sfa_node_L2 = mdp.nodes.SFANode(input_dim=switchboard_L2.out_channel_dim, output_dim=sfa_out_dim_L2)
    sfa_node_L2 = mdp.nodes.SFANode(input_dim=preserve_mask_L2.size, output_dim=sfa_out_dim_L2, block_size=block_size_L2)
    
    print "Warning!!! layer L2 using cloned SFA instead of several independent copies!!!"
    #!!!no ma, ya aniadele el atributo output_channels al PINVSwitchboard
    
    sfa_layer_L2 = mdp.hinet.CloneLayer(sfa_node_L2, n_nodes=num_nodes_SFA_L2)
else:
    print "Layer L2 with ", num_nodes_SFA_L2, " independent SFA nodes will be created"
    
    SFA_nodes_L2 = range(num_nodes_SFA_L2)
    for i in range(num_nodes_SFA_L2):
        SFA_nodes_L2[i] = mdp.nodes.SFANode(input_dim=preserve_mask_L2.size, output_dim=sfa_out_dim_L2, block_size=block_size_L2)
    sfa_layer_L2 = mdp.hinet.Layer(SFA_nodes_L2)


t7 = time.time()
print "SFA Layer L2 created in %0.3f ms"% ((t7-t6)*1000.0)


#Create L3 sfa node
#sfa_out_dim_L3 = 150
#sfa_out_dim_L3 = 78
sfa_out_dim_L3 = 20

print "Creating final SFA node L3"
#sfa_node_L3 = mdp.nodes.SFANode(input_dim=preserve_mask_L2.size, output_dim=sfa_out_dim_L3)
#sfa_node_L3 = mdp.nodes.SFANode()
sfa_node_L3 = mdp.nodes.SFANode(output_dim=sfa_out_dim_L3, block_size=block_size_L3)

t8 = time.time()
print "SFA Node L3 created in %0.3f ms"% ((t8-t7)*1000.0)

#Join Switchboard and SFA layer in a single flow
#flow = mdp.Flow([switchboard_L0, sfa_layer_L0, switchboard_L1, sfa_layer_L1, switchboard_L2, sfa_layer_L2, sfa_node_L3], verbose=True)
flow = mdp.Flow([switchboard_L0, sfa_layer_L0, switchboard_L1, sfa_layer_L1, switchboard_L2, sfa_layer_L2, sfa_node_L3], verbose=True)
#flow = mdp.Flow([switchboard_L0, sfa_layer_L0, switchboard_L1, sfa_layer_L1])

print "sfa_node_L3 output_dim = ", sfa_node_L3.output_dim

print "Training hierarchy ..."
#WARNING!
#subimages_p = subimages.copy()
subimages_p = subimages
flow.train(subimages_p)
y = flow.execute(subimages_p[0:1])
hierarchy_out_dim = y.shape[1]
t8 = time.time()

print "Execution over training set..."
print "Input Signal: Training Data"
#WARNING
#subimages_training = subimages.copy()
subimages_training = subimages
num_images_training = num_images
sl_seq = sl_seq_training = flow.execute(subimages_training)


print "Loading test images, known ids..."
im_seq_base_dir = "/local/tmp/escalafl/Alberto/testing_seenid"
image_files0_seenid = glob.glob(im_seq_base_dir + "/random000*tif")
image_files0_seenid.sort()
image_files1_seenid = glob.glob(im_seq_base_dir + "/random001*tif")
image_files1_seenid.sort()
image_files2_seenid = glob.glob(im_seq_base_dir + "/random002*tif")
image_files2_seenid.sort()
image_files3_seenid = glob.glob(im_seq_base_dir + "/random003*tif")
image_files3_seenid.sort()
image_files4_seenid = glob.glob(im_seq_base_dir + "/random004*tif")
image_files4_seenid.sort()
image_files5_seenid = glob.glob(im_seq_base_dir + "/random005*tif")
image_files5_seenid.sort()
image_files6_seenid = glob.glob(im_seq_base_dir + "/random006*tif")
image_files6_seenid.sort()
image_files7_seenid = glob.glob(im_seq_base_dir + "/random007*tif")
image_files7_seenid.sort()
image_files8_seenid = glob.glob(im_seq_base_dir + "/random008*tif")
image_files8_seenid.sort()
image_files9_seenid = glob.glob(im_seq_base_dir + "/random009*tif")
image_files9_seenid.sort()
image_files10_seenid = glob.glob(im_seq_base_dir + "/random010*tif")
image_files10_seenid.sort()
image_files11_seenid = glob.glob(im_seq_base_dir + "/random011*tif")
image_files11_seenid.sort()
image_files12_seenid = glob.glob(im_seq_base_dir + "/random012*tif")
image_files12_seenid.sort()
image_files13_seenid = glob.glob(im_seq_base_dir + "/random013*tif")
image_files13_seenid.sort()
image_files14_seenid = glob.glob(im_seq_base_dir + "/random014*tif")
image_files14_seenid.sort()
image_files15_seenid = glob.glob(im_seq_base_dir + "/random015*tif")
image_files15_seenid.sort()

image_files_seenid = []
image_files_seenid.extend(image_files0_seenid)
image_files_seenid.extend(image_files1_seenid)
image_files_seenid.extend(image_files2_seenid)
image_files_seenid.extend(image_files3_seenid)
#image_files_seenid.extend(image_files4_seenid)
#image_files_seenid.extend(image_files5_seenid)
#image_files_seenid.extend(image_files6_seenid)
#image_files_seenid.extend(image_files7_seenid)
#image_files_seenid.extend(image_files8_seenid)
#image_files_seenid.extend(image_files9_seenid)
#image_files_seenid.extend(image_files10_seenid)
#image_files_seenid.extend(image_files11_seenid)
#image_files_seenid.extend(image_files12_seenid)
#image_files_seenid.extend(image_files13_seenid)
#image_files_seenid.extend(image_files14_seenid)
#image_files_seenid.extend(image_files15_seenid)
num_images_seenid = len(image_files_seenid)

subimages_seenid = numpy.zeros((num_images_seenid, subimage_width * subimage_height))
act_im_num = 0
for image_file in image_files_seenid:
    im = Image.open(image_file)
    im = im.convert("L")
    im_arr = numpy.asarray(im)
    im_small = im_arr[subimage_first_row:(subimage_first_row+subimage_height*subimage_pixelsampling):subimage_pixelsampling,
                              subimage_first_column:(subimage_first_column+subimage_width*subimage_pixelsampling):subimage_pixelsampling].astype(float)
#    im_ori.append(im_small)
    subimages_seenid[act_im_num] = im_small.flatten()
    act_im_num = act_im_num+1
    del im_small
    del im_arr
    del im
    
print num_images_seenid, " Images loaded in %0.3f ms"% ((t1-t0)*1000.0)
print "Execution over known id testing set..."
print "Input Signal: Known Id test images"
sl_seq_seenid = flow.execute(subimages_seenid)


print "Loading test images, unknown ids..."
im_seq_base_dir = "/local/tmp/escalafl/Alberto/testing_newid"
image_files16_newid = glob.glob(im_seq_base_dir + "/random016*tif")
image_files16_newid.sort()
image_files17_newid = glob.glob(im_seq_base_dir + "/random017*tif")
image_files17_newid.sort()
image_files18_newid = glob.glob(im_seq_base_dir + "/random018*tif")
image_files18_newid.sort()
image_files19_newid = glob.glob(im_seq_base_dir + "/random019*tif")
image_files19_newid.sort()

image_files_newid = []
image_files_newid.extend(image_files16_newid)
image_files_newid.extend(image_files17_newid)
image_files_newid.extend(image_files18_newid)
image_files_newid.extend(image_files19_newid)
num_images_newid = len(image_files_newid)

subimages_newid = numpy.zeros((num_images_newid, subimage_width * subimage_height))
act_im_num = 0
for image_file in image_files_newid:
    im = Image.open(image_file)
    im = im.convert("L")
    im_arr = numpy.asarray(im)
    im_small = im_arr[subimage_first_row:(subimage_first_row+subimage_height*subimage_pixelsampling):subimage_pixelsampling,
                              subimage_first_column:(subimage_first_column+subimage_width*subimage_pixelsampling):subimage_pixelsampling].astype(float)
#    im_ori.append(im_small)
    subimages_newid[act_im_num] = im_small.flatten()
    act_im_num = act_im_num+1
    del im_small
    del im_arr
    del im
    
print num_images_newid, " Images loaded in %0.3f ms"% ((t1-t0)*1000.0)
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

