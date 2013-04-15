#Basic Experiments showing inversion of (linear) SFA applied to sequences of images
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 19 Mai 2009
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
        
#mat = numpy.zeros((8,6))              
#ac = numpy.array([3, 1, 2, 4, 4, 1, 0, 5])
#for i in range(mat.shape[0]):             
#    mat[i,ac[i]] = 1                           
#    mat2 = numpy.matrix(mat)
#
#pinv = (mat2.T * mat2).I * mat2.T

#data2 = numpy.matrix(data)
#mat2 * data2.T
#scrambled = mat2 * data2.T
#scrambled
#scrambled2 = scrambled.copy()
#scrambled2[6] = 0
#scrambled2[7] = 0
#pinv * scrambled2

print "***********************"
print "Testing Pseudo Invertible Switchboard: PInvSwitchboard (fixed small test values)"
print "***********************"

connections = numpy.array([3, 1, 2, 4, 4, 1])
pinv_switchboard = PInvSwitchboard(input_dim=6, connections=connections)
#Note: do not let data be only ints, or result will also be int!!!
data = numpy.array([10, 4, 8, 2, 15, 20.0])
data = data.reshape((1,6))
print "data shape is", data.shape
print "data:", data
scrambled = pinv_switchboard.execute(data)
print "scrambled:", scrambled
recovered = pinv_switchboard.inverse(scrambled)
print "recovered:", recovered

scrambled2 = scrambled + 1.0 * numpy.random.normal(scale=1.0, size=(scrambled.shape))
print "scrambled+noise:", scrambled2
recovered2 = pinv_switchboard.inverse(scrambled2)
print "recovered (+noise):", recovered2


print "***********************"
print "Testing Pseudo Invertible Switchboard and compute_lattice_matrix_connections_with_input_dim"
print "***********************"

sfa_out_dim_L0 = 10

#Create Switchboard L1
x_field_channels_L1=3
y_field_channels_L1=3
x_field_spacing_L1=2
y_field_spacing_L1=2
in_channel_dim_L1=sfa_out_dim_L0

y_in_channels_L1, x_in_channels_L1 = 7, 7

v1_L1 = [x_field_spacing_L1, 0]
v2_L1 = [x_field_spacing_L1, y_field_spacing_L1]

preserve_mask_L1 = numpy.ones((y_field_channels_L1, x_field_channels_L1, in_channel_dim_L1)) > 0.5

print "About to create (lattice based) intermediate layer widht=%d, height=%d"%(x_field_channels_L1,y_field_channels_L1) 
print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)


lat_mat_L0 = numpy.zeros((y_in_channels_L1, x_in_channels_L1, 2))
print "Shape of lat_mat_L0 is:", lat_mat_L0.shape

y_in_channels_L1, x_in_channels_L1, tmp = lat_mat_L0.shape

#preserve_mask_L1_3D = wider(preserve_mask_L1, scale_x=in_channel_dim)
(mat_connections_L1, lat_mat_L1) = compute_lattice_matrix_connections_with_input_dim(v1_L1, v2_L1, preserve_mask_L1, x_in_channels_L1, y_in_channels_L1, in_channel_dim_L1)
print "matrix connections L1:"
print mat_connections_L1
switchboard_L1 = PInvSwitchboard(x_in_channels_L1 * y_in_channels_L1 * in_channel_dim_L1, mat_connections_L1)

switchboard_L1.connections

num_nodes_SFA_L1 = lat_mat_L1.size / 2

num_nodes_SFA_L1 = lat_mat_L1.size / 2
print "Layer L1 with ", num_nodes_SFA_L1, " cloned SFA nodes will be created"

sfa_out_dim_L1 = 30

sfa_node_L1 = mdp.nodes.SFANode(input_dim=preserve_mask_L1.size, output_dim=sfa_out_dim_L1)
sfa_layer_L1 = mdp.hinet.CloneLayer(sfa_node_L1, n_nodes=num_nodes_SFA_L1)

print "training..."
im = numpy.array(range(0, y_in_channels_L1 * x_in_channels_L1 *in_channel_dim_L1)) * 5.0 / (y_in_channels_L1 * x_in_channels_L1 *in_channel_dim_L1)
x0 = im + numpy.random.normal(loc=0.0, scale=0.5, size=(100, y_in_channels_L1 * x_in_channels_L1 *in_channel_dim_L1))
y0 = switchboard_L1.execute(x0)        
sfa_layer_L1.train(y0)
sfa_layer_L1.stop_training()

print "executing..."
x0 = im + numpy.random.normal(loc=0.0, scale=0.5, size=(1, y_in_channels_L1 * x_in_channels_L1 *in_channel_dim_L1))

print "x0=", x0
y0 = switchboard_L1.execute(x0)        
print "y0=", y0
z0 = sfa_layer_L1.execute(y0)
print "z0=", z0
y1 = sfa_layer_L1.inverse(z0)
print "y0 - y1 = ", y0-y1
x1 = switchboard_L1.inverse(y1)    
print "x0 - x1 = ", x0-x1
print "Second execution!!!"
quit()


print "***********************"
print "Experiment: Observing SFA coefficients and inversion"
print "***********************"
scale_disp = 3

#SEQUENCE OF IMAGES
image_width  = 256
image_height = 192

subimage_width  = image_width/5
subimage_height = image_height/2 
#subimage_width  = 49
#subimage_height = 91
subimage_width  = 17
subimage_height = 17

#pixelsampling: 1=each pixel, 2= one yes and one not
subimage_pixelsampling=4

subimage_first_row= image_height/2-subimage_height*subimage_pixelsampling/2
subimage_first_column=image_width/2-subimage_width*subimage_pixelsampling/2

print "Images: width=%d, height=%d, subimage_width=%d,subimage_height=%d"%(image_width,image_height, subimage_width,subimage_height)

#Open sequence of images
im_seq_base_dir = "/home/escalafl/datasets/face_gen"
image_files = glob.glob(im_seq_base_dir + "/*tif")
image_files.sort()
num_images = len(image_files)

subimages = numpy.zeros((num_images, subimage_width * subimage_height))
act_im_num = 0
im_ori = []
for image_file in image_files:
    im = Image.open(image_file)
    im = im.convert("L")
    im_arr = numpy.asarray(im) + numpy.random.normal(scale=3.0, size=(image_height,image_width))
    
    im_small = im_arr[subimage_first_row:(subimage_first_row+subimage_height*subimage_pixelsampling):subimage_pixelsampling,
                              subimage_first_column:(subimage_first_column+subimage_width*subimage_pixelsampling):subimage_pixelsampling].astype(float)
    im_ori.append(im_small)
    subimages[act_im_num] = im_small.flatten()
    act_im_num = act_im_num+1


#Add Gaussian noise
#random.gauss(x,x)

#Create Figure
f1 = plt.figure()
plt.suptitle("Sequence of Images...")

#display Sequence of images
#Alternative: im1.show(command="xv")
a11 = plt.subplot(2,2,1)
plt.title("Sequence of Images")
subimagesdisp = subimages
a11.imshow(subimagesdisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)

#display first image
#Alternative: im1.show(command="xv")
a21 = plt.subplot(2,2,3)
plt.title("Last image in Sequence")
im_smalldisp = im_small
a21.imshow(im_smalldisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)

#Create single SFA Node
sfa_out_dim_L0 = 250
sfa_node_full = mdp.nodes.SFANode(output_dim=sfa_out_dim_L0)
sfa_node_full.train(subimages)
sfa_node_full.stop_training()

sl_seq = sfa_node_full.execute(subimages)
print sl_seq.shape
sfa_pretty_coefficients(sfa_node_full, sl_seq)  

sl_seq = sfa_node_full.execute(subimages)



a12 = plt.subplot(2,2,2)
plt.title("SFA Single (left=slowest)")

sl_seqdisp = sl_seq
sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
a12.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp)+str2(comp_eta(sl_seq)[0:4]))


a22 = plt.subplot(2,2,4)
plt.title("SFA Weight")

#node.sf.shape is 100 x 20
display_sfa = sfa_node_full.sf[:,0].reshape((subimage_height, subimage_width))
a22.imshow(display_sfa, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)


#Retrieve Image in Sequence
display = subimages

figure = f1
display_length = num_images
display_width  = subimage_width
display_height = subimage_height
def on_press(event):
    global display, figure, a21, a22, subimage_width, subimage_height, display_length, sfa_node_full, sfa_out_dim_L0
    print 'you pressed', event.button, event.xdata, event.ydata

    #For Image Sequence
    y = event.ydata
    if y < 0:
        y = 0
    if y >= display_length:
        y =  display_length -1

    display_im = display[y].reshape((display_height, subimage_width))
    a21.imshow(display_im, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)

    #For SFA Coefficients
    x = event.xdata
    if x < 0:
        x = 0
    if x >= sfa_out_dim_L0:
        x = sfa_out_dim_L0 -1

    display_sfa = sfa_node_full.sf[:,x].reshape((subimage_height, subimage_width))
    a22.imshow(display_sfa, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)

    figure.canvas.draw()

f1.canvas.mpl_connect('button_press_event', on_press)


#Create Figure
f2 = plt.figure()
plt.suptitle("Inverting SFA (few components)...")


pa11 = plt.subplot(2,2,1)
plt.title("SFA Single (left=slowest)")
sl_seqdisp = sl_seq
sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
pa11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
plt.xlabel("min=%.2f, max=%.2f, scale=%.2f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp)+str2(comp_eta(sl_seq)[0:4]))


#display first image
#Alternative: im1.show(command="xv")
pa12 = plt.subplot(2,2,2)
plt.title("Reconstructed Image")

pa21 = plt.subplot(2,2,3)
plt.title("Reconstruction Error")


pa22 = plt.subplot(2,2,4)
plt.title("Orginal Image")


#Retrieve Image in Sequence
display2 = sl_seq
display3 = subimages

figure2 = f2
display_length2 = num_images
display_width2  = subimage_width
display_height2 = subimage_height
def on_press2(event):
    global display2, figure2, pa12, pa21, pa22, plt, subimage_width, subimage_height, display_length2, sfa_node_full, sfa_out_dim_L0, subimages
    print 'you pressed', event.button, event.xdata, event.ydata

    error_scale_disp=1.5
    #For Image Sequence
    if event.button == 1:
        y = event.ydata
        if y < 0:
            y = 0
        if y >= display_length2:
            y =  display_length2 -1

#Display Reconstructed Image
        data = display2[y].reshape((1, sfa_out_dim_L0))
        display_im = sfa_node_full.inverse(data)
        display_im = display_im.reshape((subimage_height, subimage_width))
        pa12.imshow(display_im, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)        
#Display Reconstruction Error
        error_im = subimages[y].reshape((subimage_height, subimage_width)) - display_im 
        error_im_disp = scale_to(error_im, error_im.mean(), error_im.max()-error_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
        pa21.imshow(error_im_disp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
        plt.axis = pa21
        pa21.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y1=%d" % (error_im.min(), error_im.max(), error_im.std(), scale_disp, y))
#Display Original Image
        subimage_im = subimages[y].reshape((subimage_height, subimage_width)) 
        pa22.imshow(subimage_im.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
        figure2.canvas.draw()
        
        

    if event.button == 2:
        for y in range(0, display_length2, 8):
            data = display2[y].reshape((1, sfa_out_dim_L0))
            display_im = sfa_node_full.inverse(data)
            display_im = display_im.reshape((subimage_height, subimage_width))
            pa12.imshow(display_im, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
            error_im = subimages[y].reshape((subimage_height, subimage_width)) - display_im 
            error_im_disp = scale_to(error_im, error_im.mean(), error_im.max()-error_im.min(), 127.5, 255.0, error_scale_disp, 'tanh')
            pa21.imshow(error_im.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
            pa21.set_xlabel("min=%.2f, max=%.2f, std=%.2f, scale=%.2f, y1=%d" % (error_im.min(), error_im.max(), error_im.std(), scale_disp, y))
            subimage_im = subimages[y].reshape((subimage_height, subimage_width)) 
            pa22.imshow(subimage_im.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)       
            figure2.canvas.draw()
            time.sleep(0.0001)

f2.canvas.mpl_connect('button_press_event', on_press2)


plt.show()
print "Finished!"



#print "Lattice based processing"
##Create Switchboard L0
##width=51, height=96
##x_field_channels=21
##y_field_channels=
##x_field_spacing=6
##y_field_spacing=6
##in_channel_dim=1
##subimage_width = subimage_height = 14 + 7
#x_field_channels_L0=14
#y_field_channels_L0=14
#x_field_spacing_L0=7
#y_field_spacing_L0=7
#in_channel_dim_L0=1
#
#v1 = (x_field_spacing_L0, 0)
#v2 = (x_field_spacing_L0, y_field_spacing_L0)
#
#preserve_mask_L0 = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
## 6 x 12
#print "About to create (lattice based) perceptive field of widht=%d, height=%d"%(x_field_channels_L0,y_field_channels_L0) 
#print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L0,y_field_spacing_L0,in_channel_dim_L0)
#
#(mat_connections, lat_mat_L0) = compute_lattice_matrix_connections(v1, v2, preserve_mask_L0, subimage_width, subimage_height, in_channel_dim_L0)
#print "matrix connections:"
#print mat_connections
#
#switchboard_L0 = mdp.hinet.Switchboard(subimage_width * subimage_height, mat_connections)
#switchboard_L0.connections
#
##Create single SFA Node
##Warning Signal too short!!!!!sfa_out_dim_L0 = 20
#sfa_out_dim_L0 = 10
#sfa_node_L0 = mdp.nodes.SFANode(input_dim=preserve_mask_L0.size, output_dim=sfa_out_dim_L0)
#
##Create array of sfa_nodes (just one node, but clonned)
#sfa_layer_L0 = mdp.hinet.CloneLayer(sfa_node_L0, n_nodes=lat_mat_L0.size / 2)
#
#
##Create Switchboard L1
#x_field_channels_L1=4
#y_field_channels_L1=4
#x_field_spacing_L1=2
#y_field_spacing_L1=2
#in_channel_dim_L1=sfa_out_dim_L0
#
#print "About to create intermediate layer widht=%d, height=%d"%(x_field_channels_L1,y_field_channels_L1) 
#print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)
#switchboard_L1 = mdp.hinet.Rectangular2dSwitchboard(12, 6, x_field_channels_L1,y_field_channels_L1,x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)
#switchboard_L1.connections
#
##Create L1 sfa node
#sfa_out_dim_L1 = 12
#sfa_node_L1 = mdp.nodes.SFANode(input_dim=switchboard_L1.out_channel_dim, output_dim=sfa_out_dim_L1)
#
##Create array of sfa_nodes (just one node, but clonned)
##This one should not be clonned!!!
#sfa_layer_L1 = mdp.hinet.CloneLayer(sfa_node_L1, n_nodes=switchboard_L1.output_channels)
#
#
##Create L1 sfa node
#sfa_out_dim_L2 = 4
#sfa_node_L2 = mdp.nodes.SFANode(output_dim=sfa_out_dim_L2)
#
#
##Join Switchboard and SFA layer in a single flow
#flow = mdp.Flow([switchboard_L0, sfa_layer_L0, switchboard_L1, sfa_layer_L1, sfa_node_L2])
#
#flow.train(subimages)
#sl_seq = flow.execute(subimages)
#
#a11 = plt.subplot(2,2,4)
#plt.title("Output Unit L2. Identical Lattice Switch")
#sl_seqdisp = sl_seq[:, range(0,sfa_out_dim_L2)]
#sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
#a11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
#plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp)+str3(comp_eta(sl_seq)[0:5]))
#
#
#print "Finished!"
#
##Create Figure
#f2 = plt.figure()
#plt.suptitle("Different Masks at input Images...")
#
#print "Changing Lattices..."
#x_field_channels_L0=14
#y_field_channels_L0=14
#x_field_spacing_L0=7
#y_field_spacing_L0=7
#in_channel_dim_L0=1
#
#v1 = (x_field_spacing_L0, 0)
#v2 = (x_field_spacing_L0, y_field_spacing_L0)
#
#print "Creating Masks..."
#x_center = (x_field_channels_L0-1) / 2.0
#x_dmax = (x_field_channels_L0-1) / 2.0
#y_center = (y_field_channels_L0-1) / 2.0
#y_dmax = (y_field_channels_L0-1) / 2.0
#
#preserve_mask_L0_circ = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
##faktor=1 => ellipse of same height and width, smaller faktor => smaller ellipse
##faktor = 0.60
#faktor = 0.50
#for xx_mask in range(x_field_channels_L0):
#    for yy_mask in range(y_field_channels_L0):
#        heights = ((xx_mask - x_center) / x_dmax) ** 2 + \
#                  ((yy_mask - y_center) / y_dmax) ** 2
#        if heights > 2 * (faktor**2):
#            preserve_mask_L0_circ[yy_mask, xx_mask] = False
#print "Size of circular perceptive field is ", preserve_mask_L0_circ.sum()
#
#preserve_mask_L0_prism = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
##faktor=1 => ellipse of same height and width, smaller faktor => smaller ellipse
##faktor = 0.50
#faktor = 0.40
#for xx_mask in range(x_field_channels_L0):
#    for yy_mask in range(y_field_channels_L0):
#        heights = numpy.abs((xx_mask - x_center) / x_dmax) + \
#                  numpy.abs((yy_mask - y_center) / y_dmax)
#        if heights > 2 * faktor:
#            preserve_mask_L0_prism[yy_mask, xx_mask] = False
#print "Size of prismal perceptive field is ", preserve_mask_L0_prism.sum()
#
#preserve_mask_L0_rect = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
##faktor=1 => ellipse of same height and width, smaller faktor => smaller ellipse
##faktor = 0.70
#faktor = 0.50
#for xx_mask in range(x_field_channels_L0):
#    for yy_mask in range(y_field_channels_L0):
#        heights = numpy.abs((xx_mask - x_center) / x_dmax) + \
#                  numpy.abs((yy_mask - y_center) / y_dmax)
#        if numpy.abs((xx_mask - x_center) / x_dmax) > faktor:
#            preserve_mask_L0_rect[yy_mask, xx_mask] = False
#        if numpy.abs((yy_mask - y_center) / y_dmax) > faktor:
#            preserve_mask_L0_rect[yy_mask, xx_mask] = False
#print "Size of rectangular perceptive field is ", preserve_mask_L0_rect.sum()
#
#preserve_mask_L0_cross = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
##faktor=1 => ellipse of same height and width, smaller faktor => smaller ellipse
##faktor = 0.25
#faktor = 0.20
#for xx_mask in range(x_field_channels_L0):
#    for yy_mask in range(y_field_channels_L0):
#        heights = numpy.abs((xx_mask - x_center) / x_dmax) + \
#                  numpy.abs((yy_mask - y_center) / y_dmax)
#        if numpy.abs((xx_mask - x_center) / x_dmax) > faktor and \
#           numpy.abs((yy_mask - y_center) / y_dmax) > faktor:
#            preserve_mask_L0_cross[yy_mask, xx_mask] = False
#print "Size of cross perceptive field is ", preserve_mask_L0_cross.sum()
#            
#preserve_mask_L0_random = numpy.ones((y_field_channels_L0, x_field_channels_L0)) > 0.5
##faktor=1 => all, 0 => nothing, 0.5 half the points
##faktor = 0.50
#faktor = 0.30
#preserve_mask_L0_random = numpy.random.uniform(size=(y_field_channels_L0, x_field_channels_L0))
##find a value faktor2, so that the proportion of ones in the mask is approx. faktor
#i = 1
#minf = 0.0
#maxf = 1.0
#while i < 12:
#    faktor2= (minf + maxf) / 2.0
#    tmpmat = (preserve_mask_L0_random <= faktor2)
#    tmpsum = tmpmat.sum()
#    if(tmpsum * 1.0 / preserve_mask_L0_random.size >  faktor):
#        maxf = faktor2
#    else:
#        minf = faktor2
#    i = i + 1
##
#preserve_mask_L0_random = (preserve_mask_L0_random <= faktor2)
#print "Size of random perceptive field is ", preserve_mask_L0_random.sum()
#
#
#print "Drawing Masks"
#im_small_c = im_small.copy()
#im_small_c[0*y_field_channels_L0+0:0*y_field_channels_L0+y_field_channels_L0, 0:x_field_channels_L0][preserve_mask_L0] = 255
#im_small_c[1*y_field_channels_L0+0:1*y_field_channels_L0+y_field_channels_L0, 0:x_field_channels_L0][preserve_mask_L0_circ] = 220
#im_small_c[2*y_field_channels_L0+0:2*y_field_channels_L0+y_field_channels_L0, 0:x_field_channels_L0][preserve_mask_L0_prism] = 180
#im_small_c[3*y_field_channels_L0+0:3*y_field_channels_L0+y_field_channels_L0, 0:x_field_channels_L0][preserve_mask_L0_rect] = 140
#im_small_c[4*y_field_channels_L0+0:4*y_field_channels_L0+y_field_channels_L0, 0:x_field_channels_L0][preserve_mask_L0_cross] = 220
#im_small_c[5*y_field_channels_L0+0:5*y_field_channels_L0+y_field_channels_L0, 0:x_field_channels_L0][preserve_mask_L0_random] = 255
#
#
##display first image
##Alternative: im1.show(command="xv")
#a11 = plt.subplot(2,3,1)
#plt.title("Last image in Sequence")
#im_smalldisp = im_small_c
#a11.imshow(im_smalldisp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
#
#
#print "Applying SFA for each mask"
#sfa_number=2
#for preserve_mask_L0_form in (preserve_mask_L0_circ, preserve_mask_L0_prism, preserve_mask_L0_rect, \
#                              preserve_mask_L0_cross, preserve_mask_L0_random):
#    print "About to create perceptive field of widht=%d, height=%d"%(x_field_channels_L0,y_field_channels_L0) 
#    print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L0,y_field_spacing_L0,in_channel_dim_L0)
#    
#    (mat_connections, lat_mat_L0) = compute_lattice_matrix_connections(v1, v2, preserve_mask_L0_form, subimage_width, subimage_height, in_channel_dim_L0)
#    print "matrix connections:"
#    print mat_connections
#    
#    switchboard_L0 = mdp.hinet.Switchboard(subimage_width * subimage_height, mat_connections)
#    switchboard_L0.connections
#    
#    #Create single SFA Node
#    #Warning!!!! sfa_out_dim_L0 = 20
#    sfa_out_dim_L0 = 10
#    sfa_node_L0 = mdp.nodes.SFANode(input_dim=preserve_mask_L0_form.sum(), output_dim=sfa_out_dim_L0)
#    
#    #Create array of sfa_nodes (just one node, but clonned)
#    sfa_layer_L0 = mdp.hinet.CloneLayer(sfa_node_L0, n_nodes=lat_mat_L0.size / 2)
#
#    #Create Switchboard L1
#    x_field_channels_L1=4
#    y_field_channels_L1=4
#    x_field_spacing_L1=2
#    y_field_spacing_L1=2
#    in_channel_dim_L1=sfa_out_dim_L0
#    
#    print "About to create intermediate layer widht=%d, height=%d"%(x_field_channels_L1,y_field_channels_L1) 
#    print "With a spacing of horiz=%d, vert=%d, and %d channels"%(x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)
#    switchboard_L1 = mdp.hinet.Rectangular2dSwitchboard(12, 6, x_field_channels_L1,y_field_channels_L1,x_field_spacing_L1,y_field_spacing_L1,in_channel_dim_L1)
#    switchboard_L1.connections
#    
#    #Create L1 sfa node
#    sfa_out_dim_L1 = 20
#    sfa_node_L1 = mdp.nodes.SFANode(input_dim=switchboard_L1.out_channel_dim, output_dim=sfa_out_dim_L1)
#    
#    #Create array of sfa_nodes (just one node, but clonned)
#    #This one should not be clonned!!!
#    sfa_layer_L1 = mdp.hinet.CloneLayer(sfa_node_L1, n_nodes=switchboard_L1.output_channels)
#    
#    #Create L1 sfa node
#    sfa_out_dim_L2 = 4
#    sfa_node_L2 = mdp.nodes.SFANode(output_dim=sfa_out_dim_L2)
#        
#    #Join Switchboard and SFA layer in a single flow
#    flow = mdp.Flow([switchboard_L0, sfa_layer_L0, switchboard_L1, sfa_layer_L1, sfa_node_L2])
#    
#    flow.train(subimages)
#    sl_seq = flow.execute(subimages)
#    
#    a11 = plt.subplot(2,3,sfa_number)
#    plt.title("SFA + Lattice. Single Unit L2 (left=slowest)")
#    sl_seqdisp = sl_seq[:, range(0,sfa_out_dim_L2)]
#    sl_seqdisp = scale_to(sl_seqdisp, sl_seq.mean(), sl_seq.max()-sl_seq.min(), 127.5, 255.0, scale_disp, 'tanh')
#    a11.imshow(sl_seqdisp.clip(0,255), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
##    plt.xlabel("min=%.3f, max=%.3f, scale=%.3f\n e[]=" % (sl_seq.min(), sl_seq.max(), scale_disp)+str3(comp_eta(sl_seq)[0:4]))
#    plt.xlabel("Mask Size=%d\n e[]=" % preserve_mask_L0_form.sum() + str3(comp_eta(sl_seq)[0:5]))
#
#    sfa_number = sfa_number + 1

plt.show()
print "Finished showing!"



#For inverting switchboard
#mat = numpy.zeros((8,6))              
#ac = numpy.array([3, 1, 2, 4, 4, 1, 0, 5])
#for i in range(mat.shape[0]):             
#    mat[i,ac[i]] = 1                           
#    mat2 = numpy.matrix(mat)
#
#data2 = numpy.matrix(data)
#mat2 * data2.T
#scrambled = mat2 * data2.T
#scrambled
#scrambled2 = scrambled.copy()
#scrambled2[6] = 0
#scrambled2[7] = 0
#pinv = (mat2.T * mat2).I * mat2.T
#pinv * scrambled2





