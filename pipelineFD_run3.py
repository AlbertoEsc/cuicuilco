#! /usr/bin/env python

#Special purpose hierarchical network pipeline for data processing
#Each network is a hierarchical implementation of Slow Feature Analysis (dimensionality reduction) followed by a regression algorithm
#Now with performance analysis based on FRGC metadata, work in progress
#Now with extensive error measurements
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 7 Juni 2010
#Ruhr-University-Bochum, Institute of Neural Computation, Group of Prof. Dr. Wiskott

import numpy
import scipy

display_plots = True and False
if display_plots:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

import PIL
import Image
import mdp
import more_nodes
import patch_mdp

#import svm as libsvm
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
#from matplotlib.ticker import MultipleLocator
import copy
import string
import xml_frgc_tools as frgc
import getopt


command_line_interface = True #and False
load_FRGC_images = True
save_frgc_data = True
adaptive_grid = True
smallest_face = 0.2 # 20% of image size
grid_step = 1.85 #ideally 2.0, but values < 2 improve detection rate by overlapping of boxes
grid_centering = True
plots_created = False
write_results = True
if not command_line_interface:
    write_results = False
right_screen_eye_first = False
verbose_networks = True and False

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

#Returns the eye coordinates in the same scale as the box, already considering correction face_sampling
#First left eye, then right eye. Notice, left_x > right_x and eye_y < center_y
#Given approximate box coordinates, corresponding to a box with some face_sampling, approximates the
#positions of the eyes according to the normalization criteria.
#face_sampling < 1 means that the face is larger inside the box
def compute_approximate_eye_coordinates(box_coordinates, face_sampling=0.825):
    x0, y0, x1, y1 = box_coordinates
    fc_x = (x0+x1)/2.0
    fc_y = (y0+y1)/2.0
    
    #eye deltas with respect to the face center
    eye_dx = 37.0/2.0 * numpy.abs(x1-x0)/ 128 / face_sampling
    eye_dy = 42.0/2.0 * numpy.abs(y1-y0)/ 128 / face_sampling
    eye_left_x = fc_x + eye_dx
    eye_right_x = fc_x - eye_dx
    eye_y = fc_y - eye_dy

    return numpy.array([eye_left_x, eye_y, eye_right_x, eye_y])

#In addition to the eye coordinates, it gives two boxes containing the left and right eyes
def compute_approximate_eye_boxes_coordinates(box_coordinates, face_sampling=0.825):
    x0, y0, x1, y1 = box_coordinates
    fc_x = (x0+x1)/2.0
    fc_y = (y0+y1)/2.0
    
    #eye deltas with respect to the face center
    eye_dx = 37.0/2.0 * numpy.abs(x1-x0)/ 128 / face_sampling
    eye_dy = 42.0/2.0 * numpy.abs(y1-y0)/ 128 / face_sampling
    box_width = 64 * numpy.abs(x1-x0)/ 128 * 0.825 / face_sampling
    box_height = 64 * numpy.abs(y1-y0)/ 128 * 0.825 / face_sampling
        
    eye_left_x = fc_x + eye_dx
    eye_right_x = fc_x - eye_dx
    eye_y = fc_y - eye_dy
    box_y0 = eye_y - box_height/2
    box_y1 = eye_y + box_height/2
    box_left_x0 = eye_left_x - box_width/2
    box_left_x1 = eye_left_x + box_width/2
    box_right_x0 = eye_right_x - box_width/2
    box_right_x1 = eye_right_x + box_width/2

    return numpy.array([eye_left_x, eye_y, eye_right_x, eye_y]), numpy.array([box_left_x0, box_y0, box_left_x1, box_y1]), numpy.array([box_right_x0, box_y0, box_right_x1, box_y1])

#Face midpoint is the average of the point between the eyes and the mouth
def compute_face_midpoint(eye_left_x, eye_left_y, eye_right_x, eye_right_y, mouth_x, mouth_y):
    eye_center_x = (eye_left_x+eye_right_x)/2.0
    eye_center_y = (eye_left_y+eye_right_y)/2.0
    midpoint_x = (eye_center_x + mouth_x)/2.0
    midpoint_y = (eye_center_y + mouth_y)/2.0
    return midpoint_x, midpoint_y

#Error in the (Euclidean) distance relative to the distance between the eyes
def relative_error_detection(app_eye_coords, eye_coords):
    dist_left = eye_coords[0:2]-app_eye_coords[0:2] #left eye
    dist_left = numpy.sqrt((dist_left**2).sum())
    dist_right = eye_coords[2:4]-app_eye_coords[2:4] #right eye
    dist_right = numpy.sqrt((dist_right**2).sum())
    dist_eyes = eye_coords[0:2]-eye_coords[2:4]
    dist_eyes = numpy.sqrt((dist_eyes**2).sum())
    return max(dist_left, dist_right) / dist_eyes
    
def face_detected(app_eye_coords, eye_coords, factor=0.25):
    rel_error = relative_error_detection(app_eye_coords, eye_coords)
    if rel_error < factor:
        return True
    else:
        return False

def FAR(faces_wrongly_detected, total_nofaces):
    return faces_wrongly_detected * 1.0 / total_nofaces

def FRR(faces_wrongly_rejected, total_faces):
    return faces_wrongly_rejected * 1.0 / total_faces

def purgueDetectedFacesEyes(detected_faces_eyes):
    if len(detected_faces_eyes) > 1:
        unique_faces_eyes = []
        unique_faces_eyes.append(detected_faces_eyes[0])
        for face_eye_coords in detected_faces_eyes:
            min_d = 100
            for face_eye_coords2 in unique_faces_eyes:
                error = relative_error_detection(face_eye_coords[4:8], face_eye_coords2[4:8])
                if error < min_d:
                    min_d = error
            if min_d > 0.25: #entries are different enough
                unique_faces_eyes.append(face_eye_coords)
        return unique_faces_eyes
    else:
        return list(detected_faces_eyes)

numpy.random.seed(123456)

verbose_pipeline = False

if verbose_pipeline:
    print "LOADING PIPELINE DESCRIPTION FILE"        

pipeline_filenames = cache.find_filenames_beginning_with(pipeline_base_dir, "Pipeline", recursion=False, extension=".txt")
if verbose_pipeline:
    print "%d pipelines found:"%len(pipeline_filenames), pipeline_filenames

if len(pipeline_filenames) <= 0:
    print "ERROR: No pipelines found in directory", pipeline_base_dir
    quit()

enable_select_pipeline = False
for i, pipeline_filename in enumerate(pipeline_filenames):
    pipeline_base_filename = string.split(pipeline_filename, sep=".")[0] #Remove extension          
#    (NetworkType, Imsize) = pipeline_info = cache_read.load_obj_from_cache(base_dir="/", base_filename=pipeline_base_filename, verbose=True)
    if verbose_pipeline:
        print "Pipeline %d: %s"%(i, pipeline_base_filename)
                    
if enable_select_pipeline ==True:
    selected_pipeline = int( raw_input("Please select a pipeline [0--%d]:"%(len(pipeline_filenames)-1) ))
else:
    selected_pipeline = 0

if verbose_pipeline:
    print "Pipeline %d was selected"%selected_pipeline           

pipeline_filename = pipeline_filenames[selected_pipeline]
#pipeline_base_filename = string.split(pipeline_filename, sep=".")[0]

enable_eyes = True

t0 = time.time()

pipeline_file = open(pipeline_filename, "rb")

num_networks = int(pipeline_file.readline())
if verbose_pipeline:
    print "Pipeline contains %d network/classifier pairs"%num_networks
#num_networks = 1

tmp_string = pipeline_file.readline()
tmp_strings = string.split(tmp_string, " ")
net_Dx = int(tmp_strings[0])
net_Dy = int(tmp_strings[1])
net_mins = float(tmp_strings[2])
net_maxs = float(tmp_strings[3])

#Now read data for eye networks
#This is the scale in which the image patches are generated from the input image (usually 64x64)
#Pixel functions use this scale
subimage_width = int(tmp_strings[4])
subimage_height = int(tmp_strings[5])

#This is the scale in which the labels are given (usually 128x128)
#Functions related to regression/classification use this scale
regression_width = int(tmp_strings[6])
regression_height = int(tmp_strings[7])

tmp_string = pipeline_file.readline()
tmp_strings = string.split(tmp_string, " ")
eye_Dx = int(tmp_strings[0])
eye_Dy = int(tmp_strings[1])
eye_mins = float(tmp_strings[2])
eye_maxs = float(tmp_strings[3])

#This is the scale in which the image patches are generated from the input image (usually 32x32)
#Pixel functions use this scale
eye_subimage_width = int(tmp_strings[4])
eye_subimage_height = int(tmp_strings[5])

#This is the scale in which the labels are given (usually 128x128)
#Functions related to regression/classification use this scale
eye_regression_width = int(tmp_strings[6])
eye_regression_height = int(tmp_strings[7])

#regression_width = regression_height = 128 #Regression data assumes subimage has this size

network_types = []
network_filenames = []
classifier_filenames = [] 
for i in range(num_networks):
    network_type = pipeline_file.readline().rstrip()
    network_types.append(network_type)
    network_filename = pipeline_file.readline().rstrip()[0:-5]
    network_filenames.append(network_filename)    
    classifier_filename = pipeline_file.readline().rstrip()[0:-5]
    classifier_filenames.append(classifier_filename)

if verbose_networks:    
    print "network types:", network_types
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

if command_line_interface:
    load_FRGC_images=False
        
if load_FRGC_images:
    frgc_metadata_file = "/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_2.0_Metadata_corrected.xml"
#   What about ppn files???
    dict_coordinate_data = frgc.load_FRGC_coordinate_data(metadata_file=frgc_metadata_file)       
    print "Entries in dict_coordinate_data=", len(dict_coordinate_data)

    frgc_biometric_base_dir = "/home/escalafl/workspace/test_SFA2/src/xml_source/"
    dict_biometric_signatures6 = frgc.load_FRGC_biometric_signatures(file_biometric_signatures=frgc_biometric_base_dir+"FRGC_Exp_2.0.6_Orig.xml")
    dict_biometric_signatures = dict_biometric_signatures6

##    dict_biometric_signatures5 = frgc.load_FRGC_biometric_signatures(file_biometric_signatures=frgc_biometric_base_dir+"FRGC_Exp_2.0.5_Orig.xml")
##    merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures5)
##    dict_biometric_signatures4 = frgc.load_FRGC_biometric_signatures(file_biometric_signatures=frgc_biometric_base_dir+"FRGC_Exp_2.0.4_Orig.xml")
##    merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures4)
##    dict_biometric_signatures3 = frgc.load_FRGC_biometric_signatures(file_biometric_signatures=frgc_biometric_base_dir+"FRGC_Exp_2.0.3_Orig.xml")
##    merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures3)
##    dict_biometric_signatures2 = frgc.load_FRGC_biometric_signatures(file_biometric_signatures=frgc_biometric_base_dir+"FRGC_Exp_2.0.2_Orig.xml")
##    merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures2)
##    dict_biometric_signatures1 = frgc.load_FRGC_biometric_signatures(file_biometric_signatures=frgc_biometric_base_dir+"FRGC_Exp_2.0.1_Orig.xml")
##    merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures1)    

    base_dir = "/local/tmp/FRGC-2.0-dist"
#    out_dir = "/local/tmp/escalafl/Alberto/FRGC_Normalized"
    image_filenames = []
    max_count = 1000
    frgc_original_coordinates = []

    count = 0
    for recording_id, recording_data in dict_coordinate_data.items():   
        (subject_id, LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y) = recording_data   
        (LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y) = map(int, (LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y))
        coordinates = (LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y)
    ###### code taken from xml_frgc_tools, see for details
#        print "RightEyeCenter_x=", RightEyeCenter_x
#        print "RightEyeCenter_x.__class__=", RightEyeCenter_x.__class__
        eyes_x_m = (RightEyeCenter_x + LeftEyeCenter_x) / 2.0
        eyes_y_m = (RightEyeCenter_y + LeftEyeCenter_y) / 2.0

        midpoint_eyes_mouth_x = (eyes_x_m + Mouth_x) / 2.0
        midpoint_eyes_mouth_y = (eyes_y_m + Mouth_y) / 2.0
        
        dist_eyes = numpy.sqrt((LeftEyeCenter_x - RightEyeCenter_x)**2 + (LeftEyeCenter_y - RightEyeCenter_y)**2) 
    
        #Triangle formed by the eyes and the mouth.
        height_triangle = numpy.sqrt((eyes_x_m - Mouth_x)**2 + (eyes_y_m - Mouth_y)**2) 
      
        #Assumes eye line is perpendicular to the line from eyes_m to mouth
        current_area = dist_eyes * height_triangle / 2.0
        desired_area = 37.0 * 42.0 / 2.0

        # if normalization_method == "mid_eyes_mouth":
        scale_factor =  numpy.sqrt(current_area / desired_area )
        #Warning, is it subimage or regression???
        #regression is fine: subimage is used only for the fisical sampling of the box, but its logical size is given by regression
#        ori_width = subimage_width*scale_factor 
#        ori_height = subimage_height*scale_factor
        ori_width = regression_width*scale_factor * 0.825
        ori_height = regression_height*scale_factor * 0.825

#WARNING, using subpixel coordinates!   
#        box_x0 = int(midpoint_eyes_mouth_x-ori_width/2)
#        box_x1 = int(midpoint_eyes_mouth_x+ori_width/2)
#        box_y0 = int(midpoint_eyes_mouth_y-ori_height/2)
#        box_y1 = int(midpoint_eyes_mouth_y+ori_height/2)
        box_x0 = midpoint_eyes_mouth_x-ori_width/2
        box_x1 = midpoint_eyes_mouth_x+ori_width/2
        box_y0 = midpoint_eyes_mouth_y-ori_height/2
        box_y1 = midpoint_eyes_mouth_y+ori_height/2
        #############
        more_coordinates = (midpoint_eyes_mouth_x, midpoint_eyes_mouth_y, box_x0, box_y0, box_x1, box_y1)
        all_coordinates = list(coordinates)+list(more_coordinates) # 8 coordinates +  6 coordinates
        if max_count != None and count > max_count:
            break

        int_coords = numpy.array(map(int, all_coordinates))
                
##        int_coords = []
##        for i in coordinates:
##            int_coords.append(int(i))
    #   
        if recording_id in dict_biometric_signatures:
#            print "Retrieving filename for", recording_id
            modality, file_name, file_format = dict_biometric_signatures[recording_id]
#            print "Filename=",file_name, "with file format", file_format
            
            image_filenames.append(base_dir + "/" + file_name)
#Warning, disabling int coordinates
#           frgc_original_coordinates.append(int_coords)
            frgc_original_coordinates.append(all_coordinates)
            count += 1
        else:
            print "Recording", recording_id, "missing biometric signature"
    frgc_original_coordinates = numpy.array(frgc_original_coordinates)



    if save_frgc_data:
        print "saving frgc_image_files and ground truth data"
        frgc_images_filename="frgc_images.txt"
        frgc_images_file = open(frgc_images_filename, "w")
        for i, filename in enumerate(image_filenames):
            frgc_images_file.write(filename+"\n")
            frgc_images_file.write("output/output%05d.txt\n"%i)
        frgc_images_file.close()

        frgc_groundtruth_filename="frgc_groundtruth.txt"
        frgc_groundtruth_file = open(frgc_groundtruth_filename, "w")
        for i in range(len(image_filenames)):
            f = frgc_original_coordinates[i]
            frgc_groundtruth_file.write("%d, %d, %d, %d, %f, %f, %f, %f\n"%(f[10], f[11], f[12], f[13], f[2], f[3], f[0], f[1]))
        frgc_groundtruth_file.close()
else:               
    image_filename_prefix = "image"
    image_filenames = cache.find_filenames_beginning_with(images_base_dir, image_filename_prefix, recursion=False, extension=".jpg")
    image_filenames.sort()

#This is used for speed benchmarking, for detection accuracy see below
fr_performance = {}

true_positives = numpy.zeros(num_networks, dtype='int')
active_boxes = numpy.zeros(num_networks, dtype='int')
num_boxes = numpy.zeros(num_networks, dtype='int')
false_positives = numpy.zeros(num_networks, dtype='int')
false_negatives = numpy.zeros(num_networks, dtype='int')
offending_images = []
for i in range(num_networks):
    offending_images.append([])

def usage():
    usage_txt = "\n ********************** \n USAGE INFORMATION \n \
    FaceDetect: A program for face detection from frontal images  \n \
    Program Usage: \n \
    A) python FaceDetect.py image_filename results_filename \n \
    example: $python FaceDetect.py images/image0000.jpg output/output0000.txt \n \
    many image formats are supported. The output file is a text file that has \n \
    zero or more lines of the form: left, top, right, bottom, xl, yl, xr, yr, \n \
    where each entry is an INTEGER value \n \n \
    B) (batch mode) python FaceDetect.py --batch batch_filename \n \
    where batch_filename is a text file containing many pairs of \n \
    image_filename/results_filename. \n \
    example $python FaceDetect.py --batch batch_images.txt \n \
    where batch_images.txt is: \n \
    images/image0000.jpg \n \
    output/output0000.txt \n \
    images/image0001.jpg \n \
    output/output0001.txt \n \
    \n \
    Batch mode is much faster than the one-filename approach because the \n \
    software libraries are loaded only once. \n \
    \n \
    Switches: \n \
    *the switch: --smallest_face=\%f allows to specify the (approximate) size of the \n \
    smallest face that might appear in terms of the \n \
    size of the corresponding input image. \n \
    example: $python FaceDetect.py images/image0000.jpg output/output0000.txt --smallest_face=0.2 \n \
    means that the smallest detected faces will have a size of at least \n \
    0.2*min(image_width, image_height) pixels. \n \
    The default value is 0.2. \n \n \
    *the switch right_screen_eye_first inverts the normal ordering of the eyes when writing the output file. \n \
    example: $python FaceDetect.py example_images/image000.jpg results/output000.txt \n \
    writes ... to sample_output/output0000.txt \n \
    while $python FaceDetect.py example_images/image000.jpg results/output000.txt --right_screen_eye_first \n \
    writes ... \n \n \
    \n \
    Bugs/Suggestions/Comments/Questions: please write to alberto.escalante\@ini.rub.de \n "
    print usage_txt

def read_batch_file(batch_filename):
    batch_file = open(batch_filename, "rb")
    lines = batch_file.readlines()
    batch_file.close()

    if len(lines)%2 != 0:
        print "Incorrect (odd) number of entries in batch file:"
        print "Each line in the batch file should be an input image_filename followed with another line containing the corresponding output_filename"
        exit(0)

    image_filenames = []
    output_filenames = []

    for i in range(len(lines)/2):
        image_filename = lines[2*i].rstrip()
        output_filename = lines[2*i+1].rstrip()
        image_filenames.append(image_filename)
        output_filenames.append(output_filename)
    return image_filenames, output_filenames

image_filenames = []
output_filenames = []
image_numbers = []
if command_line_interface:
    argv = None
    if argv is None:
        argv = sys.argv
    if len(argv) >= 3:
        try:
            opts, args = getopt.getopt(argv[1:], "b:", ["batch=","smallest_face=","right_screen_eye_first"])
            print "opts=", opts
            print "args=", args
            if len(args)==2:
                input_image = args[0]
                image_filenames = [input_image]
                image_numbers = numpy.arange(1)
                output_file = args[1]
                output_filenames = [output_file]
                print "Input image filename:", input_image
                print "Results filename:", output_file
            
            for opt, arg in opts:
                print "opt=", opt
                print "arg=", arg
                if opt in ('-b', '--batch'):
                    print "batch processing using file:", arg
                    if len(args)==2:
                        print "Error: input image / output file was already set: ", input_image, output_file
                        usage()
                        sys.exit(2)

                    image_filenames, output_filenames = read_batch_file(arg)
                    image_numbers = numpy.arange(len(image_filenames))
                    print image_filenames
                    print output_filenames
                elif opt in ('--smallest_face'):
                    smallest_face = float(arg) 
                    print "changing default size of smallest face to be found to %f * min(image_height, image_width)"%smallest_face
                elif opt in ('--right_screen_eye_first'):
                    right_screen_eye_first = True
                    print "changing default eye ordering. Now the eye most to the right on the screen appears on the output before the other eye"
        except getopt.GetoptError:
            print "Error"
            usage()
            sys.exit(2)
    else:
            usage()
            sys.exit(2)
    #quit()
        
else: #Use FRGC images
    print "Images:", image_filenames
    num_images = len(image_filenames)
    if num_images <= 0:
        raise Exception("No images Found")
    image_numbers = [34, 45, 47, 48, 49, 61, 74, 77]
    image_numbers = [762, 773, 777, 779, 850, 852, 871, 920, 921, 984]
    image_numbers = [871, 920, 921, 984]

#offending net 12/1000
#    image_numbers = [45, 47, 48, 49, 61, 102, 103, 104, 105, 136, 149, 150, 152, 153, 173, 175, 193, 196, 206, 230, 245, 261, 272, 282, 284, 292, 338, 380, 381, 411, 426, 427, 428, 437, 445, 489, 493, 499, 566, 591, 635, 636, 651, 661, 741, 750, 758, 762, 773, 777, 779, 850, 852, 871, 920, 921, 968, 984, 986]
    image_numbers = numpy.arange(0, 1000) #7,8,9
#image_numbers= [87]

    
    
    
#Sampling values with respect to regression_width and height values
#sampling_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
default_sampling_values = [0.475, 0.95, 1.9, 3.8]
default_sampling_values = [3.0] # [2.0, 3.8] # 2.0 to 4.0 for FRGC, 1.6

#Error measures for each image, sampling_value, network_number, box_number
#Include rel_ab_error, rel_scale_error, rel_eyepos_error, rel_eye_error

#Perhaps exchange sampling and image???
max_num_plots=10
num_plots=0

for im_number in image_numbers:
    if load_FRGC_images:
        frgc_image_coordinates = frgc_original_coordinates[im_number]
    #TODO: sampling values should be a function of the image size!!!
    detected_faces_eyes = []


    #for now images has a single image
    images = load_images([image_filenames[im_number]], format="L")
    #print images
    #quit()
    im_height = images[0].size[1]
    im_width = images[0].size[0]

    t2 = time.time()
#    benchmark.append(("Image loading a sampling value %f"%sampling_value, t2-t1))

    if adaptive_grid:
        min_side = min(im_height, im_width)
        min_box_side = min_side * smallest_face * 0.825 #smallest_face
        min_sampling_value = min_box_side * 1.0 / regression_width
        sampling_values = []
        sampling_value = min_sampling_value
        while (regression_width * sampling_value < im_width) and (regression_height * sampling_value < im_height):
            sampling_values.append(sampling_value)
            sampling_value *= grid_step
        max_box_side = min_side * 0.825 / numpy.sqrt(2)
        sampling_values.append(max_box_side / regression_width)
    else:
        sampling_values = default_sampling_values

#    print sampling_values
#    quit()



    for sampling_value in sampling_values:        
        
    #Patch width and height in image coordinates
        patch_width = regression_width * sampling_value 
        patch_height = regression_height * sampling_value 
        
        #TODO: add random initialization between 0 and net_Dx * 2.0 * patch_width/regression_width, the same for Y
        #These coordinates refer to the scaled image
        if verbose_networks:
            print "net_Dx=", net_Dx, "net_Dy=", net_Dy
        patch_horizontal_separation = net_Dx * 2.0 * patch_width/regression_width
        rest_horizontal = im_width - int(im_width / patch_horizontal_separation)*patch_horizontal_separation
        patch_vertical_separation = net_Dy * 2.0 * patch_height/regression_height
        rest_vertical = im_height - int(im_height / patch_vertical_separation)*patch_vertical_separation

        posX_values = numpy.arange(rest_horizontal/2, im_width-(patch_width-1), patch_horizontal_separation)
        posY_values = numpy.arange(rest_vertical/2, im_height-(patch_height-1), patch_vertical_separation)
        #A face must be detected by a box with a center distance and scale radio
        #interest points differ from center in these values
        max_Dx_diff = net_Dx * patch_width/regression_width
        max_Dy_diff = net_Dy * patch_height/regression_height
        min_scale_radio = 1/numpy.sqrt(2.0)
        max_scale_radio = numpy.sqrt(2.0)
        
        if verbose_networks:
            print "max_Dx_diff=", max_Dx_diff,"max_Dy_diff=",  max_Dy_diff 
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
        base_side = numpy.sqrt(base_magnitude)
        
        #print "subimage_coordinates", subimage_coordinates
        #base_estimation = orig_subimage_coordinates + 0.0
        #num_images is assumed to be 1 here, this might belong to the TODO
        orig_image_indices = numpy.zeros( 1 * orig_num_subimages, dtype="int")
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


        num_plots += 1
        if num_plots > max_num_plots:
            display_plots = False
        
        if display_plots:
            plots_created = True
            f0 = plt.figure()
            plt.suptitle("Iterative Face Detection")
            sel_image = 0
            p11 = plt.subplot(3,5,1)
            p12 = plt.subplot(3,5,2)
            p13 = plt.subplot(3,5,3)
            p14 = plt.subplot(3,5,4)
            p15 = plt.subplot(3,5,5)

            p21 = plt.subplot(3,5,6)
            p22 = plt.subplot(3,5,7)
            p23 = plt.subplot(3,5,8)
            p24 = plt.subplot(3,5,9)
            p25 = plt.subplot(3,5,10)

            p31 = plt.subplot(3,5,11)
            p32 = plt.subplot(3,5,12)
            p33 = plt.subplot(3,5,13)
            p34 = plt.subplot(3,5,14)
            p35 = plt.subplot(3,5,15)
        
            im_disp = numpy.asarray(images[0])
    #        print "len(images)",len(images)
    #        quit()
            p11.imshow(im_disp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
        #color=(r_color[sig], g_color[sig], b_color[sig])
        
            for ii, (x0, y0, x1, y1) in enumerate(orig_subimage_coordinates):
            #    p11.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=colors[ii] )
                p11.plot([x0, x1], [y0, y1], color=orig_colors[ii] )
        
        ###from matplotlib.lines import Line2D
        ##line = mpl.lines.Line2D([10,20,30,40,500], [10,40, 35, 20, 500],
        ##              linewidth=4, color='green', transform=f0.transFigure)
        ##f0.lines.append(line)
        
        if display_plots:
            subplots = [p12, p13, p14, p15, p21, p22, p23, p24, p25, p31, p32, p33, p34, p35]
        else:
            subplots = [None,]*14

        t3 = time.time()
        benchmark.append(("Window Creation", t3-t2))
        
        t_latest = t3
        for num_network in range(num_networks-2):
            #Get arrays
            subimages = extract_subimages(images, curr_image_indices, curr_subimage_coordinates, (subimage_width, subimage_height) )
            if len(subimages) > 0:
                num_boxes[num_network] += len(subimages)                
                subimages_arr = images_asarray(subimages)+0.0
                t_afterloading = time.time()
            
                sl = networks[num_network].execute(subimages_arr, benchmark=benchmark)
                if verbose_networks:
                    print "Network %d processed all subimages"%num_network
                reg_num_signals = classifiers[num_network].input_dim
                t_class = time.time()
                reg_out = classifiers[num_network].GaussianRegression(sl[:,0:reg_num_signals])
                
                #print "reg_out=", reg_out
            
#                if num_network in [0, 4, 8, 12]:
#                    network_type = "discrimination"
#                elif num_network in [1, 5, 9]:
#                    network_type = "posX"
#                elif num_network in [2, 6, 10]:
#                    network_type = "posY"
#                elif num_network in [3, 7, 11]:
#                    network_type = "scale"
                network_type = network_types[num_network][0:-1]

                if network_type == "Disc":
                    pass #WARNING!
                elif network_type == "PosX": #POS_X       
                    width = curr_subimage_coordinates[:, 2] - curr_subimage_coordinates[:, 0]
                    reg_out = reg_out * width / regression_width
            #        print "Regression Output scaled:", reg_out
            #        print "Correcting coordinates (X)"
                    curr_subimage_coordinates[:, 0] = curr_subimage_coordinates[:, 0] - reg_out  #X0
                    curr_subimage_coordinates[:, 2] = curr_subimage_coordinates[:, 2] - reg_out  #X1
                elif network_type == "PosY":     #POS_Y
                    height = curr_subimage_coordinates[:, 3] - curr_subimage_coordinates[:, 1]
                    reg_out = reg_out * height / regression_height
            #        print "Regression Output scaled:", reg_out
            #        print "Correcting coordinates (Y)"
                    curr_subimage_coordinates[:, 1] = curr_subimage_coordinates[:, 1] - reg_out  #Y0
                    curr_subimage_coordinates[:, 3] = curr_subimage_coordinates[:, 3] - reg_out  #Y1
                elif network_type == "Scale": #SCALE
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
                    print "Network type unknown!!!"
                    quit()
                    pass #regression does not need to modify subimage coordinates
            
                if network_type in ["PosX","PosY","Scale"]: 
                    magic = 1.2 #1.2 for group small
                    magic = 1.2 #Note, this affects false negatives!!! 1.0 is for default networks too strict!
                    magic_scale = 1.4
                    #out of image
                    out_of_borders_images = (curr_subimage_coordinates[:,0]<0) | (curr_subimage_coordinates[:,1]<0) | \
                    (curr_subimage_coordinates[:,2]>=im_width) | (curr_subimage_coordinates[:,3]>=im_height)      
                    #too large or small
                    
                    subimage_magnitudes = ((curr_subimage_coordinates[:,0:2] - curr_subimage_coordinates[:,2:4])**2).sum(axis=1)
                    subimage_sides = numpy.sqrt(subimage_magnitudes)
                    #sqrt(2)/2*orig_diagonal = 1/sqrt(2)*orig_diagonal < subimage_diagonal < sqrt(2)*orig_diagonal ???
                    too_large_small_images = (subimage_sides/base_side > max_scale_radio*magic_scale) | (subimage_sides/base_side < min_scale_radio/magic_scale)      
                    #too far away horizontally
            
                    subimage_deltas_x = (curr_subimage_coordinates[:,2] + curr_subimage_coordinates[:,0])/2 - (orig_subimage_coordinates[curr_orig_index][:,2] + orig_subimage_coordinates[curr_orig_index][:,0])/2
                    subimage_deltas_y = (curr_subimage_coordinates[:,3] + curr_subimage_coordinates[:,1])/2 - (orig_subimage_coordinates[curr_orig_index][:,3] + orig_subimage_coordinates[curr_orig_index][:,1])/2
            
                    
                    x_far_images = numpy.abs(subimage_deltas_x) > (max_Dx_diff * magic)
                    y_far_images = numpy.abs(subimage_deltas_y) > (max_Dy_diff * magic)
                    new_wrong_images = out_of_borders_images | too_large_small_images | x_far_images | y_far_images
    
                    debug_net_discrimination=False
                    if debug_net_discrimination:
#                        print "subimage_deltas_x is: ", subimage_deltas_x
#                        print "subimage_deltas_y is: ", subimage_deltas_y
                        print "wrong x_center is:", (curr_subimage_coordinates[:,2][x_far_images] + curr_subimage_coordinates[:,0][x_far_images])/2
                        print "wrong x_center was:", (orig_subimage_coordinates[:,2][curr_orig_index[x_far_images]] + orig_subimage_coordinates[:,0][curr_orig_index[x_far_images]])/2
                        print "wrong y_center is:", (curr_subimage_coordinates[:,3][y_far_images] + curr_subimage_coordinates[:,1][y_far_images])/2
                        print "wrong y_center was:", (orig_subimage_coordinates[:,3][curr_orig_index[y_far_images]] + orig_subimage_coordinates[:,1][curr_orig_index[y_far_images]])/2
                        print "new_wrong_images %d = out_of_borders_images %d + too_large_small_images %d + x_far_images %d + y_far_images %d" % \
                        (new_wrong_images.sum(), out_of_borders_images.sum(), too_large_small_images.sum(), x_far_images.sum(), y_far_images.sum())
                    else:
                        pass
                #TODO? Also discard too overlapping subimages??? yes, but do it after everything else works
                else: #Classifier
                    #TODO: Better face classifier, add more scales to each layer
                    #            cut_off_face = 0.25
                    if num_network == 0:
                        cut_off_face = 0.96 # 0.95
                    elif num_network == 4:
                        cut_off_face = 0.8 # 0.0
                    elif num_network == 8:
                        cut_off_face = 0.75 # -0.5
                    else:
                        cut_off_face = 0.65 # 0.5 too false negatives, 0.6 some, 0.7 some false positives. -0.90 
                        #-0.95 gives true_positive / false negative = 6/3
                        #-0.90 gives 7/3
                        #-0.99 gives 6/3
                        #even -0.99 seems fine!!!
    #                cut_off_face = 0.0   
                    new_wrong_images = reg_out >= cut_off_face
#                    print "new_wrong_images",new_wrong_images
            #TODO: Make shure box_side is defined if all boxes are eliminated after the first run
            
            #Update subimage patch information, on temporal variables
                new_num_subimages = curr_num_subimages - new_wrong_images.sum()
                new_subimage_coordinates = curr_subimage_coordinates[new_wrong_images==0]+0.0
                new_invalid_subimages = numpy.zeros(new_num_subimages, dtype='bool')
                new_image_indices = curr_image_indices[new_wrong_images==0] + 0
                new_orig_index = curr_orig_index[new_wrong_images==0] + 0
            

                if verbose_networks:            
                    print "%d / %d valid images"%(new_num_subimages, curr_num_subimages)
            #Overwrite current values
                curr_num_subimages = new_num_subimages
                curr_subimage_coordinates = new_subimage_coordinates
                curr_invalid_subimages = new_invalid_subimages
                curr_image_indices = new_image_indices 
                curr_orig_index = new_orig_index
            
            
                subplot = subplots[num_network]
                if subplot!=None:
                    subplot.imshow(im_disp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
                    for j, (x0, y0, x1, y1) in enumerate(curr_subimage_coordinates):
                #        if invalid_subimages[j] == False and False:
                #            subplot.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=orig_colors[curr_orig_index[j]] )
                #        if invalid_subimages[j] == False:
                        subplot.plot([x0, x1], [y0, y1], color=orig_colors[curr_orig_index[j]] )
                        
            #    print "subimage_coordinates", subimage_coordinates
            
                t_new = time.time()
                benchmark.append(("Pipeline Image Loading Step %d"%num_network, t_afterloading-t_latest))
                benchmark.append(("Pipeline Network Step %d"%num_network, t_class-t_afterloading))
                benchmark.append(("Pipeline Classifier Step %d"%num_network, t_new-t_class))
            
                t_latest = t_new
        
            ##print "Done loading subimages_train: %d Samples"%len(subimages_train)
            ##flow, layers, benchmark, Network = cache.load_obj_from_cache(network_hash, network_base_dir, "Network", verbose=True)   
            ##print "Done loading network: " + Network.name
        
                display_face = True
                display_aface = True
        
#            TODO: Make sure this works when nothing was found at all inside an image

            #Display Faces to find:
            if load_FRGC_images:
                sampled_face_coordinates = frgc_image_coordinates
                (el_x, el_y, er_x, er_y, n_x, n_y, m_x, m_y, fc_x, fc_y, b_x0, b_y0, b_x1, b_y1) = sampled_face_coordinates
                eye_coords = numpy.array([el_x, el_y, er_x, er_y])
                if display_face and display_plots:
                    #Face Box
                    subplot.plot([b_x0, b_x1, b_x1, b_x0, b_x0], [b_y0, b_y0, b_y1, b_y1, b_y0], "r")
                    #Left eye, right eye and face center
                    subplot.plot([el_x, er_x, fc_x], [el_y, er_y, fc_y], "ro")

            
            #For each face on the image,now there is only one
            #For each remaining image patch
            #Compute FRR, FAR, Error
                box_detected = False
                face_detected = False
                for j in range(len(curr_subimage_coordinates)):
        ##            print "********************"
                    orig_sub_coords = orig_subimage_coordinates[curr_orig_index[j]]
                    (ab_x0, ab_y0, ab_x1, ab_y1) = curr_sub_coords = curr_subimage_coordinates[j]
                    afc_x = (ab_x0 + ab_x1)/2.0
                    afc_y = (ab_y0 + ab_y1)/2.0
            
                    bcenter_x_orig = (orig_sub_coords[0]+orig_sub_coords[2])/2.0
                    bcenter_y_orig = (orig_sub_coords[1]+orig_sub_coords[3])/2.0
            
                    bcenter_x = (ab_x0 + ab_x1)/2.0
                    bcenter_y = (ab_y0 + ab_y1)/2.0
    
                           
    #WARNING!!!
    #                box_side =  numpy.sqrt(numpy.abs((b_x1-b_x0) * (b_y1-b_y0))) #of the real face_sampled face box, equals 0.825
    #                abox_side =  numpy.sqrt(numpy.abs((ab_x1-ab_x0) * (ab_y1-ab_y0)))
    #                box_side_orig = numpy.sqrt((orig_sub_coords[2]-orig_sub_coords[0])**2+(orig_sub_coords[3]-orig_sub_coords[1])**2)
                    box_side =  numpy.abs(b_x1-b_x0) #side of the real face_sampled face box, equals 0.825
                    abox_side =  numpy.abs(ab_x1-ab_x0)
                    box_side_orig =numpy.abs(orig_sub_coords[2]-orig_sub_coords[0])
    
                    #Errors in image pixels
                    bx_error_orig = fc_x - bcenter_x_orig 
                    by_error_orig = fc_y - bcenter_y_orig
                    bx_error = fc_x - bcenter_x 
                    by_error = fc_y - bcenter_y
                    #Errors in regression image pixels
                    rel_bx_error = (bx_error / box_side) * regression_width
                    rel_by_error = (by_error / box_side) * regression_height      
            
                    scale_error = box_side / abox_side - 1.0
                    #Error with respect to the goal sampling value of 0.825
                    rel_scale_error = scale_error * 0.825        
                    # rel_scale_error = 0.825 / box_side * abox_side - 0.825
            
                    (ael_x, ael_y, aer_x, aer_y) = sampled_app_eye_coords = compute_approximate_eye_coordinates(curr_sub_coords, face_sampling=0.825)
                    app_eye_coords = sampled_app_eye_coords 
                    #Error in image pixels
                    rel_eyes_pix_error = (app_eye_coords - eye_coords) / box_side * regression_width
                    #Normalized eye error, goal is a relative error < 0.25
                    rel_eye_error = relative_error_detection(app_eye_coords, eye_coords)
            
            
        ##            print "bx_error_orig = %f/%f/%f"%(bx_error_orig, max_Dx_diff, 1.0)   
        ##            print "by_error_orig = %f/%f/%f"%(by_error_orig, max_Dy_diff, 1.0)
        ##    #        print "bx_error = %f/%f/%f"%(bx_error, max_Dx_diff, 1.0)   
        ##    #        print "by_error = %f/%f/%f"%(by_error, max_Dy_diff, 1.0)
        ##            print "rel_eyes_pix_error = ", rel_eyes_pix_error
        ##            
        ##            #relative errors are errors in the original scales of 128 x 128 pixels & true scale = 0.825
        ##            print "rel_bx_error = ", rel_bx_error, "pixels"   
        ##            print "rel_by_error = ", rel_by_error, "pixels"
        ##            print "rel_scale_error =", rel_scale_error, "deviating from 0.825"
        ##            #relative eye error is normalized to the distance between the eyes, should be at most 0.25 for true detection
        ##            print "rel_eye_error =", rel_eye_error        
                    
                    debug_resp_box=False
        
                    #Face is within this box?
                    if numpy.abs(bx_error_orig)<max_Dx_diff and numpy.abs(by_error_orig)<max_Dy_diff and \
                    box_side / box_side_orig > min_scale_radio and box_side / box_side_orig < max_scale_radio:
                        #Bingo, this is responsible of detecting the face
                        if debug_resp_box:
                            print "Responsible box active:",
                            print "box orig_sub_coords=", orig_sub_coords
                            print "box curr_sub_coords=", curr_sub_coords
                        if box_detected == True:
                            print "WTF, face box was already detected!!!"
                        box_detected = True
                        active_boxes[num_network] += 1
                        #Error measures for each image, sampling_value, network_number, box_number
                        #Include rel_bx_error, rel_by_error, rel_scale_error, rel_eye_error, rel_eyes_pix_error,
                        fr_performance[im_number, sampling_value, num_network] = (rel_bx_error, rel_by_error, rel_scale_error, rel_eye_error, rel_eyes_pix_error)
        
                        if debug_resp_box:
                            print "frgc_original_eye_coordinates[0][0,4]=", eye_coords
                            print "fc=", fc_x, fc_y
                            print "bx_error=", bx_error
                            print "by_error=", by_error
                            print "app_eye_coords=", app_eye_coords
                            print "rel_eye_error =", rel_eye_error
                        if rel_eye_error < 0.25:
                            face_detected = True
                            print "Face was properly detected"
                            true_positives[num_network] += 1
                        else:
                            print "Face was NOT properly detected"
                            false_positives[num_network] += 1
                        
                        if display_aface and display_plots:
                            #Face Box
                            subplot.plot([ab_x0, ab_x1, ab_x1, ab_x0, ab_x0], [ab_y0, ab_y0, ab_y1, ab_y1, ab_y0], "g")
                            #Left eye, right eye and face center
                            subplot.plot([ael_x, aer_x, afc_x], [ael_y, aer_y, afc_y], "go")
                    else:
    #                    if num_network==0:
    #                        print "%f < %f? and %f < %f? and %f < %f/%f=%f < %f ?"%(bx_error_orig, max_Dx_diff, by_error_orig, max_Dy_diff, \
    #                                                                            max_scale_radio, box_side, box_side_orig, box_side /box_side_orig, min_scale_radio)
        ##                pass
        ##                print "box false positive:",
        ##                print orig_sub_coords
        ##                print "box moved to:", curr_sub_coords
        ##                print "face was originally centered in", frgc_original_coordinates[0]
                        false_positives[num_network] += 1
            #      C) If yes count as positive detection, otherwise failed detection and compute FRR     
            #      D) For remaining boxes, all of them are false detections, use them compared to total number of boxes for FAR
                if num_network==0:
                    pass
                    #print "%f < %f/%f=%f < %f?"%(max_scale_radio, box_side, box_side_orig, box_side /box_side_orig, min_scale_radio)
                if face_detected:
                    pass
                    #print "Face was correctly detected at least once"
                elif box_side / box_side_orig > min_scale_radio and box_side / box_side_orig < max_scale_radio:
                    #print "Face was not detected at all"
                    false_negatives[num_network] += 1
                    if not box_detected: 
                        offending_images[num_network].append(im_number)
                else:
                    pass #warning!
#                    print "No face present"


        eyes_coords_orig = numpy.zeros((len(curr_subimage_coordinates), 4))
        eyesL_box_orig = numpy.zeros((len(curr_subimage_coordinates), 4))
        eyesR_box_orig = numpy.zeros((len(curr_subimage_coordinates), 4))        
        for i, box_coords in enumerate(curr_subimage_coordinates):
#            detected_faces.append(compute_approximate_eye_coordinates(eye_coords, face_sampling=0.825))
            eyes_coords_orig[i], eyesL_box_orig[i], eyesR_box_orig[i] = compute_approximate_eye_boxes_coordinates(box_coords, face_sampling=0.825)
            display_eye_boxes = True
            if display_eye_boxes and display_plots:
                #left eye box
                bel_x0, bel_y0, bel_x1, bel_y1 =  eyesL_box_orig[i]
                subplot.plot([bel_x0, bel_x1, bel_x1, bel_x0, bel_x0], [bel_y0, bel_y0, bel_y1, bel_y1, bel_y0], "b")
                #right eye box
#                ber_x0, ber_y0, ber_x1, ber_y1 =  eyesR_box_orig[i]
#                subplot.plot([ber_x0, ber_x1, ber_x1, ber_x0, ber_x0], [ber_y0, ber_y0, ber_y1, ber_y1, ber_y0], "b")
 

        #Left eye only!
        eyesL_box = eyesL_box_orig
        #print "eyesL_box=",eyesL_box
        for num_network in [num_networks-1, num_networks-2]:
            eyeL_subimages = extract_subimages(images, curr_image_indices, eyesL_box, (eye_subimage_width, eye_subimage_height) )
            if len(eyeL_subimages) > 0:
                subimages_arr = images_asarray(eyeL_subimages)+0.0
                sl = networks[num_network].execute(subimages_arr, benchmark=benchmark)    
                #print "Network %d processed all subimages"%(num_networks-2)
                reg_num_signals = classifiers[num_network].input_dim
                reg_out = classifiers[num_network].GaussianRegression(sl[:,0:reg_num_signals])
    
                if network_types[num_network] == "EyeLX": #POS_X     
                    #print "EyeLX"  
                    eyes_box_width = numpy.abs(eyesL_box[:,2]-eyesL_box[:,0])
                    reg_out = reg_out * eyes_box_width / eye_regression_width
                    eyesL_box[:, 0] = eyesL_box[:, 0] - reg_out  #X0
                    eyesL_box[:, 2] = eyesL_box[:, 2] - reg_out  #X1
                elif network_types[num_network] == "EyeLY": #POS_Y       
                    #print "EyeLY"  
                    eyes_box_height = numpy.abs(eyesL_box[:,3]-eyesL_box[:,1])
                    reg_out = reg_out * eyes_box_height / eye_regression_height
                    eyesL_box[:, 1] = eyesL_box[:, 1] - reg_out  #Y0
                    eyesL_box[:, 3] = eyesL_box[:, 3] - reg_out  #Y1
                else:
                    print "Unknown network type!", network_types[num_network]
                    quit()

        #Right eye only!
        #Swap horizontal coordinates
        eyesRhack_box = eyesR_box_orig + 0.0
        eyesRhack_box[:,0] = eyesR_box_orig[:,2]
        eyesRhack_box[:,2] = eyesR_box_orig[:,0]
        
        #print "eyesRhack_box=",eyesRhack_box
        for num_network in [num_networks-1, num_networks-2]:
            eyeR_subimages = extract_subimages(images, curr_image_indices, eyesRhack_box, (eye_subimage_width, eye_subimage_height) )
            if len(eyeR_subimages) > 0:
                subimages_arr = images_asarray(eyeR_subimages)+0.0
                sl = networks[num_network].execute(subimages_arr, benchmark=benchmark)    
                #print "Network %d processed all subimages"%(num_networks-2)
                reg_num_signals = classifiers[num_network].input_dim
                reg_out = classifiers[num_network].GaussianRegression(sl[:,0:reg_num_signals])
    
                if network_types[num_network] == "EyeLX": #POS_X     
                    #print "EyeLX"  
                    eyes_box_width = numpy.abs(eyesRhack_box[:,2]-eyesRhack_box[:,0])
                    reg_out = reg_out * eyes_box_width / eye_regression_width
                    eyesRhack_box[:, 0] = eyesRhack_box[:, 0] + reg_out  #X0
                    eyesRhack_box[:, 2] = eyesRhack_box[:, 2] + reg_out  #X1
                elif network_types[num_network] == "EyeLY": #POS_Y       
                    #print "EyeLY"  
                    eyes_box_height = numpy.abs(eyesRhack_box[:,3]-eyesRhack_box[:,1])
                    reg_out = reg_out * eyes_box_height / eye_regression_height
                    eyesRhack_box[:, 1] = eyesRhack_box[:, 1] - reg_out  #Y0
                    eyesRhack_box[:, 3] = eyesRhack_box[:, 3] - reg_out  #Y1
                else:
                    print "Unknown network type!", network_types[num_network]
                    quit()

        #Undo horizontal swap of coordinates
        eyesR_box = eyesRhack_box + 0.0
        eyesR_box[:,0] = eyesRhack_box[:,2]
        eyesR_box[:,2] = eyesRhack_box[:,0]

        eyesL_coords = (eyesL_box[:,0:2]+eyesL_box[:,2:4])/2
        eyesR_coords = (eyesR_box[:,0:2]+eyesR_box[:,2:4])/2

        display_eye_boxes = True            
        if display_eye_boxes and display_plots:
            for el_x, el_y in eyesL_coords:
                subplot.plot([el_x], [el_y], "bo")
            for el_x, el_y in eyesR_coords:
                subplot.plot([el_x], [el_y], "yo")
                             
        for i, box_coords in enumerate(curr_subimage_coordinates):            
            eyes_coords = eyes_coords_orig[i]
            box_eyes_coords = numpy.array([box_coords[0], box_coords[1], box_coords[2], box_coords[3], eyesL_coords[i][0], eyesL_coords[i][1], eyesR_coords[i][0], eyesR_coords[i][1]])           
            detected_faces_eyes.append(box_eyes_coords)
        
    detected_faces_eyes = detected_faces_eyes
    #print "Faces/Eyes before purge:", detected_faces_eyes
    detected_faces_eyes_purgued = purgueDetectedFacesEyes(detected_faces_eyes)
    print "Faces/Eyes after purge:", detected_faces_eyes_purgued

    if write_results:
        print "writing face/eyes positions to disk. File:", output_filenames[im_number]
        fd = open(output_filenames[im_number], 'a')
        for face_eyes_coords in detected_faces_eyes_purgued:
            int_feyes = numpy.round(face_eyes_coords[0:8])
            if right_screen_eye_first: 
                fd.write("%d, %d, %d, %d, %d, %d, %d, %d \n"%(int_feyes[0],int_feyes[1],int_feyes[2],int_feyes[3], int_feyes[4],int_feyes[5],int_feyes[6],int_feyes[7]))
            else:
                fd.write("%d, %d, %d, %d, %d, %d, %d, %d \n"%(int_feyes[0],int_feyes[1],int_feyes[2],int_feyes[3], int_feyes[6],int_feyes[7],int_feyes[4],int_feyes[5]))                
        fd.close()
        
#        for (msg,t) in benchmark:
#            print msg, " ", t
#        
#        print "Total detection time: ", t_latest - t3, "patches processed:", orig_num_subimages

print "fr_performance[im_number, sampling_value, network] = (rel_bx_error, rel_by_error, rel_scale_error, rel_eye_error, rel_eyes_pix_error)"
for (im_number, sampling_value, net_num) in fr_performance.keys():
    if net_num == 3 and im_number >= 0:
        print (im_number, sampling_value, net_num), "=>",
        print fr_performance[(im_number, sampling_value, net_num)]    

for i in range(num_networks):
    print "Offending images after net %d:"%i, offending_images[i]

if load_FRGC_images:
    for i in range(num_networks):
        number_of_boxes= false_positives[i] + true_positives[i] #number of boxes AFTER the network, before the network num_boxes
        num_faces = true_positives[i] + false_negatives[i]
        num_nofaces = number_of_boxes - num_faces
        print "After Network %d: %d true_positives %d active_boxes %d initial boxes / %d false_positives, %d false_negatives: FAR=%f, FRR=%f"%(i, true_positives[i], active_boxes[i], \
            num_boxes[i], false_positives[i], false_negatives[i], FAR(false_positives[i], num_nofaces), FRR(false_negatives[i], num_faces))
else:
    for i in range(num_networks):
        print "After Network %d: %d initial boxes "%(i, num_boxes[i])


display_errors=True
if display_errors:
    for selected_network in range(num_networks):
#    for sampling_value in sampling_values:
        rel_bx_errors = []
        rel_by_errors = []
        rel_scale_errors = []
        rel_eye_errors = []
        rel_eyes_pix_errors = []
        
        print "************ Errors after network ", selected_network
        for (im_number, sampling_value, net_num) in fr_performance.keys():
            if net_num == selected_network:
                (rel_bx_error, rel_by_error, rel_scale_error, rel_eye_error, rel_eyes_pix_error) = fr_performance[(im_number, sampling_value, net_num)]
                rel_bx_errors.append(rel_bx_error)
                rel_by_errors.append(rel_by_error)
                rel_scale_errors.append(rel_scale_error)
                rel_eye_errors.append(rel_eye_error)
                rel_eyes_pix_errors.append(rel_eyes_pix_error)
        
        rel_bx_errors = numpy.array(rel_bx_errors)
        rel_by_errors = numpy.array(rel_by_errors)
        rel_scale_errors = numpy.array(rel_scale_errors)
        rel_eye_errors = numpy.array(rel_eye_errors)
        rel_eyes_pix_errors = numpy.array(rel_eyes_pix_errors)
        
        rel_bx_errors_std = rel_bx_errors.std()
        rel_bx_errors_std = rel_bx_errors.std()
        rel_by_errors_std = rel_by_errors.std()
        rel_scale_errors_std = rel_scale_errors.std()
        rel_eye_errors_std = rel_eye_errors.std()
        rel_eyes_pix_errors_std = rel_eyes_pix_errors.std(axis=0)
        
        rel_bx_errors_mean = rel_bx_errors.mean()
        rel_bx_errors_mean = rel_bx_errors.mean()
        rel_by_errors_mean = rel_by_errors.mean()
        rel_scale_errors_mean = rel_scale_errors.mean()
        rel_eye_errors_mean = rel_eye_errors.mean()
        rel_eyes_pix_errors_mean = rel_eyes_pix_errors.mean(axis=0)
        
        rel_bx_errors_rmse = numpy.sqrt((rel_bx_errors**2).mean())
        rel_bx_errors_rmse = numpy.sqrt((rel_bx_errors**2).mean())
        rel_by_errors_rmse = numpy.sqrt((rel_by_errors**2).mean())
        rel_scale_errors_rmse = numpy.sqrt((rel_scale_errors**2).mean())
        rel_eye_errors_rmse = numpy.sqrt((rel_eye_errors**2).mean())
        rel_eyes_pix_errors_rmse = numpy.sqrt((rel_eyes_pix_errors**2).mean(axis=0))        
        
        print "rel_bx_errors_mean =", rel_bx_errors_mean,
        print "rel_by_errors_mean =", rel_by_errors_mean,
        print "rel_scale_errors_mean =", rel_scale_errors_mean,
        print "rel_eye_errors_mean =", rel_eye_errors_mean
        print "rel_eyes_pix_errors_mean =", rel_eyes_pix_errors_mean
        
        print "rel_bx_errors_std =", rel_bx_errors_std,
        print "rel_by_errors_std =", rel_by_errors_std,
        print "rel_scale_errors_std =", rel_scale_errors_std,
        print "rel_eye_errors_std =", rel_eye_errors_std
        print "rel_eyes_pix_errors_std =", rel_eyes_pix_errors_std
        
        print "rel_bx_errors_rmse =", rel_bx_errors_rmse,
        print "rel_by_errors_rmse =", rel_by_errors_rmse,
        print "rel_scale_errors_rmse =", rel_scale_errors_rmse
        print "rel_eye_errors_rmse =", rel_eye_errors_rmse,
        print "rel_eyes_pix_errors_rmse =", rel_eyes_pix_errors_rmse


if plots_created:
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
####    -u login
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