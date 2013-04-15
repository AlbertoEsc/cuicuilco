

import numpy
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import Image
import os
import glob
import random
import time

#returns a 2D numpy array with the images in image_files
#image_files:list with the filenames of the images to be loaded
#pixelsampling_x and _y: used for sampling the input images, must be an integer
def load_images(image_files, image_width, image_height, subimage_width, subimage_height, \
                    pixelsampling_x = 1, pixelsampling_y = 1, subimage_first_row=0, subimage_first_column=0, \
                    add_noise = True, convert_format="L", translations_x=None, translations_y=None, trans_sampled=True, verbose=False):
    t0 = time.time()
    num_images = len(image_files)
    print "Loading ", num_images, "Images: width=%d, height=%d, subimage_width=%d,subimage_height=%d"%(image_width,image_height, subimage_width,subimage_height)

    if translations_x is None:
        translations_x = numpy.zeros(num_images)
    if translations_y is None:
        translations_y = numpy.zeros(num_images)

    if trans_sampled is True:
        translations_x = translations_x * pixelsampling_x
        translations_y = translations_y * pixelsampling_y
        
    if convert_format == "L":
        pixel_dimensions = 1
    elif convert_format == "RGB":
        print ":)"
        pixel_dimensions = 3
    else:
        err = "Don't know the pixel_dimensions for image format: ", convert_format
        raise Exception(err)

    subimages = numpy.zeros((num_images, subimage_width * subimage_height * pixel_dimensions))
    for act_im_num, image_file in enumerate(image_files):
        im = Image.open(image_file)   
        im = im.convert(convert_format)
        im_arr = numpy.asarray(im)
            
        im_small = im_arr[subimage_first_row   +translations_y[act_im_num]:(subimage_first_row   +translations_y[act_im_num]+subimage_height*pixelsampling_y):pixelsampling_y, \
                           subimage_first_column+translations_x[act_im_num]:(subimage_first_column+translations_x[act_im_num]+subimage_width* pixelsampling_x):pixelsampling_x].astype(float)
    
        if add_noise == True:
            if convert_format == "L":
                noise = numpy.random.normal(loc=0.0, scale=0.05, size=(subimage_height, subimage_width))
            elif convert_format == "RGB":
                noise = numpy.random.normal(loc=0.0, scale=0.05, size=(subimage_height, subimage_width, 3))
            else:
                err = "Don't kwon how to generate noise for image format: ", convert_format
                raise Exception(err)
            im_small = im_small*1.0 + noise
        subimages[act_im_num] = im_small.flatten()
        del im_small
        del im_arr
        del im
    t1 = time.time()
    if verbose:
        print num_images, " Images loaded in %0.3f ms"% ((t1-t0)*1000.0)
    return subimages

