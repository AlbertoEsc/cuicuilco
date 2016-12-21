#Basic functions for loading sequences of images
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 15 July 2009

import numpy
#import scipy
import scipy.misc

#import matplotlib as mpl
#import matplotlib.pyplot as plt
import PIL
from PIL import Image
from PIL import ImageOps
#####import Image
#import ImageOps
import os
import glob
#import random
import sfa_libs
import time
import struct
import math
import multiprocessing

#im.save(fullname)

# Image.BILINEAR, Image.NEAREST, BICUBIC,  ANTIALIAS
interpolation_format = Image.BICUBIC
#format_nearest = Image.NEAREST
def rgb_to_irb_array(im_rgb_array):
    igb_3d_array = im_rgb_array.copy()
    intensity = igb_3d_array.mean(axis=2)
    igb_3d_array[:,:,1] = (igb_3d_array[:,:,1]-intensity+170.0)/(intensity+1.0)*256.0/340/3
    igb_3d_array[:,:,2] = (igb_3d_array[:,:,2]-intensity+170.0)/(intensity+1.0)*256.0/340/3
    igb_3d_array[:,:,0] = intensity
    return igb_3d_array

def create_image_filenames(im_seq_base_dir, slow_signal=0, ids=[0], expressions=[0], morphs=[0], poses=[0], lightings=[0], step=1, offset=0, verbose=False):
    if verbose:
        print "creating image filenames..." 
    parameters = list(sfa_libs.product(ids, expressions, morphs, poses, lightings))    
    parameters.sort(lambda x, y: cmp(x[slow_signal],y[slow_signal]))

    #print "parameters=", parameters
    selection = parameters[offset::step]

    #print "selection=", selection
    
    filenames = []
    for (id, ex, morph, pose, lighting) in selection:
        file = im_seq_base_dir + "/random%03d_e%d_c%d_p%03d_i%d.tif"%(id, ex, morph, pose, lighting)
        filenames.append(file)

    #print "filenames=", filenames
    return filenames


def create_image_filenames2(im_seq_base_dir, slow_signal=0, ids=[0], ages=[999], genders=[999], racetweens = [999], expressions=[0], morphs=[0], poses=[0], lightings=[0], step=1, offset=0, verbose=False):
    if verbose:
        print "creating image filenames..." 
    parameters = list(sfa_libs.product(ids, ages, genders, racetweens, expressions, morphs, poses, lightings))    
    parameters.sort(lambda x, y: cmp(x[slow_signal],y[slow_signal]))

#    print "parameters=", parameters
    selection = parameters[offset::step]

    #print "selection=", selection
    
    filenames = []
    for (id, age, gender, racetween, ex, morph, pose, lighting) in selection:
#        file = im_seq_base_dir + "/random%03d_e%d_c%d_p%03d_i%d.tif"%(id, ex, morph, pose, lighting)
        file = im_seq_base_dir + "/output%03d_a%03d_g%03d_rt%03d_e%d_c%d_p%d_i%d.tif"%(id, age, gender, racetween, ex, morph, pose, lighting)
        filenames.append(file)

    #print "filenames=", filenames
    return filenames

def create_image_filenames3(im_seq_base_dir, im_base_name, slow_signal=0, ids=[0], ages=[999], genders=[999], \
                            racetweens = [999], expressions=[0], morphs=[0], poses=[0], lightings=[0], step=1, \
                            offset=0, verbose=False, len_ids=3, image_postfix=".tif"):
    if verbose:
        print "creating image filenames..." 

    parameters = list(sfa_libs.product(ids, ages, genders, racetweens, expressions, morphs, poses, lightings))    
    parameters.sort(lambda x, y: cmp(x[slow_signal],y[slow_signal]))

#    print "parameters=", parameters
    selection = parameters[offset::step]

    #print "selection=", selection
    
    filenames = []
    for (id, age, gender, racetween, ex, morph, pose, lighting) in selection:
        file = im_seq_base_dir + "/" + im_base_name
        if id != None:
            if len_ids == 3:
                file += "%03d"%id
            elif len_ids == 4:
                file += "%04d"%id
            elif len_ids == 5:
                file += "%05d"%id
            else:
                file += "%d"%id
        if age != None:
            file += "_a%03d"%age
        if gender != None:
            file += "_g%03d"%gender
        if racetween != None:
            file += "_rt%03d"%racetween
        if ex != None:
            file += "_e%d"%ex
        if morph != None:
            file += "_c%d"%morph
        if pose != None:
            file += "_p%d"%pose
        if lighting != None:
            if im_base_name == 'car':
                file += "_%04d"%lighting
            else:
                file += "_i%d"%lighting
        
#        file += ".tif"                        
        file += image_postfix  
        filenames.append(file)
    if verbose:
        print "filenames=", filenames
    return filenames

#TODO:Use clip here???? faster???
def cutoff(x, min_val, max_val):
    y1 = numpy.where(x >= min_val, x, min_val)
    y1 = numpy.where(y1 <= max_val, y1, max_val)
    return y1

#TODO: Perhaps it is faster to compute the subimage, and then get as array and then compute mean and std()?????
#TODO:Also consider normalization after croping
#TODO:And also consider a much more powerful histogram equalization!!!
def simple_normalization_from_coords(im, relevant_left, relevant_right, relevant_top, relevant_bottom, show_points=False, obj_avg=0.0, obj_std=0.2):
    im_ar = numpy.asarray(im)
    #print "im_ar input to norm_coords=", im_ar
    #print "im_ar.shape=", im_ar.shape
    #print "im_ar.min()=", im_ar.min()
    #print "im_ar.max()=", im_ar.max()
    if len(im_ar.shape)==3 and im_ar.shape[2] == 3:
        RGB=True
    else:
        RGB=False
    if RGB:   
        im_block = im_ar[relevant_top:relevant_bottom, relevant_left:relevant_right, :]
    else:
        im_block = im_ar[relevant_top:relevant_bottom, relevant_left:relevant_right]

    mean = im_block.mean()
    std = im_block.std()

    if RGB: #each component in an rgb image ranges from 0 to 255 (integer)
        if obj_avg is None:
            obj_avg = mean/255.0 - 0.5
        if obj_std is None:
            obj_std = std/255.0 + 0.0       

        im_ar = 255*obj_std*(im_ar-mean)/std+127.5+255*obj_avg #255*std = 50
        im_ar = cutoff(im_ar, 0, 255)
        #quit()
    else: #each component in an L image ranges from 0 to 1 (float), apparently only for saving/display, otherwise is also 0-255
        #print "looks fine"
        #quit()
        if obj_avg is None:
            obj_avg = mean/255.0 - 0.5
        if obj_std is None:
            obj_std = std/255.0 + 0.0	   
        #print "mean=", mean, "std=", std
        #print "obj_avg=", 0.5+obj_avg, "obj_std=", obj_std
        im_ar = obj_std*(im_ar-mean)/std + 0.5 + obj_avg #std = 0.2, obj_avg=0.0
        im_ar = cutoff(im_ar, 0, 1)
        #print "im_ar=", im_ar
        #quit()
    if show_points: #This is useful to see which region is considered for normalization, as well as to see/contrast the extreme intensities
        if RGB:
            im_ar[relevant_top+1,relevant_left+1,0:3]=0 #G
            im_ar[relevant_top+1,relevant_left+1,1]=255
                    
            im_ar[relevant_bottom-1,relevant_right-1,0:3]=0 #B
            im_ar[relevant_bottom-1,relevant_right-1,2]=255
                
            im_ar[relevant_top+1,relevant_right-1,0:2]=0 #R
            im_ar[relevant_top+1,relevant_right-1,0]=255
        
            im_ar[relevant_bottom-1,relevant_left+1,1:3]=255 #GB
            im_ar[relevant_bottom-1,relevant_left+1,0]=0
        else:       
            im_ar[relevant_top+1,relevant_left+1]=0                    
            im_ar[relevant_bottom-1,relevant_right-1]=0 
            im_ar[relevant_top+1,relevant_right-1]= 0 
            im_ar[relevant_bottom-1,relevant_left+1]= 1 

#    print width, height, left, right, top, bottom 
#    quit()
    if RGB:
        im_out = scipy.misc.toimage(im_ar, mode="RGB")
    else:
        im_out = scipy.misc.toimage(im_ar, mode="L")        
    return im_out

def simple_normalization_GTSRB(im, relevant_width, relevant_height):
    width, height = im.size[0], im.size[1]

    #Notice that im is centered to the object, but might be much larger!
    relevant_left = int((width-relevant_width)/2.0)
    relevant_right = relevant_left + int(relevant_width)
    
    #Bias downward
    db=0.3
    relevant_top = int((height-relevant_height)/2.0+relevant_height*db)
    relevant_bottom = relevant_top + int(relevant_height)

    return simple_normalization_from_coords(im, relevant_left, relevant_right, relevant_top, relevant_bottom, show_points=False, obj_avg=0.0, obj_std=0.2)

def simple_normalization_GenderMultiply(im):
    im_ar = numpy.asarray(im)
    
    if len(im_ar.shape)==3 and im_ar.shape[2] == 3:
        RGB=True
    else:
        RGB=False
    
    #print "Originally: im_ar.min(), im_ar.max() , im_ar.max(), im_ar.mean()=", im_ar.min(), im_ar.max(), im_ar.shape, im_ar.mean()
    if RGB:
        goal_mean = 255*0.08
    else:
        goal_mean = 0.08
    im_ar_avg = im_ar.mean()

    im_ar = im_ar*(goal_mean/im_ar_avg)
    
    if RGB:
        im_ar[im_ar < 0] = 0
        im_ar[im_ar > 255] = 255
    else:
        im_ar[im_ar < 0] = 0
        im_ar[im_ar > 1] = 1        
    #print "im_ar.min(), im_ar.max() , im_ar.mean()=", im_ar.min(), im_ar.max(), im_ar.mean()

    if RGB:
        im_out = scipy.misc.toimage(im_ar, mode="RGB")
    else:
        im_out = scipy.misc.toimage(im_ar, mode="L") 
    return im_out

def simple_normalization_Age(im, relevant_width, relevant_height, obj_avg=0.0, obj_std = 0.2):
    width, height = im.size[0], im.size[1]

    #Notice that im is centered to the object, but might be much larger!
    relevant_left = int((width-relevant_width)/2.0+0.5)
    relevant_right = relevant_left + int(relevant_width+0.5)
    
    relevant_top = int((height-relevant_height)/2.0+0.5)
    relevant_bottom = relevant_top + int(relevant_height+0.5)
    #print relevant_left, relevant_right, relevant_top, relevant_bottom
    return simple_normalization_from_coords(im, relevant_left, relevant_right, relevant_top, relevant_bottom, show_points=False, obj_avg=obj_avg, obj_std=obj_std)

#TODO: Add Scale/Zoom instead of pixel_sampling????
#use then: im.transform(out_size, Image.EXTENT, data), see xml_frgc_tests
#pixelsampling_x = 1.0 / pixel_zoom_x
#pixelsampling_y = 1.0 / pixel_zoom_y
#trans_sampled: if true translations are done at the sampled image level, otherwise at the original image
def load_image_data_monoprocessor(image_files, image_array, image_width, image_height, subimage_width, subimage_height, \
                    pre_mirroring_flags=False, pixelsampling_x = 2, pixelsampling_y = 2, subimage_first_row=0, subimage_first_column=0,  \
                    add_noise = True, convert_format="L", translations_x=None, translations_y=None, trans_sampled=True, rotation=None, contrast_enhance = None, obj_avgs=None, obj_stds=None, background_type=None, color_background_filter=None, subimage_reference_point = 0, verbose=False):
    t0 = time.time()
    num_images = len(image_files)
    print "Loading ", num_images, 
  
    if isinstance(image_width, (numpy.float, numpy.float64, numpy.int)):
        print "Images: width=%d, height=%d"%(image_width,image_height), 
    else:
        print "Images[0]: width=%d, height=%d"%(image_width[0],image_height[0]), 

    print " subimage_width=%d,subimage_height=%d"%(subimage_width,subimage_height)

    out_size = (subimage_width, subimage_height) 

    #print "pre_mirroring_flags=", pre_mirroring_flags
    #print "translations_x=", translations_x
    #print "subimage_first_column=", subimage_first_column
    if translations_x is None:
        translations_x = numpy.zeros(num_images)
    if translations_y is None:
        translations_y = numpy.zeros(num_images)

    
    print "color_background_filter is:", color_background_filter
    
    if convert_format == "L":
        pixel_dimensions = 1
        print "image_loading: pixel_format=L"
    elif convert_format == "RGB":
        print "image_loading: pixel_format=RGB"
        pixel_dimensions = 3
    else:
        err = "Don't know the pixel_dimensions for image format: ", convert_format
        raise Exception(err)
    #quit()
    
#    if rotation==None:
#        er = "rotation disabled :("
#        raise Exception(er)
    
    if isinstance(image_width, (numpy.float, numpy.float64, numpy.int)):
        image_width = numpy.ones(num_images) * image_width

    if isinstance(image_height, (numpy.float, numpy.float64, numpy.int)):
        image_height = numpy.ones(num_images) * image_height

    if isinstance(pre_mirroring_flags, (bool, numpy.bool)):
        pre_mirroring_flags = numpy.zeros(num_images, dtype="bool") | pre_mirroring_flags

    if isinstance(subimage_first_column, (numpy.float, numpy.float64, numpy.int)):
        subimage_first_column = numpy.ones(num_images) * subimage_first_column  

    if isinstance(subimage_first_row, (numpy.float, numpy.float64, numpy.int)):
        subimage_first_row = numpy.ones(num_images) * subimage_first_row  
       
    if isinstance(pixelsampling_x, (numpy.float, numpy.float64, numpy.int)):
        pixelsampling_x = numpy.ones(num_images) * pixelsampling_x 
#    print "pixelsampling_x[0]", pixelsampling_x[0]
    
    if isinstance(pixelsampling_y, (numpy.float, numpy.float64, numpy.int)):
        pixelsampling_y = numpy.ones(num_images) * pixelsampling_y 

    if isinstance(translations_x, (numpy.float, numpy.float64, numpy.int)):
        translations_x = numpy.ones(num_images) * translations_x 

    if isinstance(translations_y, (numpy.float, numpy.float64, numpy.int)):
        translations_y = numpy.ones(num_images) * translations_y

    if isinstance(rotation, (numpy.float, numpy.float64, numpy.int)):
        rotation = numpy.ones(num_images) * rotation

    if isinstance(obj_avgs, (numpy.float, numpy.float64, numpy.int)):
        obj_avgs = numpy.ones(num_images) * obj_avgs

    if obj_avgs is None:
        obj_avgs = [None, ]*num_images

    if isinstance(obj_stds, (numpy.float, numpy.float64, numpy.int)):
        obj_stds = numpy.ones(num_images) * obj_stds

    #Translations are given in sampled coordinates and thus here converted to original image coordinates
    if trans_sampled == True:
        translations_x = translations_x * pixelsampling_x
        translations_y = translations_y * pixelsampling_y

#    print "subimage_first_column.shape=", subimage_first_column.shape
#    print "translations_x.shape=", translations_x.shape
#    print "len(image_files)=", len(image_files)
#    print "subimage_first_column[0]=", subimage_first_column[0]
#    print "translations_x[0]=", translations_x[0]
#    print "subimage_width=", subimage_width
#    print "len(image_files)=", len(image_files)
        
#    act_im_num=0
#    print subimage_first_column[act_im_num] + translations_x[act_im_num]
#    print subimage_first_row[act_im_num] + translations_y[act_im_num]
#    print subimage_first_column[act_im_num] + translations_x[act_im_num] + subimage_width * pixelsampling_x[act_im_num]
#    print subimage_first_row[act_im_num] + translations_y[act_im_num] + subimage_height * pixelsampling_y[act_im_num]
#    print subimage_width
#    print "pixelsampling_x[0]", pixelsampling_x[act_im_num]
            
    if background_type is None or color_background_filter is None:
        subimages = numpy.zeros((num_images, subimage_width * subimage_height * pixel_dimensions))
        
        unique_image_files = list(set(image_files))
        act_im_num_indices = {}
        for act_im_num, image_file in enumerate(image_files):
            if image_file in act_im_num_indices: #.keys():
                act_im_num_indices[image_file].append(act_im_num)
            else:
                act_im_num_indices[image_file] = [act_im_num]
        print "loading/processing"
        for image_file in unique_image_files:
            #print "Loading unique image", image_file
            if image_array is None:
                im_orig = Image.open(image_file)   
            else:
    #                print "image_file=",image_file
    #                print "image_array=",image_array
                im_orig = scipy.misc.toimage(image_array[image_file], mode='L')
            for act_im_num in act_im_num_indices[image_file]:              
                #print "Computing image for act_im_num:", act_im_num

#        if True:
#            for act_im_num, image_file in enumerate(image_files):
                im = im_orig.copy()                 
                if pre_mirroring_flags[act_im_num]:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                    
                #Warning, make this conditional            
                #im = ImageOps.autocontrast(im, 2)
                            
                x0 = subimage_first_column[act_im_num] + translations_x[act_im_num]
                y0 = subimage_first_row[act_im_num] + translations_y[act_im_num]
                x1 = subimage_first_column[act_im_num] + translations_x[act_im_num] + subimage_width * pixelsampling_x[act_im_num]
                y1 = subimage_first_row[act_im_num] + translations_y[act_im_num] + subimage_height * pixelsampling_y[act_im_num]
                if rotation != None:
                    delta_ang = rotation[act_im_num]
                else:
                    delta_ang = None
    
    #            print "IMAGE: ", image_file           
                subimage_coordinates = (x0,y0,x1,y1)
    #            print "subimage_coordinates: ", subimage_coordinates
    #            print "pixelsampling_x[act_im_num], pixelsampling_y[act_im_num] = ", pixelsampling_x[act_im_num], pixelsampling_y[act_im_num]
    #            quit()
                
    #            if verbose:
    #                print "subimage_coordinates =", (x0,y0,x1,y1)
    #WARNING!!!!! IS THERE AN OFFSET HERE???? CHECK ROTATION CODE!!!! THIS IS WEIRD, AS USUAL!!!   
                subimage_coordinates = (x0+1,y0+1,x1+1,y1+1)          
    #            print "subimage_coordinates =", subimage_coordinates
                #BUGFIX: This should be correct
                subimage_coordinates = (x0,y0,x1,y1)
                ori_width = x1 - x0
                ori_height = y1 - y0
                
                if x1<0 or y1<0 or x0>=im.size[0] or y0>=im.size[1]:
                    err = "Image Loading Failed: Subimage seriously out of Image: ", image_file
                    print "subimage_coordinates =", (x0,y0,x1,y1)
                    print "Image size: im.size[0], im.size[1] = ", im.size[0],  im.size[1]
                    raise Exception(err)
    
    #            if x0<0 or y0<0 or x1>=im.size[0] or y1>=im.size[1]:
    #                err = "Image Loading Failed: Subimage out of Image: ", image_file
    #                print "subimage_coordinates =", (x0,y0,x1,y1)
    #                print "Image size: im.size[0], im.size[1] = ", im.size[0],  im.size[1]
    #                raise Exception(err)  
                #print "AAA"
                if delta_ang != None:
                    rotation_center_x = (x0+x1-1)/2.0
                    rotation_center_y = (y0+y1-1)/2.0

                    # print "BBB"

                    integer_rotation_center = True #and False
                    if integer_rotation_center:
                        rotation_center_x_int = int(rotation_center_x + 0.5)
                        rotation_center_y_int = int(rotation_center_y + 0.5)
                    else:
                        rotation_center_x_int = rotation_center_x
                        rotation_center_y_int = rotation_center_y
                    
                    rotation_window_width = 2 * max(im.size[0]-1-rotation_center_x_int+0.5, rotation_center_x_int+0.5) #Result is always integer and odd
                    rotation_window_height = 2 * max(im.size[1]-1-rotation_center_y_int+0.5, rotation_center_y_int+0.5) #Result is always integer and odd
                    
                    if integer_rotation_center == False:
                        rotation_window_width = int(rotation_window_width + 0.5)
                        rotation_window_height = int(rotation_window_height + 0.5)

                    Delta_x = rotation_center_x - rotation_center_x_int
                    Delta_y = rotation_center_y - rotation_center_y_int

                    rotation_angle_rad =  -delta_ang * numpy.pi/180.0 
                    Delta_x_rotated = Delta_x  * numpy.cos(rotation_angle_rad) - Delta_y * numpy.sin(rotation_angle_rad)
                    Delta_y_rotated = Delta_y  * numpy.cos(rotation_angle_rad) + Delta_x * numpy.sin(rotation_angle_rad)
    
                    rotation_crop_x0 = rotation_center_x_int-(rotation_window_width-1)/2.0
                    rotation_crop_y0 = rotation_center_y_int-(rotation_window_height-1)/2.0
                    rotation_crop_x1 = rotation_center_x_int+(rotation_window_width-1)/2.0+1
                    rotation_crop_y1 = rotation_center_y_int+(rotation_window_height-1)/2.0+1              
                    #Here crop size should not loose any pixel from the rotation window
                    crop_size = (int(rotation_window_width+0.5), int(rotation_window_height+0.5)) #Avoiding type problem, even though variables are actually integer 
                    
                    crop_coordinates = (rotation_crop_x0, rotation_crop_y0, rotation_crop_x1, rotation_crop_y1)
                    #print crop_size
    #                print crop_coordinates
    
                    #TODO:Reconsider using always bicubic interpolation                
                    #if rotation_window_width + rotation_window_height < (rotation_crop_x1 - rotation_crop_x0) + (rotation_crop_y1 - rotation_crop_y0):
                    #    im_crop_first = im.transform(crop_size, Image.EXTENT, crop_coordinates, format_nearest)
                    #else:
                    im_crop_first = im.transform(crop_size, Image.EXTENT, crop_coordinates) #, Image.BICUBIC)
                    
                    if image_file.endswith("EyeNZ2Dummy.jpg"):
                        print "SAVING image 1"
                        print "subimage_first_column[act_im_num]=", subimage_first_column[act_im_num]
                        print "ori_width, ori_height =", ori_width, ori_height
                        print "subimage_coordinates =", subimage_coordinates
                        print "rotation_center_x_int, rotation_center_y_int=", rotation_center_x_int, rotation_center_y_int
                        print "Delta_x, Delta_y=",Delta_x, Delta_y
                        im_crop_first.save("cropped.png")
                    
                    #print delta_ang
                    im_rotated = rotate_improved(im_crop_first, delta_ang, Image.BICUBIC)

                    if image_file.endswith("EyeNZ2Dummy.jpg"):
                        print "SAVING image 2"
                        im_rotated.save("rotated.png")
                    
                    #Warning, make this conditional, use equalize instead, autocontrast is too weak!           
                    if contrast_enhance in ["AgeContrastEnhancement_Avg_Std", "AgeContrastEnhancement", "GTSRBContrastEnhancement", "ContrastGenderMultiply","AgeContrastEnhancement15", "AgeContrastEnhancement20", "AgeContrastEnhancement25"]:
                        #im_contrasted = ImageOps.autocontrast(im_rotated, 2)
                        if contrast_enhance ==  "AgeContrastEnhancement_Avg_Std":
                            obj_center_width = (x1-x0 + 1)*0.7
                            obj_center_height = (y1-y0 + 1)*0.7
                            im_contrasted = simple_normalization_Age(im_rotated, obj_center_width, obj_center_height, obj_avg=obj_avgs[act_im_num], obj_std=obj_stds[act_im_num])
                        elif contrast_enhance == "AgeContrastEnhancement":
                            face_center_width = (x1-x0 + 1)*0.7
                            face_center_height = (y1-y0 + 1)*0.7
                            im_contrasted = simple_normalization_Age(im_rotated, face_center_width, face_center_height, obj_std=0.2)
                        elif contrast_enhance == "AgeContrastEnhancement15":
                            face_center_width = (x1-x0 + 1)*0.7
                            face_center_height = (y1-y0 + 1)*0.7
                            im_contrasted = simple_normalization_Age(im_rotated, face_center_width, face_center_height, obj_std=0.15)
                        elif contrast_enhance == "AgeContrastEnhancement20":
                            face_center_width = (x1-x0 + 1)*0.7
                            face_center_height = (y1-y0 + 1)*0.7
                            im_contrasted = simple_normalization_Age(im_rotated, face_center_width, face_center_height, obj_std=0.20)
                        elif contrast_enhance == "AgeContrastEnhancement25":
                            face_center_width = (x1-x0 + 1)*0.7
                            face_center_height = (y1-y0 + 1)*0.7
                            im_contrasted = simple_normalization_Age(im_rotated, face_center_width, face_center_height, obj_std=0.25)
                        elif contrast_enhance == "GTSRBContrastEnhancement":
                            object_width = (x1-x0 + 1)*0.5
                            object_height = (y1-y0 + 1)*0.7
#                            im_contrasted = simple_normalization_GTSRB(im_rotated, object_width*0.9, object_height * 0.9)
                            im_contrasted = simple_normalization_GTSRB(im_rotated, object_width, object_height)
                        elif contrast_enhance == "ContrastGenderMultiply":
                            im_contrasted = simple_normalization_GenderMultiply(im_rotated)
                            im_contrasted = simple_normalization_GenderMultiply(im_contrasted)
                            im_contrasted = simple_normalization_GenderMultiply(im_contrasted)
                        else:
                            print "Unhandled contrast enhance method"
                            quit()
                    elif contrast_enhance is None:
                        im_contrasted = im_rotated
                        Delta_x_rotated = Delta_y_rotated = 0
                    else:
                        print "Unknown contrast_enhance method!!!, ", contrast_enhance
                        quit()
                        im_contrasted = im_rotated
                    
                    if image_file.endswith("EyeNZ2Dummy.jpg"):
                        print "SAVING image 3"
                        im_contrasted.save("im_contrasted.png")

                    
                    center_x_geometric = (crop_size[0]-1)/2.0
                    center_y_geometric = (crop_size[1]-1)/2.0
                    #print "After rotation: geometric center_x, center_y=", center_x_geometric, center_y_geometric
    
                    new_center_x = center_x_geometric + Delta_x_rotated #Real value!
                    new_center_y = center_y_geometric + Delta_y_rotated #Real value!
                    #print "After rotation: new_center_x, new_center_y=", new_center_x, new_center_y
                    #print "center_x, center_y=", center_x, center_y
                    #print "ori_width, ori_height=", ori_width, ori_height
    
                    #WWW removed int
                    x0 = (new_center_x-(ori_width-1)/2.0) #WWW added -1
                    x1 = (new_center_x+(ori_width-1)/2.0)+1 #WWW added -1 and +1
                    y0 = (new_center_y-(ori_height-1)/2.0) #WWW added -1
                    y1 = (new_center_y+(ori_height-1)/2.0)+1 #WWW added -1 and +1
                        
                    #print "transform_coords = x0, x1, y0, y1=", x0, x1, y0, y1
                    subimage_coordinates = (x0,y0,x1,y1)
    
                    #WARNIGN!
                    #rotation_center_x = (crop_size[0]-1)/2.0
                    #rotation_center_y = (crop_size[1]-1)/2.0
    #                rotation_center_x = (rotation_window_width)/2.0
    #                rotation_center_y = (rotation_window_height)/2.0
                    #subimage_coordinates = (rotation_center_x-(x1-x0-1)/2.0, rotation_center_y-(y1-y0-1)/2.0, 
                    #                        rotation_center_x+(x1-x0-1)/2.0+1, rotation_center_y+(y1-y0-1)/2.0+1)
                    
                    #WARNING! Correct this in other parts where cropping is done!!!
                    #NOTE: subimage_coordinates goes from 0 to width!!! typical of python notation. Thus, the rightmost pixel is ignored, similarly downmost.
                    if contrast_enhance == "PostEqualizeHistogram":
                        #Resize, then equalize
                        im_out = im_contrasted.transform(out_size, Image.EXTENT, subimage_coordinates, interpolation_format)    #W interpolation_format       format_nearest
                        im_out = ImageOps.equalize(im_out)
                    elif contrast_enhance == "SmartEqualizeHistogram":
                        #Crop, then equalize, then resize
    #                    print "8]",
                        out_size_crop = (x1-x0+1,y1-y0+1)
                        im_out = im_contrasted.transform(out_size_crop, Image.EXTENT, subimage_coordinates, interpolation_format)    #W interpolation_format       format_nearest
                        im_out = ImageOps.equalize(im_out)
                        crop_coordinates = (0,0,x1-x0+1,y1-y0+1)
                        im_out = im_out.transform(out_size, Image.EXTENT, crop_coordinates, interpolation_format)    #W interpolation_format       format_nearest
                    else:
                        im_out = im_contrasted.transform(out_size, Image.EXTENT, subimage_coordinates, interpolation_format)
                    
                    if image_file.endswith("EyeNZ2Dummy.jpg"):
                        print "SAVING image 4"
                        print "center_x_geometric, center_y_geometric=", center_x_geometric, center_y_geometric
                        print "Delta_x_rotated, Delta_y_rotated were=", Delta_x_rotated, Delta_y_rotated
                        print "subimage_coordinates were=", subimage_coordinates
                        im_out.save("im_out.png")

                else: #No rotation is applied at all
                    #TODO: Factorize this code with the code above where rotation is used
                    #Warning, make this conditional, use equalize instead, autocontrast is too weak!           
                    im_rotated = im
                    if contrast_enhance in ["AgeContrastEnhancement_Avg_Std", "AgeContrastEnhancement", "GTSRBContrastEnhancement", "ContrastGenderMultiply", "AgeContrastEnhancement15", "AgeContrastEnhancement20", "AgeContrastEnhancement25"]:
                        #im_contrasted = ImageOps.autocontrast(im_rotated, 2)
                        if contrast_enhance ==  "AgeContrastEnhancement_Avg_Std":
                            obj_center_width = (x1-x0 + 1)*0.7
                            obj_center_height = (y1-y0 + 1)*0.7
                            im_contrasted = simple_normalization_Age(im_rotated, obj_center_width, obj_center_height, obj_avg=obj_avgs[act_im_num], obj_std=obj_stds[act_im_num])
                        elif contrast_enhance == "AgeContrastEnhancement":
                            face_center_width = (x1-x0 + 1)*0.7
                            face_center_height = (y1-y0 + 1)*0.7
                            im_contrasted = simple_normalization_Age(im_rotated, face_center_width, face_center_height, obj_std=0.2)
                        elif contrast_enhance == "AgeContrastEnhancement15":
                            face_center_width = (x1-x0 + 1)*0.7
                            face_center_height = (y1-y0 + 1)*0.7
                            im_contrasted = simple_normalization_Age(im_rotated, face_center_width, face_center_height, obj_std=0.15)
                        elif contrast_enhance == "AgeContrastEnhancement20":
                            face_center_width = (x1-x0 + 1)*0.7
                            face_center_height = (y1-y0 + 1)*0.7
                            im_contrasted = simple_normalization_Age(im_rotated, face_center_width, face_center_height, obj_std=0.20)
                        elif contrast_enhance == "AgeContrastEnhancement25":
                            face_center_width = (x1-x0 + 1)*0.7
                            face_center_height = (y1-y0 + 1)*0.7
                            im_contrasted = simple_normalization_Age(im_rotated, face_center_width, face_center_height, obj_std=0.25)
                        elif contrast_enhance == "GTSRBContrastEnhancement":
                            object_width = (x1-x0 + 1)*0.5
                            object_height = (y1-y0 + 1)*0.7
                            im_contrasted = simple_normalization_GTSRB(im_rotated, object_width, object_height)
                        elif contrast_enhance == "ContrastGenderMultiply":
                            im_contrasted = simple_normalization_GenderMultiply(im_rotated)                
                            im_contrasted = simple_normalization_GenderMultiply(im_contrasted)
                            im_contrasted = simple_normalization_GenderMultiply(im_contrasted)               
                        else:
                            print "Incorrect method (unknown constant enhancement)", contrast_enhance
                            quit()
                    elif contrast_enhance is None:
                        im_contrasted = im_rotated
                    else:
                        print "Unknown contrast_enhance method!!!, ", contrast_enhance
                        quit()
                    im_out = im_contrasted.transform(out_size, Image.EXTENT, subimage_coordinates, interpolation_format )          #W interpolation_format format_nearest
                    if contrast_enhance == "PostEqualizeHistogram":
                        im_out = ImageOps.equalize(im_out)
                        
                im_out = im_out.convert(convert_format)
                #Warning!"
                #print out_size
                #im_out = simple_normalization(im_out, out_size[0]*0.9, out_size[1]*0.5)
    
                im_small = numpy.asarray(im_out)
               
                if add_noise == True and False:
                    ex = "Was not supposed to add noise here"
                    raise Exception(ex)
                    if convert_format == "L":
                        #Using uniform noise for speed over normal noise
                        high = (3**0.5)*0.05 #standard deviation 0.05
                        noise = numpy.random.uniform(low=-high, high=high, size=(subimage_height, subimage_width))
                        #noise = numpy.random.normal(loc=0.0, scale=0.05, size=(subimage_height, subimage_width))
                    elif convert_format == "RGB":
                        noise_amplitude = 5.0 #RGB has range from 0 to 255
                        high = (3**0.5)*noise_amplitude #standard deviation 0.05
                        noise = numpy.random.uniform(low=-high, high=high, size=(subimage_height, subimage_width, 3))
#                        noise = numpy.random.normal(loc=0.0, scale=noise_amplitude, size=(subimage_height, subimage_width, 3))
                    else:
                        err = "Don't kwon how to generate noise for given image format: ", convert_format
                        raise Exception(err)
                    im_small = im_small*1.0 + noise
                    #irb_transform = True
                    #if convert_format == "RGB" and irb_transform:
                    #    im_small = rgb_to_irb_array(im_small)

                subimages[act_im_num] = im_small.flatten()
                del im_small
                del im_out
                del im
            del im_orig
    elif background_type in ["black", "blue"]:
        subimages = numpy.zeros((num_images, subimage_width * subimage_height))
        for act_im_num, image_file in enumerate(image_files):
            # print "image_file=",image_file
            im = Image.open(image_file)  
            im_rgb = im.convert("RGB") 

            x0 = subimage_first_column[act_im_num] + translations_x[act_im_num]
            y0 = subimage_first_row[act_im_num] + translations_y[act_im_num]
            x1 = subimage_first_column[act_im_num] + translations_x[act_im_num] + subimage_width*pixelsampling_x[act_im_num]
            y1 = subimage_first_row[act_im_num] + translations_y[act_im_num] + subimage_height*pixelsampling_y[act_im_num]
            subimage_coordinates = (x0,y0,x1,y1)

            if x0<0 or y0<0 or x1>=im.size[0] or y1>=im.size[1]:
                err = "Image Loading Failed: Subimage out of Image"
                raise Exception(err)
 
            im_out_rgb = im_rgb.transform(out_size, Image.EXTENT, subimage_coordinates, interpolation_format)
            im_small_rgb = numpy.asarray(im_out_rgb)+0.0

            if background_type == "black":
                background_mask = (numpy.asarray(im_small_rgb) == [0,0,0]).all(axis=2)
            elif background_type == "blue":
                background_mask = (numpy.asarray(im_small_rgb) == [76,76,204]).all(axis=2)
            else:
                ex = "Invalid background_type: ", str(background_type)
                raise Exception(ex)

            if convert_format != "RGB":
                im = im.convert(convert_format)
                im_out = im.transform(out_size, Image.EXTENT, subimage_coordinates, interpolation_format)
                im_small = numpy.asarray(im_out)+0.0
#               print ":|"
            else:
                im = im_rgb
                im_small = im_small_rgb              
                        
            background = random_filtered_noise2D(im_small.shape, color_background_filter, min=0, max=255)           
            if verbose:
                print "background.shape = ", background.shape
                print "background_mask.shape = ", background_mask.shape
                print "Z",
            
#            print "im_small:", im_small
            im_small[background_mask] = background [background_mask]
                
            if add_noise == True and False:
                ex = "Was not supposed to add noise here"
                raise Exception(ex)
                noise = numpy.random.normal(loc=0.0, scale=0.025, size=(subimage_height, subimage_width))
                im_small = im_small*1.0 + noise
            subimages[act_im_num] = im_small.flatten()
            del im, im_rgb, im_out_rgb, im_small, im_small_rgb
    else:
        print "Invalid 'background_type' and/or 'color_background_filter'"
              
    if add_noise == True: #Add noise to the whole images loaded
        if convert_format == "L":
            #Using uniform noise for speed over normal noise
            high = (3**0.5)*0.05 #standard deviation 0.05
            noise = numpy.random.uniform(low=-high, high=high, size=(num_images, subimage_width*subimage_height))
                        #noise = numpy.random.normal(loc=0.0, scale=0.05, size=(subimage_height, subimage_width))
        elif convert_format == "RGB":
            noise_amplitude = 5.0 #RGB has range from 0 to 255
            high = (3**0.5)*noise_amplitude #standard deviation 0.05
            noise = numpy.random.uniform(low=-high, high=high, size=(num_images, subimage_height* subimage_width * 3))
        else:
            err = "Don't know how to generate noise for given image format: ", convert_format
            raise Exception(err)
        subimages += noise     
        
        
    if convert_format == "RGB":
        subimages = subimages / 256.0
    t1 = time.time()
    if verbose:
        print num_images, " Images loaded in %0.3f ms"% ((t1-t0)*1000.0)
    return subimages


def vectorize(scalar_vector, length, preserve_None=False):
    if scalar_vector is None:
        if preserve_None:
            return [None,]*length
        else:
            return numpy.zeros(length)
    elif isinstance(scalar_vector, (bool, numpy.bool)):
        return numpy.zeros(length, dtype="bool") | scalar_vector
    elif isinstance(scalar_vector, (numpy.float, numpy.float64, numpy.int)):
        return numpy.ones(length) * scalar_vector
    else:
        return scalar_vector

#PARALLEL
def load_image_data(image_files, image_array, image_width, image_height, subimage_width, subimage_height, \
                    pre_mirroring_flags=False, pixelsampling_x = 2, pixelsampling_y = 2, subimage_first_row=0, subimage_first_column=0,  \
                    add_noise = True, convert_format="L", translations_x=None, translations_y=None, trans_sampled=True, rotation=None, contrast_enhance = None, obj_avgs=None, obj_stds=None, background_type=None, color_background_filter=None, subimage_reference_point = 0, verbose=False):
    num_proc = os.getenv("CUICUILCO_IMAGE_LOADING_NUM_PROC") 
    if num_proc:
        num_proc = int(num_proc)
        print "setting num_proc for image loading to", num_proc
    else:
        er = "CUICUILCO_IMAGE_LOADING_NUM_PROC undefined"
        raise Exception(er)

    if image_array is not None:
        num_proc = 1
    if num_proc == 1 or not num_proc: #single processor loading
        print "defaulting to single processor image loading"
        return load_image_data_monoprocessor(image_files, image_array, image_width, image_height, subimage_width, subimage_height, \
                                             pre_mirroring_flags, pixelsampling_x, pixelsampling_y, subimage_first_row, subimage_first_column,  \
                                             add_noise, convert_format, translations_x, translations_y, trans_sampled, rotation, contrast_enhance, obj_avgs, obj_stds, background_type, color_background_filter, subimage_reference_point, verbose)

    t0 = time.time()
    num_images = len(image_files)
    print "Loading ", num_images, 

    if isinstance(image_width, (numpy.float, numpy.float64, numpy.int)):
        print "Images: width=%d, height=%d"%(image_width,image_height), 
    else:
        print "Images[0]: width=%d, height=%d"%(image_width[0],image_height[0]), 

    print " subimage_width=%d,subimage_height=%d"%(subimage_width,subimage_height)

    #out_size = (subimage_width, subimage_height) 

    #print "pre_mirroring_flags=", pre_mirroring_flags
    #print "translations_x=", translations_x
    #print "subimage_first_column=", subimage_first_column
    translations_x = vectorize(translations_x, num_images)
    translations_y = vectorize(translations_y, num_images)
   
    print "color_background_filter is:", color_background_filter
    
    if convert_format == "L":
        pixel_dimensions = 1
        print "image_loading: pixel_format=L"
    elif convert_format == "RGB":
        print "image_loading: pixel_format=RGB"
        pixel_dimensions = 3
    else:
        err = "Don't know the pixel_dimensions for image format: ", convert_format
        raise Exception(err)
    #quit()
    
    image_width = vectorize(image_width, num_images)
    image_height = vectorize(image_height, num_images)
    pre_mirroring_flags = vectorize(pre_mirroring_flags, num_images)
    subimages_first_column = vectorize(subimage_first_column, num_images)
    subimages_first_row = vectorize(subimage_first_row, num_images)
    #print "subimages_first_row=", subimages_first_row
    pixelsamplings_x = vectorize(pixelsampling_x, num_images)
    pixelsamplings_y = vectorize(pixelsampling_y, num_images)
    translations_x = vectorize(translations_x, num_images)
    translations_y = vectorize(translations_y, num_images)
    delta_angs = vectorize(rotation, num_images, True)
    obj_avgs= vectorize(obj_avgs, num_images, True)
    obj_stds= vectorize(obj_stds, num_images)

    #Translations are given in sampled coordinates and thus converted here to original image coordinates
    if trans_sampled == True:
        translations_x = translations_x * pixelsamplings_x
        translations_y = translations_y * pixelsamplings_y

#    print "subimage_first_column.shape=", subimage_first_column.shape
#    print "translations_x.shape=", translations_x.shape
#    print "len(image_files)=", len(image_files)
#    print "subimage_first_column[0]=", subimage_first_column[0]
#    print "translations_x[0]=", translations_x[0]
#    print "subimage_width=", subimage_width
#    print "len(image_files)=", len(image_files)
        
#    act_im_num=0
#    print subimage_first_column[act_im_num] + translations_x[act_im_num]
#    print subimage_first_row[act_im_num] + translations_y[act_im_num]
#    print subimage_first_column[act_im_num] + translations_x[act_im_num] + subimage_width * pixelsampling_x[act_im_num]
#    print subimage_first_row[act_im_num] + translations_y[act_im_num] + subimage_height * pixelsampling_y[act_im_num]
#    print subimage_width
#    print "pixelsampling_x[0]", pixelsampling_x[act_im_num]
            
    if background_type != None and color_background_filter != None:
        ex ="parallel loading unsupported for such background type or background filter. Set number of processors CUICUILCO_IMAGE_LOADING_NUM_PROC to 1 or undefined "
        raise Exception(ex)
    else:
        subimages = numpy.zeros((num_images, subimage_width * subimage_height * pixel_dimensions))
        
        unique_image_files = list(set(image_files))
        act_im_num_indices = {}
        for act_im_num, image_file in enumerate(image_files):
            if image_file in act_im_num_indices: #.keys():
                act_im_num_indices[image_file].append(act_im_num)
            else:
                act_im_num_indices[image_file] = [act_im_num]

        #print "act_im_num_indices=", act_im_num_indices
        #Split unique_image_files in num_proc sets
        unique_image_files_subsets = []
        for proc_num in range(num_proc):
            unique_image_files_subsets.append([])
        
        for i, image_file in enumerate(unique_image_files):
            unique_image_files_subsets[i%num_proc].append(image_file)

        all_im_params = (pre_mirroring_flags, pixelsamplings_x, pixelsamplings_y, subimages_first_row, subimages_first_column, \
                         translations_x, translations_y, delta_angs, contrast_enhance, obj_avgs, obj_stds)
     
        queue = multiprocessing.Queue()
        jobs = []
        for proc_num in range(num_proc):
            #print "unique_image_files_subsets[proc_num]=", unique_image_files_subsets[proc_num]
            process_args = (unique_image_files_subsets[proc_num], act_im_num_indices, subimage_width, subimage_height, pixel_dimensions, add_noise, convert_format, background_type, color_background_filter, all_im_params, proc_num, queue, verbose)
            p = multiprocessing.Process(target=load_multiple_image_data, args=process_args)
            jobs.append(p)
            print "parallel image loading: starting image loading by process %d in [0,%d]"%(proc_num, num_proc-1)
            p.start()       

        print "parallel image loading: waiting for results"
        for proc, p in enumerate(jobs):
            print "waiting for result number %d"%proc
            (subset_indices, subimages_subset) =  queue.get()
            #print "incorporating results: (subset_indices, subimages_subset)=", subset_indices, subimages_subset
            subimages[subset_indices,:]=subimages_subset
        
        # sleep for 2 secs
        time.sleep(2)
        for p in jobs:
            print "parallel image loading: joining image loading processes"
            p.join()

    if add_noise == True: #Add noise to the whole images loaded
        if convert_format == "L":
            #Using uniform noise for speed over normal noise
            high = (3**0.5)*0.05 #standard deviation 0.05
            noise = numpy.random.uniform(low=-high, high=high, size=(num_images, subimage_width*subimage_height))
                        #noise = numpy.random.normal(loc=0.0, scale=0.05, size=(subimage_height, subimage_width))
        elif convert_format == "RGB":
            noise_amplitude = 5.0 #RGB has range from 0 to 255
            high = (3**0.5)*noise_amplitude #standard deviation 0.05
            noise = numpy.random.uniform(low=-high, high=high, size=(num_images, subimage_height* subimage_width * 3))
        else:
            err = "Don't know how to generate noise for given image format: ", convert_format
            raise Exception(err)
        subimages += noise         

    if convert_format == "RGB":
        subimages = subimages / 256.0

    t1 = time.time()
    print "image loading finished in %f ms, %f ms/im"%((t1-t0)*1000,(t1-t0)*1000.0/num_images)        
    return subimages
    
def load_multiple_image_data(unique_image_files_subset, act_im_num_indices, subimage_width, subimage_height, pixel_dimensions, add_noise, convert_format, background_type, color_background_filter, all_im_params, proc, queue, verbose):
    print "loading/processing %d unique source images (process %d)"%(len(unique_image_files_subset), proc)

    subset_indices=[]
    for image_file in unique_image_files_subset:
        subset_indices.extend(act_im_num_indices[image_file])

    num_images = len(subset_indices) 
    subimages = numpy.zeros((num_images, subimage_width * subimage_height * pixel_dimensions))           

    (pre_mirroring_flags, pixelsamplings_x, pixelsamplings_y, subimages_first_row, subimages_first_column, \
     translations_x, translations_y, delta_angs, contrast_enhance, objs_avg, objs_std) = all_im_params

    add_noise2 = False #Noise is added to whole matrix, to prevent randomness differences w.r.t. non parallelized version
    image_counter = 0
    for image_file in unique_image_files_subset:
        if verbose:
            print "Loading unique image", image_file
        im_orig = Image.open(image_file)   
        #this makes sure the image data is decoded now
        im_orig.load()
        if image_file.endswith("EyeNZ2Dummy.jpg"):
            save_intermediate = True
        else:
            save_intermediate = False
                    
        for act_im_num in act_im_num_indices[image_file]:
            im_params = (subimage_width, subimage_height, pre_mirroring_flags[act_im_num], pixelsamplings_x[act_im_num], pixelsamplings_y[act_im_num], subimages_first_row[act_im_num], \
                         subimages_first_column[act_im_num], add_noise2, convert_format, translations_x[act_im_num], translations_y[act_im_num], delta_angs[act_im_num], contrast_enhance, objs_avg[act_im_num], \
                         objs_std[act_im_num], background_type, color_background_filter, verbose, save_intermediate) 
            #time.sleep(1) #TURBO WARNING!!!
            subimages[image_counter] = load_single_image_data(im_orig, im_params)
            image_counter += 1

    queue.put((subset_indices, subimages))

#Translations are given in original image coordinates
def load_single_image_data(im_orig, im_params):
    (subimage_width, subimage_height, pre_mirroring_flag, pixelsampling_x, pixelsampling_y, subimage_first_row, subimage_first_column, \
                    add_noise, convert_format, translation_x, translation_y, delta_ang, contrast_enhance, obj_avg, obj_std, background_type, color_background_filter, verbose, save_intermediate) = im_params
                    
    out_size = (subimage_width, subimage_height) 
   
    if background_type is None or color_background_filter is None:
        im = im_orig
        if pre_mirroring_flag:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)                  
        #Warning, make this conditional            
        #im = ImageOps.autocontrast(im, 2)
                            
        x0 = subimage_first_column + translation_x
        y0 = subimage_first_row + translation_y
        x1 = subimage_first_column + translation_x + subimage_width * pixelsampling_x
        y1 = subimage_first_row + translation_y + subimage_height * pixelsampling_y
    
        #print "x0, y0, x1, y1=", x0, y0, x1, y1
        subimage_coordinates = (x0,y0,x1,y1)
        ori_width = x1 - x0
        ori_height = y1 - y0
                
        if x1<0 or y1<0 or x0>=im.size[0] or y0>=im.size[1]:
            err = "Image Loading Failed: Subimage seriously out of Image"
            print "subimage_coordinates =", (x0,y0,x1,y1)
            print "Image size: im.size[0], im.size[1] = ", im.size[0],  im.size[1]
            raise Exception(err)
    
        if delta_ang != None:
            rotation_center_x = (x0+x1-1)/2.0
            rotation_center_y = (y0+y1-1)/2.0

            integer_rotation_center = True #and False
            if integer_rotation_center:
                rotation_center_x_int = int(rotation_center_x + 0.5)
                rotation_center_y_int = int(rotation_center_y + 0.5)
            else:
                rotation_center_x_int = rotation_center_x
                rotation_center_y_int = rotation_center_y
            
            if integer_rotation_center:
                rotation_window_width = 2 * max(im.size[0]-1-rotation_center_x_int+0.5, rotation_center_x_int+0.5) #Result is always integer and odd 
                rotation_window_height = 2 * max(im.size[1]-1-rotation_center_y_int+0.5, rotation_center_y_int+0.5) #Result is always integer and odd                
            else:
                rotation_window_width = 2 * max(im.size[0]-1-rotation_center_x_int+0.5, rotation_center_x_int+0.5) #Result is real
                rotation_window_height = 2 * max(im.size[1]-1-rotation_center_y_int+0.5, rotation_center_y_int+0.5) #Result is real          
                rotation_window_width = int(rotation_window_width + 0.5)
                rotation_window_height = int(rotation_window_height + 0.5)
                
            Delta_x = rotation_center_x - rotation_center_x_int
            Delta_y = rotation_center_y - rotation_center_y_int

            rotation_angle_rad =  -delta_ang * numpy.pi/180.0 
            Delta_x_rotated = Delta_x  * numpy.cos(rotation_angle_rad) - Delta_y * numpy.sin(rotation_angle_rad)
            Delta_y_rotated = Delta_y  * numpy.cos(rotation_angle_rad) + Delta_x * numpy.sin(rotation_angle_rad)
    
            rotation_crop_x0 = rotation_center_x_int-(rotation_window_width-1)/2.0
            rotation_crop_y0 = rotation_center_y_int-(rotation_window_height-1)/2.0
            rotation_crop_x1 = rotation_center_x_int+(rotation_window_width-1)/2.0+1
            rotation_crop_y1 = rotation_center_y_int+(rotation_window_height-1)/2.0+1              

            #Here crop size should not loose any pixel from the rotation window
            crop_size = (int(rotation_window_width+0.5), int(rotation_window_height+0.5)) #Avoiding type problem, even though variables are actually integer        
            crop_coordinates = (rotation_crop_x0, rotation_crop_y0, rotation_crop_x1, rotation_crop_y1)
            #print crop_size
            #print crop_coordinates
    
            im_crop_first = im.transform(crop_size, Image.EXTENT, crop_coordinates) #, Image.BICUBIC)
                    
            if save_intermediate:
                print "SAVING image 1"
                print "subimage_first_column[act_im_num]=", subimage_first_column
                print "ori_width, ori_height =", ori_width, ori_height
                print "subimage_coordinates =", subimage_coordinates
                print "rotation_center_x_int, rotation_center_y_int=", rotation_center_x_int, rotation_center_y_int
                print "Delta_x, Delta_y=",Delta_x, Delta_y
                im_crop_first.save("cropped.png")
                    
            #print delta_ang
            im_rotated = rotate_improved(im_crop_first, delta_ang, Image.BICUBIC)

            if save_intermediate:
                print "SAVING image 2"
                im_rotated.save("rotated.png")
                    
            #Warning, make this conditional, use equalize instead, autocontrast is too weak!
            im_contrasted= contrast_enhance_image(im_rotated, x0, x1, y0, y1, obj_avg, obj_std, contrast_enhance)      
                   
            if save_intermediate:
                print "SAVING image 3"
                im_contrasted.save("im_contrasted.png")

                    
            center_x_geometric = (crop_size[0]-1)/2.0
            center_y_geometric = (crop_size[1]-1)/2.0
            #print "After rotation: geometric center_x, center_y=", center_x_geometric, center_y_geometric
    
            new_center_x = center_x_geometric + Delta_x_rotated #Real value!
            new_center_y = center_y_geometric + Delta_y_rotated #Real value!
            #print "After rotation: new_center_x, new_center_y=", new_center_x, new_center_y
            #print "center_x, center_y=", center_x, center_y
            #print "ori_width, ori_height=", ori_width, ori_height
    
            #WWW removed int
            x0 = (new_center_x-(ori_width-1)/2.0) #WWW added -1
            x1 = (new_center_x+(ori_width-1)/2.0)+1 #WWW added -1 and +1
            y0 = (new_center_y-(ori_height-1)/2.0) #WWW added -1
            y1 = (new_center_y+(ori_height-1)/2.0)+1 #WWW added -1 and +1
                        
            #print "transform_coords = x0, x1, y0, y1=", x0, x1, y0, y1
            subimage_coordinates = (x0,y0,x1,y1)
    
            im_out = final_sampling_and_contrast_enhance(im_contrasted, out_size, x0, x1, y0, y1, contrast_enhance)        
                    
            if save_intermediate:
                print "SAVING image 4. center_x_geometric, center_y_geometric=", center_x_geometric, center_y_geometric
                print "Delta_x_rotated, Delta_y_rotated were=", Delta_x_rotated, Delta_y_rotated
                print "subimage_coordinates were=", subimage_coordinates
                im_out.save("im_out.png")
        else: #No rotation is applied at all
            #Warning, make this conditional, use equalize instead, autocontrast is too weak!           
            im_rotated = im
            im_contrasted = contrast_enhance_image(im_rotated, x0, x1, y0, y1, obj_avg, obj_std, contrast_enhance)          
            
            im_out = final_sampling_and_contrast_enhance(im_contrasted, out_size, x0, x1, y0, y1, contrast_enhance)
            im_out = im_out.convert(convert_format)
   
        im_small = numpy.asarray(im_out)
               
        if add_noise == True and False:
            ex = "Was not supposed to add noise here..."
            raise Exception(ex)
            if convert_format == "L":
                #Using uniform noise for speed over normal noise
                high = (3**0.5)*0.05 #standard deviation 0.05
                noise = numpy.random.uniform(low=-high, high=high, size=(subimage_height, subimage_width))
                        #noise = numpy.random.normal(loc=0.0, scale=0.05, size=(subimage_height, subimage_width))
            elif convert_format == "RGB":
                noise_amplitude = 5.0 #RGB has range from 0 to 255
                high = (3**0.5)*noise_amplitude #standard deviation 0.05
                noise = numpy.random.uniform(low=-high, high=high, size=(subimage_height, subimage_width, 3))
#                        noise = numpy.random.normal(loc=0.0, scale=noise_amplitude, size=(subimage_height, subimage_width, 3))
            else:
                err = "Don't know how to generate noise for given image format: ", convert_format
                raise Exception(err)
            im_small = im_small*1.0 + noise
            #irb_transform = True
            #if convert_format == "RGB" and irb_transform:
            #    im_small = rgb_to_irb_array(im_small)

        return  im_small.flatten()

def final_sampling_and_contrast_enhance(im_contrasted, out_size, x0, x1, y0, y1, contrast_enhance):
    subimage_coordinates = (x0,y0,x1,y1)    
    if contrast_enhance == "PostEqualizeHistogram":
        #Resize, then equalize
        im_out = im_contrasted.transform(out_size, Image.EXTENT, subimage_coordinates, interpolation_format)    #W interpolation_format       format_nearest
        im_out = ImageOps.equalize(im_out)
    elif contrast_enhance == "SmartEqualizeHistogram":
        #Crop, then equalize, then resize
        out_size_crop = (x1-x0,y1-y0)
        im_out = im_contrasted.transform(out_size_crop, Image.EXTENT, subimage_coordinates, interpolation_format)    #W interpolation_format       format_nearest
        im_out = ImageOps.equalize(im_out)
        crop_coordinates = (0,0,x1-x0,y1-y0)
        im_out = im_out.transform(out_size, Image.EXTENT, crop_coordinates, interpolation_format)    #W interpolation_format       format_nearest
    else:
        im_out = im_contrasted.transform(out_size, Image.EXTENT, subimage_coordinates, interpolation_format)
    return im_out
        
def contrast_enhance_image(im_rotated, x0, x1, y0, y1, obj_avg, obj_std, contrast_enhance):
    if contrast_enhance in ["AgeContrastEnhancement_Avg_Std", "AgeContrastEnhancement", "GTSRBContrastEnhancement", "ContrastGenderMultiply", "AgeContrastEnhancement15", "AgeContrastEnhancement20", "AgeContrastEnhancement25"]:
        #im_contrasted = ImageOps.autocontrast(im_rotated, 2)
        if contrast_enhance ==  "AgeContrastEnhancement_Avg_Std":
            obj_center_width = (x1-x0 + 1)*0.7
            obj_center_height = (y1-y0 + 1)*0.7
            im_contrasted = simple_normalization_Age(im_rotated, obj_center_width, obj_center_height, obj_avg=obj_avg, obj_std=obj_std)
        elif contrast_enhance == "AgeContrastEnhancement":
            face_center_width = (x1-x0 + 1)*0.7
            face_center_height = (y1-y0 + 1)*0.7
            im_contrasted = simple_normalization_Age(im_rotated, face_center_width, face_center_height, obj_std=0.2)
        elif contrast_enhance == "AgeContrastEnhancement15":
            face_center_width = (x1-x0 + 1)*0.7
            face_center_height = (y1-y0 + 1)*0.7
            im_contrasted = simple_normalization_Age(im_rotated, face_center_width, face_center_height, obj_std=0.15)
        elif contrast_enhance == "AgeContrastEnhancement20":
            face_center_width = (x1-x0 + 1)*0.7
            face_center_height = (y1-y0 + 1)*0.7
            im_contrasted = simple_normalization_Age(im_rotated, face_center_width, face_center_height, obj_std=0.20)
        elif contrast_enhance == "AgeContrastEnhancement25":
            face_center_width = (x1-x0 + 1)*0.7
            face_center_height = (y1-y0 + 1)*0.7
            im_contrasted = simple_normalization_Age(im_rotated, face_center_width, face_center_height, obj_std=0.25)
        elif contrast_enhance == "GTSRBContrastEnhancement":
            object_width = (x1-x0 + 1)*0.5
            object_height = (y1-y0 + 1)*0.7
            im_contrasted = simple_normalization_GTSRB(im_rotated, object_width, object_height)
        elif contrast_enhance == "ContrastGenderMultiply":
            im_contrasted = simple_normalization_GenderMultiply(simple_normalization_GenderMultiply(simple_normalization_GenderMultiply(im_rotated))) #Normalized 3 times!
        else:
            ex = "Incorrect method (unknown constant enhancement) %s"%contrast_enhance
            raise Exception(ex)

    elif contrast_enhance is None:
        im_contrasted = im_rotated
        Delta_x_rotated = Delta_y_rotated = 0 #why this assignment???
    else:
        ex = "Unknown contrast_enhance method!!!  %s"%contrast_enhance
        raise Exception(ex)
    return im_contrasted



def images_asarray(images):
    num_images = len(images)
    im_width, im_height = images[0].size

    im_arr = numpy.zeros((num_images, im_width * im_height))
    for i, image in enumerate(images):
        im_arr[i] = numpy.asarray(image).flatten() 
    return im_arr

def load_images(image_files, image_format="L", sampling=1.0):
    num_images = len(image_files)
    
    im = Image.open(image_files[0]) 
    im_width, im_height = im.size

    images = []
    for i, image_file in enumerate(image_files):       
        im = Image.open(image_file)  
        im_formated = im.convert(image_format)
        images.append(im_formated)
        
    return images

#coordinates are (x0, y0, x1, y1)
#sizes are (width, height)
def extract_subimages_basic(images, image_indices, coordinates, out_size=(64,64), allow_out_of_image_sampling=True):
    if len(image_indices) != len(coordinates):
        raise Exception("number of images indices %d and number of coordinates %d do not match"%(len(image_indices), len(coordinates)) )
    
    if len(images) < 1:
        raise Exception("At least one image is needed")    
    
    subimages = []
    for i, im_index in enumerate(image_indices):
        subimage_coordinates = (x0, y0, x1, y1) = coordinates[i]
        if (x0<0 or y0<0 or x1>=images[im_index].size[0] or y1>=images[im_index].size[1]) and not allow_out_of_image_sampling:
                err = "Image Loading Failed: Subimage out of Image"
                print "subimage_coordinates =", (x0,y0,x1,y1)
                print "Image size: im.size[0], im.size[1] = ", images[im_index].size[0],  images[im_index].size[1]
                raise Exception(err)
 
        im_out = images[im_index].transform(out_size, Image.EXTENT, subimage_coordinates, interpolation_format)
        subimages.append(im_out)
    return subimages

def extract_subimages_rotate(images, image_indices, all_coordinates, delta_angs, out_size=(64,64), interpolation_format=interpolation_format):
    if len(image_indices) != len(all_coordinates):
        raise Exception("number of images indices %d and number of coordinates %d do not match"%(len(image_indices), len(all_coordinates)) )
    
    if len(images) < 1:
        raise Exception("At least one image is needed")    
    
    subimages = []
    for i, im_index in enumerate(image_indices):
        subimage_coordinates = all_coordinates[i]
        if delta_angs[i] != 0:
            im_out = extract_subimage_rotate(images[im_index], subimage_coordinates, delta_angs[i], out_size=out_size, interpolation_format=interpolation_format)
        else:
            im_out = images[im_index].transform(out_size, Image.EXTENT, subimage_coordinates, interpolation_format)
        subimages.append(im_out)
    return subimages
    
def extract_subimage_rotate(image, subimage_coordinates, delta_ang, out_size=(64,64), allow_out_of_image_sampling=True, interpolation_format=interpolation_format):
    (x0, y0, x1, y1) = subimage_coordinates
    if (x0<0 or y0<0 or x1>=image.size[0] or y1>=image.size[1]) and not allow_out_of_image_sampling:
        err = "Image Loading Failed: Subimage out of Image"
        print "subimage_coordinates =", (x0,y0,x1,y1)
        print "Image size: im.size[0], im.size[1] = ", image.size[0],  image.size[1]
        raise Exception(err)

    if delta_ang != None:
        rotation_center_x = (x0+x1-1)/2.0
        rotation_center_y = (y0+y1-1)/2.0

        integer_rotation_center = True #and False
        if integer_rotation_center:
            rotation_center_x_int = int(rotation_center_x + 0.5)
            rotation_center_y_int = int(rotation_center_y + 0.5)
        else:
            rotation_center_x_int = rotation_center_x
            rotation_center_y_int = rotation_center_y
                    
        rotation_window_width = 2 * max(image.size[0]-1-rotation_center_x_int+0.5, rotation_center_x_int+0.5) #Result is always integer and odd
        rotation_window_height = 2 * max(image.size[1]-1-rotation_center_y_int+0.5, rotation_center_y_int+0.5) #Result is always integer and odd

########
        rxA = (x0-rotation_center_x) * numpy.cos(delta_ang* numpy.pi  / 180.0) + (y0-rotation_center_y) * numpy.sin(delta_ang* numpy.pi  / 180.0)
        ryA = (y0-rotation_center_y) * numpy.cos(delta_ang* numpy.pi  / 180.0) - (x0-rotation_center_x) * numpy.sin(delta_ang* numpy.pi  / 180.0)
        rxB = -1 * (x0-rotation_center_x) * numpy.cos(delta_ang* numpy.pi  / 180.0) + (y0-rotation_center_y) * numpy.sin(delta_ang* numpy.pi  / 180.0)
        ryB = (y0-rotation_center_y) * numpy.cos(delta_ang* numpy.pi  / 180.0) - (-1)*(x0-rotation_center_x) * numpy.sin(delta_ang* numpy.pi  / 180.0)
        wx = 2 * int(max(rxA, rxB, -rxA, -rxB)+2) + 1
        wy = 2 * int(max(ryA, ryB, -ryA, -ryB)+2) + 1
        
        rotation_window_width = min(rotation_window_width, wx)
        rotation_window_height = min(rotation_window_height, wy)
        
        if integer_rotation_center == False:
            rotation_window_width = int(rotation_window_width + 0.5)
            rotation_window_height = int(rotation_window_height + 0.5)

        Delta_x = rotation_center_x - rotation_center_x_int
        Delta_y = rotation_center_y - rotation_center_y_int

        rotation_angle_rad =  -delta_ang * numpy.pi  / 180.0 
        Delta_x_rotated = Delta_x  * numpy.cos(rotation_angle_rad) - Delta_y * numpy.sin(rotation_angle_rad)
        Delta_y_rotated = Delta_y  * numpy.cos(rotation_angle_rad) + Delta_x * numpy.sin(rotation_angle_rad)
    
        rotation_crop_x0 = rotation_center_x_int-(rotation_window_width-1)/2.0
        rotation_crop_y0 = rotation_center_y_int-(rotation_window_height-1)/2.0
        rotation_crop_x1 = rotation_center_x_int+(rotation_window_width-1)/2.0+1
        rotation_crop_y1 = rotation_center_y_int+(rotation_window_height-1)/2.0+1              
        #Here crop size should not loose any pixel from the rotation window
        crop_size = (int(rotation_window_width+0.5), int(rotation_window_height+0.5)) #Avoiding type problem, even though variables are actually integer 
                    
        crop_coordinates = (rotation_crop_x0, rotation_crop_y0, rotation_crop_x1, rotation_crop_y1)
        im_crop_first = image.transform(crop_size, Image.EXTENT, crop_coordinates) #, Image.BICUBIC)
                    
        #print delta_ang
        im_rotated = rotate_improved(im_crop_first, delta_ang, Image.BICUBIC)
        im_contrasted = im_rotated
                    
        center_x_geometric = (crop_size[0]-1)/2.0
        center_y_geometric = (crop_size[1]-1)/2.0
        #print "After rotation: geometric center_x, center_y=", center_x_geometric, center_y_geometric
    
        new_center_x = center_x_geometric + Delta_x_rotated #Real value!
        new_center_y = center_y_geometric + Delta_y_rotated #Real value!
        #print "After rotation: new_center_x, new_center_y=", new_center_x, new_center_y
        #print "center_x, center_y=", center_x, center_y
        #print "ori_width, ori_height=", ori_width, ori_height
    
        ori_width = x1-x0
        ori_height = y1-y0
        x0 = (new_center_x-(ori_width-1)/2.0) #WWW added -1
        x1 = (new_center_x+(ori_width-1)/2.0)+1 #WWW added -1 and +1
        y0 = (new_center_y-(ori_height-1)/2.0) #WWW added -1
        y1 = (new_center_y+(ori_height-1)/2.0)+1 #WWW added -1 and +1
                        
        #print "transform_coords = x0, x1, y0, y1=", x0, x1, y0, y1
        subimage_coordinates = (x0,y0,x1,y1)
        im_out = im_contrasted.transform(out_size, Image.EXTENT, subimage_coordinates, interpolation_format)
    return im_out

def code_gender(gender):
    if gender is None:
        return 999
    x = int(500 + gender * 125)
    if x<0:
        x=0
    if x>999:
        x=999
    return x

def generate_color_filter4(size):
    height, width = size
    center_x = width / 2.0
    center_y = height / 2.0
#    factor = numpy.sqrt(numpy.sqrt(center_x**4 +center_y**4))
    factor = numpy.sqrt(numpy.sqrt(width**4 +height**4))
    filter = numpy.ones(size)
    for y in range(height):
        for x in range(width):
#            filter[y][x] = numpy.sqrt(numpy.sqrt((x-center_x)**2+(y-center_y)**4))/ factor
            filter[y][x] = numpy.sqrt(numpy.sqrt(x**4+y**4))/ factor
    return filter

def generate_color_filter2(size):
    height, width = size
    center_y = height / 2.0
    center_x = width / 2.0
#    factor = numpy.sqrt(center_x**2 +center_y**2)
    factor = numpy.sqrt(width**2 + height**2)
    filter = numpy.ones(size)
    for y in range(height):
        for x in range(width):
#            filter[y][x] = numpy.sqrt((x-center_x)**2+(y-center_y)**2)/ factor
            filter[y][x] = numpy.sqrt(x**2+y**2)/ factor
    return filter

#noise should be a bidimensional array    
def generate_colored_noise(filter):
    shape = filter.shape
#    pure_noise = numpy.random.normal(size=shape)
    pure_noise = numpy.random.uniform(low=-1.0, high=1.0, size=shape)
    filtered_noise = pure_noise * filter
#    filtered_noise = pure_noise
    result_noise = numpy.real(numpy.fft.ifft2(filtered_noise))
    result_noise[0][0] = 0
    result_noise = result_noise - result_noise.min()
    result_noise = (result_noise / result_noise.max()) * 255
    return result_noise

def frequencies_1D(length):
    freq = numpy.zeros(length)
    length2 = length / 2
    freq[0:length/2+1] = numpy.arange(1, length/2+2) * 1.0 / length
    freq[length/2+1:length] = freq[(length+1)/2-1:0:-1]
    return freq

def filter_colored_noise1D(length, alpha):
    freq = numpy.zeros(length)
    length2 = length / 2
    freq[0:length/2+1] = numpy.arange(1, length/2+2)
    freq[length/2+1:length] = freq[(length+1)/2-1:0:-1]
    filter = 1.0/(freq ** (alpha/2.0))
    return filter

def filter_colored_noise2D_trk(size, alpha):
    filter_v = filter_colored_noise1D(size[0], alpha)
    filter_h = filter_colored_noise1D(size[1], alpha) 
    return filter_v.reshape((size[0],1)) * filter_h.reshape((1, size[1])) 

def filter_colored_noise2D(size, alpha):
    filter_v = filter_colored_noise1D(size[0], alpha)
    filter_h = filter_colored_noise1D(size[1], alpha) 
    return numpy.sqrt(filter_v.reshape((size[0],1)) ** 2 + filter_h.reshape((1, size[1]))**2) 

def filter_colored_noise2D_imp(size, alpha=1.0):
    freqs_v = frequencies_1D(size[0])
    freqs_h = frequencies_1D(size[1])
    combined_freqs =  numpy.sqrt(freqs_v.reshape((size[0],1)) ** 2 + freqs_h.reshape((1, size[1]))**2)
    return 1.0 / (combined_freqs ** (alpha/2.0))


def change_mean(im, new_mean, new_std):
    return (im-im.mean())/im.std() * new_std + new_mean


def random_filtered_noise2D(shape, filter, min=0, max=255, mean = 127.5, std = 25, clip=True, verbose=False):
    amplitude = (max-min)
   
    if verbose:
        print "Filter size is: ", shape
        print "Filter shape is: ", filter.shape

    white_noise = numpy.random.random(shape) * amplitude + min
    Fwhite_noise = numpy.fft.fft2(white_noise)
    
    Fcolor_noise = Fwhite_noise * filter
    color_noise = numpy.fft.ifft2(Fcolor_noise).real
    color_noise = change_mean(color_noise, mean, std)
    if clip == True:
        return color_noise.clip(0, 255)
    else:
        return color_noise

def rename_filenames_facegen():
    #Renaming filenames => Fix perl script
#    im_seq_base_dir = "/home/scratch_sb/escalafl/RenderingAlberto20x500"
    im_seq_base_dir = "/local/tmp/escalafl/Alberto/Renderings20x500"
#    im_seq_base_dir = "/home/scratch_sb/escalafl/RenderingAlberto20x500"

    
    suffix = "_i0.tif"
#   Sample name: random000_e0_c0_p3_i0.tif
    identities = range(20)
    for id in identities:
        prefix = "random%03d_e0_c0_p"%(id)
        print "prefix=", prefix
#    
        image_files = glob.glob(im_seq_base_dir + "/" + prefix + "?_*tif")
        image_files
        for i in range(0,len(image_files)+0):
            os.rename(im_seq_base_dir + "/" + prefix + str(i) + suffix, im_seq_base_dir + "/" + prefix+"%03d"%i + suffix)
#
#   Sample name: random000_e0_c0_p23_i0.tif
        image_files = glob.glob(im_seq_base_dir + "/" + prefix + "??_*tif")
        print "prefix2=", prefix
        image_files
        for i in range(10,len(image_files)+10):
            os.rename(im_seq_base_dir + "/" + prefix + str(i) + suffix, im_seq_base_dir + "/" + prefix+"%03d"%i + suffix)

#Functions used in conjunction with "Natural" data set
def read_binary_header(data_base_dir="", base_filename="data_bin_1.bin"):
    filename = data_base_dir + "/" + base_filename
    fd = open(filename, 'rb')
    s = fd.read(16) #20
    fd.close()

#Note, on 64 bit machine, unsigned integer size (I) is 4 bytes, short integer (H) is... 2 bytes
    print "binary string read:", s
#    (magic_num, iteration, numSamples, numHid, sampleSpan) = struct.unpack('<IIIII', s)
    (magic_num, iteration, numSamples, numHid) = struct.unpack('<IIII', s)
    sampleSpan = -1

    if magic_num != 666:
        er = "Wrong magic number, was %d, should be 666"%magic_num
        raise(er)
    
    print "Loaded header: ", (magic_num, iteration, numSamples, numHid, sampleSpan)
    return (magic_num, iteration, numSamples, numHid, sampleSpan)

def read_natural_header(data_base_dir="", base_filename="data_bin_1.bin"):
    filename = data_base_dir + "/" + base_filename
    fd = open(filename, 'rb')
    s = fd.read(16)  #20
    fd.close()

#    (magic_num, iteration, numSamples, numHid, sampleSpan) = struct.unpack('<IIIII', s)
    (magic_num, iteration, numSamples, numHid) = struct.unpack('<IIII', s)
    sampleSpan = -1

    if magic_num != 666:
        er = "Wrong magic number, was %d, should be 666"%magic_num
        raise(er)
    
    print "Loaded header: ", (magic_num, iteration, numSamples, numHid, sampleSpan)
    return (magic_num, iteration, numSamples, numHid, sampleSpan)

def load_raw_data(data_base_dir="/scratch/escalafl/cooperations/igel/patches_8x8", base_filename="bochum_natural_8_5000.bin", input_dim = 64, dtype = "uint8", select_samples=None, verbose=False):
    filename = data_base_dir + "/" + base_filename
    fd = open(filename, 'rb')
#    s_header = fd.read(16)
#    (magic_num, iteration, numSamples, numHid, sampleSpan) = struct.unpack('<IIIII', s_header)
#    (magic_num, iteration, numSamples, numHid) = struct.unpack('<IIII', s_header)
#    sampleSpan = -1
#    if magic_num != 666:
#        er = "Wrong magic number, was %d, should be 666"%magic_num
#        raise(er)

    data = numpy.fromfile(fd, dtype=dtype, count=-1, sep='') / 255.0

    if len(data) % input_dim != 0:
        er = "number of bytes in data is not multiple of 64"
        raise Exception(er)
    numSamples = len(data) / input_dim
    data = data.reshape(numSamples, input_dim)
    fd.close()

    if verbose:
        print "data (brute) .shape=", data.shape
    if select_samples!=None:
        data = data[select_samples] 
    if verbose:
        print "Data, samples selected:", data
#    print data
#    quit()
    return data

def load_natural_data(data_base_dir="", base_filename="data_bin_1.bin", samples=None, verbose=False):
    filename = data_base_dir + "/" + base_filename
    fd = open(filename, 'rb')
    s_header = fd.read(16)
#    (magic_num, iteration, numSamples, numHid, sampleSpan) = struct.unpack('<IIIII', s_header)
    (magic_num, iteration, numSamples, numHid) = struct.unpack('<IIII', s_header)
    sampleSpan = -1
    if magic_num != 666:
        er = "Wrong magic number, was %d, should be 666"%magic_num
        raise(er)

    data = numpy.fromfile(fd, dtype=float, count=-1, sep='').reshape(numSamples, numHid)
    fd.close()

    print "data.shape=", data.shape
    print "Header: ", (magic_num, iteration, numSamples, numHid, sampleSpan)
    if samples!=None:
        data = data[samples] 
    print "Data, samples selected:", data
    return data
#(magic_num, iteration, numSamples, numHid, sampleSpan)

def transform(x, y, matrix):
    (a, b, c, d, e, f)  = matrix
    return a*x + b*y + c, d*x + e*y + f

def rotate_improved(self, angle, resample=Image.NEAREST, expand=0, force_odd_output_size=False):
    """ Function that rotates an image a given number of degrees.
    This function replaces Im.rotate from PIL and fixes a weird
    problem caused by integer coordinate computations and insufficient
    documentation. This resulted in weird center shifts.
    As far as I know the new function has no bugs.
    alberto.escalante@ini.rub.de, 17 January 2013
    """    
    im_width, im_height= self.size
    center_x = (im_width-1) / 2.0
    center_y = (im_height-1) / 2.0
                      
    angle_r = -angle * math.pi / 180
    (a,b,c,d,e,f) = [ math.cos(angle_r), math.sin(angle_r), 0.0,
                 -math.sin(angle_r), math.cos(angle_r), 0.0]
        
#    r_c_x, r_c_y = transform(center_x, center_y, (a,b,c,d,e,f))
       
    if expand: #Update size of output image
        o_pxs = []
        o_pys = []
        for px, py in ((0, 0), (im_width-1, 0), (im_width-1, im_height-1), (0, im_height-1)):
            o_px, o_py = transform(px, py, (a,b,c,d,e,f))
            o_pxs.append(o_px)
            o_pys.append(o_py)
        out_width = int(math.ceil(max(o_pxs)) - math.floor(min(o_pxs)))
        out_height = int(math.ceil(max(o_pys)) - math.floor(min(o_pys)))
        if force_odd_output_size:
            if out_width & 1== 0:
                out_width += 1
            if out_height & 1== 0:
                out_height += 1
            print "aborted..."
            quit()
    else:
        out_width = im_width
        out_height = im_height
    
#    print "rotation output: out_width=", out_width, "out_height=", out_height
    o_center_x = (out_width-1) / 2.0
    o_center_y = (out_height-1) / 2.0
        
    c = center_x-(o_center_x * a + o_center_y * b)+0.5
    f = center_y-(o_center_x * d + o_center_y * e)+0.5
    matrix = (a,b,c,d,e,f)
        
    test_im_aff = self.transform((out_width, out_height), Image.AFFINE, matrix, resample)
    return test_im_aff




 
###EXAMPLE of Pink Noise Geration:
#size = (128,64)
#alphas = [8.0, 6.0, 5.0, 4.0, 
#          3.0, 2.0, 1.5, 1.25, 
#          1.0, 0.75, 0.5, 0.25]
#amp = 25
#mean = 127.5
#plt.figure()
#plt.suptitle("Pink Noise. Amplitude = k1 * 1 / f^(alpha/2), Energy = k2 * 1 / f^alpha")
#
#for i, alpha in enumerate(alphas):
#    plt.subplot(3,4,i+1)
#    filter = filter_colored_noise2D_imp(size, alpha) # 1/f^(3/2)
#    yy =  random_filtered_noise2D(size, filter)
#    yy4 = change_mean(yy, mean, amp)
##    yy2 = (yy - yy.mean()) 
##    yy3 = yy2 / yy2.std()
##    yy4 = yy3 * amp + mean
#    plt.imshow(yy4, vmin=0, vmax=255, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)  
#    plt.xlabel("Alpha=%0.1f, std=%0.1f, mean=%0.1f"%(alpha, amp, mean))
#
#plt.show()

#plt.plot(numpy.arange(len(y1)), y1, "b.",)
#plt.subplot(1,3,2)
#plt.plot(numpy.arange(len(y2)), y2, "b.",)
#plt.subplot(1,3,3)
#plt.plot(numpy.arange(len(y3)), y3, "b.",)
#plt.show()

##EXAMPLE:

#alpha = 1.0 #1.0 = pink noise, 0.0 = white noise
#filter = filter_colored_noise1D(135, alpha)
#size = (64,128)
#filter = filter_colored_noise2D(size, alpha)
#yy =  random_filtered_noise2D(size, filter)
#
#plt.figure()
#plt.imshow(yy, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)  
#plt.draw()                                                                                                      
#plt.show()
#    
#def powernoise1D(alpha, N):
#    opt_randpow = False
#    opt_normal = False
#    N2 = N/2-1
#    f = numpy.arange(2, (N2+1)+1 )
#    A2 = 1.0/(f ** (alpha/2))
#    if not opt_randpow:
#        p2 = (numpy.random.random(N2)-0.5)*2*numpy.pi
#        d2 = A2 * numpy.exp(1j * p2)
#    else:
#        p2 = numpy.random.random(N2) + i * numpy.random.random(N2)
#        d2 = A2 * p2
#    d2_conj_rev = numpy.conj(d2)[::-1]
#
#    print [1.0, ]
#    print d2
#    print 1.0/((N2+2.0)**alpha)
#    print d2_conj_rev
#      
#    d = numpy.concatenate(([1.0,], d2, [1.0/((N2+2.0)**alpha),], d2_conj_rev))
#    x = numpy.fft.ifft(d).real
#    if opt_normal:
#        x = ((x - x.min())/(x.max() - x.min()) - 0.5) * 2
#    return x



#plt.figure()
#plt.subplot(1,3,1)
#plt.plot(numpy.arange(len(y1)), y1, "b.",)
#plt.subplot(1,3,2)
#plt.plot(numpy.arange(len(y2)), y2, "b.",)
#plt.subplot(1,3,3)
#plt.plot(numpy.arange(len(y3)), y3, "b.",)
#plt.show()

#filter = numpy.ones((16,16))
#for x in range(16):
#    for y in range(16):
#        filter[y][x] = numpy.sqrt((x-8)**2+(y-8)**2)/ (8 * 1.72)
#
#filter2 = numpy.ones((16,16))
#for x in range(16):
#    for y in range(16):
#        filter2[y][x] = ((x-8)**2+(y-8)**2)/ ((8 * 1.72)**2)
#
#noise = numpy.random.normal(size=(16,16))
#col_noise = noise * filter
#col_noise2 = noise * filter2
#
#im0 = numpy.fft.ifft2(noise)
#im1 = numpy.fft.ifft2(col_noise)
#im2 = numpy.fft.ifft2(col_noise2)
#
#
#filter[0][0]
#filter[8][8]
#filter[15][15]
#plt.figure()
#plt.subplot(1,3,1)
#plt.imshow(numpy.absolute(im0), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)  
#plt.subplot(1,3,2)
#plt.imshow(numpy.absolute(im1), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)  
#plt.subplot(1,3,3)
#plt.imshow(numpy.absolute(im2), aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)  
#plt.draw()                                                                                                      
#plt.show()










#im_seq_base_dir = "/local/tmp/escalafl/Alberto/RenderingsGender50x"
#ids=range(0,49)
#ages=[999]
#
#genders = (map(code_gender, numpy.arange(-3,3,0.1)))
#print genders
#genders = (map(code_gender, numpy.arange(-3,3,0.4005)))
#print genders

#
#racetweens = [999]
#expressions=[0]
#morphs=[0]
#poses=[0]
#lightings=[0]
#slow_signal=0
#step=1
#offset=0
#
#print "genders=", genders
#image_files = create_image_filenames2(im_seq_base_dir, slow_signal, ids, ages, genders, racetweens, \
#                            expressions, morphs, poses, lightings, step, offset)
#print image_files
#
#num_images = len(image_files)
#translation = 4
#
#scale_disp = 3
#image_width  = 256
#image_height = 192
#subimage_width  = 128
#subimage_height = 96 
#pixelsampling_x = 1
#pixelsampling_y = 1
#subimage_pixelsampling= 1
#subimage_first_row= image_height/2-subimage_height*pixelsampling_y/2
#subimage_first_column=image_width/2-subimage_width*pixelsampling_x/2+ 5*pixelsampling_x
#add_noise = True
#convert_format="L"
##translations_x=None
##translations_y=None
#translations_x=numpy.random.random_integers(-translation, translation, num_images) 
#translations_y=numpy.random.random_integers(-translation, translation, num_images)
#trans_sampled=True
#
#data = load_image_data(image_files, image_width, image_height, subimage_width, subimage_height, \
#                    pixelsampling_x, pixelsampling_y, subimage_first_row, subimage_first_column, \
#                    add_noise, convert_format, translations_x, translations_y, trans_sampled)



