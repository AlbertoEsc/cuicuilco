#Functions to generate normalized datasets
#Library version first adapted on 2.9.2010
#Alberto Escalante. alberto.escalante@ini.rub.de

import string
import sys
from lxml import etree
import PIL
import Image
import numpy
import scipy


def im_transform_randombackground(im, out_size, transf, data, filter):
    if transf != Image.EXTENT:
        print "transformation not supported in im_transform_randombackground:", transf
        return None

#    x0, y0, x1, y1 = map(int, data)
    x0, y0, x1, y1 = data
    im_width, im_height = im.size
    if x0>=0 and y0>=0 and x1 <= im_width-1 and y1 <= im_height-1:
        return im.transform(out_size, Image.EXTENT, data, filter)

    patch_width = out_size[0]
    patch_height = out_size[1]    
    noise = numpy.random.randint(256, size=(patch_height, patch_width))

    out_im = scipy.misc.toimage(noise, mode="L")

    xp0, yp0, xp1, yp1 = x0, y0, x1, y1
    Xp0, Yp0, Xp1, Yp1 = 0, 0, patch_width-1, patch_height-1
    
    if x0 < 0:
        xp0=0
        Xp0=(xp0-x0) * patch_width/(x1-x0)
    if y0 < 0:
        yp0=0
        Yp0=(yp0-y0) * patch_height/(y1-y0)
    if x1 > im_width-1:
        xp1= im_width-1
        Xp1=(xp1-x0) * patch_width/(x1-x0)
    if y1 > im_height-1:
        yp1= im_height-1
        Yp1=(yp1-y0) * patch_height/(y1-y0)

#    Warning, are integer coordinates really needed?
    Xp0, Yp0, Xp1, Yp1 = map(int, [Xp0, Yp0, Xp1, Yp1])
#   is this test correct? i guess the variables are inverted!
#   it was: if xp0 < 0 or yp0 < 0 or xp1 > im_width-1 or yp1 > im_height-1:
#   Modified version:
    if xp1 <= 0 or yp1 <= 0 or xp0 >= im_width-1 or yp0 >= im_height-1:
        print "transformation warning: patch fully out of image"
        return out_im
    
    
    #Get Image patch from original image
    out_size2 = (Xp1 - Xp0, Yp1 - Yp0)
    data2 = (xp0, yp0, xp1, yp1)
    print out_size2, data2

    if out_size2[0] > 0 and out_size2[1] > 0:     
        im_tmp = im.transform(out_size2, transf, data2, filter)    
        #Copy sampled image patch into noise image
        out_im.paste(im_tmp, (Xp0, Yp0, Xp1, Yp1))

    return out_im

#Coordinates: [eyeL, eyeR, mouth] (no nose!)
def normalize_image(filename, coordinates, normalization_method = "eyes_mouth_area", centering = "mid_eyes_mouth", out_size = (256,192), convert_format="L", verbose=False, allow_random_background=True):
    LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Mouth_x, Mouth_y = coordinates

    try:
        im = Image.open(filename)
        im = im.convert(convert_format)
    except:
        print "failed opening image", filename
        return None

    eyes_x_m = (RightEyeCenter_x + LeftEyeCenter_x) / 2.0
    eyes_y_m = (RightEyeCenter_y + LeftEyeCenter_y) / 2.0

    midpoint_eyes_mouth_x = (eyes_x_m + Mouth_x) / 2.0
    midpoint_eyes_mouth_y = (eyes_y_m + Mouth_y) / 2.0

    dist_eyes = numpy.sqrt((LeftEyeCenter_x - RightEyeCenter_x)**2 + (LeftEyeCenter_y - RightEyeCenter_y)**2) 
    
    #Triangle formed by the eyes and the mouth.
    height_triangle = numpy.sqrt((eyes_x_m - Mouth_x)**2 + (eyes_y_m - Mouth_y)**2) 
      
    if LeftEyeCenter_x > RightEyeCenter_x:
        print "Warning: the eyes are ordered incorrectly!!! in ", filename
        exit()

    #Assumes eye line is perpendicular to the line from eyes_m to mouth
    current_area = dist_eyes * height_triangle / 2.0
    desired_area = 37.0 * 42.0 / 2.0 

    print "Normalization:"+normalization_method, "Centering="+centering

    #Find scale => ori_width, ori_height
    if normalization_method == "eyes_mouth_area":        
        scale_factor =  numpy.sqrt(current_area / desired_area )
        ori_width = out_size[0]*scale_factor 
        ori_height = out_size[1]*scale_factor
    elif normalization_method == "eyes_only":
        ori_width = out_size[0]/38.0 * dist_eyes    
        ori_height = out_size[1]/out_size[0]*ori_width
    else:
        er = "Error in normalization: Unknown Method:" + str(normalization_method)
        raise er

    if centering=="mid_eyes_mouth":
        center_x = midpoint_eyes_mouth_x
        center_y = midpoint_eyes_mouth_y
    elif centering == "eyes_only":
        center_x = eyes_x_m
        center_y = eyes_y_m
    elif centering == "eyeL":
        center_x = LeftEyeCenter_x
        center_y = LeftEyeCenter_y
    elif centering == "eyeR":
        center_x = RightEyeCenter_x
        center_y = RightEyeCenter_y
    elif centering == "noFace":
        angle = numpy.random.uniform(0, 2*numpy.pi)
        center_x = midpoint_eyes_mouth_x + 0.75*ori_width * numpy.cos(angle)
        center_y = midpoint_eyes_mouth_y + 0.75*ori_height * numpy.sin(angle)
        ori_width = ori_width / 2
        ori_height = ori_height / 2
    else:
        er = "Error in centering: Unknown Method:"+str(centering)
        raise er        
        
    x0 = int(center_x-ori_width/2)
    x1 = int(center_x+ori_width/2)
    y0 = int(center_y-ori_height/2)
    y1 = int(center_y+ori_height/2)        
        
    data = (x0,y0,x1,y1)

    if allow_random_background:
        im_out = im_transform_randombackground(im, out_size, Image.EXTENT, data, Image.BILINEAR)
    else:
        if x0<0 or y0<0 or x1>=im.size[0] or y1>=im.size[1]:
            print "Normalization Failed: Not enough background to cut"
            im_out = None
        else:
            im_out = im.transform(out_size, Image.EXTENT, data, Image.BILINEAR)

    return im_out

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Incorrect usage: %s coordinate_file output_pattern mode"%sys.argv[0]
    
    #CREATE NORMALIZATION FILE
    normalization_filename = sys.argv[1] #CAS_PEAL_coordinates.txt
    output_pattern = sys.argv[2] # image%05d.jpg
    mode = sys.argv[3]

    if mode == "mid_eyes_mouth":
        normalization_method = "eyes_mouth_area"
        centering="mid_eyes_mouth"
        out_dir = "normalized/"
        num_tries = 1        
        allow_random_background=True
    elif mode == "background":
        normalization_method = "eyes_mouth_area"
        centering="noFace"
        out_dir = "noFace/"
        num_tries = 10
        allow_random_background=False
    else:
        print "unknown normalization/centering mode, aborting"
        quit()
        
    normalization_file = open(normalization_filename, "r")
    count = 0
    working = 1
    max_count = 20000
    while working==1 and count < max_count:
        filename = normalization_file.readline().rstrip()
        if filename == "":
            working = 0
        else: 
            coords_str = normalization_file.readline()
            coords = string.split(coords_str, sep=" ")
            float_coords = map(float, coords)
            dist_eyes = numpy.sqrt((float_coords[2]-float_coords[0]) ** 2 + (float_coords[3]-float_coords[1])**2)
            
            if dist_eyes < 15: #20
                print "image ", filename, "has a too small face: dist_eyes = %f pixels"%dist_eyes
            else:
                for repetition in range(num_tries):    
                    im2 = normalize_image(filename, float_coords, normalization_method=normalization_method, centering=centering, out_size = (256,192), convert_format="L", verbose=False, allow_random_background=allow_random_background)
                    if im2 == None:
                        print "image ", filename, "was not properly normalized"
                    else:
                        im2.save(out_dir+output_pattern%count, "JPEG", quality=90)
                        count += 1
    normalization_file.close() 

#Example python facenormalization.py Caltech_coordinates.txt image%05d.jpg mid_eyes_mouth
#Output files are in the directory normalized for this mode, otherwise in directory no Face
