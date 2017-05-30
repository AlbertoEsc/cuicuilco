#Functions to load FRGC metadata
#Library version first adapted on 12.07.2010
#Alberto Escalante. alberto.escalante@ini.rub.de

import string
import sys
from lxml import etree
import PIL
from PIL import Image
import numpy
import scipy

#Function that reads FRGC coordinate data from a single file
#The output is a dictionary dict_coordinate_data with the following format:
#dict_coordinate_data[recording_id] =(subject_id, LeftEyeCenter_x, \
#                    LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y)

def load_FRGC_coordinate_data(metadata_file="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_2.0_Metadata_corrected.xml"):
    coordinate_data_tree = etree.parse(metadata_file)
    coordinate_data_root = coordinate_data_tree.getroot()

    if coordinate_data_root.tag != "CoordinateData":
        print "ERROR, Unknown Tree/Root:" + coordinate_data_root.tag
        sys.exit()
    
    dict_coordinate_data  = {}
    attempted = 0
    for recording in coordinate_data_root:
        attempted += 1
        subject_id = LeftEyeCenter_x = LeftEyeCenter_y = RightEyeCenter_x = RightEyeCenter_y = Nose_x = Nose_y = Mouth_x = Mouth_y = None
        if recording.tag == "Recording":
            print "C", 
            recording_id= recording.get("recording_id")
            subject_id= recording.get("subject_id")
            capturedate= recording.get("capturedate")
    
            for point in recording:
                if point.tag == "LeftEyeCenter":           
                    LeftEyeCenter_x =point.get("x")
                    LeftEyeCenter_y =point.get("y")
                elif point.tag == "RightEyeCenter":           
                    RightEyeCenter_x =point.get("x")
                    RightEyeCenter_y =point.get("y")
                elif point.tag == "Nose":
                    Nose_x =point.get("x")
                    Nose_y =point.get("y")
                elif point.tag == "Mouth":
                    Mouth_x =point.get("x")
                    Mouth_y =point.get("y")
                else:
                    print "Unknown bodypart: " + point.tag,

            if subject_id != None and LeftEyeCenter_x != None and LeftEyeCenter_y != None and \
            RightEyeCenter_x != None and RightEyeCenter_y != None and Nose_x != None and \
            Nose_y != None and Mouth_x != None and Mouth_y != None:
                dict_coordinate_data[recording_id] = (subject_id, LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y)
            else:
                print "Some coordinate was not found!!! recording_id:", recording_id, (subject_id, LeftEyeCenter_x, \
                    LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y)
        else:
            print "Unknown 'recording' tag <" + recording.tag + "> found while traversing root"    
    print "Finished loading coordinate data!!!"
    print "%d successful coordinate entries from %d"%(len(dict_coordinate_data), attempted)
    return dict_coordinate_data

#Function that reads FRGC biometric signatures (containing filename of recordings)
#The output is a dictionary d with the following format:
#dict_biometric_signatures[name] = (modality, file_name, file_format)
def load_FRGC_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.6_Orig.xml", verbose=False):
    biometric_signature_tree = etree.parse(file_biometric_signatures)
    biometric_signature_root = biometric_signature_tree.getroot()
    
    #if biometric_signature_root.tag != "biometric-signature-set":
    #    print "ERROR, Unknown Tree/Root: " + biometric_signature_root.tag
    #    sys.exit()
    
    
    dict_biometric_signatures  = {}
    file_formats = set()
    for bio_signature in biometric_signature_root:
    #    subject_id = LeftEyeCenter_x = LeftEyeCenter_y = RightEyeCenter_x = RightEyeCenter_y = Nose_x = Nose_y = Mouth_x = Mouth_y = None
        modality = file_name = file_format = None
        if string.find(bio_signature.tag, "complex-biometric-signature") >= 0 or string.find(bio_signature.tag,"biometric-signature")>=0:
            if verbose:
                print "B", 
    #        recording_id= recording.get("recording_id")
    #        subject_id= recording.get("subject_id")
    #        capturedate= recording.get("capturedate")
    
            for presentation in bio_signature:
                name=presentation.get("name")
                modality=presentation.get("modality")
                file_name=presentation.get("file-name")
                file_format=presentation.get("file-format")            
            dict_biometric_signatures[name] = (modality, file_name, file_format)
            file_formats.add(file_format)
        else:
            print "Unknown 'bio_signature' " + bio_signature.tag + " found while traversing root"
    
    print "Finished loading signature data (filenames!!!)!!!"
    print "File_formats found: ", file_formats
    return dict_biometric_signatures

def process_image_facecenter(base_dir, file_name, file_format, coordinates, normalization_method = "mid_eyes_mouth", out_size = (256,192), convert_format="L", verbose=False):
    LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y = coordinates

    im = Image.open(base_dir+"/"+file_name)
    im = im.convert(convert_format)

    eyes_x_m = (RightEyeCenter_x + LeftEyeCenter_x) / 2.0
    eyes_y_m = (RightEyeCenter_y + LeftEyeCenter_y) / 2.0

    midpoint_eyes_mouth_x = (eyes_x_m + Mouth_x) / 2.0
    midpoint_eyes_mouth_y = (eyes_y_m + Mouth_y) / 2.0

    dist_eyes = numpy.sqrt((LeftEyeCenter_x - RightEyeCenter_x)**2 + (LeftEyeCenter_y - RightEyeCenter_y)**2) 
    
    #Triangle formed by the eyes and the mouth.
    height_triangle = numpy.sqrt((eyes_x_m - Mouth_x)**2 + (eyes_y_m - Mouth_y)**2) 
      
    if dist_eyes < 0:
        print "Warning: the eyes are ordered incorrectly!!! in ", file_name
        dist_eyes=dist_eyes * -1

    #Assumes eye line is perpendicular to the line from eyes_m to mouth
    current_area = dist_eyes * height_triangle / 2.0
    desired_area = 37.0 * 42.0 / 2.0 

    if normalization_method == "mid_eyes_mouth":
        scale_factor =  numpy.sqrt(current_area / desired_area )
        print "Normalization:"+normalization_method
        ori_width = out_size[0]*scale_factor 
        ori_height = out_size[1]*scale_factor
    
        x0 = int(midpoint_eyes_mouth_x-ori_width/2)
        x1 = int(midpoint_eyes_mouth_x+ori_width/2)
        y0 = int(midpoint_eyes_mouth_y-ori_height/2)
        y1 = int(midpoint_eyes_mouth_y+ori_height/2)
    elif normalization_method == "eyes_only":
        print "Normalization:"+normalization_method
        ori_width = out_size[0]/38.0 * dist_eyes    
        ori_height = out_size[1]/out_size[0]*ori_width
    
        x0 = int(eyes_x_m-ori_width/2)
        x1 = int(eyes_x_m+ori_width/2)
        y0 = int(eyes_y_m-ori_height/2)
        y1 = int(eyes_y_m+ori_height/2)
    else:
        print "Normalization: Unknown Method:", normalization_method
    data = (x0,y0,x1,y1)

    if x0<0 or y0<0 or x1>=im.size[0] or y1>=im.size[1]:
        print "Normalization Failed: Not enough background to cut"
        im_out = None
    else:
        im_out = im.transform(out_size, Image.EXTENT, data)

    return im_out

#Safely merges information of two dictionaries, into the first one. 
def merge_dictionaries(original_dict, additional_dict, abort_on_warning=False, verbose=False):
    for item in additional_dict:
        if item in original_dict:
            if original_dict[item] != additional_dict[item]:
                print "Warning, inconsistent entries for item !!!!"
                print "key= ", item, "as", original_dict[item], "and as", original_dict[item]
                if abort_on_warning:
                    er="Aborting. Reason inconsistent entries in dictionaries"
                    raise Exception(er)                   
        else:
            if verbose:
                print "+",
            original_dict[item] = additional_dict[item]

#This is how the first normalized images of FRGC were generated:
normalization_procedure = 1
if __name__ == "__main__" and normalization_procedure == 1:
#   What about ppn files???
    dict_coordinate_data = load_FRGC_coordinate_data(metadata_file="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_2.0_Metadata_corrected.xml")
    
    print "Entries in dict_coordinate_data=", len(dict_coordinate_data)
    
    dict_biometric_signatures6 = load_FRGC_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.6_Orig.xml")
    dict_biometric_signatures = dict_biometric_signatures6
    dict_biometric_signatures5 = load_FRGC_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.5_Orig.xml")
    merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures5)
    dict_biometric_signatures4 = load_FRGC_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.4_Orig.xml")
    merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures4)
    dict_biometric_signatures3 = load_FRGC_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.3_Orig.xml")
    merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures3)
    dict_biometric_signatures2 = load_FRGC_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.2_Orig.xml")
    merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures2)
    dict_biometric_signatures1 = load_FRGC_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.1_Orig.xml")
    merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures1)
    
    print "Entries in dict_coordinate_data=", len(dict_coordinate_data)
    print "Entries in dict_biometric_signatures=", len(dict_biometric_signatures)
    
    im_average = None
    compute_average = False
    max_count = None
    count=0
    print "converting..."
    base_dir = "/local2/FRGC-2.0-dist"
    out_dir = "/local2/tmp/escalafl/Alberto/FRGC_Normalized"
    for recording_id, recording_data in dict_coordinate_data.items():   
        (subject_id, LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y) = recording_data   
        coordinates = (LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y)
    #
        if max_count != None and count > max_count:
            break
        int_coords = []
        for i in coordinates:
            int_coords.append(int(i))
    #   
        if recording_id in dict_biometric_signatures:
            print "Retrieving filename for", recording_id
            modality, file_name, file_format = dict_biometric_signatures[recording_id]
            print "Filename=",file_name, "with file format", file_format
    #WARNING, "RGB"->"L", out_size = (256,192)
            if compute_average:
                im_out = process_image_facecenter(base_dir, file_name, file_format, int_coords, out_size = (192,192), convert_format="RGB")
            else:
                im_out = process_image_facecenter(base_dir, file_name, file_format, int_coords, out_size = (256,192), convert_format="L")                

            if im_out != None:
                if compute_average:
                    if im_average == None:
                        im_average = numpy.asarray(im_out)[:] * 1.0
                    else:
                        im_average = im_average + numpy.asarray(im_out)    #WARNING
                else:
                    im_out.save(out_dir + "/image%05d.jpg"%count, "JPEG")
                count += 1
            else:
                print "Image %s / %s was not properly loaded"%(base_dir, file_name)
        else:
            print "Recording", recording_id, "missing biometric signature"
    
    if compute_average:
        im_average = im_average * 1.0/ count
        average_I = scipy.misc.toimage(im_average.reshape(192, 192, 3), mode="RGB")
        average_I.save("average_image_seenidRGB7.jpg")    
    print "Count =", count



