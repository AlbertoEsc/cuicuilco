import string
import sys
from lxml import etree
import PIL
import Image
import numpy
import scipy

def load_coordinate_data(metadata_file="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_2.0_Metadata_corrected.xml"):
    coordinate_data_tree = etree.parse(metadata_file)
    coordinate_data_root = coordinate_data_tree.getroot()

    if coordinate_data_root.tag != "CoordinateData":
        print "ERROR, Unknown Tree/Root:" + coordinate_data_root.tag
        sys.exit()
    
    dict_coordinate_data  = {}
    for recording in coordinate_data_root:
        subject_id = LeftEyeCenter_x = LeftEyeCenter_y = RightEyeCenter_x = RightEyeCenter_y = Nose_x = Nose_y = Mouth_x = Mouth_y = None
        if recording.tag == "Recording":
            print ":)", 
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
                    print "Unknown bodypart: " + point.tag
            if subject_id != None and LeftEyeCenter_x != None and LeftEyeCenter_y != None and \
            RightEyeCenter_x != None and RightEyeCenter_y != None and Nose_x != None and \
            Nose_y != None and Mouth_x != None and Mouth_y != None:
                dict_coordinate_data[recording_id] = (subject_id, LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y)
            else:
                print "Some coordinate was not found!!! recording_id:", recording_id 
        else:
            print "Unknown 'recording' " + recording.tag + " found while traversing root"    
    print "Finished loading coordinate data!!!"
    return dict_coordinate_data


def load_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.6_Orig.xml", verbose=False):
    biometric_signature_tree = etree.parse(file_biometric_signatures)
    biometric_signature_root = biometric_signature_tree.getroot()
    
    #if biometric_signature_root.tag != "biometric-signature-set":
    #    print "ERROR, Unknown Tree/Root: " + biometric_signature_root.tag
    #    sys.exit()
    
    dict_biometric_signatures  = {}
    for bio_signature in biometric_signature_root:
    #    subject_id = LeftEyeCenter_x = LeftEyeCenter_y = RightEyeCenter_x = RightEyeCenter_y = Nose_x = Nose_y = Mouth_x = Mouth_y = None
        modality = file_name = file_format = None
        if string.find(bio_signature.tag, "complex-biometric-signature") >= 0 or string.find(bio_signature.tag,"biometric-signature")>=0:
            if verbose:
                print ":)", 
    #        recording_id= recording.get("recording_id")
    #        subject_id= recording.get("subject_id")
    #        capturedate= recording.get("capturedate")
    
            for presentation in bio_signature:
                name=presentation.get("name")
                modality=presentation.get("modality")
                file_name=presentation.get("file-name")
                file_format=presentation.get("file-format")            
            dict_biometric_signatures[name] = (modality, file_name, file_format)
        else:
            print "Unknown 'bio_signature' " + bio_signature.tag + " found while traversing root"
    
    print "Finished loading signature data (filenames!!!)!!!"
    return dict_biometric_signatures

def process_image(base_dir, file_name, file_format, coordinates, normalization_method = "eyes_mouth_area", centering="mid_eyes_mouth", out_size = (256,192), convert_format="L", verbose=False):
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

    #Assumes eye line is perpendicular to the line from eyes_m to mouth
    current_area = dist_eyes * height_triangle / 2.0
    desired_area = 37.0 * 42.0 / 2.0 
      
    if dist_eyes < 0:
        print "Warning: the eyes are ordered incorrectly!!! in ", file_name
        dist_eyes=dist_eyes * -1

    #Find scale => ori_width, ori_height
    if normalization_method == "eyes_mouth_area":        
        scale_factor =  numpy.sqrt(current_area / desired_area )
        print "Normalization:"+normalization_method
        ori_width = out_size[0]*scale_factor 
        ori_height = out_size[1]*scale_factor
    elif normalization_method == "eyes_only":
        print "Normalization:"+normalization_method
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
    
    if x0<0 or y0<0 or x1>=im.size[0] or y1>=im.size[1]:
        print "Normalization Failed: Not enough background to cut:", data
        im_out = None
    else:
        im_out = im.transform(out_size, Image.EXTENT, data)

    return im_out


def merge_dictionaries(original_dict, additional_dict, verbose=False):
    for item in additional_dict:
        if item in original_dict:
            if original_dict[item] != additional_dict[item]:
                print "Warning, inconsistent entries for item !!!!"
                print "key= ", item, "as", original_dict[item], "and as", original_dict[item]
        else:
            if verbose:
                print "+",
            original_dict[item] = additional_dict[item]


dict_coordinate_data = load_coordinate_data(metadata_file="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_2.0_Metadata_corrected.xml")

print "Entries in dict_coordinate_data=", len(dict_coordinate_data)

dict_biometric_signatures6 = load_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.6_Orig.xml")
dict_biometric_signatures = dict_biometric_signatures6
dict_biometric_signatures5 = load_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.5_Orig.xml")
merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures5)
dict_biometric_signatures4 = load_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.4_Orig.xml")
merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures4)
dict_biometric_signatures3 = load_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.3_Orig.xml")
merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures3)
dict_biometric_signatures2 = load_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.2_Orig.xml")
merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures2)
dict_biometric_signatures1 = load_biometric_signatures(file_biometric_signatures="/home/escalafl/workspace/test_SFA2/src/xml_source/FRGC_Exp_2.0.1_Orig.xml")
merge_dictionaries(dict_biometric_signatures, dict_biometric_signatures1)

print "Entries in dict_coordinate_data=", len(dict_coordinate_data)
print "Entries in dict_biometric_signatures=", len(dict_biometric_signatures)

im_average = None
max_count = None
count=0
print "converting..."
base_dir = "/local/tmp/FRGC-2.0-dist/" # /local2/FRGC-2.0-dist"
#out_dir = "/local/escalafl/Alberto/FRGC_NoFace" # "/local2/tmp/escalafl/Alberto/FRGC_cddsffds"
out_dir = "/local/escalafl/Alberto/FRGC_EyeR" # "/local2/tmp/escalafl/Alberto/FRGC_cddsffds"
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
        im_out = process_image(base_dir, file_name, file_format, int_coords, normalization_method = "eyes_mouth_area", centering="eyeR", out_size = (256,192), convert_format="L")
        if im_out != None:
            if im_average == None:
                im_average = numpy.asarray(im_out)[:] * 1.0
            else:
                im_average = im_average + numpy.asarray(im_out)
#WARNING
#            im_out.save(out_dir + "/image%05d.jpg"%count, "JPEG")
            count += 1
    else:
        print "Recording", recording_id, "missing biometric signature"

##im_average = im_average * 1.0/ count
##average_I = scipy.misc.toimage(im_average.reshape(192, 192, 3), mode="RGB")
##average_I.save("average_image_seenidRGB7.jpg")

print "Count =", count



