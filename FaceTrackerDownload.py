import os
import csv

faceindex_filename = "/local/escalafl/Alberto/FaceTracker/facetracer/faceindex.txt"
image_filenameFormat = "/local/escalafl/Alberto/FaceTracker/images/image%05.jpg"
tmpimage_filename = "/local/escalafl/Alberto/FaceTracker/tmp_image"

facesFile = open(faceindex_filename)
facesReader = csv.reader(facesFile, delimiter=' ') # csv parser for annotations file
       
rename_images = True 
convert_jpg = True

for ii, row in enumerate(facesReader):                  
    if len(row) != 3:
        print "Not an image row: ", row
    elif row[0][0] != "#":
        face_id = int(row[0])
        image_filename = image_filenameFormat%face_id
        image_url = row[1]
        extension = image_url[-4:]
        cmd = "wget -o %s%s %"%(tmpimage_filename, extension, image_url)
        print "about to execute: [%s]"%cmd

#        if extension != ".jpg" and extension != ".JPG":
        cmd2 = "convert %s%s %s"%(tmpimage_filename, extension, image_filename)
        print "about to execute: [%s]"%cmd

    else:
        print "Ignored comment row: ", row

facesFile.close()