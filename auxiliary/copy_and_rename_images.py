#Function useful to copy images from multiple directories to a single one with a unique name convension
#Library version first adapted on 7.9.2010
#Alberto Escalante. alberto.escalante@ini.rub.de
#Ruhr-University-Bochum, Institute of Neurocomputation, Group of Prof. Dr. Wiskott

import os
import sys
import shutil

#Example:
#python /home/escalafl/work3/cuicuilco/src/auxiliary/copy_and_rename_images.py "/scratch/escalafl/faces/Caltech/CaltechNormalClean/" "/scratch/escalafl/faces/Caltech/CaltechNormalClean2/" 0 "image%05d.jpg"


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "Incorrect usage: %s base_orig_dir base_dest_dir base_nr filename_pattern"%sys.argv[0]
        quit()
        
    base_orig_dir = sys.argv[1] 
    base_dest_dir = sys.argv[2] 
    base_nr = int(sys.argv[3]) 
    filename_pattern = sys.argv[4] # image%05d.jpg

    print "copying files from %s to %s"%(base_orig_dir, base_dest_dir)
    print "first number is %d"%base_nr

    dirList=os.listdir(base_orig_dir)
    dirList.sort()
    for i, fname in enumerate(dirList):
        fin = base_orig_dir+fname
        fout = base_dest_dir + filename_pattern%(i+base_nr)
        print fin, "->", fout
        shutil.copy(fin, fout)
    