import numpy
import os
import shutil
from  eban_SFA_libs import *

Warning!!! newer version in the kitchen computer!!!

def facegen_string_randomize_all(executable="D:\\mmueller\\fg\\bin\\FaceGen_Full_Sdk.exe", controls_ctl="si.ctl", original_fg="averag_base.fg", output_fg="output.fg", race="all"):
    if race not in ["all", "african", "european", "seAsian", "eIndian"]:
        print "WARNING: Unsupported race ", race
        race = "all"
    command = "%s %s %s %s random %s"%(executable, controls_ctl, original_fg, output_fg, race)
    return command

def facegen_string_set_age_shape(age_shape, executable="D:\\mmueller\\fg\\bin\\FaceGen_Full_Sdk.exe", controls_ctl="si.ctl", original_fg="averag_base.fg", output_fg="output.fg"):
    if age_shape < 15 or age_shape > 65:
        print "Warning age might be out of range"
    command = "%s %s %s %s age all shape %.4f"%(executable, controls_ctl, original_fg, output_fg, age_shape)
    return command

def facegen_string_set_age_texture(age_texture, executable="D:\\mmueller\\fg\\bin\\FaceGen_Full_Sdk.exe", controls_ctl="si.ctl", original_fg="averag_base.fg", output_fg="output.fg"):
    if age_texture < 15 or age_texture > 65:
        print "Warning age might be out of range"
    command = "%s %s %s %s age all texture %.4f"%(executable, controls_ctl, original_fg, output_fg, age_texture)
    return command

def facegen_string_set_gender_shape(gender_shape, executable="D:\\mmueller\\fg\\bin\\FaceGen_Full_Sdk.exe", controls_ctl="si.ctl", original_fg="averag_base.fg", output_fg="output.fg"):
    if gender_shape < -4 or gender_shape > 4:
        print "Warning gender might be out of range"
    command = "%s %s %s %s gender all shape %.4f"%(executable, controls_ctl, original_fg, output_fg, gender_shape)
    return command

def facegen_string_set_gender_texture(gender_texture, executable="D:\\mmueller\\fg\\bin\\FaceGen_Full_Sdk.exe", controls_ctl="si.ctl", original_fg="averag_base.fg", output_fg="output.fg"):
    if gender_texture < -4 or gender_texture > 4:
        print "Warning gender might be out of range"
    command = "%s %s %s %s gender all texture %.4f"%(executable, controls_ctl, original_fg, output_fg, gender_texture)
    return command

def facegen_string_set_race_from_to(race_tween, executable="D:\\mmueller\\fg\\bin\\FaceGen_Full_Sdk.exe", controls_ctl="si.ctl", original_fg="averag_base.fg", output_fg="output.fg", from_race="all", to_race="all"):
    if from_race not in ["all", "african", "european", "seAsian", "eIndian"]:
        print "WARNING: Unsupported from_race ", from_race
        from_race = "all"
    if to_race not in ["all", "african", "european", "seAsian", "eIndian"]:
        print "WARNING: Unsupported to_race ", to_race
        to_race = "all"
    
    if race_tween < -1 or race_tween > 1:
        print "Warning race_tween might be unsupported"
    command = "%s %s %s %s race %s %s %.4f"%(executable, controls_ctl, original_fg, output_fg, from_race, to_race, race_tween)
    return command


def facegen_complete_simp(randomize=False, race="all", age=None, gender=None, race_tween=None, from_race="all", to_race="all", executable="D:\\mmueller\\fg\\bin\\FaceGen_Full_Sdk.exe", controls_ctl="si.ctl", fg_base_dir=".", original_fg="averag_base.fg", output_fg="output.fg"):
    return facegen_complete(randomize=randomize, race=race, age_shape=age, age_texture=age, gender_shape=gender, gender_texture=gender, race_tween=race_tween, from_race=from_race, to_race=to_race, executable=executable, controls_ctl=controls_ctl, fg_base_dir=fg_base_dir, original_fg=original_fg, output_fg=output_fg)

#WARNING!!! Verify surce and destiny can be the same
#WARNING!!! THINK NAMING, TMP USAGE, RETURNING STRINGS
def facegen_complete(randomize=False, race="all", age_shape=None, age_texture=None, gender_shape=None, gender_texture=None, race_tween=None, from_race="all", to_race="all", executable="D:\\mmueller\\fg\\bin\\FaceGen_Full_Sdk.exe", controls_ctl="si.ctl", fg_base_dir=".", original_fg="averag_base.fg", output_fg="output.fg"):
    commands = []
    
    tmp_fg = fg_base_dir + "/" + "tmp.fg"
    shutil.copy(fg_base_dir + "/" + original_fg, tmp_fg)

    if randomize is True:
        cmd = facegen_string_randomize_all(executable, controls_ctl, original_fg=tmp_fg, output_fg=tmp_fg, race=race)
        commands.append(cmd)

    if age_shape is not None:
        cmd = facegen_string_set_age_shape(age_shape, executable, controls_ctl, original_fg=tmp_fg, output_fg=tmp_fg)
        commands.append(cmd)

    if age_texture is not None:
        cmd = facegen_string_set_age_texture(age_texture, executable, controls_ctl, original_fg=tmp_fg, output_fg=tmp_fg)
        commands.append(cmd)

    if gender_shape is not None:
        cmd = facegen_string_set_gender_shape(gender_shape, executable, controls_ctl, original_fg=tmp_fg, output_fg=tmp_fg)
        commands.append(cmd)

    if gender_texture is not None:
        cmd = facegen_string_set_gender_texture(gender_texture, executable, controls_ctl, original_fg=tmp_fg, output_fg=tmp_fg)
        commands.append(cmd)
        
    if race_tween is not None:
        cmd = facegen_string_set_race_from_to(race_tween, executable, controls_ctl, original_fg=tmp_fg, output_fg=tmp_fg, from_race=from_race, to_race=to_race)
        commands.append(cmd)

    for cmd in commands:
        print cmd
        os.system(cmd)
    shutil.copy(tmp_fg, fg_base_dir + "/" + output_fg)

#-4=very male, -1=male, 1=female
def code_gender(gender):
    if gender is None:
        return 999
    x = int(500 + gender * 125)
    if x<0:
        x=0
    if x>999:
        x=999
    return x

#-1 source, 1=destination
def code_raceTween(raceTween):
    if raceTween is None:
        return 999
    x = int(500 + raceTween * 500)
    if x<0:
        x=0
    if x>999:
        x=999
    return x

facegen_complete(randomize=True, race="all", age_shape=20, age_texture=25, gender_shape=1, gender_texture=0.9, race_tween=0.5, from_race="all", to_race="african", executable="D:\\mmueller\\fg\\bin\\FaceGen_Full_Sdk.exe", controls_ctl="si.ctl", fg_base_dir="./test_fg", original_fg="random000.fg", output_fg="output.fg")

ids=range(0,4)

age_ini = 15
age_end = 65
ages=numpy.arange(age_ini, age_end, 2)
#gender_ini= -1
#gender_end= 1
#genders = numpy.arange(gender_ini, gender_end, 0.1)
genders=[None]
#race_tween_ini= -1
#race_tween_end= 1
#race_tweens = numpy.arange(race_tween_ini, race_tween_end, 0.1)
race_tweens=[None]
parameters = list(product(ids, ages, genders, race_tweens))   

for id in ids:
    output_fg = "output%03d.fg"%id
    facegen_complete_simp(randomize=True, fg_base_dir="./test_fg", output_fg=output_fg)

for id, age, gender, race_tween in parameters:
    input_fg = "output%03d.fg"%id
    output_fg = "output%03d_a%03d_g%03d_rt%03d.fg"%(id, age, code_gender(gender), code_raceTween(race_tween))
    facegen_complete_simp(age=age, gender=gender, race_tween=race_tween, from_race="all", to_race="all", fg_base_dir="./test_fg", original_fg=input_fg, output_fg=output_fg)

    