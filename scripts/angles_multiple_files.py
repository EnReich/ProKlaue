# -*- coding: utf-8 -*-

# script to calculate the angles of 2 axis over the curse of an animation
# specify the range of the animation in the variable frames and the path to save
# and select the 2 axes
# Kronbein, dann Fesselbein
# Klauenbein, dann Kronbein

import maya.cmds as cmds
import numpy as np
import math
import os, glob, re, errno
from itertools import compress
from pk_src import misc



base_dir = os.path.abspath("C:\\Users\\Kai\\Documents\\ProKlaue - Sabrina")
scenes_dir = os.path.abspath(os.path.join(base_dir, "testdaten", "animated cs"))
ergebnis_dir = os.path.abspath(os.path.join(base_dir, "ergebnisse"))
frames = range(0, 820)
ground_specifier_count = 2  # how many words are used to describe the ground type *_beton_* -> 1, *_Barhuf_weich_* -> 2
axis_pairs = [
    ["Hufbein", "Kronbein"],
    ["Kronbein", "Fesselbein"]
]

def angles(a, b):
    return math.atan2(np.cross(a,b), np.dot(a,b))*180/math.pi

scenes_animated = glob.glob(os.path.join(scenes_dir, "*.mb"))

cmds.progressWindow(isInterruptable=True,
                    title = 'Reading angles from animated scenes',
                    progress = 0,
                    status = '',
                    maxValue = len(scenes_animated))

scene_idx = -1

for scene_animated in scenes_animated:
    # Check if the dialog has been cancelled
    if cmds.progressWindow(query=True, isCancelled=True):
        break

    scene_idx += 1
    cmds.progressWindow(edit=True, progress=scene_idx, status=('Processing: {}'.format(os.path.split(scene_animated)[1])))

    cmds.file(scene_animated, open=True, force=True)
    scene_animated_file_name = os.path.split(scene_animated)[1]
    animal_name = scene_animated_file_name.split("_")[0]
    ground_name = "_".join(scene_animated_file_name.split("_")[1:(1+ground_specifier_count)])

    list_of_objs = cmds.ls(transforms=True)

    for axis_pair in axis_pairs:
        axis1_match = [(re.search(r'(saddle)(.)*({0})(.)*({1})'.format(axis_pair[0], axis_pair[1]), obj_name)
                        is not None)
                       for  obj_name in list_of_objs]
        axis2_match = [(re.search(r'(saddle)(.)*({1})(.)*({0})'.format(axis_pair[0], axis_pair[1]), obj_name)
                        is not None)
                       for obj_name in list_of_objs]

        axis1 = list_of_objs[list(compress(xrange(len(axis1_match)), axis1_match))[0]]
        axis2 = list_of_objs[list(compress(xrange(len(axis2_match)), axis2_match))[0]]

        # path = os.path.join(ergebnis_dir, animal_name, ground_name,
        #                     "rot-{0}-{1}.csv".format("_".join(axis1.split("_")[1:3]), "_".join(axis2.split("_")[1:3])))
        path = os.path.join(ergebnis_dir, animal_name, ground_name, "rot-{0}-{1}.csv".format(axis_pair[0], axis_pair[1]))
        path = os.path.abspath(path)
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        o_file = open(path, 'w')

        o_file.write("frame,R1,R2,RF,distance,X1,Y1,Z1,X2,Y2,Z2,Ax1,Ax2,AxF,Ax1Ref,Ax2Ref\n")

        for frame in frames:
            if cmds.progressWindow(query=True, isCancelled=True):
                break

            cmds.currentTime(frame)

            pos1 = np.array(cmds.xform(axis1, q=1, t=1, ws=1))
            pos2 = np.array(cmds.xform(axis2, q=1, t=1, ws=1))

            distance = np.linalg.norm(pos2-pos1)

            ch1 = cmds.listRelatives(axis1, c=1, typ="transform")
            ch2 = cmds.listRelatives(axis2, c=1, typ="transform")

            cone1 = cmds.listRelatives(ch1[0], c=1, typ="transform")[0]
            cone2 = cmds.listRelatives(ch2[0], c=1, typ="transform")[0]

            dir1 = cmds.xform(cone1, q=1, t=1, ws=1)-pos1
            dir2 = cmds.xform(cone2, q=1, t=1, ws=1)-pos2
            dir_floating = np.cross(dir1, dir2)

            dir1 /= np.linalg.norm(dir1)
            dir2 /= np.linalg.norm(dir2)
            dir_floating /= np.linalg.norm(dir_floating)

            cone1_ref = cmds.listRelatives(ch1[1], c=1, typ="transform")[0]
            cone2_ref = cmds.listRelatives(ch2[1], c=1, typ="transform")[0]

            dir1_ref = cmds.xform(cone1_ref, q=1, t=1, ws=1) - pos1
            dir2_ref = cmds.xform(cone2_ref, q=1, t=1, ws=1) - pos2

            dir1_ref /= np.linalg.norm(dir1_ref)
            dir2_ref /= np.linalg.norm(dir2_ref)

            dir1_other = np.cross(dir1, dir1_ref) # pi/2 ahead wrt. dir1_ref around dir1
            dir2_other = np.cross(dir2, dir2_ref) # pi/2 ahead wrt. dir2_ref around dir2

            r1 = math.acos(np.clip(np.dot(dir1_ref, dir_floating),-1,1))*misc.RAD_TO_DEG
            r2 = math.acos(np.clip(np.dot(dir2_ref, dir_floating),-1,1))*misc.RAD_TO_DEG


            if np.dot(dir1_other,dir_floating)>0:
                r1 *= -1

            if np.dot(dir2_other,dir_floating)>0:
                r2 *= -1

            r_floating = math.acos(np.clip(np.dot(dir1, dir2),-1,1))*misc.RAD_TO_DEG

            dir2_ahead = np.array((misc.getRotationAroundAxis(angle=math.pi/2, v=dir_floating, rad=True)*np.matrix(dir2).reshape(3,1))).reshape(3)
            if np.dot(dir1, dir2_ahead)>0:
                r_floating *= -1

            o_file.write('{},"{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}"\n'.format(frame, r1, r2, r_floating, distance,
                pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2], dir1, dir2, dir_floating, dir1_ref, dir2_ref))
            #o_file.flush()
        o_file.close()

cmds.progressWindow(endProgress=1)
