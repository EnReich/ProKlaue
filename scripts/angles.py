# script to calculate the angles of 2 axis over the curse of an animation
# specify the range of the animation in the variable frames and the path to save
# and select the 2 axes


import maya.cmds as cmds
import numpy as np
import math
import os
from pk_src import misc


def angles(a, b):
    return math.atan2(np.cross(a,b), np.dot(a,b))*180/math.pi


frames = range(0, 830)
sel = cmds.ls(sl=1)
axis1 = sel[0]
axis2 = sel[1]
# path = "rot-{}-{}.csv".format(axis1.replace(":", "_"), axis2.replace(":", "_"))
base_dir = "C:\\Users\\Kai\\Documents\\ProKlaue\\testdaten\\achsen\\ergebnisse"
path = "{}/rot-{}-{}.csv".format(base_dir, "_".join(axis1.split("_")[1:3]), "_".join(axis2.split("_")[1:3]))
path = os.path.abspath(path)
o_file = open(path, 'w')

# o_file.write("frame,RXY,RXZ,RYZ,X1,Y1,Z1,X2,Y2,Z2,rot1,rot2,rotDiff\n")
#
# for frame in frames:
#     cmds.currentTime(frame)
#     trans1 = cmds.xform(axis1, q=1, ws=1, m=1)
#     trans2 = cmds.xform(axis2, q=1, ws=1, m=1)
#     # remove translation
#     transl1 = trans1[12:15]
#     transl2 = trans2[12:15]
#     trans1[12:15] = [0, 0, 0]
#     trans2[12:15] = [0, 0, 0]
#     # now get the rotation matrix for the difference w.r.t. axis 1
#     rot = np.linalg.inv(np.matrix(trans1).reshape(4, 4)) * np.matrix(trans2).reshape(4, 4)
#     # now use this to calculate angles in the planes
#     x = np.matrix([1,0,0,1])
#     y = np.matrix([0,1,0,1])
#     x_rot = np.array(x*rot).flatten()
#     y_rot = np.array(y*rot).flatten()
#     x_rot_into_xy_plane = x_rot[0:2]
#     x_rot_into_xz_plane = x_rot[[0, 2]]
#     y_rot_into_yz_plane = y_rot[1:3]
#     angle_xy_plane = angles(np.matrix([1, 0]), x_rot_into_xy_plane)
#     angle_xz_plane = angles(np.matrix([1, 0]), x_rot_into_xz_plane)
#     angle_yz_plane = angles(np.matrix([1, 0]), y_rot_into_yz_plane)
#     o_file.write('{},"{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}"\n'.format(frame, angle_xy_plane, angle_xz_plane, angle_yz_plane,
#         transl1[0], transl1[1], transl1[2], transl2[0], transl2[1], transl2[2], str(np.matrix(trans1).reshape(4, 4)).replace("\n ", ","), str(np.matrix(trans1).reshape(4, 4)).replace("\n ", ","), str(rot).replace("\n ", ",")))
#     #o_file.flush()
# o_file.close()


o_file.write("frame,R1,R2,RF,distance,X1,Y1,Z1,X2,Y2,Z2,Ax1,Ax2,AxF,Ax1Ref,Ax2Ref\n")

for frame in frames:
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

    dir1_other = np.cross(dir1, dir1_ref) # pi/2 ahead
    dir2_other = np.cross(dir2, dir2_ref) # pi/2 ahead

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
