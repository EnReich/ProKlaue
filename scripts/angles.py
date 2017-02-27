# script to calculate the angles of 2 axis over the curse of an animation
# specify the range of the animation in the variable frames and the path to save
# and select the 2 axes


import maya.cmds as cmds
import numpy as np
import math
import os


def angles(a, b):
    return math.atan2(np.cross(a,b), np.dot(a,b))*180/math.pi


frames = range(0, 670)
sel = cmds.ls(sl=1)
axis1 = sel[0]
axis2 = sel[1]
path = "rot-{}-{}.csv".format(axis1.replace(":", "_"), axis2.replace(":", "_"))
path = os.path.abspath(path)
o_file = open(path, 'w')
o_file.write("frame,RXY,RXZ,RYZ,X1,Y1,Z1,X2,Y2,Z2,rot1,rot2,rotDiff\n")

for frame in frames:
    cmds.currentTime(frame)
    trans1 = cmds.xform(axis1, q=1, ws=1, m=1)
    trans2 = cmds.xform(axis2, q=1, ws=1, m=1)
    # remove translation
    transl1 = trans1[12:15]
    transl2 = trans2[12:15]    
    trans1[12:15] = [0, 0, 0]
    trans2[12:15] = [0, 0, 0]
    # now get the rotation matrix for the difference w.r.t. axis 1
    rot = np.linalg.inv(np.matrix(trans1).reshape(4, 4)) * np.matrix(trans2).reshape(4, 4)
    # now use this to calculate angles in the planes
    x = np.matrix([1,0,0,1])
    y = np.matrix([0,1,0,1])
    x_rot = np.array(x*rot).flatten()
    y_rot = np.array(y*rot).flatten()
    x_rot_into_xy_plane = x_rot[0:2]
    x_rot_into_xz_plane = x_rot[[0, 2]]
    y_rot_into_yz_plane = y_rot[1:3]
    angle_xy_plane = angles(np.matrix([1, 0]), x_rot_into_xy_plane)
    angle_xz_plane = angles(np.matrix([1, 0]), x_rot_into_xz_plane)
    angle_yz_plane = angles(np.matrix([1, 0]), y_rot_into_yz_plane)
    o_file.write('{},"{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}"\n'.format(frame, angle_xy_plane, angle_xz_plane, angle_yz_plane,
        transl1[0], transl1[1], transl1[2], transl2[0], transl2[1], transl2[2], str(np.matrix(trans1).reshape(4, 4)).replace("\n ", ","), str(np.matrix(trans1).reshape(4, 4)).replace("\n ", ","), str(rot).replace("\n ", ",")))
    #o_file.flush()
o_file.close()
