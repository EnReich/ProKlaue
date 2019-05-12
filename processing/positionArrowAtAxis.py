# This script positions an arrow (or an arbitrary object for that matter) at an axis in the form of



import maya.cmds as cmds
import numpy as np
import os
from pk_src import misc

# where is the arrow object located
path_to_arrow_object = os.path.abspath(os.path.expanduser("~/../Documents/ProKlaue/res/arrow.mb"))

# how far at the axis should the arrow be placed (0 at the origin, 1 at the begin of the cone)
factor_transpose = 1.55 #1.2
#scale = [3,2,3]
#rotate = [0,90,180]

scale = [1.3,0.35,1.3]
rotate = [0,0,0] # for the flexion axis / z the last number is -180 or 180 (degrees) if one wants to
                 # display a rotation in the opposite mathematical sense as used in the plots later
deg = True  #rotations in degrees

# set the material for the arrow here
shading_grp_for_arrow = "shadingGrpAxisRef" # shadingGrpAxisRef or shadingGrpAxis1 or shadingGrpAxis2





objs = cmds.ls(sl=1)
#arrow = objs[1]
axis = objs[0]

new_nodes = cmds.file(path_to_arrow_object, i=True, rnn=True, mnc=True)
tf_nodes = [n for n in new_nodes if cmds.objectType(n) == "transform"]
# imported_axis = tf_nodes[-1].split("|")[-1]
arrow = tf_nodes[0].split("|")[-1]


# set the material for the arrow
cmds.sets(arrow, fe= shading_grp_for_arrow)

#cmds.makeIdentity(arrow, a=True)

axis_m = np.matrix(cmds.xform(axis, m=True, q=True, ws=True)).reshape(4,4).transpose()
arrow_m = np.matrix(cmds.xform(arrow, m=True, q=True, ws=True)).reshape(4,4).transpose()

axis_scale = cmds.xform(axis, s=True, q=True, ws=True)
axis_m_without_scale = axis_m.copy()
axis_m_without_scale[0:3, 0] = axis_m_without_scale[0:3, 0]*(1/axis_scale[0])
axis_m_without_scale[0:3, 1] = axis_m_without_scale[0:3, 1]*(1/axis_scale[1])
axis_m_without_scale[0:3, 2] = axis_m_without_scale[0:3, 2]*(1/axis_scale[2])


max_transpose = axis_m*np.matrix([0,1,0,1]).reshape(4,1)- \
                axis_m*np.matrix([0,0,0,1]).reshape(4,1)

transpose = factor_transpose * max_transpose


m_rotate = misc.getRotationMatrix(alpha=rotate[0], beta = rotate[1], gamma=rotate[2], rad=(not deg))

arrow_m_new = np.matrix([1,0,0,transpose[0],
                         0,1,0,transpose[1],
                         0,0,1,transpose[2],
                         0,0,0,1]).reshape(4,4)*\
                axis_m_without_scale*\
                m_rotate*\
                np.matrix([scale[0],0,0,0,
                         0,scale[1],0,0,
                         0,0,scale[2],0,
                         0,0,0,1]).reshape(4,4)

cmds.xform(arrow, m=arrow_m_new.transpose().A1)


