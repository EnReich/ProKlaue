# script to reposition the raw models to the position of the clean ones, raw models need to have 1 suffix character
# but otherwise identical names to clean models

import numpy as np
o = cmds.ls(sl=1)
for obj in o:
    obj_unbereinigt = obj[:-1]
    rotM_bereinigt = np.matrix(cmds.getAttr(obj+".rotM")).reshape(4,4)
    iPos_bereinigt = cmds.getAttr(obj+".importPosition")[0]
    rotM_unbereinigt = np.matrix(cmds.getAttr(obj_unbereinigt+".rotM")).reshape(4,4)
    iPos_unbereinigt = cmds.getAttr(obj_unbereinigt+".importPosition")[0]

    #cmds.makeIdentity(obj_unbereinigt, a=1)

    # if iPos_bereinigt[0]<2 and iPos_bereinigt[1]<2 and iPos_bereinigt[2]<2:
    #     #move unbereinigt to bereinigt
    #     cmds.move(-iPos_bereinigt[0], -iPos_bereinigt[1], -iPos_bereinigt[2], obj_unbereinigt)
    #     cmds.makeIdentity(obj_unbereinigt, a=1)
    #     cmds.xform(obj_unbereinigt, m=rotM_bereinigt.A1)
    #     cmds.makeIdentity(obj_unbereinigt, a=1)
    # else:
    cmds.xform(obj_unbereinigt, m=rotM_unbereinigt.transpose().A1)
    cmds.makeIdentity(obj_unbereinigt, a=1, r=1)
    cmds.move(iPos_unbereinigt[0], iPos_unbereinigt[1], iPos_unbereinigt[2], obj_unbereinigt)
    cmds.makeIdentity(obj_unbereinigt, a=1, t=1)
    cmds.move(-iPos_bereinigt[0], -iPos_bereinigt[1], -iPos_bereinigt[2], obj_unbereinigt)
    cmds.makeIdentity(obj_unbereinigt, a=1, t=1)
    cmds.xform(obj_unbereinigt, m=rotM_bereinigt.A1)
    cmds.makeIdentity(obj_unbereinigt, a=1, r=1)
    #cmds.setAttr(obj_unbereinigt+".rotM", rotM_bereinigt.A1, type='matrix')
    #cmds.setAttr(obj_unbereinigt+".importPosition", iPos_bereinigt[0], iPos_bereinigt[1],iPos_bereinigt[2], type='float3')

    scale_factor = (100. / cmds.getVolume(obj)) ** (1. / 3)
    cmds.scale(scale_factor, scale_factor, scale_factor, obj_unbereinigt)
    cmds.scale(scale_factor, scale_factor, scale_factor, obj)
    cmds.makeIdentity(obj_unbereinigt, a=1, s=1)
    cmds.makeIdentity(obj, a=1, s=1)


