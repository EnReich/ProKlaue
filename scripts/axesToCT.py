# script to export axes from an animated scene, where the axes are calculated from the animation data
# to a scene in CT position, while keeping the relative position of the axis w.r.t. the bone
# first specify a base dir and the scene paths
# then specify the bones names (for each scene, same order)
# then specify the corresponding axes names (list of lists, for each bone one list which axes belongs to this bone,
# oder is important here!)
# now specify a frame where the position of the axis is ideal (for each axis, again a list of lists,
# frame numbers w.r.t. the animated scene, in the same order as above)

import maya.cmds as cmds
import os

base_dir = "C:/Users/Kai/Documents/ProKlaue/testdaten/achsen"           # base dir
scene_animated = "Alma_Kura_axes.mb"     # scene paths relative to base dir
scene_ct = "Alma_clean_Ursprung_rescaliert.mb"
base_dir = os.path.abspath(base_dir)        # nothing to do here
scene_animated = os.path.abspath("{}/{}".format(base_dir, scene_animated))
scene_ct = os.path.abspath("{}/{}".format(base_dir, scene_ct))
scene_ct_new = os.path.abspath("{}.mb".format(".".join(scene_ct.split(".")[:-1])))
bones_animated = ["Fesselbein_links:Mesh"]     # bones in the animated scene
bones_ct = ["Fesselbein_links:Mesh"]           # in the ct scene
axes = [["ha2_Fesselbein_links1_copy_long_axis_shape"]]             # corresponding axes
frames = [[490]]           # frames


# more variables, nothing necessarily to specify here
trans_dir = os.path.abspath("{}/trans/ct".format(base_dir))
rep_dir = os.path.abspath("{}/rep/ct".format(base_dir))
axes_export_dir = os.path.abspath("{}/ex/ct".format(base_dir))

exported_axis_files = []

# everything else will be handled by the script for you :)
# open ct scene
cmds.file(scene_ct, open=True, force=True)
# calculate position files for the bones
rep_files = cmds.repositionVertices(bones_ct, ex=True, f=rep_dir, sf=1)

# open animated scene
cmds.file(scene_animated, open=True, force=True)

for idx_bone_animated, bone_animated in enumerate(bones_animated):
    bone_ct = bones_ct[idx_bone_animated]
    rep_file_for_bone = rep_files[idx_bone_animated]
    axes_for_bone = axes[idx_bone_animated]
    frames_for_bone = frames[idx_bone_animated]
    # duplicate axes
    axes_duplicates_for_bone = cmds.duplicate(axes_for_bone)
    for idx_axis_for_bone, axis_for_bone in enumerate(axes_for_bone):
        frame = frames_for_bone[idx_axis_for_bone]
        # set current time
        cmds.currentTime(frame)
        # unparent duplicated axis
        cmds.parent(axes_duplicates_for_bone[idx_axis_for_bone], world=True)
        # calculate reposition
        trans_files = cmds.repositionVertices(bone_animated, ex=False, f=rep_file_for_bone,
                                st="{}/{}".format(trans_dir, frame), da=True)
        # apply transformation on the duplicate
        axis_duplicate = axes_duplicates_for_bone[idx_axis_for_bone]
        cmds.applyTransformationFile(axis_duplicate, f=trans_files[0])
        # export the duplicate
        cmds.select(clear=True)
        cmds.select(axis_duplicate)
        export_path = os.path.abspath("{}/ax_{}_{}.mb".format(axes_export_dir, bone_animated.replace(":", "_"), frame))
        exported_file = cmds.file(export_path, type='mayaBinary', es=True)
        exported_axis_files.append(exported_file)

cmds.file(scene_ct, open=True, force=True)
# import axes
for axis_file in exported_axis_files:
    cmds.file(axis_file, i=True)
# save under new name
cmds.file(rename=scene_ct_new)
cmds.file(save=True)
