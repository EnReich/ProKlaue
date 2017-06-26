# script to export axes from a ct scene, where the axes are positioned along the bones
# to an animated scene, while keeping the relative position of the axis w.r.t. the bone
# first specify a base dir and the scene paths
# then specify the bones names (for each scene, same order)
# then specify the corresponding axes names (list of lists, for each bone one list which axes belongs to this bone,
# oder is important here!)

import maya.cmds as cmds
import os

base_dir = "C:/Users/Kai/Documents/ProKlaue/testdaten/achsen/testdaten"           # base dir
scene_animated = "Alma_Karera_8erkeys.mb"     # scene paths relative to base dir
scene_ct = "Alma_ct.mb"
base_dir = os.path.abspath(base_dir)        # nothing to do here
scene_animated = os.path.abspath("{}/{}".format(base_dir, scene_animated))
scene_ct = os.path.abspath("{}/{}".format(base_dir, scene_ct))
scene_animated_new = os.path.abspath("{}_cs.mb".format(".".join(scene_animated.split(".")[:-1])))
bones_animated = [u'Hornkapsel_links:Mesh', u'Klauenbein_links:Mesh', u'Kronbein_links:Mesh', u'Fesselbein_links:Mesh',
                  u'Hornkapsel_rechts:Mesh', u'Klauenbein_rechts:Mesh', u'Kronbein_rechts:Mesh', u'Fesselbein_rechts:Mesh']     # bones in the animated scene
bones_ct = [u'Hornkapsel_links:Mesh', u'Klauenbein_links:Mesh', u'Kronbein_links:Mesh', u'Fesselbein_links:Mesh',
            u'Hornkapsel_rechts:Mesh', u'Klauenbein_rechts:Mesh', u'Kronbein_rechts:Mesh', u'Fesselbein_rechts:Mesh']           # in the ct scene

# axes = [[], ["Klauenbein_Kronbein_links"], ["Kronbein_Klauenbein_links", "Kronbein_Fesselbein_links"], ["Fesselbein_Kronbein_links"],
#         [], ["Klauenbein_Kronbein_rechts"], ["Kronbein_Klauenbein_rechts", "Kronbein_Fesselbein_rechts"], ["Fesselbein_Kronbein_rechts"]]             # corresponding axes

axes = [[], ["saddle_Klauenbein_links_Kronbein_links"],
        ["saddle_Kronbein_links_Klauenbein_links", "saddle_Kronbein_links_Fesselbein_links"],
        ["saddle_Fesselbein_links_Kronbein_links"],
        [], ["saddle_Klauenbein_rechts_Kronbein_rechts"],
        ["saddle_Kronbein_rechts_Klauenbein_rechts", "saddle_Kronbein_rechts_Fesselbein_rechts"],
        ["saddle_Fesselbein_rechts_Kronbein_rechts"]]             # corresponding axes

# more variables, nothing necessarily to specify here
trans_dir = os.path.abspath("{}/trans/anim".format(base_dir))
rep_dir = os.path.abspath("{}/rep/anim".format(base_dir))
axes_export_dir = os.path.abspath("{}/ex/anim".format(base_dir))

exported_axis_files = {}

# everything else will be handled by the script for you :)
# open ct scene
cmds.file(scene_ct, open=True, force=True)
# calculate position files for the bones
rep_files = cmds.repositionVertices(bones_ct, ex=True, f=rep_dir, sf=1)
#export axes
axes_set = set([ax for sl in axes for ax in sl])
for ax in axes_set:
    export_path = os.path.abspath("{}/ax_{}.mb".format(axes_export_dir, ax))
    if cmds.listRelatives(ax, parent=True) is not None:
        cmds.parent(ax, world=True)
    cmds.select(clear=True)
    cmds.select(ax)
    exported_file = cmds.file(export_path, type='mayaBinary', es=True)
    exported_axis_files[ax] = exported_file

# open animated scene
cmds.file(scene_animated, open=True, force=True)

# for each bone calculate reposition and for each axis import, reposition and parent it
for idx_bone_animated, bone_animated in enumerate(bones_animated):
    bone_ct = bones_ct[idx_bone_animated]
    rep_file_for_bone = rep_files[idx_bone_animated]
    axes_for_bone = axes[idx_bone_animated]

    for idx_axis_for_bone, axis_for_bone in enumerate(axes_for_bone):
        # calculate reposition
        trans_files = cmds.repositionVertices(bone_animated, ex=False, f=rep_file_for_bone,
                                              st="{}".format(trans_dir), da=True)

        # import axis
        axis_file = exported_axis_files[axis_for_bone]
        new_nodes = cmds.file(axis_file, i=True, rnn=True, mnc=True)
        tf_nodes = [n for n in new_nodes if cmds.objectType(n) == "transform"]
        # imported_axis = tf_nodes[-1].split("|")[-1]
        imported_axis = tf_nodes[0].split("|")[-1]

        # apply transformation on the imported
        cmds.applyTransformationFile(imported_axis, f=trans_files[0], inverse=True)

        # parent axis
        # cmds.parent(imported_axis, world=True)
        cmds.parent(imported_axis, bone_animated)

# save under new name
cmds.file(rename=scene_animated_new)
cmds.file(save=True)

