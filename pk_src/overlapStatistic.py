"""
Calculates a statistic for the overlapping volume of multiple transform objects by intersection of the tetrahedra given
by the delaunay triangulation of the convex decomposition of both objects.

Objects are normed for a given volume before approximation with VHACD.

The parameter arguments are the same as for the command :ref:`vhacd` and :ref:`intersection` and will be simply handed
down to the command invocation.

**see also:** :ref:`collision_tet_tet`, :ref:`intersection_tet_tet`, :ref:`getVolume`, :ref:`vhacd`, :ref:`intersection`

**command:** cmds.overlapStatistic([obj], kcd=True, nrm=True, nvol=100)

**Args:**
    :obj: string with object's name inside maya
    :norm(nrm): norm the volume of the objects (default True)
    :normVolume(nvl): scale the objects so that they have a normed volume of the given value, only applies if norm Flag is set True (default 100)
    :saveFile(sf): save the statistics in a file at the given path
    :keepConvexDecomposition(kcd): should the convex decompositions (intermediate data) be kept (True, default) or deleted (False)
    :tmp(temporaryDir): directory to save temporary created files (needs read and write access). If no path is given the temporary files will be written into the user home directory
    :executable(exe): absolute path to the executable V-HACD file. If no path is given maya/bin/plug-ins/bin/testVHACD is used.
    :resolution(res): maximum number of voxels generated during the voxelization stage (10.000 - 64.000.000, default: 100.000)
    :depth(d): maximum number of clipping stages. During each split stage, all the model parts (with a concavity higher than the user defined threshold) are clipped according the "best" clipping plane (1 - 32, default: 20)
    :concavity(con): maximum concavity (0.0 - 1.0, default: 0.001)
    :planeDownsampling(pd): controls the granularity of the search for the "best" clipping plane (1 - 16, default: 4)
    :convexHullDownsampling(chd): controls the precision of the convex-hull generation process during the clipping plane selection stage (1 - 16, default: 4)
    :alpha(a): controls the bias toward clipping along symmetry planes (0.0 - 1.0, default: 0.05)
    :beta(b): controls the bias toward clipping along revolution axes (0.0 - 1.0, default: 0.05)
    :gamma(g): maximum allowed concavity during the merge stage (0.0 - 1.0, default: 0.0005)
    :normalizeMesh(pca): enable/disable normalizing the mesh before applying the convex decomposition (True/False, default: False)
    :mode(m): voxel-based approximate convex decomposition (0, default) or tetrahedron-based approximate convex decomposition (1)
    :maxNumVerticesPerCH(vtx): controls the maximum number of triangles per convex-hull (4 - 1024, default: 64)
    :minVolumePerCH(vol): controls the adaptive sampling of the generated convex-hulls (0.0 - 0.01, default: 0.0001)

:returns: statistic measures for the pairwise intersection volumes in a upper triangle matrix.

:raises RunTimeError: if volume of convex decomposition is smaller than volume of given object. Indicates that there are holes inside convex decomposition which obviously leads to incorrect results for the intersection volume. Solution is to choose smaller depth paramter
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.cmds as cmds
import misc
import numpy as np
import os
import sys
from pk_src import vhacd


class overlapStatistic(OpenMayaMPx.MPxCommand):
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)
        # make sure to use the newest version of the files (to avoid out-of-date definitions)
        # reload(sys.modules["pk_src.collision_tet_tet"])
        # reload(sys.modules["pk_src.intersection_tet_tet"])
        # reload(sys.modules["pk_src.intersection"])

    def doIt(self, argList):
        # get objects from argument list (size >= 2)
        try:
            obj = misc.getArgObj(self.syntax(), argList)
            if (len(obj) < 2):
                cmds.error("Select at least 2 objects!")
                return
            if (cmds.objectType(obj[0]) != 'transform' or cmds.objectType(obj[1]) != 'transform'):
                cmds.error("Object is not of type transform!")
                return
        except:
            cmds.warning("No objects selected or only one object given!")
            return

        argData = om.MArgParser(self.syntax(), argList)

        # read all arguments and set default values
        keepCD = argData.flagArgumentBool('keepConvexDecomposition', 0) if (argData.isFlagSet('keepConvexDecomposition')) else True
        nvol = argData.flagArgumentBool('normVolume', 0) if (
            argData.isFlagSet('normVolume')) else 100
        norm = argData.flagArgumentBool('norm', 0) if (
            argData.isFlagSet('norm')) else True
        s_file = os.path.abspath(argData.flagArgumentString('saveFile', 0)) if (argData.isFlagSet('saveFile')) else ""
        save = False
        if s_file != "":
            o_file = open(s_file, 'w')
            o_file.write("i0, i1, name0, name1, its_volume, dice\n")
            save = True

        # get all flags for vhacd over static parsing method
        vhacd_par = vhacd.vhacd.readArgs(argData)
        if (vhacd_par is None):
            cmds.error("V-HACD: one or more arguments are invalid!")
            return

        #norm the volume
        if norm:
            for o in obj:
                v = cmds.getVolume(o)
                scale_factor = (nvol/ v) ** (1./3)
                cmds.xform(o, scale=[scale_factor, scale_factor, scale_factor])
                cmds.makeIdentity(o, apply=True)

        intersection_matrix = np.matrix(str(cmds.intersection(obj, kcd=keepCD, matlabOutput=True, **vhacd_par)))
        dice_matrix = np.matrix(intersection_matrix)

        for i in range(0, intersection_matrix.shape[0]):
            for j in range(i, intersection_matrix.shape[1]):
                its_volume = intersection_matrix[i, j]
                dice_matrix[i, j] = (its_volume*2)/(intersection_matrix[i, i]+intersection_matrix[j, j])
                if save:
                    o_file.write(",".join((str(i), str(j), obj[i], obj[j], str(its_volume), str(dice_matrix[i, j]))))
                    o_file.write("\n")
                    o_file.flush()

        if save:
            o_file.close()

        self.setResult(str(dice_matrix))



# creator function
def overlapStatisticCreator():
    return OpenMayaMPx.asMPxPtr(overlapStatistic())


# syntax creator function
def overlapStatisticSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("kcd", "keepConvexDecomposition", om.MSyntax.kBoolean)
    syntax.addFlag("nrm", "norm", om.MSyntax.kBoolean)
    syntax.addFlag("nvl", "normVolume", om.MSyntax.kDouble)
    syntax.addFlag("sf", "saveFile", om.MSyntax.kString)

    # flags for vhacd
    syntax.addFlag("tmp", "temporaryDir", om.MSyntax.kString)
    syntax.addFlag("exe", "executable", om.MSyntax.kString)
    syntax.addFlag("res", "resolution", om.MSyntax.kLong)
    syntax.addFlag("d", "depth", om.MSyntax.kLong)
    syntax.addFlag("con", "concavity", om.MSyntax.kDouble)
    syntax.addFlag("pd", "planeDownsampling", om.MSyntax.kLong)
    syntax.addFlag("chd", "convexHullDownsampling", om.MSyntax.kLong)
    syntax.addFlag("a", "alpha", om.MSyntax.kDouble)
    syntax.addFlag("b", "beta", om.MSyntax.kDouble)
    syntax.addFlag("g", "gamma", om.MSyntax.kDouble)
    syntax.addFlag("pca", "normalizeMesh", om.MSyntax.kBoolean)
    syntax.addFlag("m", "mode", om.MSyntax.kBoolean)
    syntax.addFlag("vtx", "maxNumVerticesPerCH", om.MSyntax.kLong)
    syntax.addFlag("vol", "minVolumePerCH", om.MSyntax.kDouble)
    return syntax


# create button for shelf
def addButton(parentShelf):
    pass
