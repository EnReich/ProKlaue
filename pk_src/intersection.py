"""
Calculates an approximate intersection volume of two transform objects by intersection of the tetrahedra given by the delaunay triangulation of the convex decomposition of both objects. Because the convex hull usually is a poor approximation of the original object, the `V-HACD <https://github.com/kmammou/v-hacd>`_ library is used to find an approximate convex decomposition of the object itself and then use the convex parts to create triangulated tetrahedra to approximate the intersection volume by pairwise intersection tests.

So both objects will be decomposed in convex parts, every part will be triangulated with the delaunay triangulation and their tetrahedra will be intersected pairwise. The sum of the intersection volume of all pairwise tetrahedra is the sum of the intersection of both convex decompositions. To speed up the calculation there is a first evaluation which determines all intersection candidates for a tetrahedron from the first convex hull with possible tetrahedra of the second convex hull. Candidates are all those tetrahedra which lie within or intersect the axis aligned bounding box (their minimal and maximal range in each axis need to overlap). Secondly there is a more accurate collision test with all the candidates which determines which candidates actually intersect the current tetrahedron (using the axis separating theorem, see :ref:`collision_tet_tet`). Finally all remaining tetrahedra will be intersected and their intersection volume will be calculated (see :ref:`intersection_tet_tet`).

All necessary functions are implemented without any Maya commands to speed up calculation; from past experience it can be concluded that Maya commands are much slower than a fresh implementation in Python.
An Interval Tree approach was tested and did not lead to performance improvements.

Command only accepts 'transform' nodes and multiple objects can be selected. The output will be a matrix with the volume of the convex hulls for each object in the main diagonal and in the upper triangular matrix are the pairwise intersections between each combination of objects. Additionally the volumes of the objects itself (not their convex decompositions) will be printed during runtime.

The parameter arguments are the same as for the command :ref:`vhacd` and will be simply handed down to the vhacd command invocation.

**see also:** :ref:`collision_tet_tet`, :ref:`intersection_tet_tet`, :ref:`getVolume`, :ref:`vhacd`

**command:** cmds.intersection([obj], kcd = True)

**Args:**
    :obj: string with object's name inside maya
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

:returns: string containing a nxn matrix (n: number of objects) with intersection volumes.

:raises RunTimeError: if volume of convex decomposition is smaller than volume of given object. Indicates that there are holes inside convex decomposition which obviously leads to incorrect results for the intersection volume. Solution is to choose smaller depth paramter

**Example:**
    .. code-block:: python

        cmds.polyTorus()
        # Result: [u'pTorus1', u'polyTorus1'] #
        cmds.polyTorus()
        # Result: [u'pTorus2', u'polyTorus2'] #
        cmds.xform(ro = [90,0,0])
        cmds.makeIdentity(a = 1, r = 1)
        cmds.polyTorus()
        # Result: [u'pTorus3', u'polyTorus3'] #
        cmds.xform(ro = [0,0,90])
        cmds.makeIdentity(a = 1, r = 1)
        select -r pTorus3 pTorus2 pTorus1 ;
        cmds.intersection(kcd = 0)
        volume 0: 4.77458035128
        volume 1: 4.77458035128
        volume 2: 4.77458035128
        # Result: [[ 5.97810349  1.91942589  1.92072237]
         [ 0.          5.97378722  1.91621988]
         [ 0.          0.          5.97281007]] #
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.api.OpenMaya as om2
import maya.cmds as cmds
import numpy as np
import operator
import misc
import sys
import re
import copy
from pk_src import collision_tet_tet
from pk_src import intersection_tet_tet
from pk_src import vhacd

class intersection(OpenMayaMPx.MPxCommand):
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)
        # make sure to use the newest version of the files (to avoid out-of-date definitions)
        reload(sys.modules["pk_src.collision_tet_tet"])
        reload(sys.modules["pk_src.intersection_tet_tet"])

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
            # check if given objects are the vhacd outputs
            #if (reduce(lambda x,y: x+y, [True for o in obj if re.search("_vhacd$",o)]) != len(obj)):
            #    cmds.error("Expected VHACD objects (group names ending with '_vhacd[0-9]*')")
            #    return
        except:
            cmds.warning("No objects selected or only one object given!")
            return

        argData = om.MArgParser (self.syntax(), argList)

        # read all arguments and set default values
        keepCD = argData.flagArgumentBool('keepConvexDecomposition', 0) if (argData.isFlagSet('keepConvexDecomposition')) else True
        # get all flags for vhacd over static parsing method
        vhacd_par = vhacd.vhacd.readArgs(argData)
        if (vhacd_par is None):
            cmds.error("V-HACD: one or more arguments are invalid!")
            return

        # list of convexHulls (each list entry will be a list of convex parts of each object)
        convexHulls = []
        # get delaunay triangulation of each selected object and save results in list
        # delaunay[0:n] --> object data; delaunay[i][0:m] --> all tetrahedra of object i; delaunay[i][j][0:4] --> 3D points of j-th tetrahedra of i-th object
        delaunay = []

        # get convex decomposition (cd) and save results in list (names of each group)
        cd = cmds.vhacd(obj, **vhacd_par)

        # for each convex decomposition get the actual mesh nodes and triangulate them
        for i,o in enumerate(cd):
            # actual mesh objects are child nodes of child groups under the main group
            meshes = cmds.listRelatives(cmds.listRelatives(o))
            # mesh itself has some corrupt triangle definition (only 1 triangle) --> recalculate convex hull to avoid problems (probable cause: Maya 2012 ma-parser bug)
            meshes = [cmds.convexHull(mesh) for mesh in meshes]
            # add all convex hulls of one object to list of hulls (list of lists where each sub list contains the names of the convex hull decomposition parts)
            convexHulls.append(meshes)

            # get delaunay triangulation of each convex sub part as list of tetrahedra (tmp holds all tetrahedra of ONE single object; all convex hulls are put together without further distinction between sub parts)
            tmp = []
            for ch in convexHulls[-1]:
                tmp.extend ([np.fromstring(x, sep = ",").reshape(4,3) for x in cmds.delaunay(ch)])

            # put all delaunay triangulations of each complete object together into list (list of objects where each object consists of a list of tetrahedra)
            delaunay.append(tmp)

        # preprocessing of min and max values in each dimension for each tetrahedra
        x_minMax = []
        y_minMax = []
        z_minMax = []
        for d in delaunay:
            x_minMax.append([ [min(tetra[:,0]), max(tetra[:,0])]  for tetra in d ])
            y_minMax.append([ [min(tetra[:,1]), max(tetra[:,1])]  for tetra in d ])
            z_minMax.append([ [min(tetra[:,2]), max(tetra[:,2])]  for tetra in d ])

        # iterate over all pairwise object combinations and save volumes in nxn-table
        volumes = np.zeros((len(delaunay), len(delaunay)))
        # class object for collision test
        colTest = collision_tet_tet.Collision_tet_tet()
        # class object for intersection
        inter = intersection_tet_tet.intersection_tet_tet()

        # over each pair of objects
        for i in range(len(delaunay)):
            for j in range(i, len(delaunay)):
                # if i equals j then just calculate volume of current object and volume of convex hull
                if (i == j):
                    vol = cmds.getVolume(obj[i])
                    # calc. volumes of each convex hull part and add up all values
                    volumes[i,j] = reduce(lambda x,y: x+y, [cmds.getVolume(ch) for ch in convexHulls[i]])
                    # check volume against convex hull --> indicates hole inside convex hull approximation
                    if (vol > volumes[i,j]):
                        cmds.error("Volume Error: volume of object '{0}' is bigger than volume of convex decomposition! This indicates holes inside vhacd-result. Please check decomposition and adjust parameter for vhacd (e.g. smaller depth parameter)".format(obj[i]))

                    print("volume {}: {}".format(i, vol))
                    continue
                # over each pair of tetrahedra
                for n1 in range(len(delaunay[i])):
                    # set tetrahedron of first object
                    colTest.setV1(delaunay[i][n1])
                    inter.setV1(delaunay[i][n1])

                    # get list of candidate tetrahedra, i.e. those tetrahedra whose bounding box overlaps with the bounding box of current tetrahedron
                    candidates = [x for x in range(len(delaunay[j])) if not (x_minMax[i][n1][0] >= x_minMax[j][x][1] or x_minMax[i][n1][1] <= x_minMax[j][x][0])]
                    candidates = [y for y in candidates if not (y_minMax[i][n1][0] >= y_minMax[j][y][1] or y_minMax[i][n1][1] <= y_minMax[j][y][0])]
                    candidates = [z for z in candidates if not (z_minMax[i][n1][0] >= z_minMax[j][z][1] or z_minMax[i][n1][1] <= z_minMax[j][z][0])]

                    # check each candidate tetrahedra
                    for n2 in candidates:
                        # check if the two current tetrahedra can intersect each other
                        colTest.setV2(delaunay[j][n2])
                        if (not colTest.check()):
                            continue

                        inter.setV2(delaunay[j][n2])

                        # actual intersection (returns a list of triangles with 3d coordinates)
                        isSet = inter.intersect()
                        vols = [misc.signedVolumeOfTriangle(tri[0], tri[1], tri[2]) for tri in isSet]
                        # get volume of intersection
                        if (len(vols)):
                            volumes[i,j] += reduce(lambda x,y: x+y, vols)
        # if user wants to keep convex decomposition, create a group with all parts for each object
        if (keepCD):
            for i,g in enumerate(convexHulls):
                cmds.group(g, n = obj[i] + '_vhacd', w = 1)
        else:
            # else delete all convex hull objects
            [cmds.delete(group) for group in convexHulls]
        # delete convex decomposition structures (original vhacd output)
        cmds.delete(cd)

        self.setResult(str(volumes))

# creator function
def intersectionCreator():
    return OpenMayaMPx.asMPxPtr( intersection() )

# syntax creator function
def intersectionSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("kcd", "keepConvexDecomposition", om.MSyntax.kBoolean)
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
