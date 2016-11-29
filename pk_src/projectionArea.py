"""
Given a plane (4 vertices, 1 facet) and an triangulated mesh to calculte the projected Area
onto the plane (projection in the inverse direction of the face normal of the plane).
The distance for the threshold is measured from the vertice is measures from the plane to the centroid of each face.
A ray is constructed from each of the faces to the plane and only those faces without any other intersection than
the plane (only faces directly visible from the plane) are considered part of the altitude map.
Points with a larger distance than a given threshold will be discarded.

**see also:** :ref:`axisParallelPlane`

**command:** cmds.altitudeMap(obj, plane, threshold = 10.0, cosinusmax = 0.0)

**Args:**
    :file(f): path to save altitude map to ASCII-file; if string is empty, no data will be written
    :threshold(t): threshold of maximum distance from plane; all facets with larger distance will be discarded (default 10.0)
    :cosinusmax(c): cosinus of the angle threshhold between the plane and the faces that should be considered (default 0.0, for backface culling). Can be used to exclude/include steep/backwards oriented faces. Input any value >1 to include evry facet in the given threshold
    :select(s): boolean (default False) indicating whether the outline points should be selected if present in the current object (created intersection points can and will not be selected)

:returns: the area of the projection of the mesh onto the plane
"""


import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.api.OpenMaya as om2
import maya.cmds as cmds
import operator
import misc
import math
import string
import numpy as np
import contourShape

dot = lambda x, y: sum(map(operator.mul, x, y))
"""dot product as lambda function to speed up calculation"""
normalize = lambda v: map(operator.div, v, [math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])] * 3)
"""normalization of 3D-vector as lambda function to speed up calculation"""
EPSILON = 1e-5


class projectionArea(OpenMayaMPx.MPxCommand):
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def doIt(self, argList):
        # get objects from argument list
        try:
            obj = misc.getArgObj(self.syntax(), argList)
            if (len(obj) != 2):
                cmds.warning("There must be exactly 2 selected objects (object and plane)!")
                return

            # get vertices and face normals of object and plane (in local space to use transformation matrices of each
            # frame and avoid overhead using the getPoints-method)
            # use 4D-Vectors for transformation of points
            obj_vtx, plane_vtx = misc.getPointsAsList(obj[0], worldSpace=True), \
                misc.getPointsAsList(obj[1], worldSpace=True)
            obj_n, plane_n = misc.getFaceNormals2(obj[0], worldSpace=True), \
                misc.getFaceNormals2(obj[1], worldSpace=True)
            # check if one of the selected object has plane properties (4 vertices, 1 normal)
            if not ((len(obj_vtx) == 4 or len(plane_vtx) == 4) and (len(obj_n) == 1 or len(plane_n) == 1)):
                cmds.warning("None of the objects is a plane (4 vtx, 1 normal)!")
                return
            # in case the selection order is wrong, switch the variables (obj[0] is the object, obj[1] is the plane)
            if len(obj_vtx) == 4 and len(obj_n) == 1:
                obj[0], obj[1] = obj[1], obj[0]
                obj_vtx, plane_vtx = plane_vtx, obj_vtx
                obj_n, plane_n = plane_n, obj_n

            plane_n = plane_n[0]
            plane_n = 1.0*np.array(plane_n)
            plane_n /= math.sqrt(sum(plane_n*plane_n))

            obj_n = [np.array(n) / math.sqrt(sum(np.array(n)*n)) for n in obj_n]

            # plane_centroid = map(operator.div, reduce(lambda x, y: map(operator.add, x, y), plane_vtx), [4.0] * 4)
        except:
            cmds.warning("No object selected!")
            return

        # parse arguments
        argData = om.MArgParser(self.syntax(), argList)
        s_file = argData.flagArgumentString('file', 0) if (argData.isFlagSet('file')) else ""
        threshold = argData.flagArgumentDouble('threshold', 0) if (argData.isFlagSet('threshold')) else 10.0
        angle_culling = argData.flagArgumentDouble('radiant', 0) if (argData.isFlagSet('radiant')) else 0
        # animation = argData.flagArgumentBool('anim', 0) if (argData.isFlagSet('anim')) else False
        select = argData.flagArgumentBool('select', 0) if (argData.isFlagSet('select')) else False

        # get triangles of object model
        obj_tri = misc.getTriangles(obj[0])
        # make sure each face normal belongs to one triangle
        if (len(obj_n) != len(obj_tri)):
            cmds.warning(
                "Number of face normals and number of triangles are not equal! Please triangulate mesh and retry!")
            return

        # remove all triangles whose face normal points in same direction as plane normal (backface culling)
        obj_tri_bc = [obj_tri[i] for i, n in enumerate(obj_n) if dot(plane_n, n) < angle_culling]

        # remove all tringles that are further away than threshhold
        obj_tri_th = [tri for tri in obj_tri_bc
                      if 0 <= np.dot(plane_n, (np.array(
                misc.centroidTriangle([obj_vtx[tri[i]] for i in range(3)]))-plane_vtx[0])) < threshold]
        # obj_tri_th = obj_tri_bc

        # get rotation pivot point
        rp = cmds.xform(obj[1], q=1, rp=1, ws = 1)

        # get rotation of plane
        # if not in xyz order, convert
        roo = cmds.xform(obj[1], q=True, ws=True, roo=True)

        r = cmds.xform(obj[1], q=True, ro=True, ws=True)

        # rotate points around pivot point such that the object lies onto the
        # x-z plane (y-coordinates of plane would be equal at evry point)
        # rotate in reverse order around the rotation pivot with negative angles of the respective values for the plane
        obj_vtx_rotated = contourShape.rotate(obj_vtx, rp, -r[0], -r[1], -r[2], order=roo[::-1], rad=False)

        triRefs = misc.getTriangleEdgesReferences(obj_tri_th)

        outerEdges = contourShape.getOuterEdges(triRefs)

        segments = contourShape.getSegments(obj_vtx_rotated, outerEdges, triRefs, obj_tri_th)

        # select remaining segments used for area calculation to give a visual feedback
        print 'count of outer segments:'
        print len(segments)

        nSegments = contourShape.getPolygon(segments)

        print 'count of segments in outline:'
        print len(nSegments)

        if s_file != "":
            o_file = open(s_file, 'w')
            o_file.write("seg_ind,p0_ind,p0_x,p0_y,p0_z,p1_ind,p1_x,p1_y,p1_z,turn\n")
            for i, seg in enumerate(nSegments):
                o_file.write(",".join((str(i), str(seg.left.ind), str(seg.left.x), str(seg.left.y), str(seg.left.z),
                                       str(seg.right.ind), str(seg.right.x), str(seg.right.y), str(seg.right.z),
                                       str(seg.turn))))
                o_file.write("\n")
                o_file.flush()
            o_file.close()

        if select:
            ind = set([])
            # createdPoints = []
            for seg in nSegments:
                # print seg.left.ind
                # print seg.right.ind
                if not seg.left.created:
                    ind.add(seg.left.ind)
                # else:
                #     createdPoints.append(seg.left)
                if not seg.right.created:
                    ind.add(seg.right.ind)
                    # else:
                    #     createdPoints.append(seg.right)

            cmds.select(clear=True)
            for i in ind:
                cmds.select(obj[0] + '.vtx[' + str(i) + ']', add=True)

        area = contourShape.signedArea(nSegments)
        self.setResult(area)


# creator function
def projectionAreaCreator():
    return OpenMayaMPx.asMPxPtr(projectionArea())


# syntax creator function
def projectionAreaSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("f", "file", om.MSyntax.kString)
    syntax.addFlag("t", "threshold", om.MSyntax.kDouble)
    syntax.addFlag("c", "cosinusmax", om.MSyntax.kDouble)
    syntax.addFlag("s", "select", om.MSyntax.kBoolean)
    syntax.addFlag("a", "anim", om.MSyntax.kBoolean)
    return syntax


# create button for shelf
def addButton(parentShelf):
    pass
