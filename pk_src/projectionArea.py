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

            # get vertices and face normals of object and plane (in local space to use transformation matrices of each frame and avoid overhead using the getPoints-method)
            # use 4D-Vectors for transformation of points
            obj_vtx, plane_vtx = misc.getPointsAsList(obj[0], worldSpace=True), misc.getPointsAsList(obj[1], worldSpace=True)
            obj_n, plane_n = misc.getFaceNormals(obj[0], worldSpace=True), misc.getFaceNormals(obj[1], worldSpace=True)
            # check if one of the selected object has plane properties (4 vertices, 1 normal)
            if not ((len(obj_vtx) == 4 or len(plane_vtx) == 4) and (len(obj_n) == 1 or len(plane_n) == 1)):
                cmds.warning("None of the objects is a plane (4 vtx, 1 normal)!")
                return
            # in case the selection order is wrong, switch the variables (obj[0] is the object, obj[1] is the plane)
            if (len(obj_vtx) == 4 and len(obj_n) == 1):
                obj[0], obj[1] = obj[1], obj[0]
                obj_vtx, plane_vtx = plane_vtx, obj_vtx
                obj_n, plane_n = plane_n, obj_n

            plane_n = plane_n[0]
            # plane_centroid = map(operator.div, reduce(lambda x, y: map(operator.add, x, y), plane_vtx), [4.0] * 4)
        except:
            cmds.warning("No object selected!")
            return

        # parse arguments
        argData = om.MArgParser(self.syntax(), argList)
        s_file = argData.flagArgumentString('file', 0) if (argData.isFlagSet('file')) else ""
        threshold = argData.flagArgumentDouble('threshold', 0) if (argData.isFlagSet('threshold')) else 10.0
        animation = argData.flagArgumentBool('anim', 0) if (argData.isFlagSet('anim')) else False

        # get triangles of object model
        obj_tri = misc.getTriangles(obj[0])
        # make sure each face normal belongs to one triangle
        if (len(obj_n) != len(obj_tri)):
            cmds.warning(
                "Number of face normals and number of triangles are not equal! Please triangulate mesh and retry!")
            return

        # remove all triangles whose face normal points in same direction as plane normal (backface culling)
        obj_tri_bc = [obj_tri[i] for i, n in enumerate(obj_n) if dot(plane_n, n) < 0]

        #get rotate pivot point
        rp = cmds.xform(obj[1], q=1, rp=1, ws = 1)

        #get rotation of plane
        r = cmds.xform(obj[1], q=1, ro=1)


        #rotate points around pivot point such that the object lies onto the x-y plane (z-coordinates of plane would be equal at evry point)
        obj_vtx_rotated = contourShape.rotate(obj_vtx, rp, r[0]-90, r[1], r[2], rad=False)

        triRefs = misc.getTriangleEdgesReferences(obj_tri_bc)

        outerEdges = contourShape.getOuterEdges(triRefs)

        print obj_vtx_rotated
        print outerEdges

        poly = contourShape.getPolygon(obj_vtx_rotated, outerEdges, triRefs)
        print poly

        area = contourShape.signedArea(poly)

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
    syntax.addFlag("a", "anim", om.MSyntax.kBoolean)
    return syntax


# create button for shelf
def addButton(parentShelf):
    pass
