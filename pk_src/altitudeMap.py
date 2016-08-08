"""
Uses an axis parallel plane (app) and the object model connected to this plane to create an altitude map, i.e. a set of perpendicular distances from the faces of the object to the plane. The distance is measures from the plane to the centroid of each face. A ray is constructed from each of the faces to the plane and only those faces without any other intersection than the plane (only faces directly visible from the plane) are considered part of the altitude map.
Points with a larger distance than a given threshold will be discarded.

**see also:** :ref:`axisParallelPlane`

**command:** cmds.altitudeMap([obj, plane], file = "", threshold = 10.0)

**Args:**
    :file(f): path to save altitude map to ASCII-file; if string is empty, no data will be written
    :threshold(t): threshold of maximum distance from plane; all points with larger distance will be discarded (default 10.0)

:returns: list of centroid points of faces, their indices in the mesh vtx-list and their distances to the plane, i.e. '[[n_1, x_1, y_1, z_1, d_1], [n_2, x_2, y_2, z_2, d_2], ...]'

**Example:**
    .. code-block:: python

        cmds.polyCube()
        # Result: [u'pCube1', u'polyCube1'] #
        cmds.xform(ro = [10, 20, 30])
        cmds.axisParallelPlane(p = 'xz', pos = 'min')
        # Result: [u'pCube1_app', u'polyPlane1'] #
        cmds.polyTriangulate("pCube1")
        # Result: [u'polyTriangulate1'] #
        cmds.altitudeMap('pCube1', 'pCube1_app')
        # Result: [[4, -0.3983890351647723, 0.0597721458595899, -0.3785089467837944, 0.7449914933365491],
        # [5, 0.019866728794979728, -0.07780045709588722, -0.5469076316145288, 0.607418890381072],
        # [6, 0.021764807311746848, -0.5225944965679014, -0.1788206947620432, 0.16262485090905787],
        # [7, 0.4192048032181355, -0.3599696226914842, 0.01564478359550836, 0.32524972478547504],
        # [10, -0.3964909566480051, -0.3850218936124242, -0.010422009931308688, 0.300197453864535],
        # [11, -0.4173067247013684, -0.08482441678052995, 0.35244215325697736, 0.6003949306964292]] # 
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.api.OpenMaya as om2
import maya.cmds as cmds
import operator
import misc
import string
import numpy as np

dot = lambda x,y: sum(map(operator.mul, x, y))
"""dot product as lambda function to speed up calculation"""
EPSILON = 1e-5

class altitudeMap(OpenMayaMPx.MPxCommand):

    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def doIt(self, argList):
        # get objects from argument list
        try:
            obj = misc.getArgObj(self.syntax(), argList)
            if (len(obj) != 2):
                cmds.warning("There must be exactly 2 selected objects (object and plane)!")
                return
            # get vertices and face normals of object and plane
            obj_vtx, plane_vtx = [[p.x, p.y, p.z] for p in misc.getPoints(obj[0])], [[p.x, p.y, p.z] for p in misc.getPoints(obj[1])]
            obj_n, plane_n = misc.getFaceNormals(obj[0]), misc.getFaceNormals(obj[1])
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
            plane_centroid = map(operator.div, reduce(lambda x,y: map(operator.add, x,y), plane_vtx), [4.0]*3)
        except:
            cmds.warning("No object selected!")
            return

        argData = om.MArgParser (self.syntax(), argList)
        s_file = argData.flagArgumentString('file', 0) if (argData.isFlagSet('file')) else ""
        threshold = argData.flagArgumentDouble('threshold', 0) if (argData.isFlagSet('threshold')) else 10.0

        # get triangles of object model
        obj_tri = misc.getTriangles(obj[0])
        # make sure each face normal belongs to one triangle
        if (len(obj_n) != len(obj_tri)):
            cmds.warning("Number of face normals and number of triangles are not equal! Please triangulate mesh and retry!")
            return

        # remove all triangles whose face normal points in same direction as plane normal (backface culling), but keep index of normal/triangle
        obj_tri = [np.append(i, obj_tri[i]) for i,n in enumerate(obj_n) if dot(plane_n, n) < 0]

        # get centroid points of each triangle (to construct a ray in direction of the plane)
        obj_centroid = [[tri[0]] + misc.centroidTriangle([obj_vtx[tri[1]], obj_vtx[tri[2]], obj_vtx[tri[3]]]) for tri in obj_tri]

        # build acceleration structure
        mfnObject = misc.getMesh(obj[0])
        accel = mfnObject.autoUniformGridParams()
        # list to store points and their distance to the plane
        altitudeMap = []

        # now construct ray from face centroids (in inverse direction of plane normal) and intersect with object mesh
        # if intersection is found with object, ray cannot intersect plane; in case of no intersection, calculate distance of centroid to plane
        ray_dir = om2.MFloatVector(map(operator.mul, plane_n, [-1]*3))
        for c in obj_centroid:
            # move origin a little bit away from centroid to avoid self intersection
            origin = om2.MFloatPoint(c[1:]) + EPSILON*ray_dir

            # method returns list ([hitPoint], hitParam, hitFace, hitTriangle, hitBary2, hitBary2)
            # value hitFace equals -1 iff no intersection was found
            hitResult = mfnObject.closestIntersection(origin, ray_dir, om2.MSpace.kWorld, threshold, False, accelParams = accel)
            # if no intersection is found, get distance of centroid to plane
            if (hitResult[3] == -1):
                # distance from centroid to plane is the length of the projection v (centroid plane to centroid triangle) onto unit normal vector of plane
                d = dot(map(operator.sub, c[1:], plane_centroid), plane_n)
                # add centroid as 3D point and distance to plane to altitude map
                if (d <= threshold):
                    altitudeMap.append ( c + [d] )

        # if file name is given, write altitude map
        if (s_file != ""):
            o_file = open(s_file, 'w')
            # for each item in altitude map, remove brackets and comma
            o_file.write("N\tTx\tTy\tTz\td\n")
            for item in altitudeMap:
                o_file.write("%s\n" % string.replace(string.replace(string.strip(str(item), '[]'), ',', ''), ' ', '\t'))
            print('Altitude Map written to file \'%s\'!' % s_file)
        self.setResult(str(altitudeMap))

# creator function
def altitudeMapCreator():
    return OpenMayaMPx.asMPxPtr( altitudeMap() )

# syntax creator function
def altitudeMapSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("f", "file", om.MSyntax.kString)
    syntax.addFlag("t", "threshold", om.MSyntax.kDouble)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass
