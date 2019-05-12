"""
Given a plane (4 vertices, 1 facet) and a triangulated mesh, export the coordinates of all vertices from front faces
wrt. the plane. Coordinates are locally projected onto the plane (default) or globally.
The threshold can be used to exclude faces further away from the plane than a given distance.
The distance for the threshold is measured from the plane to the centroid of the triangle.
Triangles in negative direction from the plane are by default not considered (W.r.t. the plane normal).

**see also:** :ref:`axisParallelPlane`

**command:** cmds.frontVertices(obj, plane, local=True, threshold = float('inf'), cosinusmax = 0.0, negative=-0.05)

**Args:**
    :threshold(t): threshold of maximum distance from plane; all facets with larger distance will be discarded
        (default POSITIVE_INFINITY)
    :cosinusmax(c): cosinus of the angle threshhold between the plane and the faces that should be considered
        (default 0.0, for backface culling). Can be used to exclude/include steep/backwards oriented faces.
    :local(l): local coordinates projected onto the plane or
    :file(f): file path to save the coordinates
    :triangleSegments(tsf): file path to save the front facing triangles in the threshold as segments
    :negative(n): consider faces in negative direction with this distance threshold (default -0.05)

:returns: the area of the projection of the mesh onto the plane
"""

 # :edgeFile(ef): file path to save all edges
 # :outerEdgeFile(oef): file path to save outer edges
 # :triangleFile(ef): file path to save all edges
 # :outerTriangleFile(oef): file path to save outer edges
 # :segmentsFile(sf): file path to save the outer segments


import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.cmds as cmds
import operator
import math
import numpy as np
from pk_src import contourShape, misc

dot = lambda x, y: sum(map(operator.mul, x, y))
"""dot product as lambda function to speed up calculation"""
normalize = lambda v: map(operator.div, v, [math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])] * 3)
"""normalization of 3D-vector as lambda function to speed up calculation"""
EPSILON = 1e-5


class frontVertices(OpenMayaMPx.MPxCommand):
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
        tri_segments_file = argData.flagArgumentString('triangleSegments', 0) if (argData.isFlagSet('triangleSegments')) else ""
        # segments_file = argData.flagArgumentString('segmentsFile', 0) if (argData.isFlagSet('segmentsFile')) else ""
        # edge_file = argData.flagArgumentString('edgeFile', 0) if (argData.isFlagSet('edgeFile')) else ""
        # outer_edge_file = argData.flagArgumentString('outerEdgeFile', 0) if (argData.isFlagSet('outerEdgeFile')) else ""
        # triangle_file = argData.flagArgumentString('triangleFile', 0) if (argData.isFlagSet('triangleFile')) else ""
        # outer_triangle_file = argData.flagArgumentString('outerTriangleFile', 0) if (argData.isFlagSet('outerTriangleFile')) else ""
        threshold = argData.flagArgumentDouble('threshold', 0) if (argData.isFlagSet('threshold')) else float('inf')
        angle_culling = argData.flagArgumentDouble('cosinusmax', 0) if (argData.isFlagSet('cosinusmax')) else 0.0
        local = argData.flagArgumentBool('local', 0) if (argData.isFlagSet('local')) else True
        negative = argData.flagArgumentDouble('negative', 0) if (argData.isFlagSet('negative')) else -0.05

        # get triangles of object model
        obj_tri = misc.getTriangles(obj[0])
        # make sure each face normal belongs to one triangle
        if (len(obj_n) != len(obj_tri)):
            cmds.warning(
                "Number of face normals and number of triangles are not equal! Please triangulate mesh and retry!")
            return

        # remove all triangles whose face normal points in same direction as plane normal (backface culling)
        obj_tri_bc = [obj_tri[i] for i, n in enumerate(obj_n) if dot(plane_n, n) < angle_culling]

        # remove all triangles that are further away than threshhold
        obj_tri_th = [tri for tri in obj_tri_bc
                      if negative <= np.dot(plane_n, (np.array(
                misc.centroidTriangle([obj_vtx[tri[i]] for i in range(3)])) - plane_vtx[0])) < threshold]
        # obj_tri_th = obj_tri_bc

        if local:
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
            obj_vtx_transformed = obj_vtx_rotated-np.array(rp)

        else:
            obj_vtx_transformed = obj_vtx


        vtx_export_idx = np.unique(np.array(obj_tri_th).flatten())
        vtx_export = np.array(obj_vtx_transformed)[vtx_export_idx]

        if s_file != "":
            o_file = open(s_file, 'w')
            o_file.write("ind,x,y,z\n")
            for i, pt in enumerate(vtx_export):
                o_file.write('"{}","{}","{}","{}"\n'.format(vtx_export_idx[i], pt[0], pt[1], pt[2]))
            o_file.close()

        if tri_segments_file != "":
            o_file_tri_seg = open(tri_segments_file, 'w')
            o_file_tri_seg.write("SID,PID,PInd,x,y,z\n")
            SID = 0
            for tri in obj_tri_th:
                for PID, p in enumerate(tri):
                    p_cords = obj_vtx_transformed[p]
                    o_file_tri_seg.write(
                        '"{}","{}","{}","{}","{}","{}"\n'.format(SID, PID, p, p_cords[0], p_cords[1], p_cords[2]))
                SID += 1
            o_file_tri_seg.close()

        # if segments_file !="":
        #     o_file_seg = open(segments_file, 'w')
        #     triRefs = misc.getTriangleEdgesReferences(obj_tri_th)
        #     outerEdges = contourShape.getOuterEdges(triRefs)
        #     rightEnds = {}
        #     leftEnds ={}
        #     complete = set()
        #     for edg in outerEdges:
        #         edg_list = list(edg)
        #         edg0 = edg_list[0]
        #         edg1 = edg_list[1]
        #
        #
        #         if edg0 in rightEnds and edg1 in leftEnds:
        #             if not rightEnds[edg0]==leftEnds[edg1]:
        #                 rightEnds[edg0].extend(leftEnds[edg1])
        #                 d = rightEnds.pop(edg0)
        #                 rightEnds[d[-1]] = d
        #                 leftEnds.pop(edg1)
        #             else:
        #                 complete.add(edg1)
        #
        #         elif edg0 in leftEnds and edg1 in rightEnds:
        #             if not rightEnds[edg1] == leftEnds[edg0]:
        #                 rightEnds[edg1].extend(leftEnds[edg0])
        #                 d = rightEnds.pop(edg1)
        #                 rightEnds[d[-1]] = d
        #                 leftEnds.pop(edg0)
        #             else:
        #                 complete.add(edg0)
        #
        #         elif edg0 in leftEnds and edg1 in leftEnds:
        #             d0_old = leftEnds[edg0]
        #             d1_old = leftEnds[edg1]
        #
        #             leftEnds.pop(edg0)
        #             leftEnds.pop(edg1)
        #
        #             rightEnds.pop(d0_old[-1])
        #             rightEnds.pop(d1_old[-1])
        #
        #             d0_old.extendleft(d1_old)
        #             d_new = d0_old
        #             leftEnds[d_new[0]]   = d_new
        #             rightEnds[d_new[-1]] = d_new
        #
        #
        #         elif edg0 in rightEnds and edg1 in rightEnds:
        #             d0_old = rightEnds[edg0]
        #             d1_old = rightEnds[edg1]
        #
        #             rightEnds.pop(edg0)
        #             rightEnds.pop(edg1)
        #
        #             leftEnds.pop(d0_old[0])
        #             leftEnds.pop(d1_old[0])
        #
        #             d0_old.extend(reversed(d1_old))
        #             d_new = d0_old
        #             leftEnds[d_new[0]] = d_new
        #             rightEnds[d_new[-1]] = d_new
        #
        #         elif edg0 in rightEnds:
        #             d = rightEnds[edg0]
        #             d.extend([edg1])
        #             rightEnds.pop(edg0)
        #             rightEnds[d[-1]] = d
        #
        #         elif edg1 in rightEnds:
        #             d = rightEnds[edg1]
        #             d.extend([edg0])
        #             rightEnds.pop(edg1)
        #             rightEnds[d[-1]] = d
        #
        #         elif edg0 in leftEnds:
        #             d = leftEnds[edg0]
        #             d.extendleft([edg1])
        #             leftEnds.pop(edg0)
        #             leftEnds[d[0]] = d
        #
        #         elif edg1 in leftEnds:
        #             d = leftEnds[edg1]
        #             d.extendleft([edg0])
        #             leftEnds.pop(edg1)
        #             leftEnds[d[0]] = d
        #
        #         else:
        #             d = collections.deque([edg0, edg1])
        #             leftEnds[d[0]]=d
        #             rightEnds[d[-1]]=d
        #
        #     o_file_seg.write("SID,PID,PInd,x,y,z\n")
        #     SID = 0
        #     for leftEnd, dq in leftEnds.iteritems():
        #         print leftEnd in complete
        #         for PID, p in enumerate(dq):
        #             p_cords = obj_vtx_rotated[p]
        #             o_file_seg.write('"{}","{}","{}","{}","{}","{}"\n'.format(SID, PID, p, p_cords[0], p_cords[1], p_cords[2]))
        #         SID += 1
        #     o_file_seg.close()
        #
        #

        self.setResult(s_file)


# creator function
def frontVerticesCreator():
    return OpenMayaMPx.asMPxPtr(frontVertices())


# syntax creator function
def frontVerticesSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("f", "file", om.MSyntax.kString)
    syntax.addFlag("t", "threshold", om.MSyntax.kDouble)
    syntax.addFlag("c", "cosinusmax", om.MSyntax.kDouble)
    syntax.addFlag("l", "local", om.MSyntax.kBoolean)
    syntax.addFlag("n", "negative", om.MSyntax.kDouble)
    syntax.addFlag("tsf", "triangleSegments", om.MSyntax.kString)
    #syntax.addFlag("sf", "segmentsFile", om.MSyntax.kString)
    # syntax.addFlag("ef", "edgeFile", om.MSyntax.kString)
    # syntax.addFlag("oef", "outerEdgeFile", om.MSyntax.kString)
    # syntax.addFlag("tf", "triangleFile", om.MSyntax.kString)
    # syntax.addFlag("otf", "outerTrianlgeFile", om.MSyntax.kString)
    return syntax


# create button for shelf
def addButton(parentShelf):
    pass
